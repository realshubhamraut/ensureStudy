"""
Notes Routes - Upload, process, and search digitized notes
"""
from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
from app import db
from app.models.user import User
from app.models.classroom import Classroom, StudentClassroom
from app.models.notes import NoteProcessingJob, DigitizedNotePage, NoteEmbedding, NoteSearchHistory
from app.utils.jwt_handler import verify_token
from app.utils.storage import get_storage_service
from datetime import datetime
import os
import uuid
import requests

notes_bp = Blueprint("notes", __name__, url_prefix="/api/notes")

# AI Service URL
AI_SERVICE_URL = os.getenv("AI_SERVICE_URL", "http://localhost:8001")

# Storage service (handles S3 and local)
STORAGE = get_storage_service()

# INTEGRATION NOTE: Extended to support document file types for classroom materials
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'mov', 'avi', 'webm', 'mkv'}
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'heic'}
ALLOWED_DOCUMENT_EXTENSIONS = {'pdf', 'pptx', 'docx', 'doc'}  # NEW: Document types
ALL_ALLOWED_EXTENSIONS = ALLOWED_VIDEO_EXTENSIONS | ALLOWED_IMAGE_EXTENSIONS | ALLOWED_DOCUMENT_EXTENSIONS
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB


def get_current_user():
    """Get current user from JWT token"""
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        return None
    token = auth_header.split(" ")[1]
    try:
        payload = verify_token(token)
        if not payload:
            return None
        return User.query.get(payload.get("user_id"))
    except Exception as e:
        print(f"[Notes Auth] Token verification failed: {e}")
        return None


def student_required(f):
    """Decorator to require student role"""
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        user = get_current_user()
        if not user:
            return jsonify({"error": "Authentication required"}), 401
        if user.role != "student":
            return jsonify({"error": "Student access required"}), 403
        return f(user, *args, **kwargs)
    return decorated



def allowed_file(filename, allowed_extensions):
    """Check if file has allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions


# ==================== Presigned URL for Direct Upload ====================

@notes_bp.route("/presigned-upload", methods=["POST"])
@student_required
def get_presigned_upload_url(user):
    """
    Get a presigned URL for direct S3 upload from browser.
    Returns upload URL and form fields for multipart upload.
    Falls back to regular upload endpoint if S3 not available.
    """
    try:
        data = request.get_json()
        filename = data.get("filename")
        content_type = data.get("content_type", "application/octet-stream")
        classroom_id = data.get("classroom_id")
        title = data.get("title", "Untitled Notes")
        
        if not filename or not classroom_id:
            return jsonify({"error": "filename and classroom_id are required"}), 400
        
        # Create job record first
        job_id = str(uuid.uuid4())
        ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
        
        # INTEGRATION NOTE: Extended to support document file types
        if ext in ALLOWED_VIDEO_EXTENSIONS:
            source_type = 'video'
        elif ext in ALLOWED_IMAGE_EXTENSIONS:
            source_type = 'images'
        elif ext in ALLOWED_DOCUMENT_EXTENSIONS:
            source_type = 'document'
        else:
            return jsonify({"error": f"File type not allowed. Allowed: {ALL_ALLOWED_EXTENSIONS}"}), 400
        
        # Try to get presigned URL
        presigned = STORAGE.generate_presigned_upload_url(
            job_id=job_id,
            filename=filename,
            content_type=content_type
        )
        
        if presigned:
            # Create pending job
            job = NoteProcessingJob(
                id=job_id,
                student_id=user.id,
                classroom_id=classroom_id,
                title=title,
                source_type=source_type,
                source_url=f"s3://{STORAGE.bucket_name}/{presigned['file_key']}",
                source_filename=filename,
                status='awaiting_upload',
                current_step='Waiting for file upload'
            )
            db.session.add(job)
            db.session.commit()
            
            return jsonify({
                "use_presigned": True,
                "job_id": job_id,
                "upload_url": presigned["url"],
                "upload_fields": presigned["fields"],
                "file_key": presigned["file_key"],
                "confirm_url": f"/api/notes/presigned-confirm/{job_id}"
            })
        else:
            # Fallback: instruct client to use regular upload
            return jsonify({
                "use_presigned": False,
                "message": "Use regular /api/notes/upload endpoint",
                "upload_url": "/api/notes/upload"
            })
            
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500


@notes_bp.route("/presigned-confirm/<job_id>", methods=["POST"])
@student_required
def confirm_presigned_upload(user, job_id):
    """
    Confirm that a presigned upload completed successfully.
    This triggers the processing pipeline.
    """
    try:
        job = NoteProcessingJob.query.filter_by(id=job_id, student_id=user.id).first()
        
        if not job:
            return jsonify({"error": "Job not found"}), 404
        
        if job.status != 'awaiting_upload':
            return jsonify({"error": "Job already confirmed"}), 400
        
        data = request.get_json() or {}
        file_size = data.get("file_size", 0)
        
        # Update job status
        job.status = 'pending'
        job.source_size_bytes = file_size
        job.current_step = 'Uploaded, waiting for processing'
        db.session.commit()
        
        # Trigger AI service processing
        try:
            requests.post(
                f"{AI_SERVICE_URL}/api/notes/process",
                json={
                    "job_id": job_id,
                    "source_type": job.source_type,
                    "source_url": job.source_url,
                    "storage_type": "s3"
                },
                timeout=5
            )
        except Exception as e:
            print(f"Could not notify AI service: {e}")
        
        # INTEGRATION NOTE: Also publish to Kafka for the document processing pipeline
        # This enables the LangGraph agent to process documents asynchronously
        try:
            from backend.kafka.producers.document_event_producer import publish_document_for_processing
            publish_document_for_processing(
                document_id=job_id,
                student_id=user.id,
                classroom_id=job.classroom_id,
                source_url=job.source_url,
                file_type=job.source_type,
                title=job.title
            )
        except Exception as e:
            print(f"Could not publish to Kafka: {e}")
        
        return jsonify({
            "success": True,
            "job": job.to_dict(),
            "message": "Upload confirmed. Processing will begin shortly."
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500


# ==================== Upload & Job Management ====================

@notes_bp.route("/upload", methods=["POST"])
@student_required
def upload_notes(user):
    """
    Upload video or images for notes digitization
    Accepts: multipart/form-data with 'file' or 'files[]'
    """
    try:
        print(f"[Notes Upload] User: {user.id}, Role: {user.role}")
        
        classroom_id = request.form.get("classroom_id")
        title = request.form.get("title", "Untitled Notes")
        description = request.form.get("description", "")
        
        print(f"[Notes Upload] Classroom ID: {classroom_id}, Title: {title}")
        
        if not classroom_id:
            return jsonify({"error": "classroom_id is required"}), 400
        
        # Verify student is in this classroom (or skip for now to allow testing)
        enrollment = StudentClassroom.query.filter_by(
            student_id=user.id,
            classroom_id=classroom_id,
            is_active=True
        ).first()
        
        # For testing: allow upload even without enrollment
        if not enrollment:
            print(f"[Notes Upload] Warning: No enrollment found, allowing anyway for testing")
            # return jsonify({"error": "You are not enrolled in this classroom"}), 403
        
        # Check for file upload
        if 'file' not in request.files and 'files[]' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        
        # Handle single video or multiple images
        files = request.files.getlist('files[]') if 'files[]' in request.files else [request.files['file']]
        
        if not files or files[0].filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Determine source type
        first_file = files[0]
        filename = secure_filename(first_file.filename)
        ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
        
        # INTEGRATION NOTE: Extended to support document file types
        if ext in ALLOWED_VIDEO_EXTENSIONS:
            source_type = 'video'
            if len(files) > 1:
                return jsonify({"error": "Only one video file allowed"}), 400
        elif ext in ALLOWED_IMAGE_EXTENSIONS:
            source_type = 'images'
        elif ext in ALLOWED_DOCUMENT_EXTENSIONS:
            source_type = 'document'
            if len(files) > 1:
                return jsonify({"error": "Only one document file allowed at a time"}), 400
        else:
            return jsonify({"error": f"File type not allowed. Allowed: {ALL_ALLOWED_EXTENSIONS}"}), 400
        
        # Create upload directory (using storage service's local folder)
        job_id = str(uuid.uuid4())
        upload_dir = os.path.join(STORAGE.local_folder, "notes", job_id)
        os.makedirs(upload_dir, exist_ok=True)
        
        # Save files
        saved_files = []
        total_size = 0
        
        for file in files:
            if file and file.filename:
                fname = secure_filename(file.filename)
                file_path = os.path.join(upload_dir, fname)
                file.save(file_path)
                saved_files.append(file_path)
                total_size += os.path.getsize(file_path)
        
        if total_size > MAX_FILE_SIZE:
            # Clean up
            for f in saved_files:
                os.remove(f)
            return jsonify({"error": f"Total file size exceeds {MAX_FILE_SIZE // (1024*1024)}MB limit"}), 400
        
        # Create job record
        job = NoteProcessingJob(
            id=job_id,
            student_id=user.id,
            classroom_id=classroom_id,
            title=title,
            description=description,
            source_type=source_type,
            source_url=upload_dir,
            # INTEGRATION NOTE: Set filename based on source type
            source_filename=filename if source_type in ('video', 'document') else f"{len(saved_files)} images",
            source_size_bytes=total_size,
            status='pending',
            progress_percent=0,
            current_step='Uploaded, waiting for processing'
        )
        
        db.session.add(job)
        db.session.commit()
        
        # Trigger AI service processing (async)
        try:
            requests.post(
                f"{AI_SERVICE_URL}/api/notes/process",
                json={
                    "job_id": job_id,
                    "source_type": source_type,
                    "source_path": upload_dir,
                    "files": saved_files
                },
                timeout=5
            )
        except Exception as e:
            # Processing will be picked up by background worker
            print(f"Could not notify AI service: {e}")
        
        return jsonify({
            "success": True,
            "job": job.to_dict(),
            "message": "Notes uploaded successfully. Processing will begin shortly."
        }), 201
        
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500


@notes_bp.route("/jobs", methods=["GET"])
@student_required
def list_jobs(user):
    """List all processing jobs for current student"""
    try:
        classroom_id = request.args.get("classroom_id")
        status = request.args.get("status")
        
        query = NoteProcessingJob.query.filter_by(student_id=user.id)
        
        if classroom_id:
            query = query.filter_by(classroom_id=classroom_id)
        if status:
            query = query.filter_by(status=status)
        
        jobs = query.order_by(NoteProcessingJob.created_at.desc()).all()
        
        return jsonify({
            "jobs": [j.to_dict() for j in jobs],
            "total": len(jobs)
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@notes_bp.route("/jobs/<job_id>", methods=["GET"])
@student_required
def get_job(user, job_id):
    """Get job details with pages"""
    try:
        job = NoteProcessingJob.query.filter_by(id=job_id, student_id=user.id).first()
        
        if not job:
            return jsonify({"error": "Job not found"}), 404
        
        include_pages = request.args.get("include_pages", "true").lower() == "true"
        
        return jsonify({
            "job": job.to_dict(include_pages=include_pages)
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@notes_bp.route("/jobs/<job_id>/status", methods=["GET"])
@student_required
def get_job_status(user, job_id):
    """Get job processing status (for polling)"""
    try:
        job = NoteProcessingJob.query.filter_by(id=job_id, student_id=user.id).first()
        
        if not job:
            return jsonify({"error": "Job not found"}), 404
        
        return jsonify({
            "id": job.id,
            "status": job.status,
            "progress_percent": job.progress_percent,
            "current_step": job.current_step,
            "total_pages": job.total_pages,
            "processed_pages": job.processed_pages,
            "error_message": job.error_message
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@notes_bp.route("/jobs/<job_id>", methods=["DELETE"])
@student_required
def delete_job(user, job_id):
    """Delete a job and all associated data"""
    try:
        job = NoteProcessingJob.query.filter_by(id=job_id, student_id=user.id).first()
        
        if not job:
            return jsonify({"error": "Job not found"}), 404
        
        # Delete from vector DB (call AI service)
        try:
            requests.delete(
                f"{AI_SERVICE_URL}/api/notes/embeddings/{job_id}",
                timeout=10
            )
        except:
            pass  # Continue even if AI service fails
        
        # Delete files using storage service (handles both S3 and local)
        STORAGE.delete_job_files(job_id)
        
        # Delete from database (cascade deletes pages and embeddings)
        db.session.delete(job)
        db.session.commit()
        
        return jsonify({"success": True, "message": "Job deleted successfully"})
        
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500


# ==================== Pages ====================

@notes_bp.route("/pages/<job_id>", methods=["GET"])
@student_required
def get_pages(user, job_id):
    """Get all pages for a job"""
    try:
        job = NoteProcessingJob.query.filter_by(id=job_id, student_id=user.id).first()
        
        if not job:
            return jsonify({"error": "Job not found"}), 404
        
        pages = DigitizedNotePage.query.filter_by(job_id=job_id)\
            .order_by(DigitizedNotePage.page_number).all()
        
        include_text = request.args.get("include_text", "true").lower() == "true"
        
        return jsonify({
            "pages": [p.to_dict(include_text=include_text) for p in pages],
            "total": len(pages)
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@notes_bp.route("/pages/<job_id>/<int:page_number>", methods=["GET"])
@student_required
def get_page(user, job_id, page_number):
    """Get a specific page"""
    try:
        job = NoteProcessingJob.query.filter_by(id=job_id, student_id=user.id).first()
        
        if not job:
            return jsonify({"error": "Job not found"}), 404
        
        page = DigitizedNotePage.query.filter_by(
            job_id=job_id,
            page_number=page_number
        ).first()
        
        if not page:
            return jsonify({"error": "Page not found"}), 404
        
        return jsonify({"page": page.to_dict()})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@notes_bp.route("/images/<job_id>/<path:image_path>", methods=["GET"])
def serve_image(job_id, image_path):
    """
    Serve an image file from a job's directory
    Supports token via query param for image loading in details modal
    
    URL patterns:
    - /api/notes/images/{job_id}/enhanced/page_001.png
    - /api/notes/images/{job_id}/frames/frame_001.png
    """
    try:
        # Support token via query param
        token = request.args.get("token")
        auth_header = request.headers.get("Authorization")
        
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]
        
        if not token:
            return jsonify({"error": "Authentication required"}), 401
        
        try:
            payload = verify_token(token)
            if not payload:
                return jsonify({"error": "Invalid token"}), 401
            user = User.query.get(payload.get("user_id"))
            if not user:
                return jsonify({"error": "User not found"}), 401
        except Exception:
            return jsonify({"error": "Invalid token"}), 401
        
        # Verify job belongs to user
        job = NoteProcessingJob.query.filter_by(id=job_id, student_id=user.id).first()
        
        if not job:
            return jsonify({"error": "Job not found"}), 404
        
        from flask import send_file
        import mimetypes
        
        # Build full path - job.source_url is the job directory
        job_dir = job.source_url
        full_path = os.path.join(job_dir, image_path)
        
        # Security check - ensure path is within job directory
        full_path = os.path.realpath(full_path)
        job_dir_real = os.path.realpath(job_dir)
        
        if not full_path.startswith(job_dir_real):
            return jsonify({"error": "Invalid path"}), 403
        
        if not os.path.exists(full_path):
            return jsonify({"error": "Image not found"}), 404
        
        # Determine mimetype
        mime_type, _ = mimetypes.guess_type(full_path)
        if not mime_type:
            mime_type = "application/octet-stream"
        
        return send_file(
            full_path,
            mimetype=mime_type,
            as_attachment=False
        )
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@notes_bp.route("/pdf/<job_id>", methods=["GET"])
def get_pdf(job_id):
    """Serve the combined PDF for a job (supports token via query param for iframe)"""
    try:
        # Support token via query param for iframe embedding
        token = request.args.get("token")
        auth_header = request.headers.get("Authorization")
        
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]
        
        if not token:
            return jsonify({"error": "Authentication required"}), 401
        
        try:
            payload = verify_token(token)
            if not payload:
                return jsonify({"error": "Invalid token"}), 401
            user = User.query.get(payload.get("user_id"))
            if not user:
                return jsonify({"error": "User not found"}), 401
        except Exception:
            return jsonify({"error": "Invalid token"}), 401
        
        job = NoteProcessingJob.query.filter_by(id=job_id, student_id=user.id).first()
        
        if not job:
            return jsonify({"error": "Job not found"}), 404
        
        if not job.pdf_path:
            return jsonify({"error": "PDF not generated for this job"}), 404
        
        from flask import send_file
        
        if not os.path.exists(job.pdf_path):
            return jsonify({"error": "PDF file not found on disk"}), 404
        
        return send_file(
            job.pdf_path,
            mimetype='application/pdf',
            as_attachment=False,
            download_name=f"{job.title}.pdf"
        )
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==================== Search ====================

@notes_bp.route("/search", methods=["POST"])
@student_required
def search_notes(user):
    """
    Semantic search across all notes
    Uses RAG for intelligent search
    """
    try:
        data = request.get_json()
        query = data.get("query", "").strip()
        classroom_id = data.get("classroom_id")
        limit = min(data.get("limit", 10), 50)
        
        if not query:
            return jsonify({"error": "Query is required"}), 400
        
        # Log search
        search_log = NoteSearchHistory(
            student_id=user.id,
            classroom_id=classroom_id,
            query=query
        )
        db.session.add(search_log)
        
        # Call AI service for semantic search
        try:
            response = requests.post(
                f"{AI_SERVICE_URL}/api/notes/search",
                json={
                    "query": query,
                    "student_id": user.id,
                    "classroom_id": classroom_id,
                    "limit": limit
                },
                timeout=30
            )
            
            if response.status_code == 200:
                results = response.json()
                search_log.result_count = len(results.get("results", []))
                db.session.commit()
                return jsonify(results)
            else:
                # Fallback to basic text search
                return _fallback_text_search(user.id, classroom_id, query, limit)
                
        except Exception as e:
            print(f"AI service error: {e}")
            return _fallback_text_search(user.id, classroom_id, query, limit)
        
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500


def _fallback_text_search(student_id, classroom_id, query, limit):
    """Fallback to basic SQL text search if AI service unavailable"""
    base_query = db.session.query(DigitizedNotePage).join(NoteProcessingJob)\
        .filter(NoteProcessingJob.student_id == student_id)
    
    if classroom_id:
        base_query = base_query.filter(NoteProcessingJob.classroom_id == classroom_id)
    
    # Simple ILIKE search
    pages = base_query.filter(
        DigitizedNotePage.extracted_text.ilike(f"%{query}%")
    ).limit(limit).all()
    
    return jsonify({
        "results": [
            {
                "page": p.to_dict(),
                "job_id": p.job_id,
                "relevance_score": 0.5,  # Placeholder score for text match
                "match_type": "text"
            }
            for p in pages
        ],
        "total": len(pages),
        "search_type": "fallback_text"
    })


# ==================== Classroom Notes Gallery ====================

@notes_bp.route("/classroom/<classroom_id>", methods=["GET"])
@student_required
def get_classroom_notes(user, classroom_id):
    """Get all notes for a classroom (including other students' if shared)"""
    try:
        print(f"[Notes List] User: {user.id}, Classroom: {classroom_id}")
        
        # Skip enrollment check for testing
        # enrollment = StudentClassroom.query.filter_by(
        #     student_id=user.id,
        #     classroom_id=classroom_id,
        #     is_active=True
        # ).first()
        # 
        # if not enrollment:
        #     return jsonify({"error": "You are not enrolled in this classroom"}), 403
        
        # Get ALL jobs for this user in this classroom (not just completed)
        jobs = NoteProcessingJob.query.filter_by(
            classroom_id=classroom_id,
            student_id=user.id
        ).order_by(NoteProcessingJob.created_at.desc()).all()
        
        print(f"[Notes List] Found {len(jobs)} jobs")
        
        return jsonify({
            "notes": [j.to_dict(include_pages=False) for j in jobs],
            "total": len(jobs)
        })
        
    except Exception as e:
        print(f"[Notes List] Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ==================== Internal API (for AI Service callbacks) ====================

@notes_bp.route("/internal/update-job", methods=["POST"])
def update_job_internal():
    """
    Internal endpoint for AI service to update job status
    Protected by internal API key
    """
    # Simple API key check for internal services
    api_key = request.headers.get("X-Internal-API-Key")
    expected_key = os.getenv("INTERNAL_API_KEY", "dev-internal-key")
    
    if api_key != expected_key:
        return jsonify({"error": "Unauthorized"}), 401
    
    try:
        data = request.get_json()
        job_id = data.get("job_id")
        
        job = NoteProcessingJob.query.get(job_id)
        if not job:
            return jsonify({"error": "Job not found"}), 404
        
        # Update job fields
        if "status" in data:
            job.status = data["status"]
            if data["status"] == "processing" and not job.started_at:
                job.started_at = datetime.utcnow()
            elif data["status"] in ["completed", "failed"]:
                job.completed_at = datetime.utcnow()
        
        if "progress_percent" in data:
            job.progress_percent = data["progress_percent"]
        if "current_step" in data:
            job.current_step = data["current_step"]
        if "total_pages" in data:
            job.total_pages = data["total_pages"]
        if "processed_pages" in data:
            job.processed_pages = data["processed_pages"]
        if "avg_confidence" in data:
            job.avg_confidence = data["avg_confidence"]
        if "error_message" in data:
            job.error_message = data["error_message"]
        if "pdf_path" in data:
            job.pdf_path = data["pdf_path"]
        
        db.session.commit()
        
        return jsonify({"success": True})
        
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500


@notes_bp.route("/internal/add-page", methods=["POST"])
def add_page_internal():
    """Internal endpoint for AI service to add processed pages"""
    api_key = request.headers.get("X-Internal-API-Key")
    expected_key = os.getenv("INTERNAL_API_KEY", "dev-internal-key")
    
    if api_key != expected_key:
        return jsonify({"error": "Unauthorized"}), 401
    
    try:
        data = request.get_json()
        
        page = DigitizedNotePage(
            job_id=data["job_id"],
            page_number=data["page_number"],
            frame_timestamp=data.get("frame_timestamp"),
            original_image_url=data.get("original_image_url"),
            enhanced_image_url=data.get("enhanced_image_url"),
            thumbnail_url=data.get("thumbnail_url"),
            extracted_text=data.get("extracted_text"),
            confidence_score=data.get("confidence_score"),
            brightness=data.get("brightness"),
            contrast=data.get("contrast"),
            sharpness=data.get("sharpness"),
            status=data.get("status", "ocr_done")
        )
        
        db.session.add(page)
        db.session.commit()
        
        return jsonify({"success": True, "page_id": page.id})
        
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500


# ==================== Human Correction ====================

@notes_bp.route("/pages/<job_id>/<int:page_number>/correct", methods=["POST"])
@student_required
def correct_page_text(user, job_id, page_number):
    """
    Human correction endpoint for OCR results.
    
    Accepts corrected text and optionally updates embeddings.
    
    Request body:
    {
        "text": "Corrected full text",
        "ocr_lines": [{"text": "...", "bbox": [x,y,w,h], "confidence": 1.0}, ...],  // optional
        "reembed": true  // optional, trigger re-embedding
    }
    """
    try:
        # Verify job ownership
        job = NoteProcessingJob.query.filter_by(id=job_id, student_id=user.id).first()
        if not job:
            return jsonify({"error": "Job not found"}), 404
        
        # Get page
        page = DigitizedNotePage.query.filter_by(
            job_id=job_id,
            page_number=page_number
        ).first()
        
        if not page:
            return jsonify({"error": "Page not found"}), 404
        
        data = request.get_json()
        corrected_text = data.get("text")
        corrected_lines = data.get("ocr_lines")
        should_reembed = data.get("reembed", False)
        
        if not corrected_text:
            return jsonify({"error": "text field is required"}), 400
        
        # Store original for audit
        original_text = page.extracted_text
        
        # Update page
        page.extracted_text = corrected_text
        page.status = "corrected"  # Mark as human-corrected
        page.confidence_score = 1.0  # Human correction = 100% confidence
        
        # Update ocr_lines if provided
        if corrected_lines:
            page.ocr_lines = corrected_lines
        
        db.session.commit()
        
        # Trigger re-embedding if requested
        if should_reembed:
            try:
                requests.post(
                    f"{AI_SERVICE_URL}/api/notes/reembed",
                    json={
                        "job_id": job_id,
                        "page_id": page.id,
                        "text": corrected_text
                    },
                    timeout=10
                )
            except Exception as e:
                print(f"Could not trigger re-embedding: {e}")
        
        return jsonify({
            "success": True,
            "page": page.to_dict(),
            "original_text_length": len(original_text) if original_text else 0,
            "corrected_text_length": len(corrected_text),
            "reembed_triggered": should_reembed
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500


@notes_bp.route("/pages/<job_id>/<int:page_number>/ocr-lines", methods=["GET"])
@student_required
def get_page_ocr_lines(user, job_id, page_number):
    """Get OCR lines with bounding boxes for a specific page (for correction UI)."""
    try:
        job = NoteProcessingJob.query.filter_by(id=job_id, student_id=user.id).first()
        if not job:
            return jsonify({"error": "Job not found"}), 404
        
        page = DigitizedNotePage.query.filter_by(
            job_id=job_id,
            page_number=page_number
        ).first()
        
        if not page:
            return jsonify({"error": "Page not found"}), 404
        
        return jsonify({
            "page_id": page.id,
            "page_number": page.page_number,
            "extracted_text": page.extracted_text,
            "ocr_lines": page.ocr_lines or [],
            "confidence_score": page.confidence_score,
            "status": page.status
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
