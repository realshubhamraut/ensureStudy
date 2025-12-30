"""
Document Internal API Routes for Core Service
Internal endpoints used by AI Service for document management.
"""
from flask import Blueprint, request, jsonify
from app import db
from app.models.document import Document, DocumentPage, DocumentChunk, DocumentQualityReport

documents_bp = Blueprint('documents', __name__, url_prefix='/api/internal/documents')


@documents_bp.route('', methods=['POST'])
def create_document():
    """Create a new document record."""
    data = request.get_json()
    
    doc = Document(
        id=data.get('id'),
        class_id=data['class_id'],
        title=data['title'],
        filename=data['filename'],
        s3_path=data['s3_path'],
        file_hash=data['file_hash'],
        file_size=data.get('file_size'),
        mime_type=data.get('mime_type'),
        uploaded_by=data['uploaded_by'],
        status=data.get('status', 'uploaded')
    )
    
    db.session.add(doc)
    db.session.commit()
    
    return jsonify(doc.to_dict()), 201


@documents_bp.route('/<doc_id>', methods=['GET'])
def get_document(doc_id):
    """Get document by ID."""
    doc = Document.query.get(doc_id)
    if not doc:
        return jsonify({'error': 'Document not found'}), 404
    
    return jsonify(doc.to_dict())


@documents_bp.route('/<doc_id>/status', methods=['PATCH'])
def update_document_status(doc_id):
    """Update document status."""
    doc = Document.query.get(doc_id)
    if not doc:
        return jsonify({'error': 'Document not found'}), 404
    
    data = request.get_json()
    
    if 'status' in data:
        doc.status = data['status']
    if 'requires_manual_review' in data:
        doc.requires_manual_review = data['requires_manual_review']
    if 'error_message' in data:
        doc.error_message = data['error_message']
    
    db.session.commit()
    
    return jsonify(doc.to_status_dict())


@documents_bp.route('/<doc_id>/pages', methods=['POST'])
def create_page(doc_id):
    """Create a document page record."""
    doc = Document.query.get(doc_id)
    if not doc:
        return jsonify({'error': 'Document not found'}), 404
    
    data = request.get_json()
    
    page = DocumentPage(
        document_id=doc_id,
        page_number=data['page_number'],
        s3_page_json_path=data.get('s3_page_json_path'),
        s3_page_image_path=data.get('s3_page_image_path'),
        ocr_confidence=data.get('ocr_confidence'),
        text_length=data.get('text_length', 0),
        block_count=data.get('block_count', 0),
        ocr_method=data.get('ocr_method')
    )
    
    db.session.add(page)
    db.session.commit()
    
    return jsonify(page.to_dict()), 201


@documents_bp.route('/<doc_id>/chunks', methods=['GET'])
def get_chunks(doc_id):
    """Get document chunks."""
    doc = Document.query.get(doc_id)
    if not doc:
        return jsonify({'error': 'Document not found'}), 404
    
    limit = request.args.get('limit', 10, type=int)
    chunks = DocumentChunk.query.filter_by(document_id=doc_id).limit(limit).all()
    
    return jsonify([c.to_dict() for c in chunks])


@documents_bp.route('/<doc_id>/chunks', methods=['POST'])
def create_chunk(doc_id):
    """Create a document chunk record."""
    data = request.get_json()
    
    chunk = DocumentChunk(
        document_id=doc_id,
        page_number=data['page_number'],
        block_id=data.get('block_id'),
        chunk_index=data['chunk_index'],
        preview_text=data.get('preview_text'),
        full_text=data.get('full_text'),
        bbox_json=data.get('bbox_json'),
        qdrant_id=data.get('qdrant_id'),
        token_count=data.get('token_count'),
        content_type=data.get('content_type', 'text'),
        embedding_hash=data.get('embedding_hash')
    )
    
    db.session.add(chunk)
    db.session.commit()
    
    return jsonify(chunk.to_dict()), 201


@documents_bp.route('/<doc_id>/quality-report', methods=['POST'])
def create_quality_report(doc_id):
    """Create document quality report."""
    data = request.get_json()
    
    # Delete existing report if any
    existing = DocumentQualityReport.query.filter_by(document_id=doc_id).first()
    if existing:
        db.session.delete(existing)
    
    report = DocumentQualityReport(
        document_id=doc_id,
        avg_ocr_confidence=data.get('avg_ocr_confidence'),
        min_ocr_confidence=data.get('min_ocr_confidence'),
        pages_processed=data.get('pages_processed', 0),
        pages_failed=data.get('pages_failed', 0),
        flagged_pages=data.get('flagged_pages'),
        total_chunks=data.get('total_chunks', 0),
        total_tokens=data.get('total_tokens', 0),
        processing_time_ms=data.get('processing_time_ms')
    )
    
    db.session.add(report)
    db.session.commit()
    
    return jsonify(report.to_dict()), 201


@documents_bp.route('/<doc_id>/quality-report', methods=['GET'])
def get_quality_report(doc_id):
    """Get document quality report."""
    report = DocumentQualityReport.query.filter_by(document_id=doc_id).first()
    if not report:
        return jsonify({'error': 'Quality report not found'}), 404
    
    return jsonify(report.to_dict())


@documents_bp.route('/by-class/<class_id>', methods=['GET'])
def get_documents_by_class(class_id):
    """Get all documents for a classroom."""
    status = request.args.get('status')
    
    query = Document.query.filter_by(class_id=class_id)
    if status:
        query = query.filter_by(status=status)
    
    docs = query.order_by(Document.uploaded_at.desc()).all()
    
    return jsonify([d.to_dict() for d in docs])
