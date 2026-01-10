"""
Document API Routes
Handles document upload, status checking, and sidebar queries.
"""
import os
import uuid
import logging
from datetime import datetime
from typing import Optional, List

from fastapi import APIRouter, HTTPException, UploadFile, File, Depends, Query
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["documents"])


# ============================================================
# Request/Response Models
# ============================================================

class DocumentUploadResponse(BaseModel):
    """Response after document upload."""
    doc_id: str
    status: str
    status_url: str
    filename: str
    file_size: int


class DocumentStatusResponse(BaseModel):
    """Document processing status."""
    doc_id: str
    status: str  # uploaded, processing, indexed, error
    requires_manual_review: bool = False
    error_message: Optional[str] = None
    page_count: int = 0
    chunk_count: int = 0


class SidebarMatch(BaseModel):
    """Single match for sidebar display."""
    chunk_id: str
    page_number: int
    bbox: Optional[List[int]] = None  # [x1, y1, x2, y2]
    text_snippet: str
    similarity: float
    ocr_confidence: Optional[float] = None


class SidebarResponse(BaseModel):
    """Sidebar API response."""
    doc_id: str
    title: str
    top_matches: List[SidebarMatch]
    preview_summary: str
    pdf_url: str
    version: int = 1


# ============================================================
# API Endpoints
# ============================================================

@router.post(
    "/classrooms/{class_id}/materials/upload",
    response_model=DocumentUploadResponse,
    summary="Upload classroom material"
)
async def upload_material(
    class_id: str,
    file: UploadFile = File(...),
    title: Optional[str] = None
):
    """
    Upload a PDF or image file as classroom material.
    
    - Stores file to S3/MinIO
    - Creates document record in Postgres
    - Enqueues background processing job
    - Returns document ID and status URL
    """
    # Validate file type
    allowed_types = ['application/pdf', 'image/png', 'image/jpeg', 'image/jpg']
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Allowed: {allowed_types}"
        )
    
    # Generate document ID
    doc_id = str(uuid.uuid4())
    
    # Read file content
    file_content = await file.read()
    file_size = len(file_content)
    
    # Use filename as title if not provided
    doc_title = title or file.filename or "Untitled Document"
    
    try:
        # Store to S3
        from ..services.s3_storage import get_storage_service
        storage = get_storage_service()
        
        upload_result = storage.upload_file(
            file_content=file_content,
            class_id=class_id,
            doc_id=doc_id,
            filename=file.filename,
            content_type=file.content_type
        )
        
        if not upload_result.success:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to store file: {upload_result.error}"
            )
        
        # Create document record via Core Service
        import httpx
        core_url = os.getenv('CORE_SERVICE_URL', 'http://localhost:9000')
        
        doc_data = {
            'id': doc_id,
            'class_id': class_id,
            'title': doc_title,
            'filename': file.filename,
            's3_path': upload_result.s3_path,
            'file_hash': upload_result.file_hash,
            'file_size': file_size,
            'mime_type': file.content_type,
            'uploaded_by': 'system',  # TODO: Get from auth
            'status': 'uploaded'
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{core_url}/api/internal/documents",
                json=doc_data,
                timeout=10.0
            )
            
            if response.status_code not in [200, 201]:
                logger.warning(f"Failed to create document record: {response.text}")
        
        # Enqueue background processing job
        try:
            from ..workers.document_tasks import process_document
            process_document.delay(doc_id)
            logger.info(f"[UPLOAD] Enqueued processing job for doc:{doc_id}")
        except Exception as e:
            logger.warning(f"[UPLOAD] Failed to enqueue job (Celery not running?): {e}")
            # Continue anyway - can process manually later
        
        logger.info(
            f"[INGEST][doc:{doc_id}] uploaded_by=system "
            f"saved {upload_result.s3_path} file_hash={upload_result.file_hash[:8]}..."
        )
        
        return DocumentUploadResponse(
            doc_id=doc_id,
            status='uploaded',
            status_url=f"/api/classrooms/{class_id}/materials/{doc_id}/status",
            filename=file.filename,
            file_size=file_size
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[UPLOAD] Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/classrooms/{class_id}/materials/{doc_id}/status",
    response_model=DocumentStatusResponse,
    summary="Get document processing status"
)
async def get_document_status(class_id: str, doc_id: str):
    """
    Get the current processing status of a document.
    
    Status values:
    - uploaded: File stored, waiting for processing
    - processing: OCR/indexing in progress
    - indexed: Successfully processed and searchable
    - error: Processing failed
    """
    try:
        import httpx
        core_url = os.getenv('CORE_SERVICE_URL', 'http://localhost:9000')
        
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{core_url}/api/internal/documents/{doc_id}",
                timeout=10.0
            )
            
            if response.status_code == 404:
                raise HTTPException(status_code=404, detail="Document not found")
            
            if response.status_code != 200:
                raise HTTPException(status_code=500, detail="Failed to fetch document status")
            
            doc = response.json()
            
            return DocumentStatusResponse(
                doc_id=doc_id,
                status=doc.get('status', 'unknown'),
                requires_manual_review=doc.get('requires_manual_review', False),
                error_message=doc.get('error_message'),
                page_count=doc.get('page_count', 0),
                chunk_count=doc.get('chunk_count', 0)
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[STATUS] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/ai-tutor/documents/{doc_id}/sidebar",
    response_model=SidebarResponse,
    summary="Get document sidebar matches"
)
async def get_document_sidebar(
    doc_id: str,
    query: Optional[str] = Query(None, description="Search query"),
    top_k: int = Query(5, ge=1, le=20, description="Number of matches to return")
):
    """
    Get matched snippets from a document for the AI Tutor sidebar.
    
    If query is provided:
    - Embeds query and retrieves top_k chunks from Qdrant
    - Returns matches with page numbers and bounding boxes
    
    If no query:
    - Returns first top_k chunks from the document
    """
    try:
        import httpx
        core_url = os.getenv('CORE_SERVICE_URL', 'http://localhost:9000')
        
        # Get document info
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{core_url}/api/internal/documents/{doc_id}",
                timeout=10.0
            )
            
            if response.status_code == 404:
                raise HTTPException(status_code=404, detail="Document not found")
            
            doc = response.json()
        
        # Get signed PDF URL
        from ..services.s3_storage import get_storage_service
        storage = get_storage_service()
        pdf_url = storage.get_presigned_url(doc.get('s3_path', '')) or ""
        
        matches = []
        
        if query:
            # Search with query
            matches = await _search_document_chunks(doc_id, query, top_k)
        else:
            # Get first chunks
            matches = await _get_document_chunks(doc_id, top_k)
        
        # Generate summary (deterministic, not LLM)
        summary = _generate_summary(doc.get('title', ''), len(matches))
        
        return SidebarResponse(
            doc_id=doc_id,
            title=doc.get('title', 'Document'),
            top_matches=matches,
            preview_summary=summary,
            pdf_url=pdf_url,
            version=doc.get('version', 1)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[SIDEBAR] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# Helper Functions
# ============================================================

async def _search_document_chunks(
    doc_id: str,
    query: str,
    top_k: int
) -> List[SidebarMatch]:
    """Search document chunks using Qdrant."""
    try:
        from ..services.qdrant_service import QdrantService
        from sentence_transformers import SentenceTransformer
        
        # Embed query
        embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        query_embedding = embedder.encode(query).tolist()
        
        # Search Qdrant
        qdrant = QdrantService()
        
        # TODO: Filter by document_id in Qdrant search
        results = qdrant.search(
            query_embedding=query_embedding,
            top_k=top_k,
            filter_conditions={'document_id': doc_id}
        )
        
        matches = []
        for r in results:
            # Parse bbox from url field (hack)
            bbox = None
            if r.payload.get('url', '').startswith('bbox:'):
                try:
                    bbox_str = r.payload['url'].replace('bbox:', '')
                    bbox = eval(bbox_str)  # [x1, y1, x2, y2]
                except:
                    pass
            
            matches.append(SidebarMatch(
                chunk_id=r.payload.get('chunk_id', ''),
                page_number=r.payload.get('page_number', 1),
                bbox=bbox,
                text_snippet=r.payload.get('chunk_text', '')[:200],
                similarity=round(r.score, 3),
                ocr_confidence=r.payload.get('ocr_confidence')
            ))
        
        return matches
        
    except Exception as e:
        logger.error(f"[SEARCH] Error: {e}")
        return []


async def _get_document_chunks(doc_id: str, top_k: int) -> List[SidebarMatch]:
    """Get first N chunks from document."""
    try:
        import httpx
        core_url = os.getenv('CORE_SERVICE_URL', 'http://localhost:9000')
        
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{core_url}/api/internal/documents/{doc_id}/chunks",
                params={'limit': top_k},
                timeout=10.0
            )
            
            if response.status_code != 200:
                return []
            
            chunks = response.json()
            
            return [
                SidebarMatch(
                    chunk_id=c.get('id', ''),
                    page_number=c.get('page_number', 1),
                    bbox=c.get('bbox'),
                    text_snippet=c.get('preview_text', ''),
                    similarity=1.0,
                    ocr_confidence=None
                )
                for c in chunks
            ]
            
    except Exception as e:
        logger.error(f"[GET_CHUNKS] Error: {e}")
        return []


def _generate_summary(title: str, match_count: int) -> str:
    """Generate deterministic summary (no LLM)."""
    if match_count == 0:
        return f"No matches found in '{title}'."
    elif match_count == 1:
        return f"Found 1 relevant section in '{title}'."
    else:
        return f"Found {match_count} relevant sections in '{title}'."
