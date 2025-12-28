"""
Notes API Endpoints for AI Service
Handles notes processing, search, and embedding management
"""
from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import logging
import asyncio

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/notes", tags=["notes"])


# ==================== Request/Response Models ====================

class ProcessNotesRequest(BaseModel):
    """Request to start notes processing"""
    job_id: str
    # INTEGRATION NOTE: Extended to support document type (PDF, PPTX, DOCX)
    source_type: str  # 'video', 'images', or 'document'
    source_path: str
    files: List[str] = []
    student_id: Optional[str] = None
    classroom_id: Optional[str] = None


class ProcessNotesResponse(BaseModel):
    """Response after starting processing"""
    success: bool
    message: str
    job_id: str


class SearchNotesRequest(BaseModel):
    """Request for semantic search in notes"""
    query: str
    student_id: str
    classroom_id: Optional[str] = None
    limit: int = 10


class SearchResult(BaseModel):
    """A single search result"""
    score: float
    text: str
    page_id: str
    job_id: str
    chunk_index: int


class SearchNotesResponse(BaseModel):
    """Response with search results"""
    results: List[Dict[str, Any]]
    total: int
    search_type: str = "semantic"


class AskQuestionRequest(BaseModel):
    """Request for RAG-based Q&A"""
    question: str
    student_id: str
    classroom_id: Optional[str] = None


class AskQuestionResponse(BaseModel):
    """Response with generated answer"""
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float


# ==================== Endpoints ====================

@router.post("/process", response_model=ProcessNotesResponse)
async def process_notes(
    request: ProcessNotesRequest,
    background_tasks: BackgroundTasks
):
    """
    Start processing a notes upload
    
    Triggers async processing pipeline:
    1. Video frame extraction (if video)
    2. Image enhancement
    3. OCR text extraction
    4. Embedding generation
    5. Vector indexing
    """
    logger.info(f"Starting notes processing for job: {request.job_id}")
    
    # Add to background tasks
    background_tasks.add_task(
        _run_processing_pipeline,
        job_id=request.job_id,
        source_type=request.source_type,
        source_path=request.source_path,
        files=request.files,
        student_id=request.student_id or "",
        classroom_id=request.classroom_id or ""
    )
    
    return ProcessNotesResponse(
        success=True,
        message="Processing started",
        job_id=request.job_id
    )


@router.post("/search", response_model=SearchNotesResponse)
async def search_notes(request: SearchNotesRequest):
    """
    Semantic search across indexed notes
    
    Uses vector similarity search in Qdrant
    """
    try:
        from app.services.notes_embedding import NotesEmbeddingService
        
        embedding_service = NotesEmbeddingService()
        results = embedding_service.search(
            query=request.query,
            student_id=request.student_id,
            classroom_id=request.classroom_id,
            limit=request.limit
        )
        
        return SearchNotesResponse(
            results=results,
            total=len(results),
            search_type="semantic"
        )
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ask", response_model=AskQuestionResponse)
async def ask_question(request: AskQuestionRequest):
    """
    Ask a question about your notes using RAG
    
    Retrieves relevant chunks and generates an answer
    """
    try:
        from app.services.notes_embedding import NotesEmbeddingService, RAGService
        
        embedding_service = NotesEmbeddingService()
        rag_service = RAGService(embedding_service)
        
        result = rag_service.answer_question(
            question=request.question,
            student_id=request.student_id,
            classroom_id=request.classroom_id
        )
        
        return AskQuestionResponse(
            answer=result["answer"],
            sources=result["sources"],
            confidence=result["confidence"]
        )
        
    except Exception as e:
        logger.error(f"Q&A error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/embeddings/{job_id}")
async def delete_embeddings(job_id: str):
    """
    Delete all embeddings for a job
    
    Called when a notes job is deleted
    """
    try:
        from app.services.notes_embedding import NotesEmbeddingService
        
        embedding_service = NotesEmbeddingService()
        deleted = embedding_service.delete_job_embeddings(job_id)
        
        return {
            "success": True,
            "job_id": job_id,
            "deleted_count": deleted
        }
        
    except Exception as e:
        logger.error(f"Delete embeddings error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """Check if notes processing services are available"""
    services_status = {}
    
    # Check OCR
    try:
        from app.services.ocr_service import OCRService
        ocr = OCRService()
        services_status["ocr"] = "available" if ocr.model is not None or ocr.tesseract_available else "unavailable"
    except:
        services_status["ocr"] = "error"
    
    # Check embeddings
    try:
        from app.services.notes_embedding import NotesEmbeddingService
        emb = NotesEmbeddingService()
        services_status["embeddings"] = "available" if emb.embedder is not None else "unavailable"
        services_status["vector_db"] = "available" if emb.qdrant_client is not None else "unavailable"
    except:
        services_status["embeddings"] = "error"
        services_status["vector_db"] = "error"
    
    # Check video processor
    try:
        from app.services.video_processor import VideoProcessor
        VideoProcessor()
        services_status["video_processor"] = "available"
    except:
        services_status["video_processor"] = "error"
    
    return {
        "status": "healthy" if all(v == "available" for v in services_status.values()) else "degraded",
        "services": services_status
    }


# ==================== Background Processing ====================

async def _run_processing_pipeline(
    job_id: str,
    source_type: str,
    source_path: str,
    files: List[str],
    student_id: str,
    classroom_id: str
):
    """
    Run the complete notes processing pipeline in background
    """
    try:
        from app.agents.notes_agent import process_notes_job
        
        await process_notes_job({
            "job_id": job_id,
            "source_type": source_type,
            "source_path": source_path,
            "files": files,
            "student_id": student_id,
            "classroom_id": classroom_id
        })
        
    except Exception as e:
        logger.error(f"Processing pipeline error for job {job_id}: {e}")
        
        # Try to update job status to failed
        try:
            import httpx
            import os
            
            core_url = os.getenv("CORE_SERVICE_URL", "http://localhost:8000")
            api_key = os.getenv("INTERNAL_API_KEY", "dev-internal-key")
            
            async with httpx.AsyncClient() as client:
                await client.post(
                    f"{core_url}/api/notes/internal/update-job",
                    json={
                        "job_id": job_id,
                        "status": "failed",
                        "error_message": str(e)
                    },
                    headers={"X-Internal-API-Key": api_key}
                )
        except:
            pass
