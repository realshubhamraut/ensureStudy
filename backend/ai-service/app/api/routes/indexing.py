"""
Material Indexing API Routes
Handles indexing of classroom materials for RAG.
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional
import logging
import httpx

from ...services.material_indexer import get_material_indexer

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/index", tags=["indexing"])


class MaterialIndexRequest(BaseModel):
    """Request to index a classroom material"""
    material_id: str
    file_url: str
    classroom_id: str
    subject: Optional[str] = None
    document_title: Optional[str] = None
    uploaded_by: Optional[str] = None


class MaterialIndexResponse(BaseModel):
    """Response from indexing request"""
    success: bool
    material_id: str
    message: str
    chunks_indexed: Optional[int] = None
    error: Optional[str] = None


class UpdateStatusRequest(BaseModel):
    """Request to update material status in core service"""
    material_id: str
    status: str  # processing, completed, failed
    chunk_count: Optional[int] = None
    error: Optional[str] = None


async def update_material_status(
    material_id: str,
    status: str,
    chunk_count: int = 0,
    error: Optional[str] = None
):
    """Update material indexing status in core service."""
    import os
    core_service_url = os.getenv("CORE_SERVICE_URL", "http://localhost:9000")
    try:
        async with httpx.AsyncClient() as client:
            response = await client.put(
                f"{core_service_url}/api/classroom/materials/{material_id}/status",
                json={
                    "indexing_status": status,
                    "chunk_count": chunk_count,
                    "indexing_error": error
                },
                timeout=10.0
            )
            if response.status_code != 200:
                logger.warning(f"[INDEX] Failed to update status: {response.status_code}")
    except Exception as e:
        logger.error(f"[INDEX] Status update error: {e}")


async def process_material_background(request: MaterialIndexRequest):
    """Background task to index material."""
    logger.info(f"[INDEX] Background indexing started for: {request.material_id}")
    
    # Update status to processing
    await update_material_status(request.material_id, "processing")
    
    try:
        indexer = get_material_indexer()
        result = await indexer.index_material(
            material_id=request.material_id,
            file_url=request.file_url,
            classroom_id=request.classroom_id,
            subject=request.subject,
            document_title=request.document_title,
            uploaded_by=request.uploaded_by
        )
        
        if result.success:
            await update_material_status(
                request.material_id,
                "completed",
                chunk_count=result.chunks_indexed
            )
            logger.info(f"[INDEX] ✅ Completed: {result.chunks_indexed} chunks indexed")
        else:
            await update_material_status(
                request.material_id,
                "failed",
                error=result.error
            )
            logger.error(f"[INDEX] ❌ Failed: {result.error}")
            
    except Exception as e:
        logger.error(f"[INDEX] ❌ Exception: {e}")
        await update_material_status(request.material_id, "failed", error=str(e))


@router.post("/material", response_model=MaterialIndexResponse)
async def index_material(
    request: MaterialIndexRequest,
    background_tasks: BackgroundTasks
):
    """
    Queue a classroom material for indexing.
    
    The indexing happens in the background. Check material status
    for completion.
    """
    logger.info(f"[INDEX] Received indexing request for: {request.material_id}")
    logger.info(f"[INDEX] Classroom: {request.classroom_id}, Subject: {request.subject}")
    
    # Add to background tasks
    background_tasks.add_task(process_material_background, request)
    
    return MaterialIndexResponse(
        success=True,
        material_id=request.material_id,
        message="Material queued for indexing"
    )


@router.post("/material/sync", response_model=MaterialIndexResponse)
async def index_material_sync(request: MaterialIndexRequest):
    """
    Index a classroom material synchronously (for testing).
    Waits for indexing to complete before responding.
    """
    logger.info(f"[INDEX] Sync indexing request for: {request.material_id}")
    
    try:
        indexer = get_material_indexer()
        result = await indexer.index_material(
            material_id=request.material_id,
            file_url=request.file_url,
            classroom_id=request.classroom_id,
            subject=request.subject,
            document_title=request.document_title,
            uploaded_by=request.uploaded_by
        )
        
        if result.success:
            return MaterialIndexResponse(
                success=True,
                material_id=request.material_id,
                message=f"Indexed {result.chunks_indexed} chunks in {result.processing_time_ms}ms",
                chunks_indexed=result.chunks_indexed
            )
        else:
            return MaterialIndexResponse(
                success=False,
                material_id=request.material_id,
                message="Indexing failed",
                error=result.error
            )
            
    except Exception as e:
        logger.error(f"[INDEX] Sync indexing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/material/{material_id}")
async def delete_material_index(material_id: str):
    """
    Delete all indexed chunks for a material.
    """
    try:
        indexer = get_material_indexer()
        success = await indexer.delete_material(material_id)
        
        return {
            "success": success,
            "material_id": material_id,
            "message": "Material index deleted" if success else "Delete failed"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/search/{classroom_id}")
async def search_classroom(
    classroom_id: str,
    query: str,
    top_k: int = 5
):
    """
    Search indexed materials for a classroom.
    """
    try:
        indexer = get_material_indexer()
        results = indexer.search_classroom_materials(
            query=query,
            classroom_id=classroom_id,
            top_k=top_k
        )
        
        return {
            "success": True,
            "classroom_id": classroom_id,
            "query": query,
            "results": results,
            "count": len(results)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
