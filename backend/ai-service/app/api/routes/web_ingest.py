"""
Web Ingest API Routes

POST /api/resources/web - Ingest academic content from web
GET /api/resources/web/sources - Get list of allowed sources
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/resources", tags=["Web Resources"])


# ============================================================================
# Request/Response Schemas
# ============================================================================

class WebIngestRequest(BaseModel):
    """Request to ingest web resources."""
    query: str
    subject: Optional[str] = None
    max_sources: int = 3
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "photosynthesis",
                "subject": "biology",
                "max_sources": 3
            }
        }


class IngestedResourceResponse(BaseModel):
    """A single ingested resource."""
    id: str
    url: str
    title: str
    source_name: str
    source_type: str
    trust_score: float
    clean_content: str
    summary: str
    word_count: int
    chunk_count: int
    fetched_at: str
    stored_in_qdrant: bool
    error: Optional[str] = None


class WebIngestResponse(BaseModel):
    """Response from web ingest."""
    success: bool
    query: str
    resources: List[IngestedResourceResponse]
    total_chunks_stored: int
    processing_time_ms: int
    error: Optional[str] = None


class AllowedSourceResponse(BaseModel):
    """Information about an allowed source."""
    domain: str
    name: str
    trust: float
    type: str


# ============================================================================
# Endpoints
# ============================================================================

@router.post("/web", response_model=WebIngestResponse)
async def ingest_web_resources(request: WebIngestRequest):
    """
    Ingest academic content from the web.
    
    - Searches only whitelisted academic sources
    - Extracts clean educational content
    - Chunks and stores in Qdrant for RAG
    - Returns clean readable content with trust scores
    
    **Allowed Sources:**
    - Wikipedia, Khan Academy, NCERT, MIT OpenCourseWare, Byjus, Toppr
    """
    try:
        from app.services.web_ingest_service import ingest_web_resources as do_ingest
        
        logger.info(f"Web ingest request: query='{request.query}', subject={request.subject}")
        
        result = await do_ingest(
            query=request.query,
            subject=request.subject,
            max_sources=request.max_sources
        )
        
        logger.info(f"Web ingest result: {len(result.resources)} resources, {result.total_chunks_stored} chunks stored")
        
        return WebIngestResponse(
            success=result.success,
            query=result.query,
            resources=[
                IngestedResourceResponse(
                    id=r.id,
                    url=r.url,
                    title=r.title,
                    source_name=r.source_name,
                    source_type=r.source_type,
                    trust_score=r.trust_score,
                    clean_content=r.clean_content,
                    summary=r.summary,
                    word_count=r.word_count,
                    chunk_count=r.chunk_count,
                    fetched_at=r.fetched_at,
                    stored_in_qdrant=r.stored_in_qdrant,
                    error=r.error
                )
                for r in result.resources
            ],
            total_chunks_stored=result.total_chunks_stored,
            processing_time_ms=result.processing_time_ms,
            error=result.error
        )
        
    except Exception as e:
        logger.error(f"Web ingest error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/web/sources", response_model=List[AllowedSourceResponse])
async def get_allowed_sources():
    """
    Get list of allowed academic sources for web ingestion.
    
    These are the only domains the web ingest will fetch from.
    """
    from app.services.web_ingest_service import ALLOWED_SOURCES
    
    return [
        AllowedSourceResponse(
            domain=domain,
            name=info["name"],
            trust=info["trust"],
            type=info["type"]
        )
        for domain, info in ALLOWED_SOURCES.items()
        if domain != "en.wikipedia.org"  # Skip duplicate
    ]


@router.get("/web/health")
async def web_ingest_health():
    """Health check for web ingest service."""
    try:
        from duckduckgo_search import DDGS
        from sentence_transformers import SentenceTransformer
        
        return {
            "status": "healthy",
            "ddg_available": True,
            "embeddings_available": True
        }
    except ImportError as e:
        return {
            "status": "degraded",
            "error": str(e)
        }
