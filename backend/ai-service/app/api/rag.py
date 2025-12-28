"""
RAG API Endpoints
"""
from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field
from typing import List, Optional
from app.rag.retriever import get_retriever
from app.utils.auth import verify_jwt_token

router = APIRouter(prefix="/api/rag", tags=["RAG"])


class Source(BaseModel):
    """Source document metadata"""
    id: int
    text: str
    source: str
    page: int
    subject: str
    topic: str
    similarity_score: float


class QueryRequest(BaseModel):
    """RAG query request"""
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: Optional[int] = Field(5, ge=1, le=20)
    subject_filter: Optional[str] = None


class RAGResponse(BaseModel):
    """RAG query response"""
    answer: str
    sources: List[Source]
    query: str
    context_used: int


class RetrieveRequest(BaseModel):
    """Retrieve-only request"""
    query: str = Field(..., min_length=1)
    top_k: Optional[int] = Field(5, ge=1, le=20)
    subject_filter: Optional[str] = None
    difficulty_filter: Optional[str] = None
    score_threshold: Optional[float] = Field(0.5, ge=0.0, le=1.0)


@router.post("/query", response_model=RAGResponse)
async def query_rag(
    request: QueryRequest,
    user_id: str = Depends(verify_jwt_token)
):
    """
    Query the RAG pipeline.
    
    Retrieves relevant documents and generates a cited answer.
    """
    try:
        retriever = get_retriever()
        
        # Get student context (could fetch from DB in production)
        student_context = {
            "user_id": user_id,
            "weak_topics": [],  # Would fetch from progress table
            "confidence_score": 50
        }
        
        result = retriever.answer_with_rag(
            query=request.query,
            student_context=student_context,
            top_k=request.top_k or 5,
            subject_filter=request.subject_filter
        )
        
        # Format sources for response
        sources = [
            Source(
                id=s.get("id", 0),
                text=s.get("text", "")[:500],  # Truncate for response
                source=s.get("source", "unknown"),
                page=s.get("page", 0),
                subject=s.get("subject", "general"),
                topic=s.get("topic", ""),
                similarity_score=s.get("similarity_score", 0.0)
            )
            for s in result.get("sources", [])
        ]
        
        return RAGResponse(
            answer=result.get("answer", ""),
            sources=sources,
            query=result.get("query", request.query),
            context_used=result.get("context_used", 0)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/retrieve")
async def retrieve_chunks(
    request: RetrieveRequest,
    user_id: str = Depends(verify_jwt_token)
):
    """
    Retrieve relevant chunks without generating an answer.
    
    Useful for debugging or when you want to process chunks yourself.
    """
    try:
        retriever = get_retriever()
        
        chunks = retriever.retrieve_chunks(
            query=request.query,
            top_k=request.top_k or 5,
            subject_filter=request.subject_filter,
            difficulty_filter=request.difficulty_filter,
            score_threshold=request.score_threshold or 0.5
        )
        
        return {
            "query": request.query,
            "chunks": chunks,
            "count": len(chunks)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/search")
async def simple_search(
    q: str = Query(..., min_length=1, description="Search query"),
    top_k: int = Query(5, ge=1, le=20),
    subject: Optional[str] = Query(None),
    user_id: str = Depends(verify_jwt_token)
):
    """
    Simple search endpoint for quick queries.
    """
    try:
        retriever = get_retriever()
        
        chunks = retriever.retrieve_chunks(
            query=q,
            top_k=top_k,
            subject_filter=subject
        )
        
        return {
            "query": q,
            "results": chunks,
            "count": len(chunks)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/collection-info")
async def get_collection_info(
    user_id: str = Depends(verify_jwt_token)
):
    """
    Get information about the Qdrant collection.
    """
    from app.rag.qdrant_setup import get_collection_info
    
    return get_collection_info()
