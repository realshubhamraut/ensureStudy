"""
Tutor API Endpoints
"""
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from app.utils.auth import verify_jwt_token
from app.agents.tutor_agent import TutorAgent

router = APIRouter(prefix="/api/tutor", tags=["Tutor"])


class ChatMessage(BaseModel):
    """Chat message"""
    role: str  # user, assistant
    content: str


class ChatRequest(BaseModel):
    """Chat request"""
    message: str = Field(..., min_length=1, max_length=2000)
    session_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None


class ChatResponse(BaseModel):
    """Chat response"""
    answer: str
    sources: List[Dict[str, Any]]
    session_id: str
    was_moderated: bool
    moderation_reason: Optional[str] = None


# Initialize agent
tutor_agent = TutorAgent()


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    user_id: str = Depends(verify_jwt_token)
):
    """
    Send a message to the AI tutor.
    
    Messages are moderated for academic content and answered with RAG.
    """
    try:
        result = await tutor_agent.execute({
            "query": request.message,
            "user_id": user_id,
            "session_id": request.session_id,
            "student_context": request.context or {}
        })
        
        data = result.get("data", {})
        
        return ChatResponse(
            answer=data.get("answer", ""),
            sources=data.get("sources", []),
            session_id=request.session_id or "new",
            was_moderated=data.get("blocked", False),
            moderation_reason=data.get("reason") if data.get("blocked") else None
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/explain")
async def explain_concept(
    topic: str,
    subject: str,
    difficulty: str = "medium",
    user_id: str = Depends(verify_jwt_token)
):
    """
    Get a detailed explanation of a concept.
    """
    try:
        from app.rag.retriever import get_retriever
        
        retriever = get_retriever()
        
        query = f"Explain {topic} in {subject} at {difficulty} level"
        result = retriever.answer_with_rag(
            query=query,
            top_k=7,
            subject_filter=subject
        )
        
        return {
            "topic": topic,
            "subject": subject,
            "difficulty": difficulty,
            "explanation": result.get("answer", ""),
            "sources": result.get("sources", [])
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/quick-answer")
async def quick_answer(
    question: str,
    user_id: str = Depends(verify_jwt_token)
):
    """
    Get a quick answer without full RAG pipeline.
    Uses smaller context for faster response.
    """
    try:
        from app.rag.retriever import get_retriever
        
        retriever = get_retriever()
        
        result = retriever.answer_with_rag(
            query=question,
            top_k=3  # Fewer chunks for faster response
        )
        
        return {
            "question": question,
            "answer": result.get("answer", ""),
            "source_count": result.get("context_used", 0)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
