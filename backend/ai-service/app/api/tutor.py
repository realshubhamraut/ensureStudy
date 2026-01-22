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
    """Chat response with TAL/ABCR metadata"""
    answer: str
    sources: List[Dict[str, Any]]
    session_id: str
    was_moderated: bool
    moderation_reason: Optional[str] = None
    
    # TAL - Topic Anchor
    topic_anchor: Optional[Dict[str, Any]] = None  # {id, title}
    
    # ABCR - Context Routing
    is_followup: bool = False
    abcr_confidence: float = 0.0
    confirm_new_topic: bool = False
    
    # MCP - Context Sources
    context_sources: List[str] = []  # ['anchor', 'classroom', 'curated', 'web']
    anchor_hits: int = 0
    web_filtered_count: int = 0


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
    Returns TAL/ABCR/MCP metadata for frontend integration.
    """
    try:
        result = await tutor_agent.execute({
            "query": request.message,
            "user_id": user_id,
            "session_id": request.session_id,
            "classroom_id": request.context.get("classroom_id", "") if request.context else "",
            "clicked_suggestion": request.context.get("clicked_suggestion", False) if request.context else False,
        })
        
        data = result.get("data", {})
        session_id = result.get("session_id", request.session_id or "new")
        
        return ChatResponse(
            answer=data.get("answer", ""),
            sources=data.get("sources", []),
            session_id=session_id,
            was_moderated=data.get("blocked", False),
            moderation_reason=data.get("reason") if data.get("blocked") else None,
            
            # TAL
            topic_anchor=data.get("topic_anchor"),
            
            # ABCR
            is_followup=data.get("is_followup", False),
            abcr_confidence=data.get("abcr_confidence", 0.0),
            confirm_new_topic=data.get("confirm_new_topic", False),
            
            # MCP
            context_sources=data.get("context_sources", []),
            anchor_hits=data.get("anchor_hits", 0),
            web_filtered_count=data.get("web_filtered_count", 0),
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
