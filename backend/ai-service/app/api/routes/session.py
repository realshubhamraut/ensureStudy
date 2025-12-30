"""
Session API Routes - Session-based query and resource management

Endpoints:
- POST /api/session/create - Create new session
- POST /api/session/{id}/query - Query with session context
- GET /api/session/{id}/resources - Export resource list
- POST /api/resources/append_check - Check dedup before insert
"""
import time
import hashlib
import logging
from typing import Optional, List
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

from ...services.session_service import get_session_service, SessionData, TurnData, ResourceData, AppendResult
from ...services.retrieval import semantic_search
from ...services.latex_converter import enhance_response_with_latex, detect_math_content
from ...services.web_ingest_service import get_web_ingest_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/session", tags=["session"])


# ============================================================================
# Request/Response Schemas
# ============================================================================

class CreateSessionRequest(BaseModel):
    """Request to create a new session"""
    user_id: str = Field(..., min_length=1)
    classroom_id: Optional[str] = None
    config: Optional[dict] = None


class CreateSessionResponse(BaseModel):
    """Response with created session"""
    success: bool
    session: Optional[dict] = None
    error: Optional[str] = None


class SessionQueryRequest(BaseModel):
    """Request for session-aware query"""
    user_id: str = Field(..., min_length=1)
    question: str = Field(..., min_length=3, max_length=1000)
    class_id: Optional[str] = None
    find_resources: bool = True
    force_session_prioritize: bool = False  # Override to force session context


class SessionQueryResponse(BaseModel):
    """Response with answer and session intelligence info"""
    success: bool
    request_id: str = ""
    session_context_decision: str = "new_topic"  # "related" or "new_topic"
    max_similarity: float = 0.0
    most_similar_turn_index: Optional[int] = None
    data: Optional[dict] = None
    turn: Optional[dict] = None
    resources_appended: int = 0
    suggested_questions: List[dict] = []  # Dynamic follow-up suggestions
    error: Optional[str] = None


class ResourceListResponse(BaseModel):
    """Response with session resource list"""
    success: bool
    session_id: str
    resources: List[dict] = []
    total: int = 0


class AppendCheckRequest(BaseModel):
    """Request to check if resource would be duplicate"""
    session_id: str
    url: Optional[str] = None
    content_hash: Optional[str] = None
    title: str = ""


class AppendCheckResponse(BaseModel):
    """Response indicating if resource would be inserted"""
    inserted: bool
    resource_id: str = ""
    reason: str  # new, duplicate_url, duplicate_hash, duplicate_vector


# ============================================================================
# Endpoints
# ============================================================================

@router.post("/create", response_model=CreateSessionResponse)
async def create_session(request: CreateSessionRequest):
    """
    Create a new tutoring session.
    
    Sessions track:
    - Query chain with embeddings
    - Resource list with deduplication
    - Relatedness detection
    """
    try:
        service = get_session_service()
        session = service.create_session(
            user_id=request.user_id,
            classroom_id=request.classroom_id,
            config=request.config
        )
        
        return CreateSessionResponse(
            success=True,
            session={
                "session_id": session.session_id,
                "user_id": session.user_id,
                "classroom_id": session.classroom_id,
                "created_at": session.created_at,
                "config": session.config
            }
        )
    except Exception as e:
        logger.error(f"[SESSION] Create failed: {e}")
        return CreateSessionResponse(success=False, error=str(e))


@router.post("/{session_id}/query", response_model=SessionQueryResponse)
async def session_query(
    session_id: str,
    request: SessionQueryRequest,
    background_tasks: BackgroundTasks
):
    """
    Query with session context and intelligence.
    
    Process:
    1. Compute query embedding
    2. Use SessionIntelligence to decide: related or new_topic
    3. Route retrieval based on decision (or force_session_prioritize)
    4. Append new resources (deduplicated)
    5. Generate LLM response with latex_blocks
    """
    from ...services.session_intelligence import get_session_intelligence
    
    start_time = time.time()
    request_id = ""
    decision_result = None
    
    try:
        service = get_session_service()
        intelligence = get_session_intelligence()
        
        # Verify session exists
        session_data = service.get_session(session_id)
        if not session_data:
            raise HTTPException(status_code=404, detail="Session not found or expired")
        
        # Get raw session for intelligence
        raw_session = service._sessions.get(session_id, {})
        
        # Compute embedding for query
        embedding = service.embedding_model.encode(request.question).tolist()
        
        # Use SessionIntelligence for decision
        try:
            decision_result, updated_state = intelligence.compute_decision(
                query_embedding=embedding,
                turn_embeddings=raw_session.get("turn_embeddings", []),
                last_topic_vector=raw_session.get("last_topic_vector"),
                last_decision=raw_session.get("last_decision", "new_topic"),
                consecutive_borderline=raw_session.get("consecutive_borderline", 0),
                session_id=session_id,
                turn_index=len(raw_session.get("turns", [])) + 1,
                query_text=request.question
            )
            request_id = decision_result.request_id
            
            # Update session with intelligence state
            raw_session["last_topic_vector"] = updated_state["last_topic_vector"]
            raw_session["last_decision"] = updated_state["last_decision"]
            raw_session["consecutive_borderline"] = updated_state["consecutive_borderline"]
            
        except Exception as e:
            logger.error(f"[ERROR] emb_service_down, defaulted_new_topic: {e}")
            decision_result = type('obj', (object,), {
                'decision': 'new_topic',
                'max_similarity': 0.0,
                'most_similar_turn_index': None,
                'request_id': 'error'
            })()
        
        # Add turn with the computed embedding
        turn = service.add_turn(session_id, request.question, embedding)
        if not turn:
            raise HTTPException(status_code=500, detail="Failed to add turn")
        
        resources_appended = 0
        
        # Get retrieval order based on decision
        retrieval_order = intelligence.get_retrieval_order(
            decision_result.decision,
            request.force_session_prioritize
        )
        
        # Log retrieval info
        session_hits = len(raw_session.get("resources", [])) if decision_result.decision == "related" or request.force_session_prioritize else 0
        logger.info(f"[RETR] session_hits={session_hits} classroom_hits=0 web_hits=0")
        
        # Fetch web resources if enabled
        if request.find_resources:
            try:
                web_service = get_web_ingest_service()
                # Placeholder for actual web fetch
                pass
            except Exception as e:
                logger.error(f"[SESSION] Web fetch failed: {e}")
        
        # Build response
        retrieval_time = int((time.time() - start_time) * 1000)
        
        # Placeholder response
        answer = f"Based on the context, here's the answer to: {request.question}"
        
        # Enhance with LaTeX if math content detected
        latex_enhancement = {}
        if detect_math_content(request.question) or detect_math_content(answer):
            latex_enhancement = enhance_response_with_latex(answer, answer)
        
        # Generate dynamic suggestions
        try:
            from ...services.suggestion_engine import get_suggestion_engine
            
            suggestion_engine = get_suggestion_engine()
            
            # Get session suggestion history (hashes of previously shown)
            suggestion_history = [
                h.get("hash") for h in raw_session.get("suggestions_history", [])
            ]
            
            # Get session resource phrases for boost
            session_resources = [
                r.get("title", "") for r in raw_session.get("resources", [])
            ]
            
            # Generate suggestions (use empty context_chunks for now - placeholder)
            suggestions, suggest_debug = suggestion_engine.generate_suggestions(
                request_id=request_id,
                session_id=session_id,
                user_question=request.question,
                answer=answer,
                context_chunks=[{"text": answer}],  # TODO: Use actual MCP chunks
                session_history=suggestion_history,
                session_resources=session_resources,
                k=6
            )
            
            # Update session suggestion history
            now_str = datetime.utcnow().isoformat()
            for sugg in suggestions:
                import hashlib
                sugg_hash = hashlib.sha256(sugg.text.lower().encode()).hexdigest()[:16]
                raw_session.setdefault("suggestions_history", []).append({
                    "hash": sugg_hash,
                    "text": sugg.text,
                    "shown_at": now_str
                })
            
            # LRU eviction if too many
            max_history = 50
            if len(raw_session.get("suggestions_history", [])) > max_history:
                raw_session["suggestions_history"] = raw_session["suggestions_history"][-max_history:]
            
            suggested_questions = [s.to_dict() for s in suggestions]
            
        except Exception as e:
            logger.warning(f"[SUGGEST] Suggestion generation failed: {e}")
            suggested_questions = []
        
        return SessionQueryResponse(
            success=True,
            request_id=request_id,
            session_context_decision=decision_result.decision,
            max_similarity=decision_result.max_similarity,
            most_similar_turn_index=decision_result.most_similar_turn_index,
            data={
                "answer_short": answer,
                "answer_detailed": None,
                "latex_blocks": latex_enhancement.get("latex_blocks", []),
                "render_hint": latex_enhancement.get("render_hint", "katex"),
                "answer_detailed_plain": latex_enhancement.get("answer_detailed_plain"),
                "confidence_score": 0.85,
                "metadata": {
                    "retrieval_time_ms": retrieval_time,
                    "session_id": session_id,
                    "retrieval_order": retrieval_order
                }
            },
            turn={
                "turn_number": turn.turn_number,
                "related": decision_result.decision == "related",
                "relatedness_score": decision_result.max_similarity
            },
            resources_appended=resources_appended,
            suggested_questions=suggested_questions
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[SESSION] Query failed: {e}")
        return SessionQueryResponse(success=False, error=str(e))


@router.get("/{session_id}/resources", response_model=ResourceListResponse)
async def get_session_resources(session_id: str):
    """
    Get session resource list.
    
    Returns all resources discovered during session,
    sorted by last_referenced_at descending.
    """
    try:
        service = get_session_service()
        
        # Verify session exists
        session = service.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found or expired")
        
        resources = service.get_resource_list(session_id)
        
        return ResourceListResponse(
            success=True,
            session_id=session_id,
            resources=[
                {
                    "resource_id": r.resource_id,
                    "type": r.resource_type,
                    "source": r.source,
                    "url": r.url,
                    "title": r.title,
                    "preview_summary": r.preview_summary,
                    "inline_render": r.inline_render,
                    "inserted_at": r.inserted_at,
                    "last_referenced_at": r.last_referenced_at
                }
                for r in resources
            ],
            total=len(resources)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[SESSION] Get resources failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{session_id}/export")
async def export_session(session_id: str):
    """
    Export full session JSON for debugging/audit.
    
    Returns complete session state including query chain and resources.
    """
    try:
        service = get_session_service()
        
        export = service.export_session_json(session_id)
        if not export:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return {"success": True, "session": export}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[SESSION] Export failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{session_id}/reset")
async def reset_session_context(session_id: str):
    """
    Reset session context for fresh topic.
    
    Clears last_topic_vector and sets last_decision to new_topic.
    Does NOT delete resource_list.
    """
    try:
        service = get_session_service()
        
        # Verify session exists
        session = service.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found or expired")
        
        # Get raw session and reset intelligence state
        raw_session = service._sessions.get(session_id)
        if raw_session:
            raw_session["last_topic_vector"] = None
            raw_session["last_decision"] = "new_topic"
            raw_session["consecutive_borderline"] = 0
            
            # Add topic segment marker
            from datetime import datetime
            raw_session.setdefault("topic_segments", []).append({
                "reset_at": datetime.utcnow().isoformat(),
                "turn_index": len(raw_session.get("turns", []))
            })
        
        logger.info(f"[SESSION] Reset context for session_id={session_id}")
        
        return {"status": "ok", "message": "Session context reset successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[SESSION] Reset failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{session_id}/status")
async def get_session_status(session_id: str):
    """
    Get session status for debugging/UI.
    
    Returns turns, last_decision, topic info.
    """
    try:
        service = get_session_service()
        
        # Verify session exists
        session = service.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found or expired")
        
        raw_session = service._sessions.get(session_id, {})
        
        return {
            "success": True,
            "session_id": session_id,
            "status": {
                "turn_count": len(raw_session.get("turns", [])),
                "resource_count": len(raw_session.get("resources", [])),
                "last_decision": raw_session.get("last_decision", "new_topic"),
                "consecutive_borderline": raw_session.get("consecutive_borderline", 0),
                "has_topic_vector": raw_session.get("last_topic_vector") is not None,
                "topic_segments": len(raw_session.get("topic_segments", [])),
                "suggestions_history_count": len(raw_session.get("suggestions_history", [])),
                "created_at": raw_session.get("created_at"),
                "last_active_at": raw_session.get("last_active_at")
            },
            "recent_turns": [
                {
                    "turn_number": t["turn_number"],
                    "question": t["question"][:50] + "..." if len(t["question"]) > 50 else t["question"],
                    "related": t.get("related", False),
                    "timestamp": t.get("timestamp")
                }
                for t in raw_session.get("turns", [])[-5:]  # Last 5 turns
            ]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[SESSION] Status failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{session_id}/suggestions")
async def get_suggestions(
    session_id: str,
    request_id: str = None,
    k: int = 6
):
    """
    Get/refresh suggestions for a session.
    
    Can be called to refresh suggestions without re-running full RAG.
    """
    try:
        from ...services.suggestion_engine import get_suggestion_engine
        
        service = get_session_service()
        suggestion_engine = get_suggestion_engine()
        
        # Verify session exists
        session = service.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found or expired")
        
        raw_session = service._sessions.get(session_id, {})
        
        # Get last turn for context
        turns = raw_session.get("turns", [])
        if not turns:
            return {"success": True, "suggested_questions": [], "message": "No turns in session"}
        
        last_turn = turns[-1]
        user_question = last_turn.get("question", "")
        
        # Get session suggestion history
        suggestion_history = [
            h.get("hash") for h in raw_session.get("suggestions_history", [])
        ]
        
        # Generate fresh suggestions
        suggestions, debug_info = suggestion_engine.generate_suggestions(
            request_id=request_id or f"refresh_{session_id[:8]}",
            session_id=session_id,
            user_question=user_question,
            answer="",  # No answer context for refresh
            context_chunks=[],
            session_history=suggestion_history,
            k=k
        )
        
        return {
            "success": True,
            "session_id": session_id,
            "suggested_questions": [s.to_dict() for s in suggestions],
            "debug": debug_info
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[SUGGEST] Get suggestions failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{session_id}/suggestions/history")
async def get_suggestions_history(session_id: str):
    """
    Get all suggestions shown in this session (for debugging/audit).
    """
    try:
        service = get_session_service()
        
        # Verify session exists
        session = service.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found or expired")
        
        raw_session = service._sessions.get(session_id, {})
        history = raw_session.get("suggestions_history", [])
        
        return {
            "success": True,
            "session_id": session_id,
            "total": len(history),
            "suggestions": history
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[SUGGEST] Get history failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# Internal Dedup Check Endpoint
# ============================================================================

internal_router = APIRouter(prefix="/api/resources", tags=["resources"])


@internal_router.post("/append_check", response_model=AppendCheckResponse)
async def check_append(request: AppendCheckRequest):
    """
    Check if resource would be inserted or is duplicate.
    
    Used by workers before inserting to avoid redundant work.
    """
    try:
        service = get_session_service()
        
        # Simulate append without actually inserting
        result = service.append_resource(
            session_id=request.session_id,
            resource_type="text",  # Will be updated on actual insert
            source="web",
            url=request.url,
            title=request.title,
            content_hash=request.content_hash
        )
        
        return AppendCheckResponse(
            inserted=result.inserted,
            resource_id=result.resource_id,
            reason=result.reason
        )
        
    except Exception as e:
        logger.error(f"[APPEND] Check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
