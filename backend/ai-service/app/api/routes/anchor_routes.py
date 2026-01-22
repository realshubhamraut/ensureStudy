"""
Anchor Management API Routes

Provides endpoints for:
- Getting active topic anchor
- Resetting/clearing anchor
- Confirming new topic
- Getting sidebar resources
"""
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

from app.utils.auth import verify_jwt_token
from app.services.topic_anchor_service import get_topic_anchor_service
from app.services.session_service import get_session_service

router = APIRouter(prefix="/api/tutor/anchor", tags=["Topic Anchor"])


# ============================================================================
# Request/Response Models
# ============================================================================

class AnchorResponse(BaseModel):
    """Active anchor information"""
    id: Optional[str] = None
    title: Optional[str] = None
    scope: List[str] = []
    entities: List[str] = []
    created_at: Optional[str] = None
    expires_at: Optional[str] = None
    is_active: bool = False


class ResetResponse(BaseModel):
    """Reset confirmation"""
    success: bool
    message: str
    previous_anchor: Optional[str] = None


class ConfirmTopicRequest(BaseModel):
    """Request to confirm new topic"""
    session_id: str
    new_topic_title: str
    keep_history: bool = False


class SidebarResource(BaseModel):
    """Resource for sidebar display"""
    id: str
    title: str
    type: str  # document, web, anchor
    url: Optional[str] = None
    relevance: float = 0.0
    is_current_anchor: bool = False


class SidebarResponse(BaseModel):
    """Sidebar content"""
    current_topic: Optional[str] = None
    resources: List[SidebarResource] = []
    turn_count: int = 0


# ============================================================================
# Endpoints
# ============================================================================

@router.get("", response_model=AnchorResponse)
async def get_anchor(
    session_id: str,
    user_id: str = Depends(verify_jwt_token)
):
    """
    Get the active topic anchor for a session.
    
    Returns anchor details including title, scope, and entities.
    """
    try:
        tal = get_topic_anchor_service()
        anchor = tal.get_anchor(session_id)
        
        if not anchor:
            return AnchorResponse(is_active=False)
        
        return AnchorResponse(
            id=anchor.id,
            title=anchor.canonical_title,
            scope=anchor.subject_scope[:5],
            entities=anchor.locked_entities[:10],
            created_at=anchor.created_at,
            expires_at=anchor.expires_at,
            is_active=True
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reset", response_model=ResetResponse)
async def reset_anchor(
    session_id: str,
    user_id: str = Depends(verify_jwt_token)
):
    """
    Reset/clear the active topic anchor.
    
    This allows the user to start a fresh topic without context from previous anchor.
    """
    try:
        tal = get_topic_anchor_service()
        
        # Get current anchor before clearing
        current = tal.get_anchor(session_id)
        previous_title = current.canonical_title if current else None
        
        # Clear anchor
        success = tal.clear_anchor(session_id, reason="user_reset")
        
        return ResetResponse(
            success=success,
            message="Topic anchor cleared" if success else "No active anchor",
            previous_anchor=previous_title
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/confirm", response_model=AnchorResponse)
async def confirm_new_topic(
    request: ConfirmTopicRequest,
    user_id: str = Depends(verify_jwt_token)
):
    """
    Confirm a new topic when ABCR detects topic change.
    
    Called by frontend when user confirms they want to switch topics.
    """
    try:
        tal = get_topic_anchor_service()
        
        # Clear existing anchor
        tal.clear_anchor(request.session_id, reason="new_topic")
        
        # Create new anchor with confirmed title
        anchor = tal.create_anchor(
            session_id=request.session_id,
            request_id="",
            canonical_title=request.new_topic_title,
            source="user_confirmed"
        )
        
        return AnchorResponse(
            id=anchor.id,
            title=anchor.canonical_title,
            scope=anchor.subject_scope[:5],
            entities=anchor.locked_entities[:10],
            created_at=anchor.created_at,
            expires_at=anchor.expires_at,
            is_active=True
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history")
async def get_topic_history(
    session_id: str,
    user_id: str = Depends(verify_jwt_token)
):
    """
    Get topic history for a session.
    
    Returns list of previous topic anchors with timestamps and end reasons.
    """
    try:
        tal = get_topic_anchor_service()
        history = tal.get_topic_history(session_id)
        
        return {
            "session_id": session_id,
            "history": [
                {
                    "anchor_id": h.anchor_id,
                    "title": h.canonical_title,
                    "started_at": h.started_at,
                    "ended_at": h.ended_at,
                    "end_reason": h.end_reason,
                    "turns_count": h.turns_count
                }
                for h in history
            ]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sidebar", response_model=SidebarResponse)
async def get_sidebar(
    session_id: str,
    user_id: str = Depends(verify_jwt_token)
):
    """
    Get sidebar content for the tutor UI.
    
    Returns current topic anchor and recent resources.
    """
    try:
        tal = get_topic_anchor_service()
        
        # Get current anchor
        anchor = tal.get_anchor(session_id)
        
        # Get recent resources from session
        resources = []
        
        # Add anchor as a resource if active
        if anchor:
            resources.append(SidebarResource(
                id=anchor.id,
                title=anchor.canonical_title,
                type="anchor",
                relevance=1.0,
                is_current_anchor=True
            ))
        
        # TODO: Get actual resources from session service
        # session_svc = get_session_service()
        # session_resources = session_svc.get_resource_list(session_id)
        
        return SidebarResponse(
            current_topic=anchor.canonical_title if anchor else None,
            resources=resources,
            turn_count=0  # TODO: Get from session
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
