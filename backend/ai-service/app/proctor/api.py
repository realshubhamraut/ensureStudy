"""
Proctoring API - FastAPI endpoints for exam proctoring

Endpoints:
- POST /api/proctor/start - Start a proctoring session
- POST /api/proctor/stream - Stream a frame for processing
- POST /api/proctor/stop - Stop session and get results
- POST /api/proctor/tab-switch - Record a tab switch event
- GET /api/proctor/status/{session_id} - Get session status
"""

import base64
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

from .session import ProctorSession

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/proctor", tags=["Proctoring"])

# In-memory session storage (replace with Redis for production)
_sessions: Dict[str, ProctorSession] = {}


# ============== Request/Response Models ==============

class StartSessionRequest(BaseModel):
    """Request to start a proctoring session"""
    assessment_id: str = Field(..., description="ID of the assessment")
    student_id: str = Field(..., description="ID of the student")


class StartSessionResponse(BaseModel):
    """Response after starting a session"""
    session_id: str
    status: str
    message: str


class StreamFrameRequest(BaseModel):
    """Request to process a webcam frame"""
    session_id: str = Field(..., description="Session ID from /start")
    frame_base64: str = Field(..., description="Base64 encoded JPEG frame")
    timestamp: float = Field(0.0, description="Frame timestamp in seconds")


class StreamFrameResponse(BaseModel):
    """Response after processing a frame"""
    processed: bool
    current_score: int
    active_flags: List[str]
    frame_count: Optional[int] = None
    quality_issues: Optional[List[str]] = None


class TabSwitchRequest(BaseModel):
    """Request to record a tab switch"""
    session_id: str


class TabSwitchResponse(BaseModel):
    """Response after recording tab switch"""
    recorded: bool
    tab_switch_count: int


class StopSessionRequest(BaseModel):
    """Request to stop a proctoring session"""
    session_id: str


class StopSessionResponse(BaseModel):
    """Final proctoring results"""
    session_id: str
    integrity_score: int
    flags: List[str]
    review_required: bool
    frames_processed: int
    duration_seconds: float


class SessionStatusResponse(BaseModel):
    """Current session status"""
    session_id: str
    is_active: bool
    frames_processed: int
    current_score: int
    active_flags: List[str]
    duration_seconds: float


class ModelStatusResponse(BaseModel):
    """Model availability status"""
    dlib_predictor: bool
    yolo_model: bool
    mediapipe: bool
    face_landmarker: bool


# ============== API Endpoints ==============

@router.post("/start", response_model=StartSessionResponse)
async def start_session(request: StartSessionRequest):
    """
    Start a new proctoring session.
    
    Creates a session that will track integrity metrics
    as frames are streamed from the webcam.
    """
    try:
        session = ProctorSession(
            assessment_id=request.assessment_id,
            student_id=request.student_id
        )
        
        _sessions[session.id] = session
        
        logger.info(f"Started proctoring session: {session.id}")
        
        return StartSessionResponse(
            session_id=session.id,
            status="active",
            message="Proctoring session started successfully"
        )
        
    except Exception as e:
        logger.error(f"Failed to start proctoring session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stream", response_model=StreamFrameResponse)
async def stream_frame(request: StreamFrameRequest):
    """
    Process a single webcam frame.
    
    Decodes the base64 frame, runs detection pipeline,
    and returns current integrity score and active flags.
    """
    session = _sessions.get(request.session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if not session.is_active:
        raise HTTPException(status_code=400, detail="Session is not active")
    
    try:
        # Decode base64 frame
        frame_bytes = base64.b64decode(request.frame_base64)
        frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
        frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid frame data")
        
        # Process frame
        result = session.process_frame(frame, request.timestamp)
        
        return StreamFrameResponse(
            processed=result.get("processed", False),
            current_score=result.get("current_score", 100),
            active_flags=result.get("active_flags", []),
            frame_count=result.get("frame_count"),
            quality_issues=result.get("quality_issues")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Frame processing error: {e}")
        raise HTTPException(status_code=500, detail=f"Frame processing error: {str(e)}")


@router.post("/tab-switch", response_model=TabSwitchResponse)
async def record_tab_switch(request: TabSwitchRequest):
    """
    Record a tab switch event.
    
    Called when the frontend detects that the user
    switched away from the exam tab.
    """
    session = _sessions.get(request.session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if not session.is_active:
        raise HTTPException(status_code=400, detail="Session is not active")
    
    session.add_tab_switch()
    
    return TabSwitchResponse(
        recorded=True,
        tab_switch_count=session.metrics.tab_switch_count
    )


@router.post("/stop", response_model=StopSessionResponse)
async def stop_session(request: StopSessionRequest, background_tasks: BackgroundTasks):
    """
    Stop a proctoring session and get final results.
    
    Calculates final integrity score, generates flags,
    and cleans up session resources.
    """
    session = _sessions.get(request.session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        # Finalize session
        result = session.finalize()
        
        # Schedule cleanup
        background_tasks.add_task(_cleanup_session, request.session_id)
        
        return StopSessionResponse(
            session_id=result["session_id"],
            integrity_score=result["integrity_score"],
            flags=result["flags"],
            review_required=result["review_required"],
            frames_processed=result["frames_processed"],
            duration_seconds=result["duration_seconds"]
        )
        
    except Exception as e:
        logger.error(f"Error stopping session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/{session_id}", response_model=SessionStatusResponse)
async def get_session_status(session_id: str):
    """
    Get current status of a proctoring session.
    """
    session = _sessions.get(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    ratios = session.metrics.get_ratios()
    current_score = session.scorer.compute(ratios)
    active_flags = session.flagger.generate(ratios)
    duration = (datetime.utcnow() - session.started_at).total_seconds()
    
    return SessionStatusResponse(
        session_id=session.id,
        is_active=session.is_active,
        frames_processed=session.metrics.frame_count,
        current_score=current_score,
        active_flags=active_flags,
        duration_seconds=duration
    )


@router.get("/models-status", response_model=ModelStatusResponse)
async def get_models_status():
    """
    Check which ML models are available.
    """
    from .models.model_loader import check_models
    
    status = check_models()
    
    return ModelStatusResponse(**status)

# ============== Audio and Face Verification Endpoints ==============

class AudioStreamRequest(BaseModel):
    """Request to process audio data"""
    session_id: str
    audio_base64: str = Field(..., description="Base64 encoded audio samples (int16)")


class AudioStreamResponse(BaseModel):
    """Response after processing audio"""
    processed: bool
    suspicious: bool
    amplitude: float
    message: str


class RegisterFaceRequest(BaseModel):
    """Request to register a face for verification"""
    session_id: str
    image_base64: str = Field(..., description="Base64 encoded JPEG image")


class RegisterFaceResponse(BaseModel):
    """Response after registering face"""
    registered: bool
    message: str


class VerifyFaceRequest(BaseModel):
    """Request to verify student identity"""
    session_id: str
    frame_base64: str = Field(..., description="Base64 encoded JPEG frame")


class VerifyFaceResponse(BaseModel):
    """Response after face verification"""
    verified: bool
    confidence: float
    message: str


@router.post("/audio-stream", response_model=AudioStreamResponse)
async def process_audio_stream(request: AudioStreamRequest):
    """
    Process audio stream data for suspicious activity detection.
    
    The frontend should capture audio samples and send them as base64.
    Detects loud noises, speech, and other suspicious audio patterns.
    """
    session = _sessions.get(request.session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if not session.is_active:
        raise HTTPException(status_code=400, detail="Session is not active")
    
    try:
        result = session.audio_detector.analyze_base64(request.audio_base64)
        
        return AudioStreamResponse(
            processed=True,
            suspicious=result.get("suspicious", False),
            amplitude=result.get("amplitude", 0.0),
            message=result.get("message", "")
        )
        
    except Exception as e:
        logger.error(f"Audio processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/register-face", response_model=RegisterFaceResponse)
async def register_face(request: RegisterFaceRequest):
    """
    Register a reference face for identity verification.
    
    Should be called at the start of an exam with the student's
    verified photo or a clear initial webcam frame.
    """
    session = _sessions.get(request.session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if not session.is_active:
        raise HTTPException(status_code=400, detail="Session is not active")
    
    try:
        result = session.face_verifier.register_face_base64(request.image_base64)
        
        return RegisterFaceResponse(
            registered=result.get("registered", False),
            message=result.get("message", "")
        )
        
    except Exception as e:
        logger.error(f"Face registration error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/verify-face", response_model=VerifyFaceResponse)
async def verify_face(request: VerifyFaceRequest):
    """
    Verify that the current webcam frame matches the registered face.
    
    Returns confidence score and verification status.
    Should be called periodically during the exam.
    """
    session = _sessions.get(request.session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if not session.is_active:
        raise HTTPException(status_code=400, detail="Session is not active")
    
    try:
        # Decode frame
        frame_bytes = base64.b64decode(request.frame_base64)
        frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
        frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid frame data")
        
        result = session.face_verifier.verify(frame)
        
        return VerifyFaceResponse(
            verified=result.get("verified", False),
            confidence=result.get("confidence", 0.0),
            message=result.get("message", "")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Face verification error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============== Background Tasks ==============

async def _cleanup_session(session_id: str):
    """Clean up session resources after delay"""
    import asyncio
    
    # Wait a bit before cleanup to allow any final requests
    await asyncio.sleep(60)
    
    if session_id in _sessions:
        del _sessions[session_id]
        logger.info(f"Cleaned up session: {session_id}")


# ============== Health Check ==============

@router.get("/health")
async def health_check():
    """Health check for proctoring module"""
    return {
        "status": "healthy",
        "active_sessions": len(_sessions),
        "module": "proctoring"
    }
