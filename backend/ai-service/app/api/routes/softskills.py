"""
Soft Skills Evaluation API Routes

Provides endpoints for evaluating communication skills:
- Fluency analysis
- Grammar checking
- Eye contact and posture (from video)
- Overall soft skills score
- Real-time frame analysis via WebSocket
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import json
import base64
import asyncio
from datetime import datetime

router = APIRouter(prefix="/softskills", tags=["Soft Skills"])


# ============================================
# Request/Response Schemas
# ============================================

class SoftSkillsEvaluationRequest(BaseModel):
    """Request for soft skills evaluation."""
    user_id: str
    transcript: str = Field(..., description="Transcribed speech from user")
    audio_duration_seconds: float = Field(default=60.0, description="Duration of audio")
    has_video: bool = Field(default=False, description="Whether video was captured")
    pause_ratio: float = Field(default=0.0, description="Ratio of pause time to total time")
    

class FluencyMetrics(BaseModel):
    """Fluency analysis results."""
    words_per_minute: float
    pause_ratio: float
    filler_word_count: int
    filler_words_used: List[str]
    score: float


class GrammarMetrics(BaseModel):
    """Grammar analysis results."""
    error_count: int
    sentence_count: int
    avg_sentence_length: float
    score: float


class VisualMetrics(BaseModel):
    """Video analysis results from computer vision."""
    eye_contact_score: float
    posture_score: float
    hand_gesture_score: float
    # Detailed metrics (optional, filled when video analyzed)
    gaze_center_ratio: Optional[float] = None
    head_forward_ratio: Optional[float] = None
    hands_visible_ratio: Optional[float] = None
    frames_analyzed: Optional[int] = None


class FrameAnalysisRequest(BaseModel):
    """Request for single frame analysis."""
    frame_base64: str = Field(..., description="Base64 encoded JPEG frame")
    session_id: Optional[str] = None


class FrameAnalysisResult(BaseModel):
    """Result from analyzing a single frame."""
    face_present: bool
    gaze_direction: str  # center, left, right
    gaze_score: float
    head_yaw: float
    head_pitch: float
    hands_visible: bool
    num_hands: int
    gesture_score: float
    body_detected: bool
    posture_score: float
    is_upright: bool
    shoulders_level: bool


class SoftSkillsResult(BaseModel):
    """Complete soft skills evaluation result."""
    fluency: FluencyMetrics
    grammar: GrammarMetrics
    visual: VisualMetrics
    overall_score: float
    breakdown: dict
    feedback: List[str]
    strengths: List[str]
    areas_for_improvement: List[str]


# ============================================
# Analysis Functions (using new services)
# ============================================

def analyze_fluency_with_service(transcript: str, duration_seconds: float, pause_ratio: float = 0.0) -> FluencyMetrics:
    """Analyze speech fluency using the fluency analyzer service."""
    try:
        from ...services.fluency_analyzer import get_fluency_analyzer
        
        analyzer = get_fluency_analyzer()
        result = analyzer.analyze(transcript, duration_seconds, pause_ratio)
        
        return FluencyMetrics(
            words_per_minute=round(result.wpm, 1),
            pause_ratio=round(result.pause_ratio, 2),
            filler_word_count=result.filler_count,
            filler_words_used=result.fillers_detected[:5],
            score=round(result.score, 1)
        )
    except Exception as e:
        print(f"[SoftSkills] Fluency analysis error: {e}")
        # Fallback to basic analysis
        return _analyze_fluency_basic(transcript, duration_seconds)


def _analyze_fluency_basic(transcript: str, duration_seconds: float) -> FluencyMetrics:
    """Basic fluency analysis fallback."""
    words = transcript.split()
    word_count = len(words)
    wpm = (word_count / duration_seconds) * 60 if duration_seconds > 0 else 0
    
    FILLER_WORDS = ["um", "uh", "like", "you know", "basically", "actually"]
    transcript_lower = transcript.lower()
    filler_count = sum(transcript_lower.count(f) for f in FILLER_WORDS)
    fillers_found = [f for f in FILLER_WORDS if f in transcript_lower]
    
    score = max(0, min(100, 80 - filler_count * 5))
    
    return FluencyMetrics(
        words_per_minute=round(wpm, 1),
        pause_ratio=0.0,
        filler_word_count=filler_count,
        filler_words_used=fillers_found[:5],
        score=round(score, 1)
    )


def analyze_grammar(transcript: str) -> GrammarMetrics:
    """Analyze grammar quality using the grammar analyzer service."""
    try:
        from ...services.grammar_analyzer import get_grammar_analyzer
        
        analyzer = get_grammar_analyzer()
        result = analyzer.analyze(transcript)
        
        return GrammarMetrics(
            error_count=result.error_count,
            sentence_count=result.sentence_count,
            avg_sentence_length=result.avg_sentence_length,
            score=result.score
        )
    except Exception as e:
        print(f"[SoftSkills] Grammar analysis error: {e}")
        # Fallback to basic analysis
        return _analyze_grammar_basic(transcript)


def _analyze_grammar_basic(transcript: str) -> GrammarMetrics:
    """Basic grammar analysis fallback."""
    sentences = [s.strip() for s in transcript.replace('?', '.').replace('!', '.').split('.') if s.strip()]
    sentence_count = len(sentences)
    
    words = transcript.split()
    word_count = len(words)
    
    avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
    
    # Simple grammar checks
    errors = 0
    transcript_lower = transcript.lower()
    
    if " i " in transcript_lower or transcript_lower.startswith("i "):
        errors += 1
    if "  " in transcript:
        errors += 1
    if transcript_lower.count("dont") > 0 or transcript_lower.count("wont") > 0:
        errors += 1
    
    error_penalty = min(40, errors * 10)
    length_score = 100 if 10 <= avg_sentence_length <= 20 else max(50, 100 - abs(avg_sentence_length - 15) * 3)
    
    score = max(0, min(100, (length_score - error_penalty)))
    
    return GrammarMetrics(
        error_count=errors,
        sentence_count=sentence_count,
        avg_sentence_length=round(avg_sentence_length, 1),
        score=round(score, 1)
    )


def analyze_visual_with_pipeline(has_video: bool, video_bytes: bytes = None) -> VisualMetrics:
    """
    Analyze visual aspects from video using the soft skills pipeline.
    """
    if not has_video or video_bytes is None:
        return VisualMetrics(
            eye_contact_score=75.0,
            posture_score=75.0,
            hand_gesture_score=70.0
        )
    
    try:
        from ...services.softskills_pipeline import get_softskills_pipeline
        import cv2
        import numpy as np
        
        pipeline = get_softskills_pipeline()
        pipeline.start_session()
        
        # Decode video and process frames
        nparr = np.frombuffer(video_bytes, np.uint8)
        
        # For now, treat as single image (MVP)
        # In production, would extract frames from video
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is not None:
            pipeline.process_frame(frame)
        
        # Get aggregated metrics
        visual_metrics = pipeline.aggregate_visual_metrics()
        
        return VisualMetrics(
            eye_contact_score=round(visual_metrics.eye_contact_score, 1),
            posture_score=round(visual_metrics.posture_score, 1),
            hand_gesture_score=round(visual_metrics.gesture_score, 1),
            gaze_center_ratio=visual_metrics.gaze_center_ratio,
            hands_visible_ratio=visual_metrics.hands_visible_ratio,
            frames_analyzed=visual_metrics.frames_analyzed
        )
        
    except Exception as e:
        print(f"[SoftSkills] Video analysis failed: {e}")
        return VisualMetrics(
            eye_contact_score=75.0,
            posture_score=75.0,
            hand_gesture_score=70.0
        )


def generate_feedback(
    fluency: FluencyMetrics,
    grammar: GrammarMetrics,
    visual: VisualMetrics,
    overall: float
) -> tuple:
    """Generate personalized feedback."""
    feedback = []
    strengths = []
    improvements = []
    
    # Fluency feedback
    if fluency.score >= 80:
        strengths.append("Excellent speech fluency and natural pace")
    elif fluency.score >= 60:
        feedback.append("Good fluency, but try to reduce filler words")
    else:
        improvements.append("Practice speaking without filler words like 'um' and 'like'")
    
    if fluency.words_per_minute < 100:
        improvements.append("Try to speak a bit faster to maintain engagement")
    elif fluency.words_per_minute > 160:
        improvements.append("Slow down slightly for better clarity")
    
    # Grammar feedback
    if grammar.score >= 80:
        strengths.append("Strong grammatical structure")
    elif grammar.score >= 60:
        feedback.append("Generally good grammar with minor issues")
    else:
        improvements.append("Focus on sentence structure and proper grammar")
    
    # Visual feedback
    if visual.eye_contact_score >= 80:
        strengths.append("Great eye contact with the camera")
    else:
        improvements.append("Try to maintain more eye contact with the interviewer")
    
    if visual.posture_score >= 80:
        strengths.append("Confident and stable posture")
    else:
        improvements.append("Sit upright and maintain a confident posture")
    
    if visual.hand_gesture_score >= 80:
        strengths.append("Natural and expressive hand gestures")
    elif visual.hand_gesture_score < 60:
        improvements.append("Keep your hands visible and use natural gestures")
    
    # Overall feedback
    if overall >= 80:
        feedback.insert(0, "Outstanding performance! You demonstrated excellent communication skills.")
    elif overall >= 60:
        feedback.insert(0, "Good job! With some practice, you can improve further.")
    else:
        feedback.insert(0, "Keep practicing! Focus on the areas highlighted below.")
    
    return feedback, strengths, improvements


# ============================================
# API Endpoints
# ============================================

@router.post("/evaluate", response_model=SoftSkillsResult)
async def evaluate_soft_skills(request: SoftSkillsEvaluationRequest):
    """Evaluate soft skills from speech transcript and optional video."""
    
    if not request.transcript or len(request.transcript.strip()) < 10:
        raise HTTPException(status_code=400, detail="Transcript too short for evaluation")
    
    # Analyze each component
    fluency = analyze_fluency_with_service(
        request.transcript, 
        request.audio_duration_seconds,
        request.pause_ratio
    )
    grammar = analyze_grammar(request.transcript)
    visual = analyze_visual_with_pipeline(request.has_video)
    
    # Calculate overall score using weighted formula
    overall_score = (
        0.30 * fluency.score +
        0.20 * grammar.score +
        0.15 * visual.eye_contact_score +
        0.10 * visual.hand_gesture_score +
        0.10 * visual.posture_score +
        0.10 * 75.0 +  # Expression placeholder
        0.05 * 75.0    # Other
    )
    
    # Generate feedback
    feedback, strengths, improvements = generate_feedback(fluency, grammar, visual, overall_score)
    
    return SoftSkillsResult(
        fluency=fluency,
        grammar=grammar,
        visual=visual,
        overall_score=round(overall_score, 1),
        breakdown={
            "fluency": {"weight": "30%", "score": fluency.score},
            "grammar": {"weight": "20%", "score": grammar.score},
            "eye_contact": {"weight": "15%", "score": visual.eye_contact_score},
            "hand_gestures": {"weight": "10%", "score": visual.hand_gesture_score},
            "posture": {"weight": "10%", "score": visual.posture_score}
        },
        feedback=feedback,
        strengths=strengths,
        areas_for_improvement=improvements
    )


@router.post("/evaluate-video")
async def evaluate_with_video(
    video: UploadFile = File(...),
    user_id: str = Form(...),
    transcript: str = Form(...),
    audio_duration_seconds: float = Form(default=60.0)
):
    """
    Evaluate soft skills with video upload.
    
    Analyzes the video using computer vision to detect:
    - Eye contact (gaze direction)
    - Head pose stability
    - Hand gestures
    - Posture
    """
    # Validate file type
    if not video.content_type.startswith('video/') and not video.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be a video or image")
    
    # Read video bytes
    video_bytes = await video.read()
    
    if len(video_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty video file")
    
    # Validate transcript
    if not transcript or len(transcript.strip()) < 10:
        raise HTTPException(status_code=400, detail="Transcript too short for evaluation")
    
    # Analyze each component
    fluency = analyze_fluency_with_service(transcript, audio_duration_seconds)
    grammar = analyze_grammar(transcript)
    
    # Analyze video with real CV detection
    visual = analyze_visual_with_pipeline(has_video=True, video_bytes=video_bytes)
    
    # Calculate overall score
    overall_score = (
        0.30 * fluency.score +
        0.20 * grammar.score +
        0.15 * visual.eye_contact_score +
        0.10 * visual.hand_gesture_score +
        0.10 * visual.posture_score +
        0.10 * 75.0 +  # Expression placeholder
        0.05 * 75.0
    )
    
    # Generate feedback
    feedback, strengths, improvements = generate_feedback(fluency, grammar, visual, overall_score)
    
    return SoftSkillsResult(
        fluency=fluency,
        grammar=grammar,
        visual=visual,
        overall_score=round(overall_score, 1),
        breakdown={
            "fluency": {"weight": "30%", "score": fluency.score},
            "grammar": {"weight": "20%", "score": grammar.score},
            "eye_contact": {"weight": "15%", "score": visual.eye_contact_score},
            "hand_gestures": {"weight": "10%", "score": visual.hand_gesture_score},
            "posture": {"weight": "10%", "score": visual.posture_score}
        },
        feedback=feedback,
        strengths=strengths,
        areas_for_improvement=improvements
    )


@router.post("/analyze-frame", response_model=FrameAnalysisResult)
async def analyze_frame(request: FrameAnalysisRequest):
    """
    Analyze a single webcam frame for soft skills metrics.
    
    Use this for real-time feedback during practice sessions.
    Returns immediate detection results for:
    - Face presence
    - Gaze direction (center/left/right)
    - Head position (yaw, pitch)
    - Hand visibility
    - Posture metrics
    """
    try:
        from ...services.softskills_pipeline import get_softskills_pipeline
        
        pipeline = get_softskills_pipeline()
        result = pipeline.process_frame_base64(request.frame_base64)
        
        return FrameAnalysisResult(
            face_present=result.face_detected,
            gaze_direction=result.gaze_direction,
            gaze_score=result.gaze_score,
            head_yaw=result.head_yaw,
            head_pitch=result.head_pitch,
            hands_visible=result.hands_visible,
            num_hands=result.num_hands,
            gesture_score=result.gesture_score,
            body_detected=result.body_detected,
            posture_score=result.posture_score,
            is_upright=result.is_upright,
            shoulders_level=result.shoulders_level
        )
        
    except Exception as e:
        print(f"[SoftSkills] Frame analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Frame analysis error: {str(e)}")


# ============================================
# WebSocket for Real-Time Analysis
# ============================================

class SessionManager:
    """Manages active WebSocket sessions."""
    
    def __init__(self):
        self.active_sessions: Dict[str, WebSocket] = {}
    
    async def connect(self, session_id: str, websocket: WebSocket):
        await websocket.accept()
        self.active_sessions[session_id] = websocket
    
    def disconnect(self, session_id: str):
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
    
    async def send_result(self, session_id: str, data: dict):
        if session_id in self.active_sessions:
            await self.active_sessions[session_id].send_json(data)


session_manager = SessionManager()


@router.websocket("/ws/{session_id}")
async def websocket_realtime_analysis(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for real-time soft skills analysis.
    
    Client sends base64-encoded frames, receives analysis results.
    
    Message format (client -> server):
    {
        "type": "frame",
        "data": "<base64 encoded JPEG>"
    }
    
    Message format (server -> client):
    {
        "type": "analysis",
        "timestamp": 1234567890,
        "face_detected": true,
        "gaze_direction": "center",
        "gaze_score": 85.0,
        "hands_visible": true,
        "gesture_score": 75.0,
        "posture_score": 80.0,
        "is_upright": true
    }
    """
    await session_manager.connect(session_id, websocket)
    
    try:
        from ...services.softskills_pipeline import get_softskills_pipeline
        
        pipeline = get_softskills_pipeline()
        pipeline.start_session()
        
        print(f"[WebSocket] Session {session_id} connected")
        
        frame_count = 0
        while True:
            # Receive message
            data = await websocket.receive_text()
            message = json.loads(data)
            
            msg_type = message.get("type")
            print(f"[WebSocket] Received message type: {msg_type}")
            
            if msg_type == "frame":
                # Process frame
                frame_base64 = message.get("data", "")
                
                if frame_base64:
                    frame_count += 1
                    print(f"[WebSocket] Processing frame {frame_count} (data length: {len(frame_base64)})")
                    
                    try:
                        result = pipeline.process_frame_base64(frame_base64)
                        
                        response = {
                            "type": "analysis",
                            "timestamp": result.timestamp_ms,
                            "face_detected": result.face_detected,
                            "gaze_direction": result.gaze_direction,
                            "gaze_score": round(result.gaze_score, 1),
                            "is_looking_at_camera": result.is_looking_at_camera,
                            "hands_visible": result.hands_visible,
                            "num_hands": result.num_hands,
                            "gesture_score": round(result.gesture_score, 1),
                            "body_detected": result.body_detected,
                            "posture_score": round(result.posture_score, 1),
                            "is_upright": result.is_upright,
                            "shoulders_level": result.shoulders_level
                        }
                        print(f"[WebSocket] Sending analysis: gaze={result.gaze_score:.1f}, posture={result.posture_score:.1f}")
                        
                        # Send result back
                        await websocket.send_json(response)
                    except Exception as e:
                        print(f"[WebSocket] Frame processing error: {e}")
                        import traceback
                        traceback.print_exc()
                else:
                    print("[WebSocket] Frame message received but no data")
            
            elif message.get("type") == "transcript":
                # Process transcript for fluency
                transcript = message.get("text", "")
                duration = message.get("duration", 60.0)
                
                if transcript:
                    fluency_result = analyze_fluency_with_service(transcript, duration)
                    
                    await websocket.send_json({
                        "type": "fluency",
                        "score": fluency_result.score,
                        "wpm": fluency_result.words_per_minute,
                        "filler_count": fluency_result.filler_word_count,
                        "fillers": fluency_result.filler_words_used
                    })
            
            elif message.get("type") == "get_summary":
                # Get aggregated metrics
                visual_metrics = pipeline.aggregate_visual_metrics()
                
                await websocket.send_json({
                    "type": "summary",
                    "frames_analyzed": visual_metrics.frames_analyzed,
                    "eye_contact_score": round(visual_metrics.eye_contact_score, 1),
                    "gaze_center_ratio": round(visual_metrics.gaze_center_ratio, 3),
                    "gesture_score": round(visual_metrics.gesture_score, 1),
                    "hands_visible_ratio": round(visual_metrics.hands_visible_ratio, 3),
                    "posture_score": round(visual_metrics.posture_score, 1),
                    "is_upright_ratio": round(visual_metrics.is_upright_ratio, 3)
                })
            
            elif message.get("type") == "end_session":
                # End session and get final scores
                visual_metrics = pipeline.aggregate_visual_metrics()
                
                await websocket.send_json({
                    "type": "session_complete",
                    "visual_metrics": visual_metrics.to_dict()
                })
                break
                
    except WebSocketDisconnect:
        print(f"[WebSocket] Session {session_id} disconnected")
    except Exception as e:
        print(f"[WebSocket] Session {session_id} error: {e}")
    finally:
        session_manager.disconnect(session_id)


@router.get("/health")
async def health_check():
    """Check if soft skills service is healthy."""
    try:
        from ...services.gaze_analyzer import MEDIAPIPE_AVAILABLE
        
        return {
            "status": "healthy",
            "mediapipe_available": MEDIAPIPE_AVAILABLE,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "degraded",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


# ============================================
# Questions API
# ============================================

import random
import os

def load_questions():
    """Load questions from JSON file."""
    try:
        data_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "data", "softskills_questions.json"
        )
        with open(data_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"[SoftSkills] Error loading questions: {e}")
        return None


class QuestionResponse(BaseModel):
    """Single question response."""
    id: int
    text: str
    category: str
    difficulty: str


class QuestionsListResponse(BaseModel):
    """List of questions response."""
    questions: List[QuestionResponse]
    total: int
    session_config: dict


@router.get("/questions", response_model=QuestionsListResponse)
async def get_questions(
    count: int = 5,
    category: Optional[str] = None,
    difficulty: Optional[str] = None
):
    """
    Get random questions for a soft skills session.
    
    - count: Number of questions to return (default 5)
    - category: Optional category filter (self_introduction, strengths_weaknesses, etc.)
    - difficulty: Optional difficulty filter (easy, medium, hard)
    """
    data = load_questions()
    if not data:
        raise HTTPException(status_code=500, detail="Could not load questions")
    
    all_questions = []
    
    for cat in data["categories"]:
        cat_id = cat["id"]
        cat_name = cat["name"]
        
        # Filter by category if specified
        if category and cat_id != category:
            continue
            
        for q in cat["questions"]:
            # Filter by difficulty if specified
            if difficulty and q.get("difficulty") != difficulty:
                continue
            
            all_questions.append({
                "id": q["id"],
                "text": q["text"],
                "category": cat_name,
                "difficulty": q.get("difficulty", "medium")
            })
    
    # Shuffle and select
    random.shuffle(all_questions)
    selected = all_questions[:count]
    
    return QuestionsListResponse(
        questions=[QuestionResponse(**q) for q in selected],
        total=len(selected),
        session_config=data.get("session_config", {})
    )


@router.get("/questions/categories")
async def get_question_categories():
    """Get available question categories."""
    data = load_questions()
    if not data:
        raise HTTPException(status_code=500, detail="Could not load questions")
    
    categories = []
    for cat in data["categories"]:
        categories.append({
            "id": cat["id"],
            "name": cat["name"],
            "count": len(cat["questions"])
        })
    
    return {
        "categories": categories,
        "scoring_weights": data.get("scoring_weights", {})
    }


@router.get("/scoring-weights")
async def get_scoring_weights():
    """Get the scoring weights used for evaluation."""
    data = load_questions()
    if data and "scoring_weights" in data:
        return data["scoring_weights"]
    
    # Default weights
    return {
        "fluency": 0.35,
        "grammar": 0.25,
        "eye_contact": 0.20,
        "hand_gestures": 0.10,
        "posture": 0.10
    }
