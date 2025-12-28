"""
Soft Skills Evaluation API Routes

Provides endpoints for evaluating communication skills:
- Fluency analysis
- Grammar checking
- Eye contact and posture (from video)
- Overall soft skills score
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field
from typing import List, Optional
import random
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
    head_deviation: str  # normal, up, down, left, right
    hands_visible: bool
    num_hands: int


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
# Analysis Functions
# ============================================

# Common filler words to detect
FILLER_WORDS = [
    "um", "uh", "like", "you know", "basically", "actually", 
    "literally", "so", "well", "I mean", "kind of", "sort of"
]


def analyze_fluency(transcript: str, duration_seconds: float) -> FluencyMetrics:
    """Analyze speech fluency."""
    words = transcript.split()
    word_count = len(words)
    
    # Words per minute
    wpm = (word_count / duration_seconds) * 60 if duration_seconds > 0 else 0
    
    # Detect filler words
    transcript_lower = transcript.lower()
    filler_count = 0
    fillers_found = []
    for filler in FILLER_WORDS:
        count = transcript_lower.count(filler)
        if count > 0:
            filler_count += count
            fillers_found.append(filler)
    
    # Estimate pause ratio (simplified - count periods/commas as pauses)
    pause_indicators = transcript.count('.') + transcript.count(',') + transcript.count('...')
    pause_ratio = min(1.0, pause_indicators / max(1, word_count / 10))
    
    # Calculate score
    # Ideal WPM is 120-150 for speaking
    wpm_score = 100 if 120 <= wpm <= 150 else max(0, 100 - abs(wpm - 135) * 0.5)
    filler_penalty = min(30, filler_count * 5)
    pause_penalty = pause_ratio * 10
    
    score = max(0, min(100, wpm_score - filler_penalty - pause_penalty))
    
    return FluencyMetrics(
        words_per_minute=round(wpm, 1),
        pause_ratio=round(pause_ratio, 2),
        filler_word_count=filler_count,
        filler_words_used=fillers_found[:5],  # Top 5
        score=round(score, 1)
    )


def analyze_grammar(transcript: str) -> GrammarMetrics:
    """Analyze grammar quality."""
    # Split into sentences
    sentences = [s.strip() for s in transcript.replace('?', '.').replace('!', '.').split('.') if s.strip()]
    sentence_count = len(sentences)
    
    words = transcript.split()
    word_count = len(words)
    
    avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
    
    # Simple grammar checks (in production, use language-tool-python or similar)
    errors = 0
    transcript_lower = transcript.lower()
    
    # Check for common errors
    if " i " in transcript_lower or transcript_lower.startswith("i "):
        errors += 1  # Should be "I"
    if "  " in transcript:
        errors += 1  # Double spaces
    if transcript_lower.count("dont") > 0 or transcript_lower.count("wont") > 0:
        errors += 1  # Missing apostrophes
    
    # Score calculation
    error_penalty = min(40, errors * 10)
    length_score = 100 if 10 <= avg_sentence_length <= 20 else max(50, 100 - abs(avg_sentence_length - 15) * 3)
    
    score = max(0, min(100, (length_score - error_penalty)))
    
    return GrammarMetrics(
        error_count=errors,
        sentence_count=sentence_count,
        avg_sentence_length=round(avg_sentence_length, 1),
        score=round(score, 1)
    )


def analyze_visual(has_video: bool, video_bytes: bytes = None) -> VisualMetrics:
    """
    Analyze visual aspects from video using computer vision.
    
    Uses the VideoAnalyzer service which integrates:
    - Face detection (dlib)
    - Gaze tracking (eye landmark analysis)
    - Head pose estimation (PnP)
    - Hand detection (MediaPipe)
    """
    if not has_video or video_bytes is None:
        # Default scores when no video
        return VisualMetrics(
            eye_contact_score=75.0,
            posture_score=75.0,
            hand_gesture_score=70.0
        )
    
    try:
        from ..services.video_analyzer import get_video_analyzer
        
        analyzer = get_video_analyzer()
        result = analyzer.analyze_video_bytes(video_bytes)
        
        return VisualMetrics(
            eye_contact_score=round(result.eye_contact_score, 1),
            posture_score=round(result.posture_score, 1),
            hand_gesture_score=round(result.hand_gesture_score, 1)
        )
        
    except Exception as e:
        # Fallback to defaults if analysis fails
        import logging
        logging.warning(f"Video analysis failed: {e}")
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
    fluency = analyze_fluency(request.transcript, request.audio_duration_seconds)
    grammar = analyze_grammar(request.transcript)
    visual = analyze_visual(request.has_video)
    
    # Calculate overall score using weighted formula
    overall_score = (
        0.35 * fluency.score +
        0.25 * grammar.score +
        0.20 * visual.eye_contact_score +
        0.10 * visual.hand_gesture_score +
        0.10 * visual.posture_score
    )
    
    # Generate feedback
    feedback, strengths, improvements = generate_feedback(fluency, grammar, visual, overall_score)
    
    return SoftSkillsResult(
        fluency=fluency,
        grammar=grammar,
        visual=visual,
        overall_score=round(overall_score, 1),
        breakdown={
            "fluency": {"weight": "35%", "score": fluency.score},
            "grammar": {"weight": "25%", "score": grammar.score},
            "eye_contact": {"weight": "20%", "score": visual.eye_contact_score},
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
    if not video.content_type.startswith('video/'):
        raise HTTPException(status_code=400, detail="File must be a video")
    
    # Read video bytes
    video_bytes = await video.read()
    
    if len(video_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty video file")
    
    # Validate transcript
    if not transcript or len(transcript.strip()) < 10:
        raise HTTPException(status_code=400, detail="Transcript too short for evaluation")
    
    # Analyze each component
    fluency = analyze_fluency(transcript, audio_duration_seconds)
    grammar = analyze_grammar(transcript)
    
    # Analyze video with real CV detection
    visual = analyze_visual(has_video=True, video_bytes=video_bytes)
    
    # Calculate overall score using weighted formula
    overall_score = (
        0.35 * fluency.score +
        0.25 * grammar.score +
        0.20 * visual.eye_contact_score +
        0.10 * visual.hand_gesture_score +
        0.10 * visual.posture_score
    )
    
    # Generate feedback
    feedback, strengths, improvements = generate_feedback(fluency, grammar, visual, overall_score)
    
    return SoftSkillsResult(
        fluency=fluency,
        grammar=grammar,
        visual=visual,
        overall_score=round(overall_score, 1),
        breakdown={
            "fluency": {"weight": "35%", "score": fluency.score},
            "grammar": {"weight": "25%", "score": grammar.score},
            "eye_contact": {"weight": "20%", "score": visual.eye_contact_score},
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
    - Head position (normal/up/down/left/right)
    - Hand visibility
    """
    import base64
    import cv2
    import numpy as np
    
    try:
        # Decode base64 frame
        frame_bytes = base64.b64decode(request.frame_base64)
        frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
        frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid frame data")
        
        # Analyze frame
        from ..services.video_analyzer import get_video_analyzer
        
        analyzer = get_video_analyzer()
        result = analyzer.analyze_frame(frame)
        
        return FrameAnalysisResult(
            face_present=result.face_present,
            gaze_direction=result.gaze_direction,
            head_deviation=result.head_deviation,
            hands_visible=result.hands_visible,
            num_hands=result.num_hands
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Frame analysis error: {str(e)}")
