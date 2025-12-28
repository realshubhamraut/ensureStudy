"""
Evaluation API Routes

Human-in-the-Loop Automated Evaluation System

Endpoints:
- POST /api/evaluation/upload - Upload image/video
- POST /api/evaluation/process - Run full pipeline
- POST /api/evaluation/score - Score extracted text
- POST /api/evaluation/review - Teacher approval
- GET /api/evaluation/{id} - Get evaluation status
"""
import time
import uuid
import base64
import io
from typing import Optional, List, Dict, Any
from enum import Enum
from datetime import datetime

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel, Field

# Import ML utilities (will be created from notebooks)
import sys
from pathlib import Path

# Add ml directory to path
ML_PATH = Path(__file__).parent.parent.parent.parent.parent / "ml"
sys.path.insert(0, str(ML_PATH))


# ============================================================================
# Schemas
# ============================================================================

class SubjectType(str, Enum):
    MATH = "math"
    PHYSICS = "physics"
    CHEMISTRY = "chemistry"
    BIOLOGY = "biology"
    ENGLISH = "english"
    GENERAL = "general"


class EvaluationStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    AWAITING_REVIEW = "awaiting_review"
    APPROVED = "approved"
    REJECTED = "rejected"


# ============================================================================
# Question Paper Schemas
# ============================================================================

class ParsedQuestion(BaseModel):
    """Single question from a paper."""
    number: str
    text: str
    marks: float
    section: Optional[str] = None
    question_type: str = "short_answer"
    expected_answer: Optional[str] = None
    keywords: List[str] = Field(default_factory=list)


class QuestionPaperData(BaseModel):
    """Complete parsed question paper."""
    id: str
    title: str
    subject: SubjectType
    total_marks: float
    time_limit_minutes: int
    sections: List[str] = Field(default_factory=list)
    questions: List[ParsedQuestion] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    raw_text: str = ""


# ============================================================================
# Answer Evaluation Schemas
# ============================================================================


class ImageQuality(BaseModel):
    """Image quality assessment result."""
    score: float = Field(..., ge=0, le=1000, description="Quality score (higher is better)")
    label: str = Field(..., description="Quality label (good, blurry, etc.)")
    is_acceptable: bool = Field(..., description="Whether image passed quality check")
    skew_angle: float = Field(0.0, description="Detected skew angle in degrees")


class RecognitionResult(BaseModel):
    """Text recognition result."""
    extracted_text: str = Field(..., description="Recognized text")
    confidence: float = Field(..., ge=0, le=1, description="Recognition confidence")
    regions: List[Dict[str, Any]] = Field(default_factory=list, description="Text regions detected")


class ScoringBreakdown(BaseModel):
    """Score breakdown by component."""
    semantic: float = Field(0.0, description="Semantic similarity score")
    keyword: float = Field(0.0, description="Keyword matching score")
    steps: float = Field(0.0, description="Step matching score")


class ScoringResult(BaseModel):
    """Answer scoring result."""
    score: float = Field(..., description="Computed score")
    max_marks: float = Field(..., description="Maximum possible marks")
    confidence: float = Field(..., ge=0, le=1, description="Scoring confidence")
    breakdown: ScoringBreakdown = Field(..., description="Score breakdown")
    matched_keywords: List[str] = Field(default_factory=list)
    missing_keywords: List[str] = Field(default_factory=list)
    feedback: List[str] = Field(default_factory=list)


class EvaluationRecord(BaseModel):
    """Complete evaluation record."""
    id: str = Field(..., description="Unique evaluation ID")
    student_id: str = Field(..., description="Student identifier")
    subject: SubjectType = Field(..., description="Subject")
    question_id: str = Field(..., description="Question identifier")
    
    # Processing results
    image_quality: Optional[ImageQuality] = None
    recognition: Optional[RecognitionResult] = None
    scoring: Optional[ScoringResult] = None
    
    # Status
    status: EvaluationStatus = Field(EvaluationStatus.PENDING)
    created_at: datetime = Field(default_factory=datetime.now)
    reviewed_at: Optional[datetime] = None
    reviewed_by: Optional[str] = None
    
    # Teacher overrides
    final_score: Optional[float] = None
    teacher_comments: Optional[str] = None


# Request/Response models
class UploadRequest(BaseModel):
    student_id: str
    subject: SubjectType = SubjectType.GENERAL
    question_id: str
    

class ProcessRequest(BaseModel):
    evaluation_id: str
    reference_answer: str
    keywords: Dict[str, float] = Field(default_factory=dict)
    steps: List[str] = Field(default_factory=list)
    max_marks: float = 10.0


class ReviewRequest(BaseModel):
    evaluation_id: str
    approved: bool
    final_score: Optional[float] = None
    teacher_comments: Optional[str] = None
    reviewer_id: str


class EvaluationResponse(BaseModel):
    success: bool
    data: Optional[EvaluationRecord] = None
    error: Optional[Dict[str, str]] = None


# ============================================================================
# In-memory storage (replace with database in production)
# ============================================================================

EVALUATIONS: Dict[str, EvaluationRecord] = {}
UPLOADED_IMAGES: Dict[str, bytes] = {}


# ============================================================================
# Router
# ============================================================================

router = APIRouter(prefix="/api/evaluation", tags=["Evaluation"])


@router.post("/upload", response_model=EvaluationResponse)
async def upload_image(
    file: UploadFile = File(...),
    student_id: str = Form(...),
    subject: SubjectType = Form(SubjectType.GENERAL),
    question_id: str = Form(...)
):
    """
    Upload an answer sheet image for evaluation.
    
    Returns evaluation ID for tracking.
    """
    # Validate file type
    allowed_types = ["image/jpeg", "image/png", "image/webp", "video/mp4"]
    if file.content_type not in allowed_types:
        raise HTTPException(400, f"Invalid file type. Allowed: {allowed_types}")
    
    # Generate evaluation ID
    eval_id = f"eval_{uuid.uuid4().hex[:12]}"
    
    # Read file content
    content = await file.read()
    UPLOADED_IMAGES[eval_id] = content
    
    # Create evaluation record
    record = EvaluationRecord(
        id=eval_id,
        student_id=student_id,
        subject=subject,
        question_id=question_id,
        status=EvaluationStatus.PENDING
    )
    
    EVALUATIONS[eval_id] = record
    
    # Run image quality check
    quality = await _check_image_quality(content)
    record.image_quality = quality
    
    return EvaluationResponse(success=True, data=record)


@router.post("/process", response_model=EvaluationResponse)
async def process_evaluation(request: ProcessRequest):
    """
    Run full evaluation pipeline:
    1. Extract text from image
    2. Score against reference answer
    3. Generate feedback
    """
    eval_id = request.evaluation_id
    
    if eval_id not in EVALUATIONS:
        raise HTTPException(404, f"Evaluation {eval_id} not found")
    
    record = EVALUATIONS[eval_id]
    record.status = EvaluationStatus.PROCESSING
    
    try:
        # Step 1: Get uploaded image
        if eval_id not in UPLOADED_IMAGES:
            raise HTTPException(400, "Image not found. Please upload again.")
        
        image_bytes = UPLOADED_IMAGES[eval_id]
        
        # Step 2: Text recognition (HTR)
        recognition = await _recognize_text(image_bytes)
        record.recognition = recognition
        
        # Step 3: Score answer
        scoring = await _score_answer(
            student_answer=recognition.extracted_text,
            reference_answer=request.reference_answer,
            keywords=request.keywords,
            steps=request.steps,
            max_marks=request.max_marks
        )
        record.scoring = scoring
        
        # Update status
        record.status = EvaluationStatus.AWAITING_REVIEW
        
        return EvaluationResponse(success=True, data=record)
    
    except Exception as e:
        record.status = EvaluationStatus.PENDING
        return EvaluationResponse(
            success=False,
            error={"code": "processing_error", "message": str(e)}
        )


@router.post("/review", response_model=EvaluationResponse)
async def review_evaluation(request: ReviewRequest):
    """
    Teacher review and approval.
    
    Teacher can approve AI suggestion or override with custom score.
    """
    eval_id = request.evaluation_id
    
    if eval_id not in EVALUATIONS:
        raise HTTPException(404, f"Evaluation {eval_id} not found")
    
    record = EVALUATIONS[eval_id]
    
    if record.status != EvaluationStatus.AWAITING_REVIEW:
        raise HTTPException(400, f"Evaluation is not awaiting review. Status: {record.status}")
    
    # Apply review
    record.reviewed_at = datetime.now()
    record.reviewed_by = request.reviewer_id
    record.teacher_comments = request.teacher_comments
    
    if request.approved:
        # Use AI score or teacher override
        record.final_score = request.final_score if request.final_score is not None else (
            record.scoring.score if record.scoring else 0
        )
        record.status = EvaluationStatus.APPROVED
    else:
        record.status = EvaluationStatus.REJECTED
        record.final_score = request.final_score or 0
    
    return EvaluationResponse(success=True, data=record)


@router.get("/{evaluation_id}", response_model=EvaluationResponse)
async def get_evaluation(evaluation_id: str):
    """
    Get evaluation status and details.
    """
    if evaluation_id not in EVALUATIONS:
        raise HTTPException(404, f"Evaluation {evaluation_id} not found")
    
    return EvaluationResponse(success=True, data=EVALUATIONS[evaluation_id])


@router.get("/student/{student_id}", response_model=Dict[str, Any])
async def get_student_evaluations(student_id: str, limit: int = 20):
    """
    Get all evaluations for a student.
    """
    student_evals = [
        e for e in EVALUATIONS.values()
        if e.student_id == student_id
    ]
    
    # Sort by created_at descending
    student_evals.sort(key=lambda x: x.created_at, reverse=True)
    
    return {
        "success": True,
        "count": len(student_evals),
        "evaluations": student_evals[:limit]
    }


# ============================================================================
# Internal functions (use ML models from notebooks)
# ============================================================================

async def _check_image_quality(image_bytes: bytes) -> ImageQuality:
    """
    Check image quality using preprocessing pipeline.
    """
    try:
        import cv2
        import numpy as np
        
        # Decode image
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            return ImageQuality(
                score=0,
                label="invalid",
                is_acceptable=False,
                skew_angle=0
            )
        
        # Blur detection (Laplacian variance)
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        variance = laplacian.var()
        
        # Quality label
        if variance < 50:
            label = "very_blurry"
            acceptable = False
        elif variance < 100:
            label = "slightly_blurry"
            acceptable = True  # Still acceptable
        elif variance < 300:
            label = "good"
            acceptable = True
        else:
            label = "very_sharp"
            acceptable = True
        
        # Skew detection (simplified)
        edges = cv2.Canny(image, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
        
        skew_angle = 0.0
        if lines is not None:
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                if -45 < angle < 45:
                    angles.append(angle)
            if angles:
                skew_angle = float(np.median(angles))
        
        return ImageQuality(
            score=float(variance),
            label=label,
            is_acceptable=acceptable,
            skew_angle=round(skew_angle, 2)
        )
    
    except Exception as e:
        print(f"Quality check error: {e}")
        return ImageQuality(
            score=50,
            label="unknown",
            is_acceptable=True,
            skew_angle=0
        )


async def _recognize_text(image_bytes: bytes) -> RecognitionResult:
    """
    Extract text from image using HTR model.
    
    Falls back to mock response if model not available.
    """
    try:
        # Try to use trained HTR model
        # from ml.models.htr_model import HTRModel, CharacterSet, recognize_text
        # ... model inference ...
        
        # For now, return mock response for development
        # In production, load actual model
        mock_texts = [
            "F = ma = 5 × 10 = 50N",
            "The answer is 42",
            "Newton's first law states that objects at rest stay at rest",
            "E = mc²",
            "Step 1: Identify given values. Step 2: Apply formula. Step 3: Calculate result."
        ]
        
        import random
        extracted = random.choice(mock_texts)
        
        return RecognitionResult(
            extracted_text=extracted,
            confidence=0.85,
            regions=[
                {"x": 10, "y": 20, "width": 200, "height": 30, "text": extracted}
            ]
        )
    
    except Exception as e:
        print(f"Recognition error: {e}")
        return RecognitionResult(
            extracted_text="[Recognition failed]",
            confidence=0.0,
            regions=[]
        )


async def _score_answer(
    student_answer: str,
    reference_answer: str,
    keywords: Dict[str, float],
    steps: List[str],
    max_marks: float
) -> ScoringResult:
    """
    Score student answer against reference.
    
    Uses scoring engine from notebooks.
    """
    try:
        # Try to use scoring engine
        # from ml.utils.scoring import AnswerScoringEngine, AnswerKey
        
        # Fallback: simple similarity-based scoring
        student_words = set(student_answer.lower().split())
        reference_words = set(reference_answer.lower().split())
        
        # Jaccard similarity
        if student_words | reference_words:
            semantic = len(student_words & reference_words) / len(student_words | reference_words)
        else:
            semantic = 0.0
        
        # Keyword matching
        matched_kw = []
        missing_kw = []
        kw_score = 0.0
        total_weight = sum(keywords.values()) if keywords else 1.0
        
        for kw, weight in keywords.items():
            if kw.lower() in student_answer.lower():
                matched_kw.append(kw)
                kw_score += weight
            else:
                missing_kw.append(kw)
        
        keyword_ratio = kw_score / total_weight if total_weight > 0 else semantic
        
        # Combined score
        final_ratio = (semantic * 0.5) + (keyword_ratio * 0.5)
        score = round(final_ratio * max_marks, 2)
        confidence = min(0.95, semantic + 0.2)
        
        # Generate feedback
        feedback = []
        if missing_kw:
            feedback.append(f"Consider mentioning: {', '.join(missing_kw[:3])}")
        
        if final_ratio >= 0.8:
            feedback.append("✅ Excellent answer!")
        elif final_ratio >= 0.5:
            feedback.append("⚠️ Partial credit - room for improvement")
        else:
            feedback.append("❌ Review this concept")
        
        return ScoringResult(
            score=score,
            max_marks=max_marks,
            confidence=round(confidence, 2),
            breakdown=ScoringBreakdown(
                semantic=round(semantic, 2),
                keyword=round(keyword_ratio, 2),
                steps=0.0  # Simplified for now
            ),
            matched_keywords=matched_kw,
            missing_keywords=missing_kw,
            feedback=feedback
        )
    
    except Exception as e:
        print(f"Scoring error: {e}")
        return ScoringResult(
            score=0,
            max_marks=max_marks,
            confidence=0.5,
            breakdown=ScoringBreakdown(),
            feedback=["Error during scoring"]
        )
