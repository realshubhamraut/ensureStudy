"""
Grading API Routes
Endpoints for AI-powered assignment grading

POST /api/grade/submission - Grade a student submission
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional
import asyncio

from ...services.grading_service import grading_service, GradingResult

router = APIRouter(prefix="/api/grade", tags=["Grading"])


class GradeSubmissionRequest(BaseModel):
    """Request to grade a submission"""
    assignment_id: str
    submission_id: str
    teacher_pdf_url: str
    student_pdf_urls: List[str]
    max_points: int = 100
    classroom_id: Optional[str] = None
    student_id: Optional[str] = None


class GradeSubmissionResponse(BaseModel):
    """Response from grading request"""
    success: bool
    message: str
    submission_id: str
    grading_started: bool = False
    result: Optional[dict] = None
    error: Optional[str] = None


async def run_grading_task(request: GradeSubmissionRequest):
    """Background task to run grading"""
    try:
        await grading_service.grade_submission(
            assignment_id=request.assignment_id,
            submission_id=request.submission_id,
            teacher_pdf_url=request.teacher_pdf_url,
            student_pdf_urls=request.student_pdf_urls,
            max_points=request.max_points,
            classroom_id=request.classroom_id,
            student_id=request.student_id
        )
    except Exception as e:
        print(f"[Grading] Background task error: {e}")


@router.post("/submission", response_model=GradeSubmissionResponse)
async def grade_submission(
    request: GradeSubmissionRequest,
    background_tasks: BackgroundTasks
):
    """
    Start grading a student submission.
    
    This endpoint immediately returns and runs grading in background.
    Results are sent to core service via callback when complete.
    
    Workflow:
    1. Core service calls this when student submits
    2. This kicks off background grading
    3. Grading service extracts questions from teacher PDF
    4. Grading service extracts/OCRs student submission
    5. AI grades each answer
    6. Results sent to core service callback
    7. Student receives notification
    """
    try:
        # Validate request
        if not request.teacher_pdf_url:
            raise HTTPException(status_code=400, detail="Teacher PDF URL required")
        
        if not request.student_pdf_urls:
            raise HTTPException(status_code=400, detail="Student PDF URL(s) required")
        
        print(f"[Grading] Received grading request for submission {request.submission_id}")
        
        # Start grading in background
        background_tasks.add_task(run_grading_task, request)
        
        return GradeSubmissionResponse(
            success=True,
            message="Grading started in background",
            submission_id=request.submission_id,
            grading_started=True
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"[Grading] Error starting grading: {e}")
        return GradeSubmissionResponse(
            success=False,
            message="Failed to start grading",
            submission_id=request.submission_id,
            grading_started=False,
            error=str(e)
        )


@router.post("/submission/sync", response_model=GradeSubmissionResponse)
async def grade_submission_sync(request: GradeSubmissionRequest):
    """
    Grade a submission synchronously (for testing).
    Waits for grading to complete before returning.
    """
    try:
        print(f"[Grading] Sync grading for submission {request.submission_id}")
        
        result = await grading_service.grade_submission(
            assignment_id=request.assignment_id,
            submission_id=request.submission_id,
            teacher_pdf_url=request.teacher_pdf_url,
            student_pdf_urls=request.student_pdf_urls,
            max_points=request.max_points,
            classroom_id=request.classroom_id,
            student_id=request.student_id
        )
        
        return GradeSubmissionResponse(
            success=True,
            message="Grading complete",
            submission_id=request.submission_id,
            grading_started=True,
            result=result.to_dict()
        )
        
    except Exception as e:
        print(f"[Grading] Sync grading error: {e}")
        return GradeSubmissionResponse(
            success=False,
            message="Grading failed",
            submission_id=request.submission_id,
            grading_started=True,
            error=str(e)
        )


@router.get("/health")
async def grading_health():
    """Check grading service health"""
    pdf_available = grading_service.embedding_model is not None
    return {
        "status": "ok",
        "embedding_model_loaded": pdf_available,
        "service": "grading"
    }
