"""
Question Generation & Assessment API Routes
============================================

FastAPI routes for:
- Question generation from content
- Question generation from classroom materials
- Answer evaluation
- Assessment management
"""

from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/questions", tags=["Questions"])


# ============================================================================
# Request/Response Models
# ============================================================================

class GenerateQuestionsRequest(BaseModel):
    """Request to generate questions from provided content"""
    content: str = Field(..., description="Text content to generate questions from")
    question_type: str = Field(default="mcq", description="mcq, descriptive, or short_answer")
    num_questions: int = Field(default=5, ge=1, le=20, description="Number of questions")
    difficulty: str = Field(default="medium", description="easy, medium, or hard")


class GenerateFromMaterialsRequest(BaseModel):
    """Request to generate questions from classroom materials"""
    classroom_id: str = Field(..., description="Classroom ID to search materials in")
    topic_id: Optional[str] = Field(None, description="Optional topic ID to filter")
    question_type: str = Field(default="mcq")
    num_questions: int = Field(default=5, ge=1, le=20)
    difficulty: str = Field(default="medium")


class GenerateMixedAssessmentRequest(BaseModel):
    """Request to generate a mixed assessment"""
    content: str = Field(..., description="Content to generate from")
    mcq_count: int = Field(default=5, ge=0, le=15)
    descriptive_count: int = Field(default=2, ge=0, le=10)
    short_answer_count: int = Field(default=3, ge=0, le=10)
    difficulty: str = Field(default="medium")


class MCQAnswer(BaseModel):
    """MCQ answer for evaluation"""
    question_id: str
    student_answer: str  # A, B, C, or D
    correct_answer: str


class DescriptiveAnswer(BaseModel):
    """Descriptive answer for evaluation"""
    question_id: str
    question_text: str
    student_answer: str
    key_points: Optional[List[str]] = None
    time_taken_seconds: Optional[int] = None


class EvaluateMCQRequest(BaseModel):
    """Request to evaluate MCQ answers"""
    answers: List[MCQAnswer]


class EvaluateDescriptiveRequest(BaseModel):
    """Request to evaluate descriptive answers"""
    answers: List[DescriptiveAnswer]
    reference_content: str = Field(..., description="Reference content for evaluation")


class QuestionResponse(BaseModel):
    """Single question response"""
    question_type: str
    question_text: str
    options: List[Dict[str, str]] = []
    correct_answer: Optional[str] = None  # Hidden for students
    explanation: Optional[str] = None
    key_points: List[str] = []
    difficulty: str
    marks: int
    time_estimate_seconds: int


class EvaluationResponse(BaseModel):
    """Single evaluation response"""
    question_id: str
    question_type: str
    is_correct: bool
    score: float
    max_score: float
    correctness_score: float
    relevance_score: float
    completeness_score: float
    feedback: str
    improvements: str
    topics_to_study: List[str]


# ============================================================================
# Question Generation Routes
# ============================================================================

@router.post("/generate", response_model=Dict[str, Any])
async def generate_questions(request: GenerateQuestionsRequest):
    """
    Generate questions from provided content.
    
    Supports MCQ, descriptive, and short answer question types.
    """
    try:
        from ..services.question_generator import get_question_generator
        
        generator = get_question_generator()
        
        questions = await generator.generate_questions(
            content=request.content,
            question_type=request.question_type,
            num_questions=request.num_questions,
            difficulty=request.difficulty
        )
        
        return {
            "success": True,
            "question_type": request.question_type,
            "difficulty": request.difficulty,
            "count": len(questions),
            "questions": [q.to_dict() for q in questions]
        }
        
    except Exception as e:
        logger.error(f"Error generating questions: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate-from-materials", response_model=Dict[str, Any])
async def generate_from_materials(request: GenerateFromMaterialsRequest):
    """
    Generate questions from classroom materials stored in Qdrant.
    
    Searches the classroom's indexed materials and generates questions.
    """
    try:
        from ..services.question_generator import get_question_generator
        
        generator = get_question_generator()
        
        questions = await generator.generate_from_classroom_materials(
            classroom_id=request.classroom_id,
            topic_id=request.topic_id,
            question_type=request.question_type,
            num_questions=request.num_questions,
            difficulty=request.difficulty
        )
        
        if not questions:
            return {
                "success": True,
                "message": "No questions could be generated. Please ensure the classroom has indexed materials.",
                "count": 0,
                "questions": []
            }
        
        return {
            "success": True,
            "classroom_id": request.classroom_id,
            "topic_id": request.topic_id,
            "question_type": request.question_type,
            "difficulty": request.difficulty,
            "count": len(questions),
            "questions": [q.to_dict() for q in questions]
        }
        
    except Exception as e:
        logger.error(f"Error generating from materials: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate-mixed", response_model=Dict[str, Any])
async def generate_mixed_assessment(request: GenerateMixedAssessmentRequest):
    """
    Generate a mixed assessment with different question types.
    
    Creates a complete assessment with MCQ, descriptive, and short answer questions.
    """
    try:
        from ..services.question_generator import get_question_generator
        
        generator = get_question_generator()
        
        result = await generator.generate_mixed_assessment(
            content=request.content,
            mcq_count=request.mcq_count,
            descriptive_count=request.descriptive_count,
            short_answer_count=request.short_answer_count,
            difficulty=request.difficulty
        )
        
        # Convert to serializable format
        questions = {
            "mcq": [q.to_dict() for q in result["mcq"]],
            "descriptive": [q.to_dict() for q in result["descriptive"]],
            "short_answer": [q.to_dict() for q in result["short_answer"]]
        }
        
        total_count = sum(len(qs) for qs in questions.values())
        
        return {
            "success": True,
            "difficulty": request.difficulty,
            "total_questions": total_count,
            "questions": questions
        }
        
    except Exception as e:
        logger.error(f"Error generating mixed assessment: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Answer Evaluation Routes
# ============================================================================

@router.post("/evaluate-mcq", response_model=Dict[str, Any])
async def evaluate_mcq_answers(request: EvaluateMCQRequest):
    """
    Evaluate MCQ answers.
    
    Fast evaluation - compares student answers to correct answers directly.
    """
    try:
        from ..services.answer_evaluator import get_answer_evaluator
        
        evaluator = get_answer_evaluator()
        
        results = []
        for ans in request.answers:
            result = evaluator.evaluate_mcq(
                question_id=ans.question_id,
                student_answer=ans.student_answer,
                correct_answer=ans.correct_answer
            )
            results.append(result.to_dict())
        
        # Calculate summary
        num_correct = sum(1 for r in results if r["is_correct"])
        total = len(results)
        
        return {
            "success": True,
            "num_questions": total,
            "num_correct": num_correct,
            "num_incorrect": total - num_correct,
            "percentage": round((num_correct / total) * 100, 1) if total > 0 else 0,
            "evaluations": results
        }
        
    except Exception as e:
        logger.error(f"Error evaluating MCQ: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/evaluate-descriptive", response_model=Dict[str, Any])
async def evaluate_descriptive_answers(request: EvaluateDescriptiveRequest):
    """
    Evaluate descriptive answers using LLM.
    
    Provides detailed feedback, scoring on correctness/relevance/completeness,
    and suggestions for improvement.
    """
    try:
        from ..services.answer_evaluator import get_answer_evaluator
        
        evaluator = get_answer_evaluator()
        
        # Convert to expected format
        answers = [
            {
                "question_id": ans.question_id,
                "question_type": "descriptive",
                "question_text": ans.question_text,
                "student_answer": ans.student_answer,
                "key_points": ans.key_points,
                "time_taken_seconds": ans.time_taken_seconds
            }
            for ans in request.answers
        ]
        
        results = await evaluator.evaluate_batch(
            answers=answers,
            reference_content=request.reference_content
        )
        
        # Calculate summary
        summary = evaluator.calculate_total_score(results)
        
        return {
            "success": True,
            "summary": summary,
            "evaluations": [r.to_dict() for r in results]
        }
        
    except Exception as e:
        logger.error(f"Error evaluating descriptive: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Question Bank Routes (CRUD with Core Service)
# ============================================================================

class SaveQuestionsRequest(BaseModel):
    """Request to save generated questions to database"""
    classroom_id: str
    question_bank_name: str
    topic_id: Optional[str] = None
    subtopic_id: Optional[str] = None
    questions: List[Dict[str, Any]]
    created_by: Optional[str] = None


@router.post("/save-to-bank", response_model=Dict[str, Any])
async def save_questions_to_bank(request: SaveQuestionsRequest):
    """
    Save generated questions to the question bank in the core service.
    
    This calls the core service API to persist questions.
    """
    import os
    import httpx
    
    try:
        core_service_url = os.getenv("CORE_SERVICE_URL", "http://localhost:8000")
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            # First, create a question bank
            bank_response = await client.post(
                f"{core_service_url}/api/topics/question-banks",
                json={
                    "classroom_id": request.classroom_id,
                    "name": request.question_bank_name,
                    "description": f"Auto-generated question bank with {len(request.questions)} questions",
                    "source_type": "generated",
                    "created_by": request.created_by
                }
            )
            
            if bank_response.status_code != 201:
                logger.warning(f"Failed to create question bank: {bank_response.text}")
                # Continue anyway, questions can exist without a bank
                question_bank_id = None
            else:
                question_bank_id = bank_response.json().get("id")
            
            # Save each question
            saved_count = 0
            for q in request.questions:
                question_data = {
                    "question_bank_id": question_bank_id,
                    "topic_id": request.topic_id,
                    "subtopic_id": request.subtopic_id,
                    "question_type": q.get("question_type", "mcq"),
                    "question_text": q.get("question_text", ""),
                    "options": q.get("options", []),
                    "correct_answer": q.get("correct_answer", ""),
                    "explanation": q.get("explanation", ""),
                    "key_points": q.get("key_points", []),
                    "difficulty": q.get("difficulty", "medium"),
                    "marks": q.get("marks", 1),
                    "time_estimate_seconds": q.get("time_estimate_seconds", 60),
                    "source_content_preview": q.get("source_content", "")[:500],
                    "created_by": request.created_by
                }
                
                q_response = await client.post(
                    f"{core_service_url}/api/topics/questions",
                    json=question_data
                )
                
                if q_response.status_code == 201:
                    saved_count += 1
                else:
                    logger.warning(f"Failed to save question: {q_response.text}")
        
        return {
            "success": True,
            "question_bank_id": question_bank_id,
            "total_questions": len(request.questions),
            "saved_count": saved_count
        }
        
    except Exception as e:
        logger.error(f"Error saving questions: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """Health check endpoint for questions API"""
    return {
        "status": "healthy",
        "service": "questions-api"
    }
