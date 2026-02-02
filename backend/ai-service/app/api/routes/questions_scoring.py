"""
Questions & Scoring API Routes

Handles:
- Generate questions for topics
- Submit MCQ answers
- Submit descriptive answers
- Score and update topic mastery
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import logging
import httpx
import os

from app.services.question_scoring_service import (
    get_question_generator,
    get_answer_scorer,
    ScoringResult
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/questions", tags=["Questions & Scoring"])


# ============================================================================
# Request/Response Models
# ============================================================================

class GenerateQuestionsRequest(BaseModel):
    topic_id: str
    topic_name: str
    topic_description: Optional[str] = ""
    key_concepts: List[str] = []
    difficulty: str = "medium"
    mcq_count: int = 5
    descriptive_count: int = 2


class MCQResponse(BaseModel):
    id: str
    question_text: str
    options: List[Dict[str, Any]]
    difficulty: str
    marks: int = 1


class DescriptiveQuestionResponse(BaseModel):
    id: str
    question_text: str
    difficulty: str
    marks: int = 10


class SubmitMCQRequest(BaseModel):
    user_id: str
    question_id: str
    selected_option: str  # A, B, C, D
    response_time_ms: Optional[int] = None
    source: str = "assessment"  # assessment, practice
    source_session_id: Optional[str] = None


class SubmitDescriptiveRequest(BaseModel):
    user_id: str
    question_id: str
    answer: str
    response_time_ms: Optional[int] = None
    source: str = "mock_interview"
    source_session_id: Optional[str] = None


class SubmitResponseResult(BaseModel):
    score_awarded: float
    max_score: float
    score_percentage: float
    is_correct: Optional[bool]
    feedback: str
    mastery_updated: bool = False


class BatchSubmitMCQRequest(BaseModel):
    user_id: str
    source: str = "assessment"
    source_session_id: Optional[str] = None
    answers: List[Dict[str, Any]]  # [{question_id, selected_option, response_time_ms}]


# ============================================================================
# API Endpoints
# ============================================================================

@router.post("/generate", response_model=Dict[str, Any])
async def generate_questions(request: GenerateQuestionsRequest):
    """
    Generate MCQ and descriptive questions for a topic.
    
    Returns questions that can be reviewed/saved by teacher.
    """
    generator = get_question_generator()
    
    # Generate MCQs
    mcqs = generator.generate_mcq(
        topic_name=request.topic_name,
        topic_description=request.topic_description,
        key_concepts=request.key_concepts,
        difficulty=request.difficulty,
        count=request.mcq_count
    )
    
    # Generate descriptive
    descriptives = generator.generate_descriptive(
        topic_name=request.topic_name,
        topic_description=request.topic_description,
        key_concepts=request.key_concepts,
        difficulty=request.difficulty,
        count=request.descriptive_count
    )
    
    # Format response
    return {
        "topic_id": request.topic_id,
        "topic_name": request.topic_name,
        "mcq_questions": [
            {
                "question_text": q.question_text,
                "options": [{"id": o.id, "text": o.text, "is_correct": o.is_correct} for o in q.options],
                "correct_answer": q.correct_answer,
                "explanation": q.explanation,
                "difficulty": q.difficulty
            }
            for q in mcqs
        ],
        "descriptive_questions": [
            {
                "question_text": q.question_text,
                "expected_answer": q.expected_answer,
                "key_points": q.key_points,
                "explanation": q.explanation,
                "difficulty": q.difficulty
            }
            for q in descriptives
        ],
        "total_mcq": len(mcqs),
        "total_descriptive": len(descriptives)
    }


@router.post("/generate-and-save", response_model=Dict[str, Any])
async def generate_and_save_questions(request: GenerateQuestionsRequest):
    """
    Generate questions and save them to database.
    
    1. Generate questions
    2. Save to TopicQuestion table via Core API
    3. Return saved questions with IDs
    """
    generator = get_question_generator()
    core_api_url = os.getenv("CORE_API_URL", "http://localhost:8000")
    
    # Generate questions
    mcqs = generator.generate_mcq(
        topic_name=request.topic_name,
        topic_description=request.topic_description,
        key_concepts=request.key_concepts,
        difficulty=request.difficulty,
        count=request.mcq_count
    )
    
    descriptives = generator.generate_descriptive(
        topic_name=request.topic_name,
        topic_description=request.topic_description,
        key_concepts=request.key_concepts,
        difficulty=request.difficulty,
        count=request.descriptive_count
    )
    
    saved_mcqs = []
    saved_descriptives = []
    
    async with httpx.AsyncClient() as client:
        # Save MCQs
        for q in mcqs:
            payload = {
                "classroom_topic_id": request.topic_id,
                "question_type": "mcq",
                "question_text": q.question_text,
                "options": [{"id": o.id, "text": o.text, "is_correct": o.is_correct} for o in q.options],
                "correct_answer": q.correct_answer,
                "explanation": q.explanation,
                "difficulty": q.difficulty,
                "marks": 1
            }
            
            try:
                resp = await client.post(
                    f"{core_api_url}/api/classroom/topics/{request.topic_id}/questions",
                    json=payload,
                    timeout=30.0
                )
                if resp.status_code in (200, 201):
                    saved_mcqs.append(resp.json())
            except Exception as e:
                logger.warning(f"Failed to save MCQ: {e}")
        
        # Save descriptives
        for q in descriptives:
            payload = {
                "classroom_topic_id": request.topic_id,
                "question_type": "descriptive",
                "question_text": q.question_text,
                "expected_answer": q.expected_answer,
                "key_points": q.key_points,
                "explanation": q.explanation,
                "difficulty": q.difficulty,
                "marks": 10
            }
            
            try:
                resp = await client.post(
                    f"{core_api_url}/api/classroom/topics/{request.topic_id}/questions",
                    json=payload,
                    timeout=30.0
                )
                if resp.status_code in (200, 201):
                    saved_descriptives.append(resp.json())
            except Exception as e:
                logger.warning(f"Failed to save descriptive: {e}")
    
    return {
        "topic_id": request.topic_id,
        "mcq_saved": len(saved_mcqs),
        "descriptive_saved": len(saved_descriptives),
        "mcq_questions": saved_mcqs,
        "descriptive_questions": saved_descriptives
    }


@router.post("/submit-mcq", response_model=SubmitResponseResult)
async def submit_mcq_answer(request: SubmitMCQRequest):
    """
    Submit an MCQ answer for scoring.
    
    1. Fetch question from database
    2. Score answer
    3. Create response record
    4. Update student topic score
    """
    core_api_url = os.getenv("CORE_API_URL", "http://localhost:8000")
    scorer = get_answer_scorer()
    
    async with httpx.AsyncClient() as client:
        # Get question details
        try:
            resp = await client.get(
                f"{core_api_url}/api/classroom/questions/{request.question_id}",
                timeout=30.0
            )
            if resp.status_code != 200:
                raise HTTPException(404, "Question not found")
            
            question = resp.json()
        except httpx.RequestError as e:
            raise HTTPException(500, f"Failed to fetch question: {e}")
        
        # Score the answer
        result = scorer.score_mcq(
            selected_option=request.selected_option,
            correct_answer=question.get("correct_answer", "A"),
            marks=question.get("marks", 1)
        )
        
        # Save response record
        response_payload = {
            "user_id": request.user_id,
            "question_id": request.question_id,
            "response_type": "mcq",
            "selected_option": request.selected_option,
            "is_correct": result.is_correct,
            "score_awarded": result.score_awarded,
            "max_score": result.max_score,
            "score_percentage": result.score_percentage,
            "response_time_ms": request.response_time_ms,
            "source": request.source,
            "source_session_id": request.source_session_id
        }
        
        try:
            resp = await client.post(
                f"{core_api_url}/api/classroom/responses",
                json=response_payload,
                timeout=30.0
            )
        except Exception as e:
            logger.warning(f"Failed to save response: {e}")
        
        # Update topic score
        topic_id = question.get("classroom_topic_id")
        if topic_id:
            try:
                await client.post(
                    f"{core_api_url}/api/classroom/topic-scores/update-mcq",
                    json={
                        "user_id": request.user_id,
                        "topic_id": topic_id,
                        "correct": result.is_correct,
                        "marks": question.get("marks", 1)
                    },
                    timeout=30.0
                )
            except Exception as e:
                logger.warning(f"Failed to update topic score: {e}")
    
    return SubmitResponseResult(
        score_awarded=result.score_awarded,
        max_score=result.max_score,
        score_percentage=result.score_percentage,
        is_correct=result.is_correct,
        feedback=result.feedback,
        mastery_updated=True
    )


@router.post("/submit-descriptive", response_model=SubmitResponseResult)
async def submit_descriptive_answer(request: SubmitDescriptiveRequest):
    """
    Submit a descriptive answer for LLM scoring.
    
    1. Fetch question with key points
    2. Score with LLM
    3. Create response record
    4. Update student topic score
    """
    core_api_url = os.getenv("CORE_API_URL", "http://localhost:8000")
    scorer = get_answer_scorer()
    
    async with httpx.AsyncClient() as client:
        # Get question details
        try:
            resp = await client.get(
                f"{core_api_url}/api/classroom/questions/{request.question_id}",
                params={"include_answer": "true"},
                timeout=30.0
            )
            if resp.status_code != 200:
                raise HTTPException(404, "Question not found")
            
            question = resp.json()
        except httpx.RequestError as e:
            raise HTTPException(500, f"Failed to fetch question: {e}")
        
        # Score the answer with LLM
        result = scorer.score_descriptive(
            student_answer=request.answer,
            question_text=question.get("question_text", ""),
            key_points=question.get("key_points", []),
            marks=question.get("marks", 10)
        )
        
        # Save response record
        response_payload = {
            "user_id": request.user_id,
            "question_id": request.question_id,
            "response_type": "descriptive",
            "descriptive_response": request.answer,
            "matched_key_points": result.matched_key_points,
            "score_awarded": result.score_awarded,
            "max_score": result.max_score,
            "score_percentage": result.score_percentage,
            "ai_feedback": result.feedback,
            "ai_confidence": result.confidence,
            "response_time_ms": request.response_time_ms,
            "source": request.source,
            "source_session_id": request.source_session_id
        }
        
        try:
            resp = await client.post(
                f"{core_api_url}/api/classroom/responses",
                json=response_payload,
                timeout=30.0
            )
        except Exception as e:
            logger.warning(f"Failed to save response: {e}")
        
        # Update topic score
        topic_id = question.get("classroom_topic_id")
        if topic_id:
            try:
                await client.post(
                    f"{core_api_url}/api/classroom/topic-scores/update-descriptive",
                    json={
                        "user_id": request.user_id,
                        "topic_id": topic_id,
                        "score_awarded": result.score_awarded,
                        "max_score": result.max_score
                    },
                    timeout=30.0
                )
            except Exception as e:
                logger.warning(f"Failed to update topic score: {e}")
    
    return SubmitResponseResult(
        score_awarded=result.score_awarded,
        max_score=result.max_score,
        score_percentage=result.score_percentage,
        is_correct=None,
        feedback=result.feedback,
        mastery_updated=True
    )


@router.post("/submit-batch-mcq", response_model=Dict[str, Any])
async def submit_batch_mcq(request: BatchSubmitMCQRequest):
    """
    Submit multiple MCQ answers at once (for assessment completion).
    
    Returns summary of scores and updates all topic scores.
    """
    results = []
    total_score = 0.0
    total_max = 0.0
    correct_count = 0
    
    for answer in request.answers:
        submit_req = SubmitMCQRequest(
            user_id=request.user_id,
            question_id=answer["question_id"],
            selected_option=answer["selected_option"],
            response_time_ms=answer.get("response_time_ms"),
            source=request.source,
            source_session_id=request.source_session_id
        )
        
        try:
            result = await submit_mcq_answer(submit_req)
            results.append({
                "question_id": answer["question_id"],
                "is_correct": result.is_correct,
                "score_awarded": result.score_awarded
            })
            total_score += result.score_awarded
            total_max += result.max_score
            if result.is_correct:
                correct_count += 1
        except Exception as e:
            logger.error(f"Failed to score MCQ {answer['question_id']}: {e}")
            results.append({
                "question_id": answer["question_id"],
                "error": str(e)
            })
    
    return {
        "user_id": request.user_id,
        "total_questions": len(request.answers),
        "correct_count": correct_count,
        "total_score": total_score,
        "total_max": total_max,
        "percentage": (total_score / total_max * 100) if total_max > 0 else 0,
        "results": results
    }


@router.get("/topic/{topic_id}", response_model=Dict[str, Any])
async def get_topic_questions(
    topic_id: str,
    question_type: Optional[str] = Query(None, description="mcq or descriptive"),
    include_answers: bool = Query(False, description="Include correct answers")
):
    """
    Get all questions for a topic.
    """
    core_api_url = os.getenv("CORE_API_URL", "http://localhost:8000")
    
    async with httpx.AsyncClient() as client:
        try:
            params = {"include_answers": str(include_answers).lower()}
            if question_type:
                params["question_type"] = question_type
            
            resp = await client.get(
                f"{core_api_url}/api/classroom/topics/{topic_id}/questions",
                params=params,
                timeout=30.0
            )
            
            if resp.status_code != 200:
                raise HTTPException(resp.status_code, "Failed to fetch questions")
            
            return resp.json()
        except httpx.RequestError as e:
            raise HTTPException(500, f"Failed to fetch questions: {e}")
