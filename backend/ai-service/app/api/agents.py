"""
Agents API Endpoints
"""
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from app.utils.auth import verify_jwt_token

router = APIRouter(prefix="/api/agents", tags=["Agents"])


class StudyPlanRequest(BaseModel):
    """Request for study plan generation"""
    weak_topics: List[Dict[str, Any]]
    student_schedule: Dict[str, List[int]] = {}
    upcoming_exams: List[Dict[str, Any]] = []


class AssessmentRequest(BaseModel):
    """Request for assessment generation"""
    weak_topics: List[Dict[str, Any]]
    num_questions: int = Field(10, ge=1, le=50)
    difficulty: str = "medium"


class ModerationRequest(BaseModel):
    """Request for content moderation"""
    message: str


@router.post("/study-plan")
async def generate_study_plan(
    request: StudyPlanRequest,
    user_id: str = Depends(verify_jwt_token)
):
    """
    Generate a personalized study plan.
    """
    try:
        from app.agents.study_planner import StudyPlannerAgent
        
        agent = StudyPlannerAgent()
        
        result = await agent.execute({
            "weak_topics": request.weak_topics,
            "student_schedule": request.student_schedule,
            "upcoming_exams": request.upcoming_exams
        })
        
        return result.get("data", {})
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate-assessment")
async def generate_assessment(
    request: AssessmentRequest,
    user_id: str = Depends(verify_jwt_token)
):
    """
    Generate an adaptive assessment for weak topics.
    """
    try:
        from app.agents.assessment_agent import AssessmentAgent
        
        agent = AssessmentAgent()
        
        result = await agent.execute({
            "weak_topics": request.weak_topics,
            "num_questions": request.num_questions,
            "difficulty": request.difficulty
        })
        
        return result.get("data", {})
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/moderate")
async def moderate_content(
    request: ModerationRequest,
    user_id: str = Depends(verify_jwt_token)
):
    """
    Check if content is appropriate for academic context.
    """
    try:
        from app.agents.moderation import ModerationAgent
        
        agent = ModerationAgent()
        
        result = await agent.execute({
            "message": request.message,
            "user_id": user_id
        })
        
        return result.get("data", {})
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/available")
async def list_available_agents(
    user_id: str = Depends(verify_jwt_token)
):
    """
    List all available agents and their capabilities.
    """
    return {
        "agents": [
            {
                "name": "tutor",
                "description": "Main tutoring agent for answering questions",
                "capabilities": ["answer_questions", "explain_concepts", "provide_citations"]
            },
            {
                "name": "study_planner",
                "description": "Generate personalized study schedules",
                "capabilities": ["identify_weak_topics", "allocate_study_hours", "create_timetables"]
            },
            {
                "name": "assessment",
                "description": "Generate adaptive assessments",
                "capabilities": ["generate_mcqs", "adjust_difficulty", "target_weak_topics"]
            },
            {
                "name": "notes_generator",
                "description": "Create interactive study notes",
                "capabilities": ["generate_notes", "extract_definitions", "create_summaries"]
            },
            {
                "name": "moderation",
                "description": "Filter non-academic content",
                "capabilities": ["classify_intent", "block_off_topic", "log_decisions"]
            }
        ]
    }
