"""
Topic Scores Aggregation API

Aggregates confidence scores from:
- Assessments (40% weight)
- Mock Interviews (40% weight)
- Study Completion (20% weight)
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
from datetime import datetime, date, timedelta
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/curriculum", tags=["curriculum-scores"])


# ============================================================================
# Models
# ============================================================================

class TopicScore(BaseModel):
    topic_id: str
    topic_name: str
    confidence_score: float  # 0-100
    assessment_score: Optional[float] = None
    interview_score: Optional[float] = None
    completion_score: Optional[float] = None
    scheduled_date: Optional[str] = None
    status: str = "not_started"  # not_started, in_progress, completed
    last_activity: Optional[str] = None


class ScheduledTopic(BaseModel):
    topic_id: str
    topic_name: str
    date: str  # ISO date
    confidence_score: float
    status: str
    unit: Optional[str] = None
    chapter: Optional[str] = None


class WeeklySchedule(BaseModel):
    week_start: str
    week_end: str
    days: Dict[str, List[ScheduledTopic]]  # date -> topics


class RescheduleRequest(BaseModel):
    topic_id: str
    curriculum_id: str
    new_date: str


class TopicScoresResponse(BaseModel):
    user_id: str
    topics: List[TopicScore]
    overall_confidence: float


# ============================================================================
# In-Memory Storage (Replace with DB in production)
# ============================================================================

# Store topic schedules: {curriculum_id: {topic_id: scheduled_date}}
_topic_schedules: Dict[str, Dict[str, str]] = {}

# Store assessment scores: {user_id: {topic_name: [scores]}}
_assessment_scores: Dict[str, Dict[str, List[float]]] = {}

# Store mock interview scores: {user_id: {topic_name: [scores]}}
_interview_scores: Dict[str, Dict[str, List[float]]] = {}


# ============================================================================
# Score Calculation
# ============================================================================

def calculate_topic_confidence(
    user_id: str,
    topic_name: str
) -> tuple[float, float, float, float]:
    """
    Calculate topic confidence from multiple sources.
    
    Returns: (overall, assessment_avg, interview_avg, completion)
    """
    # Get assessment scores for this topic
    assessment_scores = _assessment_scores.get(user_id, {}).get(topic_name.lower(), [])
    assessment_avg = sum(assessment_scores) / len(assessment_scores) if assessment_scores else 0
    
    # Get mock interview scores for this topic
    interview_scores = _interview_scores.get(user_id, {}).get(topic_name.lower(), [])
    interview_avg = sum(interview_scores) / len(interview_scores) if interview_scores else 0
    
    # Completion score (placeholder - integrate with actual study tracking)
    completion = 0
    
    # Calculate weighted average
    # Only count sources that have data
    weights = []
    scores = []
    
    if assessment_scores:
        weights.append(0.4)
        scores.append(assessment_avg)
    if interview_scores:
        weights.append(0.4)
        scores.append(interview_avg)
    if completion > 0:
        weights.append(0.2)
        scores.append(completion)
    
    if weights:
        # Normalize weights
        total_weight = sum(weights)
        overall = sum(s * (w / total_weight) for s, w in zip(scores, weights))
    else:
        overall = 0
    
    return overall, assessment_avg, interview_avg, completion


# ============================================================================
# API Endpoints
# ============================================================================

@router.get("/topic-scores/{user_id}", response_model=TopicScoresResponse)
async def get_topic_scores(user_id: str, curriculum_id: Optional[str] = None):
    """
    Get aggregated topic scores for a user.
    Scores are calculated from assessments and mock interviews.
    """
    from app.services.curriculum_storage import get_curriculum_storage
    storage = get_curriculum_storage()
    
    topics_with_scores = []
    
    # Get user's curricula
    user_curricula = storage.get_user_curricula(user_id)
    
    if curriculum_id:
        user_curricula = [c for c in user_curricula if c.get("id") == curriculum_id]
    
    for curriculum in user_curricula:
        for topic in curriculum.get("topics", []):
            topic_id = topic.get("id", "")
            topic_name = topic.get("name", "")
            
            # Calculate confidence
            overall, assess, interview, completion = calculate_topic_confidence(user_id, topic_name)
            
            # Get scheduled date
            scheduled = _topic_schedules.get(curriculum.get("id", ""), {}).get(topic_id)
            
            topics_with_scores.append(TopicScore(
                topic_id=topic_id,
                topic_name=topic_name,
                confidence_score=round(overall, 1),
                assessment_score=round(assess, 1) if assess else None,
                interview_score=round(interview, 1) if interview else None,
                completion_score=round(completion, 1) if completion else None,
                scheduled_date=scheduled,
                status="completed" if overall >= 80 else "in_progress" if overall > 0 else "not_started"
            ))
    
    # Calculate overall confidence
    if topics_with_scores:
        overall_confidence = sum(t.confidence_score for t in topics_with_scores) / len(topics_with_scores)
    else:
        overall_confidence = 0
    
    return TopicScoresResponse(
        user_id=user_id,
        topics=topics_with_scores,
        overall_confidence=round(overall_confidence, 1)
    )


@router.get("/schedule/{curriculum_id}", response_model=WeeklySchedule)
async def get_weekly_schedule(
    curriculum_id: str,
    week_offset: int = 0  # 0 = current week, -1 = last week, 1 = next week
):
    """
    Get weekly schedule for a curriculum.
    """
    from app.services.curriculum_storage import get_curriculum_storage
    storage = get_curriculum_storage()
    
    curriculum = storage.get_curriculum(curriculum_id)
    if not curriculum:
        raise HTTPException(status_code=404, detail="Curriculum not found")
    
    user_id = curriculum.get("user_id", "")
    
    # Calculate week boundaries
    today = date.today()
    start_of_week = today - timedelta(days=today.weekday())  # Monday
    start_of_week += timedelta(weeks=week_offset)
    end_of_week = start_of_week + timedelta(days=6)
    
    # Initialize days
    days = {}
    for i in range(7):
        day_date = start_of_week + timedelta(days=i)
        days[day_date.isoformat()] = []
    
    # Get scheduled topics
    topic_schedules = _topic_schedules.get(curriculum_id, {})
    
    # Build topics lookup by both id and name
    topics_list = curriculum.get("topics", [])
    topics_by_id = {t.get("id", ""): t for t in topics_list if t.get("id")}
    topics_by_name = {t.get("name", ""): t for t in topics_list if t.get("name")}
    
    # If no schedules exist, auto-schedule from daily_goals
    if not topic_schedules:
        daily_goals = curriculum.get("daily_goals", [])
        
        for goal in daily_goals:
            goal_date = goal.get("date", "")
            if goal_date:
                for topic_name in goal.get("topics", []):
                    # Try to find topic by name first, then by ID
                    topic = topics_by_name.get(topic_name) or topics_by_id.get(topic_name, {})
                    topic_id = topic.get("id", topic_name)  # Use topic_name as ID if no topic found
                    topic_schedules[topic_id] = goal_date
                    # Store the actual name we found
                    if topic_id not in topics_by_id:
                        # Create a placeholder topic entry
                        topics_by_id[topic_id] = {"id": topic_id, "name": topic_name}
        
        _topic_schedules[curriculum_id] = topic_schedules
    
    # Populate weekly schedule
    topics_by_id = {t["id"]: t for t in curriculum.get("topics", [])}
    
    for topic_id, scheduled_date in topic_schedules.items():
        if scheduled_date in days:
            topic = topics_by_id.get(topic_id, {})
            topic_name = topic.get("name", topic_id)
            
            # Get confidence score
            overall, _, _, _ = calculate_topic_confidence(user_id, topic_name)
            
            days[scheduled_date].append(ScheduledTopic(
                topic_id=topic_id,
                topic_name=topic_name,
                date=scheduled_date,
                confidence_score=round(overall, 1),
                status="completed" if overall >= 80 else "in_progress" if overall > 0 else "scheduled",
                unit=topic.get("unit"),
                chapter=topic.get("chapter")
            ))
    
    return WeeklySchedule(
        week_start=start_of_week.isoformat(),
        week_end=end_of_week.isoformat(),
        days=days
    )


@router.post("/reschedule")
async def reschedule_topic(request: RescheduleRequest):
    """
    Reschedule a topic to a new date.
    """
    curriculum_id = request.curriculum_id
    
    if curriculum_id not in _topic_schedules:
        _topic_schedules[curriculum_id] = {}
    
    _topic_schedules[curriculum_id][request.topic_id] = request.new_date
    
    logger.info(f"Rescheduled topic {request.topic_id} to {request.new_date}")
    
    return {"success": True, "message": f"Topic rescheduled to {request.new_date}"}


@router.post("/record-assessment-score")
async def record_assessment_score(
    user_id: str,
    topic_name: str,
    score: float
):
    """
    Record an assessment score for a topic.
    Called when user completes an assessment.
    """
    if user_id not in _assessment_scores:
        _assessment_scores[user_id] = {}
    
    topic_key = topic_name.lower()
    if topic_key not in _assessment_scores[user_id]:
        _assessment_scores[user_id][topic_key] = []
    
    _assessment_scores[user_id][topic_key].append(score)
    
    return {"success": True}


@router.post("/record-interview-score")
async def record_interview_score(
    user_id: str,
    topic_name: str,
    score: float
):
    """
    Record a mock interview score for a topic.
    Called when user completes a mock interview session.
    """
    if user_id not in _interview_scores:
        _interview_scores[user_id] = {}
    
    topic_key = topic_name.lower()
    if topic_key not in _interview_scores[user_id]:
        _interview_scores[user_id][topic_key] = []
    
    _interview_scores[user_id][topic_key].append(score)
    
    return {"success": True}
