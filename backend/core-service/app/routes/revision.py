"""
Revision Schedule Routes - AI-powered spaced repetition scheduling
Type 5 Learning Agent component
"""
from datetime import datetime, date, timedelta
from flask import Blueprint, request, jsonify
from app import db
from app.models.curriculum import (
    StudentTopicScore, ClassroomTopic, Chapter, StudyScheduleEntry
)
from app.models.classroom import Classroom, StudentClassroom
from app.routes.users import require_auth

revision_bp = Blueprint("revision", __name__, url_prefix="/api/curriculum")


# ============================================================================
# Spaced Repetition Algorithm (SM-2 Variant)
# ============================================================================

def calculate_review_interval(mastery_pct: float, review_count: int) -> int:
    """
    Calculate next review interval in days using SM-2 algorithm variant.
    Lower mastery = shorter intervals (more frequent review needed)
    Higher mastery = longer intervals (spaced repetition)
    """
    # Base intervals in days (Fibonacci-like progression)
    base_intervals = [1, 2, 4, 7, 14, 30, 60, 90]
    
    # Get base interval from review count
    if review_count < len(base_intervals):
        base_interval = base_intervals[review_count]
    else:
        base_interval = 90  # Cap at 90 days
    
    # Adjust by mastery (low mastery = shorter interval)
    # mastery_factor: 0.3 (weak) to 1.5 (mastered)
    if mastery_pct >= 80:
        mastery_factor = 1.5  # Mastered - long interval
    elif mastery_pct >= 60:
        mastery_factor = 1.0  # Proficient - normal interval
    elif mastery_pct >= 40:
        mastery_factor = 0.6  # Learning - shorter interval
    else:
        mastery_factor = 0.3  # Weak - very short interval
    
    adjusted_interval = max(1, int(base_interval * mastery_factor))
    return adjusted_interval


def calculate_priority_score(
    mastery_pct: float,
    days_overdue: int,
    days_since_activity: int
) -> float:
    """
    Calculate priority score (0-100) for revision scheduling.
    Higher score = higher priority for review.
    
    Components:
    - Weakness factor (100 - mastery): 50% weight
    - Overdue urgency: 30% weight
    - Recency factor: 20% weight
    """
    # Weakness factor (0-100): lower mastery = higher priority
    weakness = 100 - mastery_pct
    
    # Overdue urgency (0-100): more overdue = higher priority
    urgency = min(100, days_overdue * 10)  # Caps at 10 days overdue
    
    # Recency factor (0-100): longer since activity = higher priority
    recency = min(100, days_since_activity * 5)  # Caps at 20 days
    
    # Weighted combination
    priority = (
        weakness * 0.5 +
        urgency * 0.3 +
        recency * 0.2
    )
    
    return round(priority, 1)


def get_review_status(next_review_date: date, today: date) -> str:
    """Determine review status based on due date."""
    days_until = (next_review_date - today).days
    
    if days_until < 0:
        return "overdue"
    elif days_until == 0:
        return "due"
    elif days_until <= 2:
        return "upcoming"
    else:
        return "scheduled"


# ============================================================================
# Reusable: Get today's revision topics (used by assessment generator)
# ============================================================================

def get_todays_revision_topics(user_id: str, target_date: date, max_topics: int = 5) -> list:
    """
    Get topics that are due for revision on a specific date.
    Uses the SM-2 spaced repetition algorithm.
    
    Returns a list of dicts with:
        topic_id, topic_name, subject_name, chapter_name, mastery_percentage, reason
    
    Only includes topics with REAL activity (skips new/untouched topics).
    """
    # Get enrolled classrooms
    enrollments = StudentClassroom.query.filter_by(
        student_id=user_id, is_active=True
    ).all()
    classroom_ids = [e.classroom_id for e in enrollments]
    
    if not classroom_ids:
        return []
    
    # Get all classroom topics
    topics = ClassroomTopic.query.filter(
        ClassroomTopic.classroom_id.in_(classroom_ids),
        ClassroomTopic.is_active == True
    ).all()
    
    if not topics:
        return []
    
    # Get chapters and classrooms for context
    chapters_query = Chapter.query.filter(
        Chapter.classroom_id.in_(classroom_ids),
        Chapter.is_active == True
    ).all()
    chapters_map = {ch.id: ch for ch in chapters_query}
    
    classrooms_query = Classroom.query.filter(
        Classroom.id.in_(classroom_ids)
    ).all()
    classrooms_map = {c.id: c for c in classrooms_query}
    
    # Get student scores
    scores_query = StudentTopicScore.query.filter(
        StudentTopicScore.user_id == user_id,
        StudentTopicScore.classroom_topic_id.in_([t.id for t in topics])
    ).all()
    scores_map = {s.classroom_topic_id: s for s in scores_query}
    
    # Find topics due for revision using SM-2 algorithm
    revision_topics = []
    
    for topic in topics:
        score = scores_map.get(topic.id)
        chapter = chapters_map.get(topic.chapter_id)
        classroom = classrooms_map.get(topic.classroom_id)
        
        # ONLY include topics that have been actually attempted
        if not score or not score.last_activity_at:
            continue
        
        review_count = score.mcq_attempts + score.descriptive_attempts
        if review_count == 0:
            continue
        
        mastery = score.mastery_percentage
        if mastery <= 0:
            continue
        
        # Calculate next review date using SM-2
        interval_days = calculate_review_interval(mastery, review_count)
        last_activity = score.last_activity_at.date()
        next_review = last_activity + timedelta(days=interval_days)
        
        # Only include if due or overdue on target_date
        if next_review > target_date:
            continue
        
        # Calculate priority
        days_overdue = max(0, (target_date - next_review).days)
        days_since_activity = (target_date - last_activity).days
        priority = calculate_priority_score(mastery, days_overdue, days_since_activity)
        
        # Determine reason
        if days_overdue > 3:
            reason = "overdue"
        elif mastery < 50:
            reason = "low_mastery"
        else:
            reason = "scheduled_review"
        
        revision_topics.append({
            "topic_id": topic.id,
            "topic_name": topic.name,
            "subject_name": classroom.subject if classroom else "General",
            "chapter_name": chapter.name if chapter else "",
            "mastery_percentage": round(mastery, 1),
            "reason": reason,
            "priority": priority
        })
    
    # Sort by priority (highest first) and limit
    revision_topics.sort(key=lambda x: x["priority"], reverse=True)
    return revision_topics[:max_topics]


# ============================================================================
# API Routes
# ============================================================================

@revision_bp.route("/revision-schedule", methods=["GET"])
@require_auth
def get_revision_schedule():
    """
    Get AI-generated revision schedule for the student.
    Uses spaced repetition algorithm based on:
    - StudentTopicScore mastery levels
    - MCQ and interview performance
    - Last activity timestamps
    
    Query params:
    - week_offset: int, default 0 (current week)
    - classroom_ids: comma-separated list (optional filter)
    
    Returns:
    {
        "week_start": "2026-02-03",
        "week_end": "2026-02-09",
        "schedule": {
            "2026-02-03": [...topics],
            "2026-02-04": [...topics],
            ...
        },
        "stats": {
            "topics_due": 5,
            "topics_overdue": 2,
            "avg_mastery": 65.4
        }
    }
    """
    user_id = request.user_id
    week_offset = request.args.get("week_offset", 0, type=int)
    classroom_ids_param = request.args.get("classroom_ids", "")
    
    # Calculate week boundaries
    today = date.today()
    # Monday of current week + offset
    days_since_monday = today.weekday()
    week_start = today - timedelta(days=days_since_monday) + timedelta(weeks=week_offset)
    week_end = week_start + timedelta(days=6)
    
    # Get enrolled classrooms
    if classroom_ids_param:
        classroom_ids = [cid.strip() for cid in classroom_ids_param.split(",") if cid.strip()]
    else:
        enrollments = StudentClassroom.query.filter_by(
            student_id=user_id, is_active=True
        ).all()
        classroom_ids = [e.classroom_id for e in enrollments]
    
    if not classroom_ids:
        return jsonify({
            "week_start": week_start.isoformat(),
            "week_end": week_end.isoformat(),
            "schedule": {},
            "stats": {
                "topics_due": 0,
                "topics_overdue": 0,
                "avg_mastery": 0
            }
        }), 200
    
    # Get all classroom topics for enrolled classrooms
    topics = ClassroomTopic.query.filter(
        ClassroomTopic.classroom_id.in_(classroom_ids),
        ClassroomTopic.is_active == True
    ).all()
    
    # Get chapters for context
    chapters_query = Chapter.query.filter(
        Chapter.classroom_id.in_(classroom_ids),
        Chapter.is_active == True
    ).all()
    chapters_map = {ch.id: ch for ch in chapters_query}
    
    # Get classrooms for subject names
    classrooms_query = Classroom.query.filter(
        Classroom.id.in_(classroom_ids)
    ).all()
    classrooms_map = {c.id: c for c in classrooms_query}
    
    # Get student scores for all topics
    scores_query = StudentTopicScore.query.filter(
        StudentTopicScore.user_id == user_id,
        StudentTopicScore.classroom_topic_id.in_([t.id for t in topics])
    ).all()
    scores_map = {s.classroom_topic_id: s for s in scores_query}
    
    # Calculate revision schedule for each topic
    revision_items = []
    
    for topic in topics:
        score = scores_map.get(topic.id)
        chapter = chapters_map.get(topic.chapter_id)
        classroom = classrooms_map.get(topic.classroom_id)
        
        # ONLY include topics that have been actually attempted AND have some mastery
        # Skip topics with no activity, no attempts, or 0% mastery - these are "New" topics
        if not score or not score.last_activity_at:
            continue  # No score record or never had activity
        
        review_count = score.mcq_attempts + score.descriptive_attempts
        if review_count == 0:
            continue  # Never actually attempted - skip "New" topics
        
        mastery = score.mastery_percentage
        if mastery <= 0:
            continue  # No mastery earned yet - skip topics that would show as "New"
        
        # Topic has been attempted AND has mastery - include in revision schedule
        
        # Get the interval based on mastery level
        interval_days = calculate_review_interval(mastery, review_count)
        
        # Calculate next review date
        last_activity = score.last_activity_at.date()
        next_review = last_activity + timedelta(days=interval_days)
        
        # Calculate priority and status
        days_overdue = max(0, (today - next_review).days)
        days_since_activity = (today - last_activity).days
        priority = calculate_priority_score(mastery, days_overdue, days_since_activity)
        status = get_review_status(next_review, today)
        
        revision_items.append({
            "topic_id": topic.id,
            "topic_name": topic.name,
            "subject_name": classroom.subject if classroom else "Unknown",
            "chapter_name": chapter.name if chapter else "",
            "chapter_color": chapter.color if chapter else "#6366F1",
            "mastery_percentage": round(mastery, 1),
            "quiz_score": round((score.mcq_total_score / score.mcq_max_score * 100) if score.mcq_max_score > 0 else 0, 1),
            "interview_score": round(score.descriptive_avg_score, 1),
            "review_count": review_count,
            "scheduled_date": next_review.isoformat(),
            "status": status,
            "priority": priority,
            "last_activity": last_activity.isoformat(),
            "source": "assessment"  # Only includes topics with real assessment/interview data
        })

    
    # Sort by priority (highest first)
    revision_items.sort(key=lambda x: x["priority"], reverse=True)
    
    # Distribute topics across the week
    # Max 3 topics per day, prioritize by urgency
    schedule = {}
    for i in range(7):
        day = week_start + timedelta(days=i)
        schedule[day.isoformat()] = []
    
    # First pass: Place overdue and due topics
    for item in revision_items[:]:
        item_date = date.fromisoformat(item["scheduled_date"])
        
        # If date is in this week, place it there
        if week_start <= item_date <= week_end:
            day_key = item_date.isoformat()
            if len(schedule[day_key]) < 3:  # Max 3 per day
                schedule[day_key].append(item)
                revision_items.remove(item)
        # If overdue, place on today or tomorrow
        elif item["status"] == "overdue" and today >= week_start and today <= week_end:
            # Try today first
            today_key = today.isoformat()
            tomorrow_key = (today + timedelta(days=1)).isoformat()
            
            if today_key in schedule and len(schedule[today_key]) < 3:
                item["scheduled_date"] = today_key
                schedule[today_key].append(item)
                revision_items.remove(item)
            elif tomorrow_key in schedule and len(schedule[tomorrow_key]) < 3:
                item["scheduled_date"] = tomorrow_key
                schedule[tomorrow_key].append(item)
                revision_items.remove(item)
    
    # Second pass: Distribute remaining items evenly
    remaining_items = revision_items[:14]  # At most 14 more (2 per day)
    day_index = 0
    for item in remaining_items:
        while day_index < 7:
            day = week_start + timedelta(days=day_index)
            day_key = day.isoformat()
            if len(schedule[day_key]) < 3:
                item["scheduled_date"] = day_key
                schedule[day_key].append(item)
                break
            day_index += 1
        else:
            break  # All days full
    
    # Calculate stats
    all_scheduled = [item for items in schedule.values() for item in items]
    topics_due = sum(1 for item in all_scheduled if item["status"] in ["due", "overdue"])
    topics_overdue = sum(1 for item in all_scheduled if item["status"] == "overdue")
    
    mastery_values = [item["mastery_percentage"] for item in all_scheduled if item["mastery_percentage"] > 0]
    avg_mastery = round(sum(mastery_values) / len(mastery_values), 1) if mastery_values else 0
    
    return jsonify({
        "week_start": week_start.isoformat(),
        "week_end": week_end.isoformat(),
        "schedule": schedule,
        "stats": {
            "topics_due": topics_due,
            "topics_overdue": topics_overdue,
            "topics_scheduled": len(all_scheduled),
            "avg_mastery": avg_mastery
        }
    }), 200


@revision_bp.route("/revision-schedule/mark-complete", methods=["POST"])
@require_auth
def mark_revision_complete():
    """
    Mark a revision topic as completed for today.
    This doesn't affect scores (that's done via assessments),
    just tracks that revision was done.
    """
    user_id = request.user_id
    data = request.get_json()
    
    topic_id = data.get("topic_id")
    if not topic_id:
        return jsonify({"error": "topic_id required"}), 400
    
    # Verify topic exists and user has score
    score = StudentTopicScore.query.filter_by(
        user_id=user_id,
        classroom_topic_id=topic_id
    ).first()
    
    if score:
        # Update last activity to now (this will affect next review calculation)
        score.last_activity_at = datetime.utcnow()
        db.session.commit()
        
        return jsonify({
            "success": True,
            "message": "Revision marked complete",
            "next_review": (date.today() + timedelta(
                days=calculate_review_interval(score.mastery_percentage, 
                    score.mcq_attempts + score.descriptive_attempts + 1)
            )).isoformat()
        }), 200
    else:
        return jsonify({
            "success": False,
            "message": "No score record found. Complete an assessment first."
        }), 404
