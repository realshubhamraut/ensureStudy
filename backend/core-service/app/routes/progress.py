"""
Student Progress Routes
"""
from flask import Blueprint, request, jsonify
from datetime import datetime, date, timedelta
from uuid import uuid4
from app import db
from app.models.user import Progress
from app.routes.users import require_auth

progress_bp = Blueprint("progress", __name__, url_prefix="/api/progress")


@progress_bp.route("/", methods=["GET"])
@require_auth
def get_user_progress():
    """Get all progress records for current user"""
    user_id = request.user_id
    subject = request.args.get("subject")
    
    query = Progress.query.filter_by(user_id=user_id)
    
    if subject:
        query = query.filter_by(subject=subject)
    
    progress_records = query.order_by(Progress.updated_at.desc()).all()
    
    return jsonify({
        "progress": [p.to_dict() for p in progress_records],
        "count": len(progress_records)
    }), 200


@progress_bp.route("/weak-topics", methods=["GET"])
@require_auth
def get_weak_topics():
    """Get topics marked as weak for current user"""
    user_id = request.user_id
    
    weak_topics = Progress.query.filter_by(
        user_id=user_id,
        is_weak=True
    ).order_by(Progress.confidence_score.asc()).all()
    
    return jsonify({
        "weak_topics": [p.to_dict() for p in weak_topics],
        "count": len(weak_topics)
    }), 200


@progress_bp.route("/topic", methods=["POST"])
@require_auth
def create_or_update_progress():
    """Create or update progress for a topic"""
    user_id = request.user_id
    data = request.get_json()
    
    topic = data.get("topic")
    subject = data.get("subject")
    
    if not topic or not subject:
        return jsonify({"error": "Topic and subject required"}), 400
    
    # Find existing or create new
    progress = Progress.query.filter_by(
        user_id=user_id,
        topic=topic,
        subject=subject
    ).first()
    
    if not progress:
        progress = Progress(
            id=uuid4(),
            user_id=user_id,
            topic=topic,
            subject=subject
        )
        db.session.add(progress)
    
    # Update fields
    if "confidence_score" in data:
        progress.confidence_score = data["confidence_score"]
    
    if "assessment_score" in data:
        scores = progress.assessment_scores or []
        scores.append({
            "score": data["assessment_score"],
            "date": datetime.utcnow().isoformat()
        })
        progress.assessment_scores = scores
    
    if data.get("studied"):
        progress.times_studied = (progress.times_studied or 0) + 1
        progress.last_studied = datetime.utcnow()
    
    # Auto-detect weak topics
    if progress.confidence_score < 50:
        progress.is_weak = True
    elif progress.confidence_score > 70:
        progress.is_weak = False
    
    db.session.commit()
    
    return jsonify({"progress": progress.to_dict()}), 200


@progress_bp.route("/summary", methods=["GET"])
@require_auth
def get_progress_summary():
    """Get summary of user's progress"""
    user_id = request.user_id
    
    all_progress = Progress.query.filter_by(user_id=user_id).all()
    
    if not all_progress:
        return jsonify({
            "total_topics": 0,
            "weak_topics_count": 0,
            "average_confidence": 0,
            "subjects": {}
        }), 200
    
    weak_count = sum(1 for p in all_progress if p.is_weak)
    avg_confidence = sum(p.confidence_score for p in all_progress) / len(all_progress)
    
    # Group by subject
    subjects = {}
    for p in all_progress:
        if p.subject not in subjects:
            subjects[p.subject] = {
                "topics_count": 0,
                "weak_count": 0,
                "avg_confidence": 0,
                "total_studied": 0
            }
        subjects[p.subject]["topics_count"] += 1
        subjects[p.subject]["total_studied"] += p.times_studied or 0
        if p.is_weak:
            subjects[p.subject]["weak_count"] += 1
    
    # Calculate avg confidence per subject
    for subject in subjects:
        subject_progress = [p for p in all_progress if p.subject == subject]
        subjects[subject]["avg_confidence"] = sum(
            p.confidence_score for p in subject_progress
        ) / len(subject_progress)
    
    return jsonify({
        "total_topics": len(all_progress),
        "weak_topics_count": weak_count,
        "average_confidence": round(avg_confidence, 2),
        "subjects": subjects
    }), 200


@progress_bp.route("/study-streak", methods=["GET"])
@require_auth
def get_study_streak():
    """Get study streak based on days with activity"""
    user_id = request.user_id
    
    # Get all progress records with last_studied dates
    progress_records = Progress.query.filter(
        Progress.user_id == user_id,
        Progress.last_studied.isnot(None)
    ).all()
    
    if not progress_records:
        return jsonify({
            "currentStreak": 0,
            "longestStreak": 0,
            "totalStudyDays": 0,
            "lastStudiedDate": None
        }), 200
    
    # Get unique study dates
    study_dates = set()
    for p in progress_records:
        if p.last_studied:
            study_dates.add(p.last_studied.date())
    
    if not study_dates:
        return jsonify({
            "currentStreak": 0,
            "longestStreak": 0,
            "totalStudyDays": 0,
            "lastStudiedDate": None
        }), 200
    
    # Sort dates descending
    sorted_dates = sorted(study_dates, reverse=True)
    today = date.today()
    
    # Calculate current streak
    current_streak = 0
    check_date = today
    
    for d in sorted_dates:
        if d == check_date or d == check_date - timedelta(days=1):
            current_streak += 1
            check_date = d - timedelta(days=1)
        else:
            break
    
    # Calculate longest streak
    longest_streak = 1
    current_run = 1
    for i in range(1, len(sorted_dates)):
        if sorted_dates[i] == sorted_dates[i-1] - timedelta(days=1):
            current_run += 1
            longest_streak = max(longest_streak, current_run)
        else:
            current_run = 1
    
    return jsonify({
        "currentStreak": current_streak,
        "longestStreak": longest_streak,
        "totalStudyDays": len(study_dates),
        "lastStudiedDate": sorted_dates[0].isoformat() if sorted_dates else None
    }), 200


@progress_bp.route("/overview", methods=["GET"])
@require_auth
def get_progress_overview():
    """Get overview stats matching frontend Progress page"""
    user_id = request.user_id
    
    all_progress = Progress.query.filter_by(user_id=user_id).all()
    
    if not all_progress:
        return jsonify({
            "avgConfidence": 0,
            "topicsMastered": 0,
            "topicsNeedAttention": 0,
            "studyStreak": 0,
            "totalTopics": 0,
            "subjects": []
        }), 200
    
    # Calculate stats
    avg_confidence = round(sum(p.confidence_score for p in all_progress) / len(all_progress), 1)
    topics_mastered = len([p for p in all_progress if p.confidence_score >= 70])
    topics_need_attention = len([p for p in all_progress if p.is_weak or p.confidence_score < 50])
    
    # Calculate streak
    study_dates = set()
    for p in all_progress:
        if p.last_studied:
            study_dates.add(p.last_studied.date())
    
    current_streak = 0
    if study_dates:
        sorted_dates = sorted(study_dates, reverse=True)
        check_date = date.today()
        for d in sorted_dates:
            if d == check_date or d == check_date - timedelta(days=1):
                current_streak += 1
                check_date = d - timedelta(days=1)
            else:
                break
    
    # Group by subject
    subjects_dict = {}
    for p in all_progress:
        if p.subject not in subjects_dict:
            subjects_dict[p.subject] = {"scores": [], "count": 0}
        subjects_dict[p.subject]["scores"].append(p.confidence_score)
        subjects_dict[p.subject]["count"] += 1
    
    subjects = [
        {
            "subject": name,
            "avgConfidence": round(sum(data["scores"]) / len(data["scores"]), 1),
            "topicCount": data["count"]
        }
        for name, data in subjects_dict.items()
    ]
    
    return jsonify({
        "avgConfidence": avg_confidence,
        "topicsMastered": topics_mastered,
        "topicsNeedAttention": topics_need_attention,
        "studyStreak": current_streak,
        "totalTopics": len(all_progress),
        "subjects": sorted(subjects, key=lambda x: x["avgConfidence"], reverse=True)
    }), 200


@progress_bp.route("/topics-list", methods=["GET"])
@require_auth
def get_topics_list():
    """Get all topics matching frontend TopicProgress interface"""
    user_id = request.user_id
    
    all_progress = Progress.query.filter_by(user_id=user_id).order_by(
        Progress.confidence_score.desc()
    ).all()
    
    def format_relative_time(dt):
        if not dt:
            return "Never"
        now = datetime.utcnow()
        diff = now - dt
        
        if diff.days > 7:
            return f"{diff.days // 7} weeks ago"
        elif diff.days > 0:
            return f"{diff.days} day{'s' if diff.days > 1 else ''} ago"
        elif diff.seconds > 3600:
            hours = diff.seconds // 3600
            return f"{hours} hour{'s' if hours > 1 else ''} ago"
        else:
            return "Just now"
    
    return jsonify([
        {
            "topic": p.topic,
            "subject": p.subject,
            "confidence": round(p.confidence_score, 1),
            "isWeak": p.is_weak or p.confidence_score < 50,
            "timesStudied": p.times_studied or 0,
            "lastStudied": format_relative_time(p.last_studied)
        }
        for p in all_progress
    ]), 200


# ============================================================================
# Classroom Topic Mastery (for ProgressDashboard)
# ============================================================================

@progress_bp.route("/topic-mastery", methods=["GET"])
@require_auth
def get_topic_mastery():
    """
    Get topic mastery data for ProgressDashboard.
    Returns stats, chapters, and topics with scores.
    """
    from app.models.curriculum import StudentTopicScore, ClassroomTopic, Chapter
    from app.models.classroom import Classroom, StudentClassroom
    
    user_id = request.user_id
    classroom_id = request.args.get("classroom_id")
    
    # Get enrolled classrooms if no specific one requested
    if classroom_id:
        classroom_ids = [classroom_id]
    else:
        enrollments = StudentClassroom.query.filter_by(
            student_id=user_id, is_active=True
        ).all()
        classroom_ids = [e.classroom_id for e in enrollments]
    
    if not classroom_ids:
        return jsonify({
            "stats": {
                "total_topics": 0,
                "topics_started": 0,
                "topics_mastered": 0,
                "average_mastery": 0,
                "total_study_hours": 0,
                "current_streak": 0
            },
            "chapters": [],
            "topics": []
        }), 200
    
    # Get chapters for these classrooms
    chapters = Chapter.query.filter(
        Chapter.classroom_id.in_(classroom_ids),
        Chapter.is_active == True
    ).order_by(Chapter.order).all()
    
    # Get topics for these classrooms
    topics = ClassroomTopic.query.filter(
        ClassroomTopic.classroom_id.in_(classroom_ids),
        ClassroomTopic.is_active == True
    ).all()
    
    # Get student scores
    scores_query = db.session.query(
        StudentTopicScore,
        ClassroomTopic,
        Chapter
    ).join(
        ClassroomTopic, StudentTopicScore.classroom_topic_id == ClassroomTopic.id
    ).join(
        Chapter, ClassroomTopic.chapter_id == Chapter.id
    ).filter(
        StudentTopicScore.user_id == user_id,
        ClassroomTopic.classroom_id.in_(classroom_ids)
    ).all()
    
    # Build scores map
    scores_map = {}
    for score, topic, chapter in scores_query:
        scores_map[topic.id] = {
            "score": score,
            "topic": topic,
            "chapter": chapter
        }
    
    # Calculate chapter progress
    chapter_progress = []
    for ch in chapters:
        ch_topics = [t for t in topics if t.chapter_id == ch.id]
        ch_scores = [scores_map.get(t.id) for t in ch_topics if t.id in scores_map]
        
        mastered = sum(1 for s in ch_scores if s and s["score"].mastery_percentage >= 80)
        avg_mastery = sum(s["score"].mastery_percentage for s in ch_scores if s) / len(ch_scores) if ch_scores else 0
        
        chapter_progress.append({
            "chapter_id": ch.id,
            "name": ch.name,
            "color": ch.color,
            "topics_count": len(ch_topics),
            "topics_mastered": mastered,
            "average_mastery": round(avg_mastery, 1)
        })
    
    # Build topic list with scores
    topic_list = []
    for t in topics:
        score_data = scores_map.get(t.id)
        chapter = next((ch for ch in chapters if ch.id == t.chapter_id), None)
        
        if score_data:
            topic_list.append({
                "topic_id": t.id,
                "topic_name": t.name,
                "chapter_name": chapter.name if chapter else "Unknown",
                "chapter_color": chapter.color if chapter else "#3B82F6",
                "mastery_level": round(score_data["score"].mastery_percentage, 1),
                "quiz_score": round((score_data["score"].mcq_total_score / score_data["score"].mcq_max_score * 100) if score_data["score"].mcq_max_score > 0 else 0, 1),
                "interview_score": round(score_data["score"].descriptive_avg_score, 1),
                "total_attempts": score_data["score"].mcq_attempts + score_data["score"].descriptive_attempts,
                "last_activity": score_data["score"].last_activity_at.strftime("%Y-%m-%d") if score_data["score"].last_activity_at else "Never"
            })
        else:
            topic_list.append({
                "topic_id": t.id,
                "topic_name": t.name,
                "chapter_name": chapter.name if chapter else "Unknown",
                "chapter_color": chapter.color if chapter else "#3B82F6",
                "mastery_level": 0,
                "quiz_score": 0,
                "interview_score": 0,
                "total_attempts": 0,
                "last_activity": "Never"
            })
    
    # Calculate stats
    total_topics = len(topics)
    topics_started = sum(1 for t in topic_list if t["total_attempts"] > 0)
    topics_mastered = sum(1 for t in topic_list if t["mastery_level"] >= 80)
    average_mastery = sum(t["mastery_level"] for t in topic_list) / total_topics if total_topics > 0 else 0
    
    # TODO: Calculate real study hours and streak from activity
    total_study_hours = sum(t.estimated_hours or 1 for t in topics if t.id in scores_map) * 0.5
    
    return jsonify({
        "stats": {
            "total_topics": total_topics,
            "topics_started": topics_started,
            "topics_mastered": topics_mastered,
            "average_mastery": round(average_mastery, 1),
            "total_study_hours": round(total_study_hours, 1),
            "current_streak": 0  # TODO: Calculate from activity
        },
        "chapters": chapter_progress,
        "topics": topic_list
    }), 200
