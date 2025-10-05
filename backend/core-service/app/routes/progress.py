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
