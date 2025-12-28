"""
Student Progress Routes
"""
from flask import Blueprint, request, jsonify
from datetime import datetime
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
