"""
Leaderboard Routes
"""
from flask import Blueprint, request, jsonify
from app import db
from app.models.user import Leaderboard, User
from app.routes.users import require_auth

leaderboard_bp = Blueprint("leaderboard", __name__, url_prefix="/api/leaderboard")


@leaderboard_bp.route("/global", methods=["GET"])
@require_auth
def get_global_leaderboard():
    """Get global leaderboard"""
    limit = request.args.get("limit", 50, type=int)
    
    entries = db.session.query(Leaderboard, User).join(
        User, Leaderboard.user_id == User.id
    ).order_by(Leaderboard.global_points.desc()).limit(limit).all()
    
    leaderboard = []
    for i, (entry, user) in enumerate(entries, 1):
        leaderboard.append({
            "rank": i,
            "user_id": str(user.id),
            "username": user.username,
            "avatar_url": user.avatar_url,
            "global_points": entry.global_points,
            "level": entry.level,
            "badges": entry.badges[:3],  # Show first 3 badges
            "study_streak": entry.study_streak
        })
    
    return jsonify({"leaderboard": leaderboard}), 200


@leaderboard_bp.route("/subject/<subject>", methods=["GET"])
@require_auth
def get_subject_leaderboard(subject):
    """Get leaderboard for a specific subject"""
    limit = request.args.get("limit", 50, type=int)
    
    # Get all entries that have points in this subject
    entries = db.session.query(Leaderboard, User).join(
        User, Leaderboard.user_id == User.id
    ).all()
    
    # Filter and sort by subject points
    subject_entries = []
    for entry, user in entries:
        subject_points = entry.subject_points.get(subject, 0)
        if subject_points > 0:
            subject_entries.append({
                "entry": entry,
                "user": user,
                "points": subject_points
            })
    
    subject_entries.sort(key=lambda x: x["points"], reverse=True)
    subject_entries = subject_entries[:limit]
    
    leaderboard = []
    for i, item in enumerate(subject_entries, 1):
        leaderboard.append({
            "rank": i,
            "user_id": str(item["user"].id),
            "username": item["user"].username,
            "avatar_url": item["user"].avatar_url,
            "subject_points": item["points"]
        })
    
    return jsonify({
        "subject": subject,
        "leaderboard": leaderboard
    }), 200


@leaderboard_bp.route("/class/<class_id>", methods=["GET"])
@require_auth
def get_class_leaderboard(class_id):
    """Get leaderboard for a specific class"""
    limit = request.args.get("limit", 50, type=int)
    
    entries = db.session.query(Leaderboard, User).join(
        User, Leaderboard.user_id == User.id
    ).filter(
        User.class_id == class_id
    ).order_by(Leaderboard.class_points.desc()).limit(limit).all()
    
    leaderboard = []
    for i, (entry, user) in enumerate(entries, 1):
        leaderboard.append({
            "rank": i,
            "user_id": str(user.id),
            "username": user.username,
            "avatar_url": user.avatar_url,
            "class_points": entry.class_points,
            "study_streak": entry.study_streak
        })
    
    return jsonify({
        "class_id": class_id,
        "leaderboard": leaderboard
    }), 200


@leaderboard_bp.route("/me", methods=["GET"])
@require_auth
def get_my_leaderboard():
    """Get current user's leaderboard entry"""
    entry = Leaderboard.query.filter_by(user_id=request.user_id).first()
    
    if not entry:
        return jsonify({"error": "Leaderboard entry not found"}), 404
    
    # Calculate global rank
    higher_ranked = Leaderboard.query.filter(
        Leaderboard.global_points > entry.global_points
    ).count()
    global_rank = higher_ranked + 1
    
    return jsonify({
        "entry": entry.to_dict(),
        "global_rank": global_rank
    }), 200


@leaderboard_bp.route("/me/badges", methods=["GET"])
@require_auth
def get_my_badges():
    """Get current user's badges"""
    entry = Leaderboard.query.filter_by(user_id=request.user_id).first()
    
    if not entry:
        return jsonify({"badges": []}), 200
    
    return jsonify({
        "badges": entry.badges,
        "count": len(entry.badges)
    }), 200
