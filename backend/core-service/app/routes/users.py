"""
User Management Routes
"""
from flask import Blueprint, request, jsonify
from app import db
from app.models.user import User
from app.utils.jwt_handler import verify_token
from functools import wraps

users_bp = Blueprint("users", __name__, url_prefix="/api/users")


def require_auth(f):
    """Decorator to require authentication"""
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get("Authorization")
        if not auth_header:
            return jsonify({"error": "Missing authorization header"}), 401
        
        try:
            token = auth_header.split()[1]
            payload = verify_token(token)
            request.user_id = payload["user_id"]
            request.user_role = payload["role"]
            return f(*args, **kwargs)
        except Exception as e:
            return jsonify({"error": f"Invalid token: {str(e)}"}), 401
    
    return decorated


def require_teacher(f):
    """Decorator to require teacher role"""
    @wraps(f)
    @require_auth
    def decorated(*args, **kwargs):
        if request.user_role != "teacher":
            return jsonify({"error": "Teacher access required"}), 403
        return f(*args, **kwargs)
    return decorated


@users_bp.route("/<user_id>", methods=["GET"])
@require_auth
def get_user(user_id):
    """Get user by ID"""
    user = User.query.get(user_id)
    
    if not user:
        return jsonify({"error": "User not found"}), 404
    
    # Users can only view their own profile or teachers can view students
    if str(request.user_id) != user_id and request.user_role != "teacher":
        return jsonify({"error": "Access denied"}), 403
    
    return jsonify({"user": user.to_dict()}), 200


@users_bp.route("/<user_id>", methods=["PUT"])
@require_auth
def update_user(user_id):
    """Update user profile"""
    if str(request.user_id) != user_id:
        return jsonify({"error": "Can only update own profile"}), 403
    
    user = User.query.get(user_id)
    if not user:
        return jsonify({"error": "User not found"}), 404
    
    data = request.get_json()
    
    # Fields that can be updated
    updatable_fields = [
        "first_name", "last_name", "avatar_url", "bio",
        "audio_consent", "data_sharing_consent"
    ]
    
    for field in updatable_fields:
        if field in data:
            setattr(user, field, data[field])
    
    db.session.commit()
    
    return jsonify({"user": user.to_dict()}), 200


@users_bp.route("/class/<class_id>", methods=["GET"])
@require_teacher
def get_class_students(class_id):
    """Get all students in a class (teacher only)"""
    students = User.query.filter_by(class_id=class_id, role="student").all()
    
    return jsonify({
        "class_id": class_id,
        "students": [s.to_dict() for s in students],
        "count": len(students)
    }), 200


@users_bp.route("/search", methods=["GET"])
@require_auth
def search_users():
    """Search users by username or email"""
    query = request.args.get("q", "")
    
    if len(query) < 3:
        return jsonify({"error": "Query must be at least 3 characters"}), 400
    
    users = User.query.filter(
        (User.username.ilike(f"%{query}%")) | (User.email.ilike(f"%{query}%"))
    ).limit(20).all()
    
    # Return limited info for non-teachers
    if request.user_role == "teacher":
        return jsonify({"users": [u.to_dict() for u in users]}), 200
    else:
        return jsonify({
            "users": [{"id": str(u.id), "username": u.username} for u in users]
        }), 200
