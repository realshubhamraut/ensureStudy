"""
Student API Routes

Provides endpoints for student-specific operations:
- Link code management for parent linking
- Linked parents list
"""

from flask import Blueprint, request, jsonify
from app.models.user import User
from app.models.student_profile import StudentProfile, ParentStudentLink
from app.utils.jwt_handler import verify_token
from app import db

bp = Blueprint('students', __name__, url_prefix='/api/students')


def get_current_user_from_token():
    """Extract and verify user from Authorization header."""
    auth_header = request.headers.get("Authorization")
    
    if not auth_header:
        return None, ({"error": "Missing authorization header"}, 401)
    
    try:
        parts = auth_header.split()
        if len(parts) != 2 or parts[0].lower() != "bearer":
            return None, ({"error": "Invalid authorization format"}, 401)
        
        token = parts[1]
        payload = verify_token(token)
        
        user = User.query.get(payload["user_id"])
        if not user:
            return None, ({"error": "User not found"}, 404)
        
        return user, None
        
    except Exception as e:
        return None, ({"error": f"Invalid token: {str(e)}"}, 401)


@bp.route('/link-code', methods=['GET'])
def get_link_code():
    """Get or generate the student's parent linking code."""
    user, error = get_current_user_from_token()
    if error:
        return jsonify(error[0]), error[1]
    
    if user.role != 'student':
        return jsonify({"error": "Only students have link codes"}), 403
    
    # Get or create student profile
    profile = StudentProfile.query.filter_by(user_id=user.id).first()
    if not profile:
        profile = StudentProfile(user_id=user.id)
        profile.generate_link_code()
        db.session.add(profile)
        db.session.commit()
    
    # Generate code if missing
    if not profile.link_code:
        profile.generate_link_code()
        db.session.commit()
    
    return jsonify({
        "link_code": profile.link_code,
        "student_name": f"{user.first_name} {user.last_name}"
    })


@bp.route('/linked-parents', methods=['GET'])
def get_linked_parents():
    """Get list of parents linked to this student."""
    user, error = get_current_user_from_token()
    if error:
        return jsonify(error[0]), error[1]
    
    if user.role != 'student':
        return jsonify({"error": "Unauthorized"}), 403
    
    # Get all parent links
    links = ParentStudentLink.query.filter_by(student_id=user.id).all()
    
    parents = []
    for link in links:
        parent = User.query.get(link.parent_id)
        if parent:
            parents.append({
                "id": link.id,
                "name": f"{parent.first_name} {parent.last_name}",
                "email": parent.email,
                "relationship_type": link.relationship_type,
                "linked_at": link.created_at.isoformat() if link.created_at else None,
                "is_verified": link.is_verified
            })
    
    return jsonify({
        "parents": parents,
        "count": len(parents)
    })


# =============================================================================
# Parent-side endpoints (for parent dashboard)
# =============================================================================

@bp.route('/link-by-code', methods=['POST'])
def link_by_code():
    """Link a parent to a student using the student's link code."""
    user, error = get_current_user_from_token()
    if error:
        return jsonify(error[0]), error[1]
    
    if user.role != 'parent':
        return jsonify({"error": "Only parents can link to students"}), 403
    
    data = request.get_json()
    link_code = data.get('link_code', '').strip().upper()
    relationship_type = data.get('relationship_type', 'parent')
    
    if not link_code:
        return jsonify({"error": "Link code is required"}), 400
    
    # Find student profile with this code
    profile = StudentProfile.query.filter_by(link_code=link_code).first()
    if not profile:
        return jsonify({"error": "Invalid link code"}), 404
    
    student = User.query.get(profile.user_id)
    if not student:
        return jsonify({"error": "Student not found"}), 404
    
    # Check if already linked
    existing = ParentStudentLink.query.filter_by(
        parent_id=user.id,
        student_id=profile.user_id
    ).first()
    
    if existing:
        return jsonify({"error": "Already linked to this student"}), 400
    
    # Create link
    link = ParentStudentLink(
        parent_id=user.id,
        student_id=profile.user_id,
        relationship_type=relationship_type,
        is_verified=True
    )
    db.session.add(link)
    db.session.commit()
    
    return jsonify({
        "success": True,
        "message": f"Successfully linked to {student.first_name} {student.last_name}",
        "student": {
            "id": student.id,
            "name": f"{student.first_name} {student.last_name}",
            "email": student.email
        }
    })


@bp.route('/linked-children', methods=['GET'])
def get_linked_children():
    """Get list of children linked to this parent."""
    user, error = get_current_user_from_token()
    if error:
        return jsonify(error[0]), error[1]
    
    if user.role != 'parent':
        return jsonify({"error": "Unauthorized"}), 403
    
    # Get all children links
    links = ParentStudentLink.query.filter_by(parent_id=user.id).all()
    
    children = []
    for link in links:
        student = User.query.get(link.student_id)
        if student:
            children.append({
                "id": link.id,
                "student_id": student.id,
                "name": f"{student.first_name} {student.last_name}",
                "email": student.email,
                "relationship_type": link.relationship_type,
                "linked_at": link.created_at.isoformat() if link.created_at else None
            })
    
    return jsonify({
        "children": children,
        "count": len(children)
    })
