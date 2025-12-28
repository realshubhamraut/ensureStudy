"""
Teacher Routes - Student management, classroom access
"""
from flask import Blueprint, request, jsonify
from app import db
from app.models.user import User
from app.models.organization import Organization
from app.models.student_profile import StudentProfile, ParentStudentLink, TeacherClassAssignment
from app.utils.jwt_handler import verify_token

teacher_bp = Blueprint("teacher", __name__, url_prefix="/api/teacher")


def teacher_required(f):
    """Decorator to require teacher role"""
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get("Authorization")
        if not auth_header:
            return jsonify({"error": "Missing authorization header"}), 401
        
        try:
            token = auth_header.split()[1]
            payload = verify_token(token)
            user = User.query.get(payload["user_id"])
            
            if not user or user.role != "teacher":
                return jsonify({"error": "Teacher access required"}), 403
            
            request.current_user = user
            return f(*args, **kwargs)
        except Exception as e:
            return jsonify({"error": f"Authentication failed: {str(e)}"}), 401
    
    return decorated


# ==================== Dashboard ====================

@teacher_bp.route("/dashboard", methods=["GET"])
@teacher_required
def get_dashboard():
    """Get teacher dashboard stats"""
    user = request.current_user
    
    if not user.organization_id:
        return jsonify({"error": "Not part of an organization"}), 400
    
    # Get all students in the same organization
    students = User.query.filter_by(
        organization_id=user.organization_id, 
        role="student",
        is_active=True
    ).all()
    
    return jsonify({
        "teacher": user.to_dict(),
        "stats": {
            "total_students": len(students),
        }
    }), 200


# ==================== Student Management ====================

@teacher_bp.route("/students", methods=["GET"])
@teacher_required
def list_students():
    """List all students the teacher can access"""
    user = request.current_user
    
    if not user.organization_id:
        return jsonify({"error": "Not part of an organization"}), 400
    
    students = User.query.filter_by(
        organization_id=user.organization_id, 
        role="student",
        is_active=True
    ).all()
    
    result = []
    for student in students:
        student_data = student.to_dict()
        if student.student_profile:
            student_data["profile"] = student.student_profile.to_dict()
            
            # Get parent info if linked
            parent_link = ParentStudentLink.query.filter_by(student_id=student.id).first()
            if parent_link:
                parent = User.query.get(parent_link.parent_id)
                if parent:
                    student_data["parent"] = {
                        "id": parent.id,
                        "name": f"{parent.first_name} {parent.last_name}",
                        "email": parent.email,
                        "phone": parent.phone
                    }
        result.append(student_data)
    
    return jsonify({
        "students": result,
        "count": len(students)
    }), 200


@teacher_bp.route("/students/<student_id>", methods=["GET"])
@teacher_required
def get_student(student_id):
    """Get student details"""
    user = request.current_user
    
    if not user.organization_id:
        return jsonify({"error": "Not part of an organization"}), 400
    
    student = User.query.filter_by(
        id=student_id, 
        organization_id=user.organization_id, 
        role="student"
    ).first()
    
    if not student:
        return jsonify({"error": "Student not found"}), 404
    
    student_data = student.to_dict(include_details=True)
    
    # Get parent info if linked
    parent_link = ParentStudentLink.query.filter_by(student_id=student.id).first()
    if parent_link:
        parent = User.query.get(parent_link.parent_id)
        if parent:
            student_data["parent"] = {
                "id": parent.id,
                "name": f"{parent.first_name} {parent.last_name}",
                "email": parent.email,
                "phone": parent.phone
            }
    
    return jsonify({"student": student_data}), 200


@teacher_bp.route("/students/<student_id>", methods=["PUT"])
@teacher_required
def update_student(student_id):
    """Update student details (limited fields for teachers)"""
    user = request.current_user
    
    if not user.organization_id:
        return jsonify({"error": "Not part of an organization"}), 400
    
    student = User.query.filter_by(
        id=student_id, 
        organization_id=user.organization_id, 
        role="student"
    ).first()
    
    if not student:
        return jsonify({"error": "Student not found"}), 404
    
    data = request.get_json()
    
    # Teachers can only update certain fields
    for field in ["first_name", "last_name", "phone"]:
        if field in data:
            setattr(student, field, data[field])
    
    # Update student profile if applicable
    if student.student_profile and "profile" in data:
        profile_data = data["profile"]
        for field in ["grade_level", "board", "stream", "target_exams", "subjects"]:
            if field in profile_data:
                setattr(student.student_profile, field, profile_data[field])
    
    db.session.commit()
    
    return jsonify({
        "student": student.to_dict(include_details=True),
        "message": "Student updated successfully"
    }), 200


# ==================== Parents ====================

@teacher_bp.route("/parents", methods=["GET"])
@teacher_required
def list_parents():
    """List all parents linked to students in the organization"""
    user = request.current_user
    
    if not user.organization_id:
        return jsonify({"error": "Not part of an organization"}), 400
    
    # Get all students in org
    students = User.query.filter_by(
        organization_id=user.organization_id, 
        role="student",
        is_active=True
    ).all()
    
    student_ids = [s.id for s in students]
    
    # Get parent links
    parent_links = ParentStudentLink.query.filter(
        ParentStudentLink.student_id.in_(student_ids)
    ).all()
    
    parent_ids = list(set([pl.parent_id for pl in parent_links]))
    parents = User.query.filter(User.id.in_(parent_ids)).all()
    
    result = []
    for parent in parents:
        parent_data = parent.to_dict()
        # Get linked students
        linked_students = ParentStudentLink.query.filter_by(parent_id=parent.id).all()
        parent_data["linked_students"] = []
        for link in linked_students:
            student = User.query.get(link.student_id)
            if student:
                parent_data["linked_students"].append({
                    "id": student.id,
                    "name": f"{student.first_name} {student.last_name}",
                    "email": student.email
                })
        result.append(parent_data)
    
    return jsonify({
        "parents": result,
        "count": len(parents)
    }), 200
