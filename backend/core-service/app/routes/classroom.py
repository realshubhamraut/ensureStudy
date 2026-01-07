"""
Classroom Routes - Create, manage, and join classrooms
"""
from flask import Blueprint, request, jsonify
from app import db
from app.models.user import User
from app.models.organization import Organization
from app.models.classroom import Classroom, StudentClassroom
from app.utils.jwt_handler import verify_token

classroom_bp = Blueprint("classroom", __name__, url_prefix="/api/classroom")


def get_current_user():
    """Get current user from JWT token"""
    auth_header = request.headers.get("Authorization")
    if not auth_header:
        return None
    
    try:
        token = auth_header.split()[1]
        payload = verify_token(token)
        return User.query.get(payload["user_id"])
    except:
        return None


def teacher_required(f):
    """Decorator to require teacher role"""
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        user = get_current_user()
        if not user or user.role != "teacher":
            return jsonify({"error": "Teacher access required"}), 403
        request.current_user = user
        return f(*args, **kwargs)
    return decorated


def student_required(f):
    """Decorator to require student role"""
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        user = get_current_user()
        if not user or user.role != "student":
            return jsonify({"error": "Student access required"}), 403
        request.current_user = user
        return f(*args, **kwargs)
    return decorated


# ==================== Teacher: Create & Manage Classrooms ====================

@classroom_bp.route("", methods=["POST"])
@teacher_required
def create_classroom():
    """Teacher creates a new classroom"""
    user = request.current_user
    
    if not user.organization_id:
        return jsonify({"error": "You must be part of an organization"}), 400
    
    data = request.get_json()
    name = data.get("name")
    
    if not name:
        return jsonify({"error": "Classroom name is required"}), 400
    
    classroom = Classroom(
        name=name,
        grade=data.get("grade"),
        section=data.get("section"),
        subject=data.get("subject"),
        teacher_id=user.id,
        organization_id=user.organization_id
    )
    
    db.session.add(classroom)
    db.session.commit()
    
    return jsonify({
        "classroom": classroom.to_dict(),
        "message": f"Classroom created! Share code: {classroom.join_code}"
    }), 201


@classroom_bp.route("", methods=["GET"])
@teacher_required
def list_classrooms():
    """List teacher's classrooms"""
    user = request.current_user
    
    classrooms = Classroom.query.filter_by(teacher_id=user.id).all()
    
    return jsonify({
        "classrooms": [c.to_dict() for c in classrooms],
        "count": len(classrooms)
    }), 200


@classroom_bp.route("/<classroom_id>", methods=["GET"])
@teacher_required
def get_classroom(classroom_id):
    """Get classroom details with students"""
    user = request.current_user
    
    classroom = Classroom.query.filter_by(id=classroom_id, teacher_id=user.id).first()
    
    if not classroom:
        return jsonify({"error": "Classroom not found"}), 404
    
    # Get enrolled students
    enrollments = StudentClassroom.query.filter_by(classroom_id=classroom_id, is_active=True).all()
    students = []
    for e in enrollments:
        student = User.query.get(e.student_id)
        if student:
            students.append({
                "id": student.id,
                "email": student.email,
                "username": student.username,
                "first_name": student.first_name,
                "last_name": student.last_name,
                "joined_at": e.joined_at.isoformat() if e.joined_at else None
            })
    
    return jsonify({
        "classroom": classroom.to_dict(),
        "students": students
    }), 200


@classroom_bp.route("/<classroom_id>", methods=["PUT"])
@teacher_required
def update_classroom(classroom_id):
    """Update classroom details"""
    user = request.current_user
    
    classroom = Classroom.query.filter_by(id=classroom_id, teacher_id=user.id).first()
    
    if not classroom:
        return jsonify({"error": "Classroom not found"}), 404
    
    data = request.get_json()
    
    for field in ["name", "grade", "section", "subject", "is_active"]:
        if field in data:
            setattr(classroom, field, data[field])
    
    db.session.commit()
    
    return jsonify({
        "classroom": classroom.to_dict(),
        "message": "Classroom updated"
    }), 200


@classroom_bp.route("/<classroom_id>/syllabus", methods=["PUT"])
@teacher_required
def update_syllabus(classroom_id):
    """Upload or update classroom syllabus"""
    from datetime import datetime
    user = request.current_user
    
    classroom = Classroom.query.filter_by(id=classroom_id, teacher_id=user.id).first()
    
    if not classroom:
        return jsonify({"error": "Classroom not found"}), 404
    
    data = request.get_json()
    
    # Support both URL and text content
    if "syllabus_url" in data:
        classroom.syllabus_url = data["syllabus_url"]
        classroom.syllabus_filename = data.get("syllabus_filename", "syllabus")
    
    if "syllabus_content" in data:
        classroom.syllabus_content = data["syllabus_content"]
    
    classroom.syllabus_uploaded_at = datetime.utcnow()
    
    db.session.commit()
    
    return jsonify({
        "classroom": classroom.to_dict(),
        "message": "Syllabus updated successfully"
    }), 200


@classroom_bp.route("/<classroom_id>/syllabus", methods=["DELETE"])
@teacher_required
def delete_syllabus(classroom_id):
    """Remove classroom syllabus"""
    user = request.current_user
    
    classroom = Classroom.query.filter_by(id=classroom_id, teacher_id=user.id).first()
    
    if not classroom:
        return jsonify({"error": "Classroom not found"}), 404
    
    classroom.syllabus_url = None
    classroom.syllabus_content = None
    classroom.syllabus_filename = None
    classroom.syllabus_uploaded_at = None
    
    db.session.commit()
    
    return jsonify({"message": "Syllabus removed"}), 200


@classroom_bp.route("/<classroom_id>/syllabus", methods=["GET"])
def get_syllabus(classroom_id):
    """Get classroom syllabus (public for enrolled students)"""
    user = get_current_user()
    
    classroom = Classroom.query.get(classroom_id)
    
    if not classroom:
        return jsonify({"error": "Classroom not found"}), 404
    
    # Check access: teacher who owns it OR enrolled student
    if user:
        is_teacher = classroom.teacher_id == user.id
        is_enrolled = StudentClassroom.query.filter_by(
            student_id=user.id,
            classroom_id=classroom.id,
            is_active=True
        ).first() is not None
        
        if not (is_teacher or is_enrolled):
            return jsonify({"error": "Access denied"}), 403
    else:
        return jsonify({"error": "Authentication required"}), 401
    
    return jsonify({
        "syllabus_url": classroom.syllabus_url,
        "syllabus_content": classroom.syllabus_content,
        "syllabus_filename": classroom.syllabus_filename,
        "syllabus_uploaded_at": classroom.syllabus_uploaded_at.isoformat() if classroom.syllabus_uploaded_at else None,
        "has_syllabus": bool(classroom.syllabus_url or classroom.syllabus_content),
        "classroom_name": classroom.name,
        "teacher_name": f"{classroom.teacher.first_name or ''} {classroom.teacher.last_name or ''}".strip() or classroom.teacher.username if classroom.teacher else None
    }), 200


@classroom_bp.route("/<classroom_id>/regenerate-code", methods=["POST"])
@teacher_required
def regenerate_code(classroom_id):
    """Regenerate classroom join code"""
    user = request.current_user
    
    classroom = Classroom.query.filter_by(id=classroom_id, teacher_id=user.id).first()
    
    if not classroom:
        return jsonify({"error": "Classroom not found"}), 404
    
    new_code = classroom.regenerate_code()
    db.session.commit()
    
    return jsonify({
        "join_code": new_code,
        "message": "Join code regenerated"
    }), 200


# ==================== Student: Join Classroom ====================

@classroom_bp.route("/join", methods=["POST"])
@student_required
def join_classroom():
    """Student joins a classroom using code"""
    user = request.current_user
    
    data = request.get_json()
    join_code = data.get("code", "").upper().strip()
    
    if not join_code:
        return jsonify({"error": "Classroom code is required"}), 400
    
    # Find classroom
    classroom = Classroom.query.filter_by(join_code=join_code).first()
    
    if not classroom:
        return jsonify({"error": "Invalid classroom code"}), 404
    
    if not classroom.is_active:
        return jsonify({"error": "This classroom is not accepting new students"}), 400
    
    # Check if already enrolled
    existing = StudentClassroom.query.filter_by(
        student_id=user.id, 
        classroom_id=classroom.id
    ).first()
    
    if existing:
        if existing.is_active:
            return jsonify({"error": "You are already in this classroom"}), 400
        else:
            # Re-activate enrollment
            existing.is_active = True
            db.session.commit()
            return jsonify({
                "message": f"Welcome back to {classroom.name}!",
                "classroom": classroom.to_dict()
            }), 200
    
    # Check organization license availability
    org = classroom.organization
    
    # Check if admissions are open
    if not org.admission_open:
        return jsonify({
            "error": "Admissions are currently closed. Please contact your teacher."
        }), 400
    
    if not org.has_available_licenses():
        return jsonify({
            "error": "Your school has run out of student licenses. Please tell your teacher to contact the school administrator to purchase more credits."
        }), 400
    
    # Use a license
    org.use_license()
    
    # Associate student with organization
    user.organization_id = org.id
    
    # Create enrollment
    enrollment = StudentClassroom(
        student_id=user.id,
        classroom_id=classroom.id
    )
    
    db.session.add(enrollment)
    db.session.commit()
    
    return jsonify({
        "message": f"Welcome to {classroom.name}!",
        "classroom": classroom.to_dict()
    }), 200


@classroom_bp.route("/my-classrooms", methods=["GET"])
@student_required
def my_classrooms():
    """Get student's enrolled classrooms with teacher info and syllabus"""
    user = request.current_user
    
    enrollments = StudentClassroom.query.filter_by(student_id=user.id, is_active=True).all()
    
    classrooms = []
    for e in enrollments:
        classroom = Classroom.query.get(e.classroom_id)
        if classroom:
            # Use include_teacher=True to get teacher info including for syllabus display
            data = classroom.to_dict(include_teacher=True)
            classrooms.append(data)
    
    return jsonify({
        "classrooms": classrooms,
        "count": len(classrooms)
    }), 200


@classroom_bp.route("/leave/<classroom_id>", methods=["POST"])
@student_required
def leave_classroom(classroom_id):
    """Student leaves a classroom"""
    user = request.current_user
    
    enrollment = StudentClassroom.query.filter_by(
        student_id=user.id,
        classroom_id=classroom_id,
        is_active=True
    ).first()
    
    if not enrollment:
        return jsonify({"error": "Not enrolled in this classroom"}), 404
    
    enrollment.is_active = False
    
    # Note: We don't release the license when a student leaves
    # That would be an admin decision
    
    db.session.commit()
    
    return jsonify({"message": "Left classroom successfully"}), 200


# ==================== Materials: Upload & Manage ====================

@classroom_bp.route("/<classroom_id>/materials", methods=["POST"])
@teacher_required
def upload_material(classroom_id):
    """Teacher uploads a material to classroom"""
    from app.models.classroom import ClassroomMaterial
    import requests as http_requests
    user = request.current_user
    
    classroom = Classroom.query.filter_by(id=classroom_id, teacher_id=user.id).first()
    
    if not classroom:
        return jsonify({"error": "Classroom not found"}), 404
    
    data = request.get_json()
    
    if not data.get("name") or not data.get("url"):
        return jsonify({"error": "Name and URL are required"}), 400
    
    material = ClassroomMaterial(
        classroom_id=classroom_id,
        name=data["name"],
        file_url=data["url"],
        file_type=data.get("type", "application/octet-stream"),
        file_size=data.get("size", 0),
        subject=data.get("subject", ""),
        description=data.get("description", ""),
        uploaded_by=user.id
    )
    
    db.session.add(material)
    db.session.commit()
    
    # Trigger indexing for PDF files
    if material.file_type == 'application/pdf':
        try:
            http_requests.post(
                'http://localhost:8001/api/index/material',
                json={
                    'material_id': material.id,
                    'file_url': material.file_url,
                    'classroom_id': classroom_id,
                    'subject': material.subject,
                    'document_title': material.name,
                    'uploaded_by': user.id
                },
                timeout=5
            )
        except Exception as e:
            # Don't fail upload if indexing trigger fails
            print(f"[CLASSROOM] Warning: Failed to trigger indexing: {e}")
    
    # Create notifications for all enrolled students
    try:
        from app.models.notification import notify_classroom_members
        notify_classroom_members(
            classroom_id=classroom_id,
            notification_type="material",
            title=f"New material in {classroom.name}",
            message=f"New material uploaded: {material.name}",
            action_url=f"/classrooms/{classroom_id}",
            source_id=material.id
        )
    except Exception as e:
        print(f"Failed to send notifications: {e}")
    
    return jsonify({
        "material": material.to_dict(),
        "message": "Material uploaded successfully"
    }), 201


@classroom_bp.route("/<classroom_id>/materials", methods=["GET"])
def list_materials(classroom_id):
    """Get materials for a classroom (teacher or enrolled student)"""
    from app.models.classroom import ClassroomMaterial
    user = get_current_user()
    
    if not user:
        return jsonify({"error": "Authentication required"}), 401
    
    classroom = Classroom.query.get(classroom_id)
    
    if not classroom:
        return jsonify({"error": "Classroom not found"}), 404
    
    # Check access
    is_teacher = classroom.teacher_id == user.id
    is_enrolled = StudentClassroom.query.filter_by(
        student_id=user.id,
        classroom_id=classroom.id,
        is_active=True
    ).first() is not None
    
    if not (is_teacher or is_enrolled):
        return jsonify({"error": "Access denied"}), 403
    
    materials = ClassroomMaterial.query.filter_by(
        classroom_id=classroom_id,
        is_active=True
    ).order_by(ClassroomMaterial.uploaded_at.desc()).all()
    
    return jsonify({
        "materials": [m.to_dict() for m in materials],
        "count": len(materials)
    }), 200


@classroom_bp.route("/<classroom_id>/materials/<material_id>", methods=["DELETE"])
@teacher_required
def delete_material(classroom_id, material_id):
    """Teacher deletes a material from classroom"""
    from app.models.classroom import ClassroomMaterial
    user = request.current_user
    
    # Verify classroom ownership
    classroom = Classroom.query.filter_by(id=classroom_id, teacher_id=user.id).first()
    
    if not classroom:
        return jsonify({"error": "Classroom not found"}), 404
    
    material = ClassroomMaterial.query.filter_by(
        id=material_id,
        classroom_id=classroom_id
    ).first()
    
    if not material:
        return jsonify({"error": "Material not found"}), 404
    
    # Soft delete
    material.is_active = False
    db.session.commit()
    
    return jsonify({"message": "Material deleted"}), 200


@classroom_bp.route("/materials/<material_id>/status", methods=["PUT"])
def update_material_status(material_id):
    """Update material indexing status (called by AI service)"""
    from app.models.classroom import ClassroomMaterial
    from datetime import datetime
    
    material = ClassroomMaterial.query.get(material_id)
    
    if not material:
        return jsonify({"error": "Material not found"}), 404
    
    data = request.get_json()
    
    if "indexing_status" in data:
        material.indexing_status = data["indexing_status"]
        if data["indexing_status"] == "completed":
            material.indexed_at = datetime.utcnow()
    
    if "chunk_count" in data:
        material.chunk_count = data["chunk_count"]
    
    if "indexing_error" in data:
        material.indexing_error = data["indexing_error"]
    
    db.session.commit()
    
    return jsonify({
        "material": material.to_dict(),
        "message": "Status updated"
    }), 200


# ==================== Announcements: Stream Posts ====================

@classroom_bp.route("/<classroom_id>/announcements", methods=["POST"])
@teacher_required
def create_announcement(classroom_id):
    """Teacher posts an announcement to classroom stream"""
    from app.models.announcement import Announcement
    from app.models.notification import notify_classroom_members
    user = request.current_user
    
    classroom = Classroom.query.filter_by(id=classroom_id, teacher_id=user.id).first()
    
    if not classroom:
        return jsonify({"error": "Classroom not found"}), 404
    
    data = request.get_json()
    message = data.get("message", "").strip()
    
    if not message:
        return jsonify({"error": "Message is required"}), 400
    
    announcement = Announcement(
        classroom_id=classroom_id,
        teacher_id=user.id,
        message=message
    )
    
    db.session.add(announcement)
    db.session.commit()
    
    # Create notifications for all enrolled students
    try:
        notify_classroom_members(
            classroom_id=classroom_id,
            notification_type="announcement",
            title=f"New announcement in {classroom.name}",
            message=message[:100] + "..." if len(message) > 100 else message,
            action_url=f"/classrooms/{classroom_id}",
            source_id=announcement.id
        )
    except Exception as e:
        print(f"Failed to send notifications: {e}")
    
    return jsonify({
        "announcement": announcement.to_dict(),
        "message": "Announcement posted"
    }), 201


@classroom_bp.route("/<classroom_id>/announcements", methods=["GET"])
def list_announcements(classroom_id):
    """Get announcements for a classroom (teacher or enrolled student)"""
    from app.models.announcement import Announcement
    user = get_current_user()
    
    if not user:
        return jsonify({"error": "Authentication required"}), 401
    
    classroom = Classroom.query.get(classroom_id)
    
    if not classroom:
        return jsonify({"error": "Classroom not found"}), 404
    
    # Check access
    is_teacher = classroom.teacher_id == user.id
    is_enrolled = StudentClassroom.query.filter_by(
        student_id=user.id,
        classroom_id=classroom.id,
        is_active=True
    ).first() is not None
    
    if not (is_teacher or is_enrolled):
        return jsonify({"error": "Access denied"}), 403
    
    announcements = Announcement.query.filter_by(
        classroom_id=classroom_id
    ).order_by(Announcement.created_at.desc()).all()
    
    return jsonify({
        "announcements": [a.to_dict() for a in announcements],
        "count": len(announcements)
    }), 200


@classroom_bp.route("/<classroom_id>/announcements/<announcement_id>", methods=["DELETE"])
@teacher_required
def delete_announcement(classroom_id, announcement_id):
    """Teacher deletes an announcement"""
    from app.models.announcement import Announcement
    user = request.current_user
    
    # Verify classroom ownership
    classroom = Classroom.query.filter_by(id=classroom_id, teacher_id=user.id).first()
    
    if not classroom:
        return jsonify({"error": "Classroom not found"}), 404
    
    announcement = Announcement.query.filter_by(
        id=announcement_id,
        classroom_id=classroom_id
    ).first()
    
    if not announcement:
        return jsonify({"error": "Announcement not found"}), 404
    
    db.session.delete(announcement)
    db.session.commit()
    
    return jsonify({"message": "Announcement deleted"}), 200
