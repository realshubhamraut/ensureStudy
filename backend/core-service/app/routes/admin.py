"""
Admin Routes - Organization management, licensing, user management
"""
from flask import Blueprint, request, jsonify
from uuid import uuid4
from datetime import datetime, timedelta
from app import db
from app.models.user import User
from app.models.organization import Organization, LicensePurchase
from app.models.student_profile import StudentProfile, ParentStudentLink, TeacherClassAssignment
from app.utils.jwt_handler import verify_token

admin_bp = Blueprint("admin", __name__, url_prefix="/api/admin")


def admin_required(f):
    """Decorator to require admin role"""
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
            
            if not user or user.role != "admin":
                return jsonify({"error": "Admin access required"}), 403
            
            request.current_user = user
            return f(*args, **kwargs)
        except Exception as e:
            return jsonify({"error": f"Authentication failed: {str(e)}"}), 401
    
    return decorated


# ==================== Organization ====================

@admin_bp.route("/organization", methods=["GET"])
@admin_required
def get_organization():
    """Get current admin's organization"""
    user = request.current_user
    org = Organization.query.filter_by(admin_user_id=user.id).first()
    
    if not org:
        return jsonify({"error": "No organization found"}), 404
    
    return jsonify({"organization": org.to_dict()}), 200


@admin_bp.route("/organization", methods=["PUT"])
@admin_required
def update_organization():
    """Update organization details"""
    user = request.current_user
    org = Organization.query.filter_by(admin_user_id=user.id).first()
    
    if not org:
        return jsonify({"error": "No organization found"}), 404
    
    data = request.get_json()
    
    # Update allowed fields
    for field in ["name", "phone", "address", "city", "state"]:
        if field in data:
            setattr(org, field, data[field])
    
    db.session.commit()
    
    return jsonify({"organization": org.to_dict(), "message": "Organization updated"}), 200


@admin_bp.route("/organization/regenerate-token", methods=["POST"])
@admin_required
def regenerate_token():
    """Regenerate organization access token"""
    user = request.current_user
    org = Organization.query.filter_by(admin_user_id=user.id).first()
    
    if not org:
        return jsonify({"error": "No organization found"}), 404
    
    new_token = org.regenerate_token()
    db.session.commit()
    
    return jsonify({
        "message": "Token regenerated",
        "access_token": new_token
    }), 200


# ==================== Dashboard Stats ====================

@admin_bp.route("/dashboard", methods=["GET"])
@admin_required
def get_dashboard():
    """Get admin dashboard stats"""
    user = request.current_user
    org = Organization.query.filter_by(admin_user_id=user.id).first()
    
    if not org:
        return jsonify({"error": "No organization found"}), 404
    
    # Get counts
    total_teachers = User.query.filter_by(organization_id=org.id, role="teacher").count()
    total_students = User.query.filter_by(organization_id=org.id, role="student").count()
    total_parents = User.query.filter_by(organization_id=org.id, role="parent").count()
    
    # Recent users
    recent_users = User.query.filter_by(organization_id=org.id)\
        .order_by(User.created_at.desc())\
        .limit(5)\
        .all()
    
    return jsonify({
        "stats": {
            "license_count": org.license_count,
            "used_licenses": org.used_licenses,
            "available_licenses": org.license_count - org.used_licenses,
            "total_teachers": total_teachers,
            "total_students": total_students,
            "total_parents": total_parents,
            "total_users": total_teachers + total_students + total_parents
        },
        "subscription": {
            "status": org.subscription_status,
            "expires": org.subscription_expires.isoformat() if org.subscription_expires else None
        },
        "access_token": org.access_token,
        "recent_users": [u.to_dict() for u in recent_users]
    }), 200


# ==================== Teacher Management ====================

@admin_bp.route("/teachers", methods=["GET"])
@admin_required
def list_teachers():
    """List all teachers in organization"""
    user = request.current_user
    org = Organization.query.filter_by(admin_user_id=user.id).first()
    
    if not org:
        return jsonify({"error": "No organization found"}), 404
    
    teachers = User.query.filter_by(organization_id=org.id, role="teacher").all()
    
    return jsonify({
        "teachers": [t.to_dict() for t in teachers],
        "count": len(teachers)
    }), 200


@admin_bp.route("/teachers/<teacher_id>", methods=["DELETE"])
@admin_required
def remove_teacher(teacher_id):
    """Remove a teacher from organization"""
    user = request.current_user
    org = Organization.query.filter_by(admin_user_id=user.id).first()
    
    if not org:
        return jsonify({"error": "No organization found"}), 404
    
    teacher = User.query.filter_by(id=teacher_id, organization_id=org.id, role="teacher").first()
    
    if not teacher:
        return jsonify({"error": "Teacher not found"}), 404
    
    teacher.is_active = False
    db.session.commit()
    
    return jsonify({"message": "Teacher removed"}), 200


@admin_bp.route("/teachers/<teacher_id>", methods=["GET"])
@admin_required
def get_teacher_details(teacher_id):
    """Get detailed teacher info including classrooms"""
    user = request.current_user
    org = Organization.query.filter_by(admin_user_id=user.id).first()
    
    if not org:
        return jsonify({"error": "No organization found"}), 404
    
    teacher = User.query.filter_by(id=teacher_id, organization_id=org.id, role="teacher").first()
    
    if not teacher:
        return jsonify({"error": "Teacher not found"}), 404
    
    # Build teacher data
    teacher_data = teacher.to_dict()
    teacher_data["phone"] = teacher.phone
    
    # Add classroom assignments
    from app.models.classroom import Classroom, StudentClassroom
    
    classrooms = []
    teacher_classrooms = Classroom.query.filter_by(teacher_id=teacher.id).all()
    for classroom in teacher_classrooms:
        student_count = StudentClassroom.query.filter_by(classroom_id=classroom.id).count()
        classrooms.append({
            "id": classroom.id,
            "name": classroom.name,
            "student_count": student_count
        })
    teacher_data["classrooms"] = classrooms
    
    return jsonify({"teacher": teacher_data}), 200


# ==================== Student Management ====================

@admin_bp.route("/students", methods=["GET"])
@admin_required
def list_students():
    """List all students in organization"""
    user = request.current_user
    org = Organization.query.filter_by(admin_user_id=user.id).first()
    
    if not org:
        return jsonify({"error": "No organization found"}), 404
    
    students = User.query.filter_by(organization_id=org.id, role="student").all()
    
    result = []
    for student in students:
        student_data = student.to_dict()
        if student.student_profile:
            student_data["profile"] = student.student_profile.to_dict()
        result.append(student_data)
    
    return jsonify({
        "students": result,
        "count": len(students)
    }), 200


@admin_bp.route("/students/<student_id>", methods=["DELETE"])
@admin_required
def remove_student(student_id):
    """Remove a student from organization (releases license)"""
    user = request.current_user
    org = Organization.query.filter_by(admin_user_id=user.id).first()
    
    if not org:
        return jsonify({"error": "No organization found"}), 404
    
    student = User.query.filter_by(id=student_id, organization_id=org.id, role="student").first()
    
    if not student:
        return jsonify({"error": "Student not found"}), 404
    
    student.is_active = False
    org.release_license()  # Release the license
    db.session.commit()
    
    return jsonify({
        "message": "Student removed",
        "available_licenses": org.license_count - org.used_licenses
    }), 200


@admin_bp.route("/students/<student_id>", methods=["GET"])
@admin_required
def get_student_details(student_id):
    """Get detailed student info including parent/guardian data"""
    user = request.current_user
    org = Organization.query.filter_by(admin_user_id=user.id).first()
    
    if not org:
        return jsonify({"error": "No organization found"}), 404
    
    student = User.query.filter_by(id=student_id, organization_id=org.id, role="student").first()
    
    if not student:
        return jsonify({"error": "Student not found"}), 404
    
    # Build student data
    student_data = student.to_dict()
    student_data["phone"] = student.phone
    
    # Add profile info
    if student.student_profile:
        student_data["profile"] = student.student_profile.to_dict()
    
    # Add parent/guardian info
    parents = []
    parent_links = ParentStudentLink.query.filter_by(student_id=student.id).all()
    for link in parent_links:
        parent = User.query.get(link.parent_id)
        if parent:
            parents.append({
                "id": parent.id,
                "name": f"{parent.first_name or ''} {parent.last_name or ''}".strip() or parent.username,
                "email": parent.email,
                "phone": parent.phone,
                "relationship_type": link.relationship_type
            })
    student_data["parents"] = parents
    
    # Add classroom enrollments
    from app.models.classroom import StudentClassroom, Classroom
    enrollments = StudentClassroom.query.filter_by(student_id=student.id).all()
    classrooms = []
    for enrollment in enrollments:
        classroom = Classroom.query.get(enrollment.classroom_id)
        if classroom:
            classrooms.append({
                "id": classroom.id,
                "name": classroom.name
            })
    student_data["classrooms"] = classrooms
    
    return jsonify({"student": student_data}), 200


# ==================== Admission Control ====================

@admin_bp.route("/admission/toggle", methods=["POST"])
@admin_required
def toggle_admission():
    """Open or close admission window"""
    user = request.current_user
    org = Organization.query.filter_by(admin_user_id=user.id).first()
    
    if not org:
        return jsonify({"error": "No organization found"}), 404
    
    data = request.get_json()
    admission_open = data.get("admission_open")
    
    if admission_open is not None:
        org.admission_open = bool(admission_open)
        db.session.commit()
    
    return jsonify({
        "admission_open": org.admission_open,
        "message": f"Admissions {'opened' if org.admission_open else 'closed'}"
    }), 200


# ==================== User Edit ====================

@admin_bp.route("/users/<user_id>", methods=["PUT"])
@admin_required
def update_user(user_id):
    """Update user details (teacher or student)"""
    admin = request.current_user
    org = Organization.query.filter_by(admin_user_id=admin.id).first()
    
    if not org:
        return jsonify({"error": "No organization found"}), 404
    
    user = User.query.filter_by(id=user_id, organization_id=org.id).first()
    
    if not user:
        return jsonify({"error": "User not found"}), 404
    
    data = request.get_json()
    
    # Update allowed fields
    for field in ["first_name", "last_name", "phone", "avatar_url", "is_active"]:
        if field in data:
            setattr(user, field, data[field])
    
    # Update student profile if applicable
    if user.role == "student" and user.student_profile and "profile" in data:
        profile_data = data["profile"]
        for field in ["grade_level", "board", "stream", "target_exams", "subjects"]:
            if field in profile_data:
                setattr(user.student_profile, field, profile_data[field])
    
    db.session.commit()
    
    return jsonify({
        "user": user.to_dict(include_details=True),
        "message": "User updated successfully"
    }), 200


@admin_bp.route("/users/<user_id>", methods=["GET"])
@admin_required
def get_user(user_id):
    """Get user details"""
    admin = request.current_user
    org = Organization.query.filter_by(admin_user_id=admin.id).first()
    
    if not org:
        return jsonify({"error": "No organization found"}), 404
    
    user = User.query.filter_by(id=user_id, organization_id=org.id).first()
    
    if not user:
        return jsonify({"error": "User not found"}), 404
    
    return jsonify({"user": user.to_dict(include_details=True)}), 200


# ==================== License Management ====================

@admin_bp.route("/licenses/purchase", methods=["POST"])
@admin_required
def purchase_licenses():
    """Initiate license purchase"""
    user = request.current_user
    org = Organization.query.filter_by(admin_user_id=user.id).first()
    
    if not org:
        return jsonify({"error": "No organization found"}), 404
    
    data = request.get_json()
    quantity = data.get("quantity", 0)
    
    if quantity < 1:
        return jsonify({"error": "Quantity must be at least 1"}), 400
    
    total_amount = quantity * org.price_per_student
    
    # Create purchase record (pending)
    purchase = LicensePurchase(
        id=str(uuid4()),
        organization_id=org.id,
        quantity=quantity,
        price_per_unit=org.price_per_student,
        total_amount=total_amount,
        payment_status="pending"
    )
    
    db.session.add(purchase)
    db.session.commit()
    
    # In production, create Razorpay order here
    # For now, return purchase ID for payment flow
    
    return jsonify({
        "purchase_id": purchase.id,
        "quantity": quantity,
        "total_amount": total_amount,
        "currency": "INR",
        "message": "Purchase initiated. Complete payment to activate licenses."
    }), 200


@admin_bp.route("/licenses/confirm", methods=["POST"])
@admin_required
def confirm_purchase():
    """Confirm license purchase after payment"""
    user = request.current_user
    org = Organization.query.filter_by(admin_user_id=user.id).first()
    
    if not org:
        return jsonify({"error": "No organization found"}), 404
    
    data = request.get_json()
    purchase_id = data.get("purchase_id")
    payment_id = data.get("payment_id")  # From Razorpay
    
    purchase = LicensePurchase.query.filter_by(
        id=purchase_id, 
        organization_id=org.id,
        payment_status="pending"
    ).first()
    
    if not purchase:
        return jsonify({"error": "Purchase not found"}), 404
    
    # In production, verify payment with Razorpay here
    
    # Update purchase
    purchase.payment_status = "completed"
    purchase.payment_id = payment_id
    
    # Add licenses to organization
    org.license_count += purchase.quantity
    org.subscription_status = "active"
    
    # Set/extend subscription expiry (1 year)
    if org.subscription_expires and org.subscription_expires > datetime.utcnow():
        org.subscription_expires += timedelta(days=365)
    else:
        org.subscription_expires = datetime.utcnow() + timedelta(days=365)
    
    db.session.commit()
    
    return jsonify({
        "message": "Purchase confirmed",
        "licenses_added": purchase.quantity,
        "total_licenses": org.license_count,
        "subscription_expires": org.subscription_expires.isoformat()
    }), 200


@admin_bp.route("/licenses/history", methods=["GET"])
@admin_required
def purchase_history():
    """Get license purchase history"""
    user = request.current_user
    org = Organization.query.filter_by(admin_user_id=user.id).first()
    
    if not org:
        return jsonify({"error": "No organization found"}), 404
    
    purchases = LicensePurchase.query.filter_by(organization_id=org.id)\
        .order_by(LicensePurchase.created_at.desc())\
        .all()
    
    return jsonify({
        "purchases": [p.to_dict() for p in purchases],
        "count": len(purchases)
    }), 200
