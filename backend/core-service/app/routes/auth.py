"""
Authentication Routes - Updated for multi-tenant with token-based registration
"""
from flask import Blueprint, request, jsonify
from uuid import uuid4
from app import db
from app.models.user import User, Leaderboard
from app.models.organization import Organization
from app.models.student_profile import StudentProfile, ParentStudentLink
from app.utils.jwt_handler import create_tokens, verify_token, refresh_access_token

auth_bp = Blueprint("auth", __name__, url_prefix="/api/auth")


@auth_bp.route("/register", methods=["POST"])
def register():
    """Register a new user - supports token-based registration for orgs"""
    data = request.get_json()
    
    # Validate required fields
    required_fields = ["email", "password", "username"]
    for field in required_fields:
        if not data.get(field):
            return jsonify({"error": f"Missing required field: {field}"}), 400
    
    # Check if user already exists
    if User.query.filter_by(email=data["email"]).first():
        return jsonify({"error": "Email already registered"}), 409
    
    if User.query.filter_by(username=data["username"]).first():
        return jsonify({"error": "Username already taken"}), 409
    
    # Validate password strength
    if len(data["password"]) < 8:
        return jsonify({"error": "Password must be at least 8 characters"}), 400
    
    role = data.get("role", "student")
    organization_id = None
    organization = None
    
    # =============== Token-based registration (Teachers ONLY) ===============
    # Students no longer need token - they join classrooms with codes after signup
    access_token = data.get("access_token")
    if access_token:
        organization = Organization.query.filter_by(access_token=access_token).first()
        
        if not organization:
            return jsonify({"error": "Invalid access token"}), 400
        
        # Check if admissions are open
        if hasattr(organization, 'admission_open') and not organization.admission_open:
            return jsonify({"error": "Admissions are currently closed for this organization"}), 403
        
        if organization.subscription_status == "expired":
            return jsonify({"error": "Organization subscription has expired"}), 403
        
        # Only teachers can use org token now
        if role == "teacher":
            organization_id = organization.id
        else:
            return jsonify({"error": "Access token is only required for teacher registration"}), 400
    
    # =============== Parent registration (link code) ===============
    student_link_code = data.get("student_link_code")
    linked_student = None
    
    if role == "parent" and student_link_code:
        # Find student by link code
        student_profile = StudentProfile.query.filter_by(link_code=student_link_code.upper()).first()
        
        if not student_profile:
            return jsonify({"error": "Invalid student link code"}), 400
        
        linked_student = User.query.get(student_profile.user_id)
        if linked_student:
            organization_id = linked_student.organization_id
    
    # =============== Admin registration (new organization) ===============
    if role == "admin":
        org_name = data.get("organization_name")
        if not org_name:
            return jsonify({"error": "Organization name required for admin registration"}), 400
        
        # Create organization
        organization = Organization(
            id=str(uuid4()),
            name=org_name,
            email=data["email"],
            phone=data.get("phone"),
            subscription_status="trial",
            license_count=5  # Trial licenses
        )
        db.session.add(organization)
        db.session.flush()
        organization_id = organization.id
    
    # =============== Create User ===============
    user = User(
        id=str(uuid4()),
        email=data["email"],
        username=data["username"],
        first_name=data.get("first_name", ""),
        last_name=data.get("last_name", ""),
        role=role,
        organization_id=organization_id,  # None for students until they join classroom
        phone=data.get("phone"),
        audio_consent=data.get("audio_consent", False),
        data_sharing_consent=data.get("data_sharing_consent", False)
    )
    user.set_password(data["password"])
    
    # Link admin to organization
    if role == "admin" and organization:
        organization.admin_user_id = user.id
    
    # NOTE: License is no longer consumed at student registration
    # License is now consumed when student joins a classroom
    
    try:
        db.session.add(user)
        db.session.flush()
        
        # Create leaderboard entry
        leaderboard = Leaderboard(
            id=str(uuid4()),
            user_id=user.id,
            global_points=0,
            subject_points={},
            badges=[]
        )
        db.session.add(leaderboard)
        
        # Create student profile if student
        if role == "student":
            profile_data = data.get("profile", {})
            student_profile = StudentProfile(
                id=str(uuid4()),
                user_id=user.id,
                grade_level=profile_data.get("grade_level"),
                board=profile_data.get("board"),
                stream=profile_data.get("stream"),
                target_exams=profile_data.get("target_exams", []),
                subjects=profile_data.get("subjects", []),
                onboarding_complete=False
            )
            student_profile.generate_link_code()
            db.session.add(student_profile)
        
        # Link parent to student
        if role == "parent" and linked_student:
            link = ParentStudentLink(
                id=str(uuid4()),
                parent_id=user.id,
                student_id=linked_student.id,
                relationship_type=data.get("relationship", "parent"),
                is_verified=True
            )
            db.session.add(link)
        
        db.session.commit()
        
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": f"Database error: {str(e)}"}), 500
    
    # Generate tokens
    access_token, refresh_token = create_tokens(str(user.id), user.role)
    
    response_data = {
        "message": "User registered successfully",
        "user": user.to_dict(),
        "access_token": access_token,
        "refresh_token": refresh_token
    }
    
    # Include student link code for parents to share
    if role == "student" and user.student_profile:
        response_data["student_link_code"] = user.student_profile.link_code
    
    # Include organization info for admin
    if role == "admin" and organization:
        response_data["organization"] = organization.to_dict()
    
    return jsonify(response_data), 201


@auth_bp.route("/login", methods=["POST"])
def login():
    """Authenticate user and return tokens"""
    data = request.get_json()
    
    email = data.get("email")
    password = data.get("password")
    
    if not email or not password:
        return jsonify({"error": "Email and password required"}), 400
    
    # Find user
    user = User.query.filter_by(email=email).first()
    
    if not user or not user.check_password(password):
        return jsonify({"error": "Invalid email or password"}), 401
    
    if not user.is_active:
        return jsonify({"error": "Account is deactivated"}), 403
    
    # Generate tokens
    access_token, refresh_token = create_tokens(str(user.id), user.role)
    
    response_data = {
        "message": "Login successful",
        "user": user.to_dict(),
        "access_token": access_token,
        "refresh_token": refresh_token
    }
    
    # Add role-specific info
    if user.role == "student" and user.student_profile:
        response_data["profile"] = user.student_profile.to_dict()
        response_data["onboarding_required"] = not user.student_profile.onboarding_complete
    
    return jsonify(response_data), 200


@auth_bp.route("/me", methods=["GET"])
def get_current_user():
    """Get current authenticated user"""
    auth_header = request.headers.get("Authorization")
    
    if not auth_header:
        return jsonify({"error": "Missing authorization header"}), 401
    
    try:
        # Extract token from "Bearer <token>"
        parts = auth_header.split()
        if len(parts) != 2 or parts[0].lower() != "bearer":
            return jsonify({"error": "Invalid authorization format"}), 401
        
        token = parts[1]
        payload = verify_token(token)
        
        user = User.query.get(payload["user_id"])
        if not user:
            return jsonify({"error": "User not found"}), 404
        
        response_data = {"user": user.to_dict()}
        
        # Add role-specific data
        if user.role == "student" and user.student_profile:
            response_data["profile"] = user.student_profile.to_dict()
        
        if user.organization_id:
            org = Organization.query.get(user.organization_id)
            if org:
                response_data["organization"] = {
                    "id": org.id,
                    "name": org.name
                }
        
        return jsonify(response_data), 200
    
    except Exception as e:
        return jsonify({"error": f"Invalid token: {str(e)}"}), 401


@auth_bp.route("/refresh", methods=["POST"])
def refresh():
    """Refresh access token"""
    data = request.get_json()
    refresh_token = data.get("refresh_token")
    
    if not refresh_token:
        return jsonify({"error": "Refresh token required"}), 400
    
    new_access_token = refresh_access_token(refresh_token)
    
    if not new_access_token:
        return jsonify({"error": "Invalid or expired refresh token"}), 401
    
    return jsonify({
        "access_token": new_access_token
    }), 200


@auth_bp.route("/logout", methods=["POST"])
def logout():
    """Logout user (client should discard tokens)"""
    return jsonify({"message": "Logged out successfully"}), 200


@auth_bp.route("/change-password", methods=["POST"])
def change_password():
    """Change user password"""
    auth_header = request.headers.get("Authorization")
    
    if not auth_header:
        return jsonify({"error": "Missing authorization header"}), 401
    
    try:
        token = auth_header.split()[1]
        payload = verify_token(token)
        user = User.query.get(payload["user_id"])
        
        if not user:
            return jsonify({"error": "User not found"}), 404
        
        data = request.get_json()
        current_password = data.get("current_password")
        new_password = data.get("new_password")
        
        if not current_password or not new_password:
            return jsonify({"error": "Current and new password required"}), 400
        
        if not user.check_password(current_password):
            return jsonify({"error": "Current password is incorrect"}), 401
        
        if len(new_password) < 8:
            return jsonify({"error": "New password must be at least 8 characters"}), 400
        
        user.set_password(new_password)
        db.session.commit()
        
        return jsonify({"message": "Password changed successfully"}), 200
    
    except Exception as e:
        return jsonify({"error": f"Error: {str(e)}"}), 500


@auth_bp.route("/validate-token", methods=["POST"])
def validate_org_token():
    """Validate organization access token (for registration forms)"""
    data = request.get_json()
    access_token = data.get("access_token")
    
    if not access_token:
        return jsonify({"error": "Access token required"}), 400
    
    organization = Organization.query.filter_by(access_token=access_token).first()
    
    if not organization:
        return jsonify({"valid": False, "error": "Invalid token"}), 200
    
    return jsonify({
        "valid": True,
        "organization": {
            "name": organization.name,
            "available_licenses": organization.license_count - organization.used_licenses,
            "subscription_status": organization.subscription_status
        }
    }), 200
