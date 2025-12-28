"""
Authorization Service - Role-Based Access Control

Extends existing JWT authentication with:
- Classroom membership verification
- Document access control
- Resource ownership validation
- Role-based permission checking
"""
import logging
from typing import Optional, List
from functools import wraps
from flask import request, jsonify, g
from app import db
from app.models.user import User
from app.models.classroom import Classroom, StudentClassroom
from app.utils.jwt_handler import verify_token

logger = logging.getLogger(__name__)


# ============================================================================
# Permission Definitions
# ============================================================================

class Permissions:
    """Permission constants for RBAC"""
    # Document permissions
    DOCUMENT_UPLOAD = "document:upload"
    DOCUMENT_VIEW = "document:view"
    DOCUMENT_DELETE = "document:delete"
    DOCUMENT_VIEW_ALL = "document:view_all"
    
    # Classroom permissions
    CLASSROOM_CREATE = "classroom:create"
    CLASSROOM_MANAGE = "classroom:manage"
    CLASSROOM_VIEW_STUDENTS = "classroom:view_students"
    
    # AI Tutor permissions
    TUTOR_QUERY = "tutor:query"
    TUTOR_VIEW_HISTORY = "tutor:view_history"
    TUTOR_VIEW_ALL_HISTORY = "tutor:view_all_history"
    
    # Admin permissions
    ADMIN_VIEW_LOGS = "admin:view_logs"
    ADMIN_REINDEX = "admin:reindex"
    ADMIN_WEB_FETCH = "admin:web_fetch"


# Role to permissions mapping
ROLE_PERMISSIONS = {
    "student": [
        Permissions.DOCUMENT_UPLOAD,
        Permissions.DOCUMENT_VIEW,
        Permissions.DOCUMENT_DELETE,  # Own only
        Permissions.TUTOR_QUERY,
        Permissions.TUTOR_VIEW_HISTORY,  # Own only
    ],
    "teacher": [
        Permissions.DOCUMENT_UPLOAD,
        Permissions.DOCUMENT_VIEW,
        Permissions.DOCUMENT_VIEW_ALL,
        Permissions.DOCUMENT_DELETE,
        Permissions.CLASSROOM_CREATE,
        Permissions.CLASSROOM_MANAGE,
        Permissions.CLASSROOM_VIEW_STUDENTS,
        Permissions.TUTOR_QUERY,
        Permissions.TUTOR_VIEW_HISTORY,
        Permissions.TUTOR_VIEW_ALL_HISTORY,
        Permissions.ADMIN_WEB_FETCH,
    ],
    "admin": [
        # Admins have all permissions
        Permissions.DOCUMENT_UPLOAD,
        Permissions.DOCUMENT_VIEW,
        Permissions.DOCUMENT_VIEW_ALL,
        Permissions.DOCUMENT_DELETE,
        Permissions.CLASSROOM_CREATE,
        Permissions.CLASSROOM_MANAGE,
        Permissions.CLASSROOM_VIEW_STUDENTS,
        Permissions.TUTOR_QUERY,
        Permissions.TUTOR_VIEW_HISTORY,
        Permissions.TUTOR_VIEW_ALL_HISTORY,
        Permissions.ADMIN_VIEW_LOGS,
        Permissions.ADMIN_REINDEX,
        Permissions.ADMIN_WEB_FETCH,
    ]
}


# ============================================================================
# Authorization Service
# ============================================================================

class AuthorizationService:
    """
    Authorization service for classroom and document access control.
    
    Usage:
        auth_service = AuthorizationService()
        
        # Check if user can access classroom
        if auth_service.check_classroom_access(user_id, classroom_id):
            # Allow access
        
        # Check document permissions
        if auth_service.check_document_access(user_id, document_id, 'delete'):
            # Allow deletion
    """
    
    def has_permission(self, user: User, permission: str) -> bool:
        """Check if user has a specific permission"""
        if not user or not user.role:
            return False
        
        role_perms = ROLE_PERMISSIONS.get(user.role, [])
        return permission in role_perms
    
    def check_classroom_access(
        self,
        user_id: str,
        classroom_id: str,
        required_role: Optional[str] = None
    ) -> bool:
        """
        Check if user has access to a classroom.
        
        Args:
            user_id: User ID to check
            classroom_id: Classroom ID to access
            required_role: Optional required role (teacher, admin)
            
        Returns:
            True if access allowed, False otherwise
        """
        user = User.query.get(user_id)
        if not user:
            return False
        
        # Admin always has access
        if user.role == "admin":
            return True
        
        # Check if user is teacher of this classroom
        classroom = Classroom.query.get(classroom_id)
        if not classroom:
            return False
        
        if user.role == "teacher":
            if classroom.teacher_id == user_id:
                return True
            # Teacher might need to be explicitly enrolled
            
        # Check if required_role is met
        if required_role and user.role != required_role:
            if required_role == "teacher" and user.role != "admin":
                return False
        
        # Check student enrollment
        if user.role == "student":
            enrollment = StudentClassroom.query.filter_by(
                student_id=user_id,
                classroom_id=classroom_id,
                is_active=True
            ).first()
            return enrollment is not None
        
        return False
    
    def check_document_access(
        self,
        user_id: str,
        document_id: str,
        action: str = "read"
    ) -> bool:
        """
        Check if user can perform action on document.
        
        Args:
            user_id: User ID
            document_id: Document ID  
            action: read, write, delete
            
        Returns:
            True if action allowed
        """
        from app.models.notes import NoteProcessingJob
        
        user = User.query.get(user_id)
        if not user:
            return False
        
        # Admin can do anything
        if user.role == "admin":
            return True
        
        document = NoteProcessingJob.query.get(document_id)
        if not document:
            return False
        
        # Check classroom access first
        if not self.check_classroom_access(user_id, document.classroom_id):
            return False
        
        # For read: anyone in classroom can read
        if action == "read":
            return True
        
        # For write/delete: check ownership or teacher status
        if action in ["write", "delete"]:
            # Owner can always modify
            if document.student_id == user_id:
                return True
            
            # Teacher of classroom can modify
            classroom = Classroom.query.get(document.classroom_id)
            if user.role == "teacher" and classroom.teacher_id == user_id:
                return True
            
            return False
        
        return False
    
    def check_resource_ownership(
        self,
        user_id: str,
        resource_id: str,
        resource_type: str
    ) -> bool:
        """
        Check if user owns a resource.
        
        Args:
            user_id: User ID
            resource_id: Resource ID
            resource_type: 'document', 'query', etc.
        """
        if resource_type == "document":
            from app.models.notes import NoteProcessingJob
            resource = NoteProcessingJob.query.get(resource_id)
            return resource and resource.student_id == user_id
        
        elif resource_type == "query":
            from app.models.document_intelligence import RAGQuery
            resource = RAGQuery.query.get(resource_id)
            return resource and resource.student_id == user_id
        
        return False
    
    def get_user_classrooms(self, user_id: str) -> List[str]:
        """Get list of classroom IDs user has access to"""
        user = User.query.get(user_id)
        if not user:
            return []
        
        if user.role == "admin":
            # Admin has access to all classrooms in org
            classrooms = Classroom.query.filter_by(
                organization_id=user.organization_id
            ).all()
            return [c.id for c in classrooms]
        
        elif user.role == "teacher":
            classrooms = Classroom.query.filter_by(teacher_id=user_id).all()
            return [c.id for c in classrooms]
        
        else:  # student
            enrollments = StudentClassroom.query.filter_by(
                student_id=user_id,
                is_active=True
            ).all()
            return [e.classroom_id for e in enrollments]


# ============================================================================
# Flask Decorators
# ============================================================================

def require_auth(f):
    """Decorator to require authentication"""
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get("Authorization")
        
        if not auth_header:
            return jsonify({"error": "Missing authorization header"}), 401
        
        try:
            parts = auth_header.split()
            if len(parts) != 2 or parts[0].lower() != "bearer":
                return jsonify({"error": "Invalid authorization format"}), 401
            
            token = parts[1]
            payload = verify_token(token)
            
            user = User.query.get(payload["user_id"])
            if not user:
                return jsonify({"error": "User not found"}), 404
            
            g.current_user = user
            g.user_id = user.id
            
        except Exception as e:
            logger.warning(f"Auth failed: {e}")
            return jsonify({"error": f"Invalid token: {str(e)}"}), 401
        
        return f(*args, **kwargs)
    return decorated


def require_role(*roles):
    """Decorator to require specific roles"""
    def decorator(f):
        @wraps(f)
        @require_auth
        def decorated(*args, **kwargs):
            if not hasattr(g, 'current_user'):
                return jsonify({"error": "Not authenticated"}), 401
            
            if g.current_user.role not in roles:
                return jsonify({
                    "error": f"Requires role: {', '.join(roles)}"
                }), 403
            
            return f(*args, **kwargs)
        return decorated
    return decorator


def require_classroom_access(classroom_id_param="classroom_id"):
    """Decorator to require classroom membership"""
    def decorator(f):
        @wraps(f)
        @require_auth
        def decorated(*args, **kwargs):
            classroom_id = kwargs.get(classroom_id_param) or request.json.get(classroom_id_param)
            
            if not classroom_id:
                return jsonify({"error": "Classroom ID required"}), 400
            
            auth_service = AuthorizationService()
            if not auth_service.check_classroom_access(g.user_id, classroom_id):
                return jsonify({"error": "Access denied to this classroom"}), 403
            
            return f(*args, **kwargs)
        return decorated
    return decorator


# Singleton
_auth_service: Optional[AuthorizationService] = None


def get_authorization_service() -> AuthorizationService:
    """Get or create authorization service singleton"""
    global _auth_service
    if _auth_service is None:
        _auth_service = AuthorizationService()
    return _auth_service
