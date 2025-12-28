"""
Unit Tests for Authorization Service
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from uuid import uuid4


class TestPermissions:
    """Tests for Permissions constants"""
    
    def test_permissions_defined(self):
        from app.services.authorization_service import Permissions
        
        assert hasattr(Permissions, 'DOCUMENT_UPLOAD')
        assert hasattr(Permissions, 'DOCUMENT_VIEW')
        assert hasattr(Permissions, 'DOCUMENT_DELETE')
        assert hasattr(Permissions, 'CLASSROOM_CREATE')
        assert hasattr(Permissions, 'TUTOR_QUERY')


class TestRolePermissions:
    """Tests for role to permissions mapping"""
    
    def test_student_permissions(self):
        from app.services.authorization_service import ROLE_PERMISSIONS, Permissions
        
        student_perms = ROLE_PERMISSIONS['student']
        
        assert Permissions.DOCUMENT_UPLOAD in student_perms
        assert Permissions.DOCUMENT_VIEW in student_perms
        assert Permissions.TUTOR_QUERY in student_perms
        
        # Students should NOT have these
        assert Permissions.CLASSROOM_CREATE not in student_perms
        assert Permissions.ADMIN_VIEW_LOGS not in student_perms
    
    def test_teacher_permissions(self):
        from app.services.authorization_service import ROLE_PERMISSIONS, Permissions
        
        teacher_perms = ROLE_PERMISSIONS['teacher']
        
        assert Permissions.DOCUMENT_VIEW_ALL in teacher_perms
        assert Permissions.CLASSROOM_CREATE in teacher_perms
        assert Permissions.CLASSROOM_MANAGE in teacher_perms
        assert Permissions.TUTOR_VIEW_ALL_HISTORY in teacher_perms
    
    def test_admin_permissions(self):
        from app.services.authorization_service import ROLE_PERMISSIONS, Permissions
        
        admin_perms = ROLE_PERMISSIONS['admin']
        
        assert Permissions.ADMIN_VIEW_LOGS in admin_perms
        assert Permissions.ADMIN_REINDEX in admin_perms
        assert Permissions.ADMIN_WEB_FETCH in admin_perms


class TestAuthorizationService:
    """Tests for AuthorizationService class"""
    
    @pytest.fixture
    def auth_service(self):
        from app.services.authorization_service import AuthorizationService
        return AuthorizationService()
    
    @pytest.fixture
    def mock_user_student(self):
        user = Mock()
        user.id = str(uuid4())
        user.role = 'student'
        user.organization_id = str(uuid4())
        return user
    
    @pytest.fixture
    def mock_user_teacher(self):
        user = Mock()
        user.id = str(uuid4())
        user.role = 'teacher'
        user.organization_id = str(uuid4())
        return user
    
    @pytest.fixture
    def mock_user_admin(self):
        user = Mock()
        user.id = str(uuid4())
        user.role = 'admin'
        user.organization_id = str(uuid4())
        return user
    
    # =========================================================================
    # has_permission Tests
    # =========================================================================
    
    def test_has_permission_student_upload(self, auth_service, mock_user_student):
        from app.services.authorization_service import Permissions
        
        assert auth_service.has_permission(mock_user_student, Permissions.DOCUMENT_UPLOAD) is True
    
    def test_has_permission_student_admin_denied(self, auth_service, mock_user_student):
        from app.services.authorization_service import Permissions
        
        assert auth_service.has_permission(mock_user_student, Permissions.ADMIN_VIEW_LOGS) is False
    
    def test_has_permission_admin_all(self, auth_service, mock_user_admin):
        from app.services.authorization_service import Permissions
        
        assert auth_service.has_permission(mock_user_admin, Permissions.ADMIN_VIEW_LOGS) is True
        assert auth_service.has_permission(mock_user_admin, Permissions.ADMIN_REINDEX) is True
    
    def test_has_permission_none_user(self, auth_service):
        from app.services.authorization_service import Permissions
        
        assert auth_service.has_permission(None, Permissions.DOCUMENT_VIEW) is False
    
    # =========================================================================
    # check_classroom_access Tests
    # =========================================================================
    
    @patch('app.services.authorization_service.User')
    @patch('app.services.authorization_service.Classroom')
    @patch('app.services.authorization_service.StudentClassroom')
    def test_check_classroom_access_admin(self, mock_sc, mock_classroom, mock_user, auth_service):
        """Admin always has access"""
        admin = Mock()
        admin.role = 'admin'
        mock_user.query.get.return_value = admin
        
        result = auth_service.check_classroom_access('admin-id', 'classroom-id')
        assert result is True
    
    @patch('app.services.authorization_service.User')
    @patch('app.services.authorization_service.Classroom')
    @patch('app.services.authorization_service.StudentClassroom')
    def test_check_classroom_access_teacher_own(self, mock_sc, mock_classroom, mock_user, auth_service):
        """Teacher has access to own classroom"""
        teacher_id = str(uuid4())
        classroom_id = str(uuid4())
        
        teacher = Mock()
        teacher.role = 'teacher'
        mock_user.query.get.return_value = teacher
        
        classroom = Mock()
        classroom.teacher_id = teacher_id
        mock_classroom.query.get.return_value = classroom
        
        result = auth_service.check_classroom_access(teacher_id, classroom_id)
        assert result is True
    
    @patch('app.services.authorization_service.User')
    @patch('app.services.authorization_service.Classroom')
    @patch('app.services.authorization_service.StudentClassroom')
    def test_check_classroom_access_student_enrolled(self, mock_sc, mock_classroom, mock_user, auth_service):
        """Enrolled student has access"""
        student_id = str(uuid4())
        classroom_id = str(uuid4())
        
        student = Mock()
        student.role = 'student'
        mock_user.query.get.return_value = student
        
        classroom = Mock()
        classroom.teacher_id = str(uuid4())  # Different teacher
        mock_classroom.query.get.return_value = classroom
        
        enrollment = Mock()
        mock_sc.query.filter_by.return_value.first.return_value = enrollment
        
        result = auth_service.check_classroom_access(student_id, classroom_id)
        assert result is True
    
    @patch('app.services.authorization_service.User')
    @patch('app.services.authorization_service.Classroom')
    @patch('app.services.authorization_service.StudentClassroom')
    def test_check_classroom_access_student_not_enrolled(self, mock_sc, mock_classroom, mock_user, auth_service):
        """Non-enrolled student denied access"""
        student = Mock()
        student.role = 'student'
        mock_user.query.get.return_value = student
        
        classroom = Mock()
        classroom.teacher_id = str(uuid4())
        mock_classroom.query.get.return_value = classroom
        
        mock_sc.query.filter_by.return_value.first.return_value = None  # Not enrolled
        
        result = auth_service.check_classroom_access('student-id', 'classroom-id')
        assert result is False
    
    @patch('app.services.authorization_service.User')
    def test_check_classroom_access_user_not_found(self, mock_user, auth_service):
        """Non-existent user denied access"""
        mock_user.query.get.return_value = None
        
        result = auth_service.check_classroom_access('nonexistent', 'classroom-id')
        assert result is False
    
    # =========================================================================
    # check_document_access Tests
    # =========================================================================
    
    @patch('app.services.authorization_service.User')
    @patch('app.services.authorization_service.Classroom')
    @patch('app.services.authorization_service.StudentClassroom')
    def test_check_document_access_read_enrolled(self, mock_sc, mock_classroom, mock_user, auth_service):
        """Enrolled student can read documents"""
        with patch('app.models.notes.NoteProcessingJob') as mock_doc:
            student_id = str(uuid4())
            
            student = Mock()
            student.role = 'student'
            mock_user.query.get.return_value = student
            
            document = Mock()
            document.classroom_id = 'classroom-1'
            document.student_id = 'other-student'
            mock_doc.query.get.return_value = document
            
            classroom = Mock()
            classroom.teacher_id = 'teacher-1'
            mock_classroom.query.get.return_value = classroom
            
            enrollment = Mock()
            mock_sc.query.filter_by.return_value.first.return_value = enrollment
            
            result = auth_service.check_document_access(student_id, 'doc-1', 'read')
            assert result is True
    
    @patch('app.services.authorization_service.User')
    def test_check_document_access_admin(self, mock_user, auth_service):
        """Admin can access all documents"""
        admin = Mock()
        admin.role = 'admin'
        mock_user.query.get.return_value = admin
        
        result = auth_service.check_document_access('admin-id', 'any-doc', 'delete')
        assert result is True


class TestFlaskDecorators:
    """Tests for Flask decorators"""
    
    def test_require_role_decorator_exists(self):
        from app.services.authorization_service import require_role
        assert callable(require_role)
    
    def test_require_auth_decorator_exists(self):
        from app.services.authorization_service import require_auth
        assert callable(require_auth)
    
    def test_require_classroom_access_decorator_exists(self):
        from app.services.authorization_service import require_classroom_access
        assert callable(require_classroom_access)


class TestAuthorizationServiceSingleton:
    """Tests for singleton pattern"""
    
    def test_get_authorization_service_singleton(self):
        from app.services.authorization_service import get_authorization_service
        
        s1 = get_authorization_service()
        s2 = get_authorization_service()
        assert s1 is s2
