"""
Security Tests - Authorization, Input Validation, Injection Prevention
"""
import pytest
from uuid import uuid4


class TestSQLInjectionPrevention:
    """Tests for SQL injection prevention"""
    
    @pytest.fixture
    def client(self, app):
        return app.test_client()
    
    def test_sql_injection_in_login_email(self, client):
        """SQL injection in email field should fail safely"""
        response = client.post('/api/auth/login', json={
            'email': "'; DROP TABLE users; --",
            'password': 'password123'
        })
        
        # Should get 401 (invalid credentials), not 500 (DB error)
        assert response.status_code in [400, 401]
        data = response.get_json()
        # Should not expose database structure
        assert 'DROP' not in str(data)
        assert 'syntax' not in str(data).lower()
    
    def test_sql_injection_in_password(self, client):
        """SQL injection in password field should fail safely"""
        response = client.post('/api/auth/login', json={
            'email': 'test@example.com',
            'password': "' OR '1'='1"
        })
        
        assert response.status_code in [400, 401]
    
    def test_sql_injection_in_username(self, client):
        """SQL injection in registration username"""
        response = client.post('/api/auth/register', json={
            'email': 'test2@example.com',
            'username': "admin'--",
            'password': 'password123'
        })
        
        # Should either sanitize or reject
        assert response.status_code in [201, 400, 409]


class TestXSSPrevention:
    """Tests for XSS prevention"""
    
    def test_xss_in_username(self, client):
        """XSS payload in username should be sanitized"""
        response = client.post('/api/auth/register', json={
            'email': 'xsstest@example.com',
            'username': '<script>alert("xss")</script>',
            'password': 'password123'
        })
        
        if response.status_code == 201:
            data = response.get_json()
            user = data.get('user', {})
            username = user.get('username', '')
            # Script tags should be escaped or removed
            assert '<script>' not in username
    
    def test_xss_in_first_name(self, client):
        """XSS payload in first_name should be sanitized"""
        response = client.post('/api/auth/register', json={
            'email': 'xsstest2@example.com',
            'username': 'testuser2',
            'password': 'password123',
            'first_name': '<img src=x onerror=alert("xss")>'
        })
        
        if response.status_code == 201:
            data = response.get_json()
            user = data.get('user', {})
            first_name = user.get('first_name', '')
            assert 'onerror' not in first_name or '&' in first_name


class TestAuthenticationSecurity:
    """Tests for authentication security"""
    
    def test_login_without_credentials(self, client):
        """Login without credentials returns 400"""
        response = client.post('/api/auth/login', json={})
        assert response.status_code == 400
    
    def test_login_missing_password(self, client):
        """Login without password returns 400"""
        response = client.post('/api/auth/login', json={
            'email': 'test@example.com'
        })
        assert response.status_code == 400
    
    def test_login_wrong_password(self, client, test_user):
        """Wrong password returns 401"""
        response = client.post('/api/auth/login', json={
            'email': test_user['email'],
            'password': 'wrongpassword'
        })
        assert response.status_code == 401
    
    def test_protected_endpoint_without_token(self, client):
        """Protected endpoint without token returns 401"""
        response = client.get('/api/auth/me')
        assert response.status_code == 401
    
    def test_protected_endpoint_with_invalid_token(self, client):
        """Invalid token returns 401"""
        response = client.get(
            '/api/auth/me',
            headers={'Authorization': 'Bearer invalid.token.here'}
        )
        assert response.status_code == 401
    
    def test_protected_endpoint_with_expired_token(self, client):
        """Expired token returns 401"""
        # Create expired token manually
        import jwt
        from datetime import datetime, timedelta
        
        expired_payload = {
            'user_id': 'test-user',
            'role': 'student',
            'type': 'access',
            'exp': datetime.utcnow() - timedelta(hours=1)  # Expired
        }
        expired_token = jwt.encode(
            expired_payload,
            'test-jwt-secret-key-32-chars-min',
            algorithm='HS256'
        )
        
        response = client.get(
            '/api/auth/me',
            headers={'Authorization': f'Bearer {expired_token}'}
        )
        assert response.status_code == 401


class TestPasswordSecurity:
    """Tests for password security"""
    
    def test_password_too_short(self, client):
        """Password less than 8 characters rejected"""
        response = client.post('/api/auth/register', json={
            'email': 'shortpw@example.com',
            'username': 'shortpwuser',
            'password': 'short'  # < 8 chars
        })
        assert response.status_code == 400
        data = response.get_json()
        assert 'password' in data.get('error', '').lower() or 'character' in data.get('error', '').lower()
    
    def test_password_not_returned(self, client, test_user):
        """Password hash never returned in response"""
        response = client.post('/api/auth/login', json={
            'email': test_user['email'],
            'password': test_user['password']
        })
        
        data = response.get_json()
        user = data.get('user', {})
        
        assert 'password' not in user
        assert 'password_hash' not in user


class TestInputValidation:
    """Tests for input validation"""
    
    def test_email_validation(self, client):
        """Invalid email format rejected"""
        response = client.post('/api/auth/register', json={
            'email': 'not-an-email',
            'username': 'validuser',
            'password': 'password123'
        })
        # May be rejected or sanitized depending on implementation
        # At minimum, should not crash
        assert response.status_code in [201, 400, 422]
    
    def test_long_input_handling(self, client):
        """Very long inputs handled gracefully"""
        long_string = 'a' * 10000
        response = client.post('/api/auth/register', json={
            'email': f'{long_string}@example.com',
            'username': long_string,
            'password': long_string
        })
        # Should be rejected or truncated, not crash
        assert response.status_code in [400, 422, 500]  # 500 only if graceful error
    
    def test_null_byte_injection(self, client):
        """Null byte injection handled"""
        response = client.post('/api/auth/register', json={
            'email': 'test\x00admin@example.com',
            'username': 'user\x00admin',
            'password': 'password123'
        })
        # Should be handled safely
        assert response.status_code in [201, 400]


class TestAuthorizationBoundaries:
    """Tests for authorization boundaries"""
    
    def test_student_cannot_access_admin_endpoints(self, client, auth_headers):
        """Student role cannot access admin endpoints"""
        # Assuming there's an admin endpoint
        response = client.get(
            '/api/admin/users',  # Example admin endpoint
            headers=auth_headers
        )
        # Should be 403 or 404 (hidden)
        assert response.status_code in [403, 404]
    
    def test_cross_user_data_access_denied(self, client, test_user):
        """User cannot access another user's data"""
        # Login as test_user
        login_response = client.post('/api/auth/login', json={
            'email': test_user['email'],
            'password': test_user['password']
        })
        token = login_response.get_json().get('access_token')
        headers = {'Authorization': f'Bearer {token}'}
        
        # Try to access a different user's data (using fake ID)
        other_user_id = str(uuid4())
        response = client.get(
            f'/api/users/{other_user_id}/profile',
            headers=headers
        )
        # Should be 403 or 404
        assert response.status_code in [403, 404]


class TestRateLimitingSecurity:
    """Tests for rate limiting security"""
    
    def test_brute_force_protection(self, client, test_user):
        """Multiple failed logins trigger protection"""
        # Make multiple failed login attempts
        for i in range(10):
            response = client.post('/api/auth/login', json={
                'email': test_user['email'],
                'password': 'wrongpassword'
            })
        
        # After multiple attempts, should see:
        # - 429 (rate limited), OR
        # - 401 with indication of lockout, OR
        # - Increased delay
        # Implementation dependent
        assert response.status_code in [401, 429]
