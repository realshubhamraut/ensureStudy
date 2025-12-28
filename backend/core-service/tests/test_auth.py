"""
Tests for Authentication Routes
"""
import pytest
from uuid import uuid4


class TestAuthRegister:
    """Test user registration"""
    
    def test_register_success(self, client):
        """Test successful registration"""
        response = client.post('/api/auth/register', json={
            'email': 'newuser@example.com',
            'username': 'newuser',
            'password': 'securepassword123',
            'first_name': 'New',
            'last_name': 'User',
            'role': 'student'
        })
        
        assert response.status_code == 201
        data = response.get_json()
        assert 'user' in data
        assert 'access_token' in data
        assert 'refresh_token' in data
        assert data['user']['email'] == 'newuser@example.com'
    
    def test_register_missing_email(self, client):
        """Test registration with missing email"""
        response = client.post('/api/auth/register', json={
            'username': 'newuser',
            'password': 'securepassword123'
        })
        
        assert response.status_code == 400
        data = response.get_json()
        assert 'error' in data
    
    def test_register_duplicate_email(self, client, test_user):
        """Test registration with existing email"""
        response = client.post('/api/auth/register', json={
            'email': test_user['email'],
            'username': 'differentuser',
            'password': 'securepassword123'
        })
        
        assert response.status_code == 409
        data = response.get_json()
        assert 'already registered' in data['error'].lower()
    
    def test_register_weak_password(self, client):
        """Test registration with weak password"""
        response = client.post('/api/auth/register', json={
            'email': 'weak@example.com',
            'username': 'weakuser',
            'password': 'short'
        })
        
        assert response.status_code == 400
        data = response.get_json()
        assert 'password' in data['error'].lower()


class TestAuthLogin:
    """Test user login"""
    
    def test_login_success(self, client, test_user):
        """Test successful login"""
        response = client.post('/api/auth/login', json={
            'email': test_user['email'],
            'password': test_user['password']
        })
        
        assert response.status_code == 200
        data = response.get_json()
        assert 'access_token' in data
        assert 'refresh_token' in data
        assert data['user']['email'] == test_user['email']
    
    def test_login_wrong_password(self, client, test_user):
        """Test login with wrong password"""
        response = client.post('/api/auth/login', json={
            'email': test_user['email'],
            'password': 'wrongpassword'
        })
        
        assert response.status_code == 401
        data = response.get_json()
        assert 'invalid' in data['error'].lower()
    
    def test_login_nonexistent_user(self, client):
        """Test login with non-existent user"""
        response = client.post('/api/auth/login', json={
            'email': 'nonexistent@example.com',
            'password': 'anypassword'
        })
        
        assert response.status_code == 401
    
    def test_login_missing_credentials(self, client):
        """Test login without credentials"""
        response = client.post('/api/auth/login', json={})
        
        assert response.status_code == 400


class TestAuthMe:
    """Test current user endpoint"""
    
    def test_get_me_success(self, client, auth_headers, test_user):
        """Test getting current user"""
        response = client.get('/api/auth/me', headers=auth_headers)
        
        assert response.status_code == 200
        data = response.get_json()
        assert data['user']['email'] == test_user['email']
    
    def test_get_me_no_token(self, client):
        """Test getting current user without token"""
        response = client.get('/api/auth/me')
        
        assert response.status_code == 401
    
    def test_get_me_invalid_token(self, client):
        """Test getting current user with invalid token"""
        response = client.get('/api/auth/me', headers={
            'Authorization': 'Bearer invalidtoken'
        })
        
        assert response.status_code == 401


class TestAuthRefresh:
    """Test token refresh"""
    
    def test_refresh_success(self, client, test_user):
        """Test successful token refresh"""
        # First login to get tokens
        login_response = client.post('/api/auth/login', json={
            'email': test_user['email'],
            'password': test_user['password']
        })
        refresh_token = login_response.get_json()['refresh_token']
        
        # Refresh the token
        response = client.post('/api/auth/refresh', json={
            'refresh_token': refresh_token
        })
        
        assert response.status_code == 200
        data = response.get_json()
        assert 'access_token' in data
    
    def test_refresh_invalid_token(self, client):
        """Test refresh with invalid token"""
        response = client.post('/api/auth/refresh', json={
            'refresh_token': 'invalidtoken'
        })
        
        assert response.status_code == 401


class TestAuthChangePassword:
    """Test password change"""
    
    def test_change_password_success(self, client, auth_headers, test_user):
        """Test successful password change"""
        response = client.post('/api/auth/change-password', 
            headers=auth_headers,
            json={
                'current_password': test_user['password'],
                'new_password': 'newpassword123'
            }
        )
        
        assert response.status_code == 200
        
        # Verify can login with new password
        login_response = client.post('/api/auth/login', json={
            'email': test_user['email'],
            'password': 'newpassword123'
        })
        assert login_response.status_code == 200
    
    def test_change_password_wrong_current(self, client, auth_headers):
        """Test password change with wrong current password"""
        response = client.post('/api/auth/change-password',
            headers=auth_headers,
            json={
                'current_password': 'wrongpassword',
                'new_password': 'newpassword123'
            }
        )
        
        assert response.status_code == 401
