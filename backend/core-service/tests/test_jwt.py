"""
Tests for JWT Handler Utilities
"""
import pytest
import time
from app.utils.jwt_handler import (
    create_tokens, 
    verify_token, 
    refresh_access_token,
    create_access_token,
    create_refresh_token
)


class TestJWTCreation:
    """Test JWT token creation"""
    
    def test_create_tokens(self):
        """Test creating access and refresh tokens"""
        access_token, refresh_token = create_tokens('user-123', 'student')
        
        assert access_token is not None
        assert refresh_token is not None
        assert access_token != refresh_token
    
    def test_create_access_token(self):
        """Test creating access token with claims"""
        token = create_access_token('user-456', 'teacher')
        
        assert token is not None
        
        # Verify the token
        payload = verify_token(token)
        assert payload['user_id'] == 'user-456'
        assert payload['role'] == 'teacher'
        assert payload['type'] == 'access'
    
    def test_create_refresh_token(self):
        """Test creating refresh token"""
        token = create_refresh_token('user-789')
        
        assert token is not None
        
        payload = verify_token(token)
        assert payload['user_id'] == 'user-789'
        assert payload['type'] == 'refresh'


class TestJWTVerification:
    """Test JWT token verification"""
    
    def test_verify_valid_token(self):
        """Test verifying a valid token"""
        token = create_access_token('user-123', 'student')
        payload = verify_token(token)
        
        assert payload is not None
        assert payload['user_id'] == 'user-123'
    
    def test_verify_invalid_token(self):
        """Test verifying an invalid token"""
        with pytest.raises(Exception):
            verify_token('invalid-token-string')
    
    def test_verify_tampered_token(self):
        """Test verifying a tampered token"""
        token = create_access_token('user-123', 'student')
        tampered = token[:-5] + 'xxxxx'
        
        with pytest.raises(Exception):
            verify_token(tampered)


class TestJWTRefresh:
    """Test token refresh functionality"""
    
    def test_refresh_with_valid_token(self):
        """Test refreshing with valid refresh token"""
        _, refresh_token = create_tokens('user-123', 'student')
        
        new_access_token = refresh_access_token(refresh_token)
        
        assert new_access_token is not None
        
        payload = verify_token(new_access_token)
        assert payload['user_id'] == 'user-123'
        assert payload['type'] == 'access'
    
    def test_refresh_with_access_token_fails(self):
        """Test that refreshing with access token fails"""
        access_token, _ = create_tokens('user-123', 'student')
        
        result = refresh_access_token(access_token)
        
        assert result is None
    
    def test_refresh_with_invalid_token(self):
        """Test refreshing with invalid token"""
        result = refresh_access_token('invalid-token')
        
        assert result is None
