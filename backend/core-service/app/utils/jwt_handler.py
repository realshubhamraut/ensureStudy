"""
JWT Token Handler Utilities
"""
import os
from datetime import datetime, timedelta
from typing import Tuple, Optional, Dict, Any
import jwt


JWT_SECRET = os.getenv("JWT_SECRET", "your-super-secret-key-min-32-chars-here")
REFRESH_SECRET = os.getenv("REFRESH_TOKEN_SECRET", "your-refresh-secret-min-32-chars")
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_HOURS = int(os.getenv("JWT_EXPIRATION_HOURS", "24"))
REFRESH_TOKEN_EXPIRE_DAYS = 7


def create_tokens(user_id: str, role: str) -> Tuple[str, str]:
    """
    Create access and refresh tokens for a user.
    
    Args:
        user_id: The user's UUID string
        role: User role (student, teacher, parent)
    
    Returns:
        Tuple of (access_token, refresh_token)
    """
    now = datetime.utcnow()
    
    # Access token payload
    access_payload = {
        "user_id": user_id,
        "role": role,
        "type": "access",
        "iat": now,
        "exp": now + timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS)
    }
    
    # Refresh token payload
    refresh_payload = {
        "user_id": user_id,
        "type": "refresh",
        "iat": now,
        "exp": now + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    }
    
    access_token = jwt.encode(access_payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    refresh_token = jwt.encode(refresh_payload, REFRESH_SECRET, algorithm=JWT_ALGORITHM)
    
    return access_token, refresh_token


def verify_token(token: str, token_type: str = "access") -> Dict[str, Any]:
    """
    Verify and decode a JWT token.
    
    Args:
        token: The JWT token string
        token_type: Either "access" or "refresh"
    
    Returns:
        Decoded payload dictionary
    
    Raises:
        jwt.ExpiredSignatureError: Token has expired
        jwt.InvalidTokenError: Token is invalid
    """
    secret = JWT_SECRET if token_type == "access" else REFRESH_SECRET
    
    payload = jwt.decode(token, secret, algorithms=[JWT_ALGORITHM])
    
    # Only check token type if it exists in payload (for backwards compatibility)
    if payload.get("type") and payload.get("type") != token_type:
        raise jwt.InvalidTokenError(f"Invalid token type. Expected {token_type}")
    
    return payload


def refresh_access_token(refresh_token: str) -> Optional[str]:
    """
    Generate a new access token from a valid refresh token.
    
    Args:
        refresh_token: Valid refresh token
    
    Returns:
        New access token or None if refresh token is invalid
    """
    try:
        payload = verify_token(refresh_token, token_type="refresh")
        user_id = payload.get("user_id")
        
        if not user_id:
            return None
        
        # Generate new access token (need to fetch role from DB in real implementation)
        now = datetime.utcnow()
        access_payload = {
            "user_id": user_id,
            "role": "student",  # Should fetch from DB
            "type": "access",
            "iat": now,
            "exp": now + timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS)
        }
        
        return jwt.encode(access_payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    
    except jwt.PyJWTError:
        return None


def get_user_id_from_token(token: str) -> Optional[str]:
    """
    Extract user ID from a token without full verification (for logging purposes).
    """
    try:
        # Decode without verification
        payload = jwt.decode(token, options={"verify_signature": False})
        return payload.get("user_id")
    except jwt.PyJWTError:
        return None
