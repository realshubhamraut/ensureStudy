"""
Rate Limiter - Redis-based request throttling

Implements per-user and per-action rate limiting to prevent abuse
and control costs for expensive operations.
"""
import os
import time
import logging
from typing import Optional, Dict, Any
from functools import wraps
from flask import request, jsonify, g

logger = logging.getLogger(__name__)


# ============================================================================
# Rate Limit Configuration
# ============================================================================

RATE_LIMITS = {
    # Document operations
    "document_upload": {"max_requests": 10, "window_seconds": 3600},     # 10/hour
    "document_delete": {"max_requests": 20, "window_seconds": 3600},     # 20/hour
    
    # AI Tutor operations (expensive LLM calls)
    "ai_tutor_query_minute": {"max_requests": 5, "window_seconds": 60},   # 5/minute
    "ai_tutor_query_hour": {"max_requests": 50, "window_seconds": 3600},  # 50/hour
    
    # Web fetching (teacher only)
    "web_fetch": {"max_requests": 5, "window_seconds": 3600},             # 5/hour
    
    # General API calls
    "api_global": {"max_requests": 100, "window_seconds": 60},            # 100/minute
    
    # Authentication
    "login_attempt": {"max_requests": 5, "window_seconds": 300},          # 5 per 5 min
    "password_reset": {"max_requests": 3, "window_seconds": 3600},        # 3/hour
}


# ============================================================================
# Rate Limiter Service
# ============================================================================

class RateLimiter:
    """
    Redis-based rate limiter with sliding window.
    
    Usage:
        limiter = RateLimiter()
        
        # Check rate limit
        if limiter.is_rate_limited(user_id, "ai_tutor_query_minute"):
            return "Rate limit exceeded", 429
        
        # Record request
        limiter.record_request(user_id, "ai_tutor_query_minute")
    """
    
    def __init__(self, redis_url: str = None):
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379")
        self.redis_client = None
        self.enabled = os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true"
        self._init_redis()
    
    def _init_redis(self):
        """Initialize Redis connection"""
        if not self.enabled:
            logger.info("[RateLimiter] Disabled via RATE_LIMIT_ENABLED=false")
            return
        
        try:
            import redis
            self.redis_client = redis.from_url(
                self.redis_url,
                decode_responses=True,
                socket_timeout=5
            )
            self.redis_client.ping()
            logger.info("[RateLimiter] Redis connected")
        except ImportError:
            logger.warning("[RateLimiter] redis package not installed")
            self.enabled = False
        except Exception as e:
            logger.warning(f"[RateLimiter] Redis connection failed: {e}")
            self.redis_client = None
    
    def _get_key(self, user_id: str, action: str) -> str:
        """Generate Redis key for rate limiting"""
        return f"rate_limit:{user_id}:{action}"
    
    def check_rate_limit(
        self,
        user_id: str,
        action: str,
        max_requests: int = None,
        window_seconds: int = None
    ) -> Dict[str, Any]:
        """
        Check if user has exceeded rate limit.
        
        Returns:
            Dict with keys: allowed, remaining, reset_at, retry_after
        """
        if not self.enabled or not self.redis_client:
            return {"allowed": True, "remaining": 999, "reset_at": 0, "retry_after": 0}
        
        # Get limits from config or params
        config = RATE_LIMITS.get(action, {"max_requests": 100, "window_seconds": 60})
        max_requests = max_requests or config["max_requests"]
        window_seconds = window_seconds or config["window_seconds"]
        
        key = self._get_key(user_id, action)
        now = time.time()
        window_start = now - window_seconds
        
        try:
            pipe = self.redis_client.pipeline()
            
            # Remove old entries
            pipe.zremrangebyscore(key, 0, window_start)
            
            # Count current entries
            pipe.zcard(key)
            
            # Get oldest entry time
            pipe.zrange(key, 0, 0, withscores=True)
            
            results = pipe.execute()
            current_count = results[1]
            oldest_entry = results[2]
            
            # Calculate reset time
            if oldest_entry:
                reset_at = oldest_entry[0][1] + window_seconds
            else:
                reset_at = now + window_seconds
            
            remaining = max(0, max_requests - current_count)
            allowed = current_count < max_requests
            retry_after = 0 if allowed else int(reset_at - now)
            
            return {
                "allowed": allowed,
                "remaining": remaining,
                "reset_at": int(reset_at),
                "retry_after": retry_after,
                "limit": max_requests,
                "window": window_seconds
            }
            
        except Exception as e:
            logger.error(f"[RateLimiter] Check failed: {e}")
            return {"allowed": True, "remaining": 999, "reset_at": 0, "retry_after": 0}
    
    def record_request(self, user_id: str, action: str) -> bool:
        """Record a request for rate limiting"""
        if not self.enabled or not self.redis_client:
            return True
        
        config = RATE_LIMITS.get(action, {"max_requests": 100, "window_seconds": 60})
        key = self._get_key(user_id, action)
        now = time.time()
        
        try:
            pipe = self.redis_client.pipeline()
            pipe.zadd(key, {str(now): now})
            pipe.expire(key, config["window_seconds"] + 60)  # Cleanup buffer
            pipe.execute()
            return True
        except Exception as e:
            logger.error(f"[RateLimiter] Record failed: {e}")
            return False
    
    def is_rate_limited(self, user_id: str, action: str) -> bool:
        """Simple check if user is rate limited"""
        result = self.check_rate_limit(user_id, action)
        return not result["allowed"]
    
    def get_remaining(self, user_id: str, action: str) -> int:
        """Get remaining requests for action"""
        result = self.check_rate_limit(user_id, action)
        return result["remaining"]


# ============================================================================
# Flask Decorator
# ============================================================================

def rate_limit(action: str, max_requests: int = None, window_seconds: int = None):
    """
    Flask decorator for rate limiting.
    
    Usage:
        @rate_limit("ai_tutor_query_minute")
        def query_tutor():
            ...
    """
    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            limiter = get_rate_limiter()
            
            # Get user ID from g (set by auth middleware)
            user_id = getattr(g, 'user_id', None) or request.remote_addr
            
            result = limiter.check_rate_limit(user_id, action, max_requests, window_seconds)
            
            if not result["allowed"]:
                response = jsonify({
                    "error": "Rate limit exceeded",
                    "retry_after": result["retry_after"]
                })
                response.status_code = 429
                response.headers["Retry-After"] = str(result["retry_after"])
                response.headers["X-RateLimit-Limit"] = str(result.get("limit", 0))
                response.headers["X-RateLimit-Remaining"] = "0"
                response.headers["X-RateLimit-Reset"] = str(result["reset_at"])
                return response
            
            # Record this request
            limiter.record_request(user_id, action)
            
            # Add rate limit headers
            response = f(*args, **kwargs)
            
            # If response is tuple (data, status_code), handle accordingly
            if isinstance(response, tuple):
                return response
            
            return response
        
        return decorated
    return decorator


# ============================================================================
# Singleton
# ============================================================================

_rate_limiter: Optional[RateLimiter] = None


def get_rate_limiter() -> RateLimiter:
    """Get or create rate limiter singleton"""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter()
    return _rate_limiter
