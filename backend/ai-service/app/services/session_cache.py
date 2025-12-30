"""
Session Cache - Redis caching for fast session lookups

Features:
- TTL-based expiration
- JSON serialization
- Async operations (optional sync fallback)
- Graceful degradation if Redis unavailable
"""
import os
import json
import logging
from typing import Optional, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)

# Try to import redis, gracefully degrade if not available
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("[CACHE] redis package not installed, caching disabled")


# ============================================================================
# Configuration
# ============================================================================

DEFAULT_TTL = int(os.getenv("SESSION_CACHE_TTL", "86400"))  # 24 hours
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
CACHE_PREFIX = "session:"


# ============================================================================
# Cache Implementation
# ============================================================================

class SessionCache:
    """
    Redis-based session cache with TTL.
    
    Usage:
        cache = SessionCache()
        cache.set("session_123", {...})
        data = cache.get("session_123")
    """
    
    def __init__(self, redis_url: str = None, ttl: int = None):
        """
        Initialize cache.
        
        Args:
            redis_url: Redis connection URL
            ttl: Default TTL in seconds
        """
        self.redis_url = redis_url or REDIS_URL
        self.ttl = ttl or DEFAULT_TTL
        self._client = None
        self._available = REDIS_AVAILABLE
        
    @property
    def client(self) -> Optional[Any]:
        """Lazy load Redis client"""
        if not self._available:
            return None
            
        if self._client is None:
            try:
                self._client = redis.from_url(self.redis_url, decode_responses=True)
                # Test connection
                self._client.ping()
                logger.info(f"[CACHE] Connected to Redis: {self.redis_url}")
            except Exception as e:
                logger.error(f"[CACHE] Redis connection failed: {e}")
                self._available = False
                self._client = None
                
        return self._client
    
    @property
    def is_available(self) -> bool:
        """Check if Redis is available"""
        return self.client is not None
    
    # ========================================================================
    # Core Operations
    # ========================================================================
    
    def set(self, session_id: str, data: dict, ttl: int = None) -> bool:
        """
        Store session in cache.
        
        Args:
            session_id: Session UUID
            data: Session data dict
            ttl: Optional TTL override
            
        Returns:
            True if successful
        """
        if not self.is_available:
            return False
            
        try:
            key = f"{CACHE_PREFIX}{session_id}"
            value = json.dumps(data, default=str)
            self.client.setex(key, ttl or self.ttl, value)
            logger.debug(f"[CACHE] SET {session_id[:8]}... TTL={ttl or self.ttl}s")
            return True
        except Exception as e:
            logger.error(f"[CACHE] SET failed: {e}")
            return False
    
    def get(self, session_id: str) -> Optional[dict]:
        """
        Get session from cache.
        
        Args:
            session_id: Session UUID
            
        Returns:
            Session data dict or None
        """
        if not self.is_available:
            return None
            
        try:
            key = f"{CACHE_PREFIX}{session_id}"
            value = self.client.get(key)
            if value:
                logger.debug(f"[CACHE] HIT {session_id[:8]}...")
                return json.loads(value)
            logger.debug(f"[CACHE] MISS {session_id[:8]}...")
            return None
        except Exception as e:
            logger.error(f"[CACHE] GET failed: {e}")
            return None
    
    def delete(self, session_id: str) -> bool:
        """Delete session from cache"""
        if not self.is_available:
            return False
            
        try:
            key = f"{CACHE_PREFIX}{session_id}"
            self.client.delete(key)
            logger.debug(f"[CACHE] DEL {session_id[:8]}...")
            return True
        except Exception as e:
            logger.error(f"[CACHE] DEL failed: {e}")
            return False
    
    def touch(self, session_id: str, ttl: int = None) -> bool:
        """
        Extend TTL without changing data.
        
        Args:
            session_id: Session UUID
            ttl: New TTL in seconds
        """
        if not self.is_available:
            return False
            
        try:
            key = f"{CACHE_PREFIX}{session_id}"
            self.client.expire(key, ttl or self.ttl)
            logger.debug(f"[CACHE] TOUCH {session_id[:8]}... TTL={ttl or self.ttl}s")
            return True
        except Exception as e:
            logger.error(f"[CACHE] TOUCH failed: {e}")
            return False
    
    def exists(self, session_id: str) -> bool:
        """Check if session exists in cache"""
        if not self.is_available:
            return False
            
        try:
            key = f"{CACHE_PREFIX}{session_id}"
            return bool(self.client.exists(key))
        except Exception as e:
            logger.error(f"[CACHE] EXISTS failed: {e}")
            return False
    
    # ========================================================================
    # Batch Operations
    # ========================================================================
    
    def get_many(self, session_ids: list) -> Dict[str, dict]:
        """Get multiple sessions"""
        if not self.is_available or not session_ids:
            return {}
            
        try:
            keys = [f"{CACHE_PREFIX}{sid}" for sid in session_ids]
            values = self.client.mget(keys)
            
            result = {}
            for sid, val in zip(session_ids, values):
                if val:
                    result[sid] = json.loads(val)
                    
            return result
        except Exception as e:
            logger.error(f"[CACHE] MGET failed: {e}")
            return {}
    
    def invalidate_pattern(self, pattern: str = "*") -> int:
        """
        Delete sessions matching pattern.
        
        Args:
            pattern: Redis key pattern (e.g., "*" for all)
            
        Returns:
            Number of keys deleted
        """
        if not self.is_available:
            return 0
            
        try:
            full_pattern = f"{CACHE_PREFIX}{pattern}"
            keys = list(self.client.scan_iter(match=full_pattern))
            if keys:
                deleted = self.client.delete(*keys)
                logger.info(f"[CACHE] Invalidated {deleted} sessions matching {pattern}")
                return deleted
            return 0
        except Exception as e:
            logger.error(f"[CACHE] Invalidate failed: {e}")
            return 0
    
    # ========================================================================
    # Metrics
    # ========================================================================
    
    def get_stats(self) -> dict:
        """Get cache statistics"""
        if not self.is_available:
            return {"available": False}
            
        try:
            info = self.client.info("stats")
            memory = self.client.info("memory")
            
            # Count session keys
            session_keys = list(self.client.scan_iter(match=f"{CACHE_PREFIX}*"))
            
            return {
                "available": True,
                "session_count": len(session_keys),
                "hits": info.get("keyspace_hits", 0),
                "misses": info.get("keyspace_misses", 0),
                "memory_used": memory.get("used_memory_human", "0B"),
            }
        except Exception as e:
            logger.error(f"[CACHE] Stats failed: {e}")
            return {"available": False, "error": str(e)}


# ============================================================================
# Singleton
# ============================================================================

_cache: Optional[SessionCache] = None


def get_session_cache() -> SessionCache:
    """Get singleton cache instance"""
    global _cache
    if _cache is None:
        _cache = SessionCache()
    return _cache
