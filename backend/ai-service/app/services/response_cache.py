"""
Response Cache Service using Redis

Caches:
- LLM responses by query hash
- Web resource fetches
- Embedding results

Cost optimization: avoids redundant LLM/embedding calls
"""
import os
import json
import hashlib
import logging
from typing import Optional, Any, Dict
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class CachedResponse:
    """Cached LLM response"""
    answer_short: str
    answer_detailed: Optional[str]
    confidence: float
    reasoning: str
    suggested_topics: list
    generation_time_ms: int
    cache_hit: bool = True


class ResponseCache:
    """
    Redis-based cache for LLM responses and other expensive computations.
    
    Environment variables:
    - REDIS_URL: Redis connection URL (default: redis://localhost:6379)
    - CACHE_TTL_SECONDS: Default TTL in seconds (default: 3600 = 1 hour)
    - CACHE_ENABLED: Enable/disable caching (default: true)
    """
    
    def __init__(self):
        self.enabled = os.getenv("CACHE_ENABLED", "true").lower() == "true"
        self.default_ttl = int(os.getenv("CACHE_TTL_SECONDS", "3600"))
        self.redis_client = None
        self._initialize_redis()
    
    def _initialize_redis(self):
        """Initialize Redis connection"""
        if not self.enabled:
            logger.info("[Cache] Caching disabled")
            return
        
        try:
            import redis
            redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
            self.redis_client = redis.from_url(redis_url, decode_responses=True)
            # Test connection
            self.redis_client.ping()
            logger.info("[Cache] Redis connected successfully")
        except ImportError:
            logger.warning("[Cache] redis package not installed, caching disabled")
            self.enabled = False
        except Exception as e:
            logger.warning(f"[Cache] Redis connection failed: {e}, using in-memory cache")
            self.redis_client = None
            # Fallback to simple in-memory cache
            self._memory_cache: Dict[str, tuple] = {}  # key -> (value, expiry_timestamp)
    
    def _generate_key(self, prefix: str, *args) -> str:
        """Generate cache key from prefix and arguments"""
        content = "|".join(str(arg) for arg in args)
        hash_value = hashlib.md5(content.encode()).hexdigest()[:12]
        return f"ensure:{prefix}:{hash_value}"
    
    def get_llm_response(
        self,
        question: str,
        context_hash: str,
        subject: str
    ) -> Optional[CachedResponse]:
        """
        Get cached LLM response if available.
        
        Args:
            question: User question
            context_hash: Hash of context used
            subject: Academic subject
            
        Returns:
            CachedResponse if found, None otherwise
        """
        if not self.enabled:
            return None
        
        key = self._generate_key("llm", question, context_hash, subject)
        
        try:
            if self.redis_client:
                data = self.redis_client.get(key)
                if data:
                    parsed = json.loads(data)
                    logger.debug(f"[Cache] LLM cache hit for key {key}")
                    return CachedResponse(**parsed, cache_hit=True)
            elif hasattr(self, '_memory_cache'):
                import time
                if key in self._memory_cache:
                    value, expiry = self._memory_cache[key]
                    if time.time() < expiry:
                        return CachedResponse(**value, cache_hit=True)
                    else:
                        del self._memory_cache[key]
        except Exception as e:
            logger.warning(f"[Cache] Error getting cached response: {e}")
        
        return None
    
    def set_llm_response(
        self,
        question: str,
        context_hash: str,
        subject: str,
        response: Dict[str, Any],
        ttl: Optional[int] = None
    ) -> bool:
        """
        Cache LLM response.
        
        Args:
            question: User question
            context_hash: Hash of context used
            subject: Academic subject
            response: Response dict to cache
            ttl: TTL in seconds (default: 1 hour)
            
        Returns:
            True if cached successfully
        """
        if not self.enabled:
            return False
        
        key = self._generate_key("llm", question, context_hash, subject)
        ttl = ttl or self.default_ttl
        
        try:
            data = json.dumps(response)
            if self.redis_client:
                self.redis_client.setex(key, ttl, data)
                logger.debug(f"[Cache] Cached LLM response for key {key}")
                return True
            elif hasattr(self, '_memory_cache'):
                import time
                self._memory_cache[key] = (response, time.time() + ttl)
                return True
        except Exception as e:
            logger.warning(f"[Cache] Error caching response: {e}")
        
        return False
    
    def get_web_resources(self, query: str) -> Optional[Dict[str, Any]]:
        """Get cached web resources"""
        if not self.enabled:
            return None
        
        key = self._generate_key("web", query)
        
        try:
            if self.redis_client:
                data = self.redis_client.get(key)
                if data:
                    return json.loads(data)
            elif hasattr(self, '_memory_cache'):
                import time
                if key in self._memory_cache:
                    value, expiry = self._memory_cache[key]
                    if time.time() < expiry:
                        return value
        except Exception as e:
            logger.warning(f"[Cache] Error getting web resources: {e}")
        
        return None
    
    def set_web_resources(
        self,
        query: str,
        resources: Dict[str, Any],
        ttl: Optional[int] = None
    ) -> bool:
        """Cache web resources (longer TTL since they change less frequently)"""
        if not self.enabled:
            return False
        
        key = self._generate_key("web", query)
        ttl = ttl or (self.default_ttl * 24)  # 24 hours default for web resources
        
        try:
            data = json.dumps(resources)
            if self.redis_client:
                self.redis_client.setex(key, ttl, data)
                return True
            elif hasattr(self, '_memory_cache'):
                import time
                self._memory_cache[key] = (resources, time.time() + ttl)
                return True
        except Exception as e:
            logger.warning(f"[Cache] Error caching web resources: {e}")
        
        return False
    
    def invalidate_pattern(self, pattern: str) -> int:
        """
        Invalidate all keys matching a pattern.
        
        Args:
            pattern: Key pattern to match (e.g., "ensure:llm:*")
            
        Returns:
            Number of keys deleted
        """
        if not self.redis_client:
            return 0
        
        try:
            keys = list(self.redis_client.scan_iter(match=pattern))
            if keys:
                return self.redis_client.delete(*keys)
        except Exception as e:
            logger.warning(f"[Cache] Error invalidating pattern: {e}")
        
        return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        if not self.redis_client:
            if hasattr(self, '_memory_cache'):
                return {
                    "type": "memory",
                    "keys": len(self._memory_cache),
                    "enabled": self.enabled
                }
            return {"type": "none", "enabled": False}
        
        try:
            info = self.redis_client.info("stats")
            return {
                "type": "redis",
                "enabled": self.enabled,
                "hits": info.get("keyspace_hits", 0),
                "misses": info.get("keyspace_misses", 0),
                "keys": self.redis_client.dbsize()
            }
        except Exception:
            return {"type": "redis", "enabled": self.enabled, "error": "Could not get stats"}


# Singleton instance
_response_cache: Optional[ResponseCache] = None


def get_response_cache() -> ResponseCache:
    """Get or create the cache singleton"""
    global _response_cache
    if _response_cache is None:
        _response_cache = ResponseCache()
    return _response_cache


def generate_context_hash(context: str) -> str:
    """Generate a short hash for context comparison"""
    return hashlib.md5(context.encode()).hexdigest()[:16]
