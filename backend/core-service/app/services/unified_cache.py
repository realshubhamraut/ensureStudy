"""
Unified Cache Service - 6-Layer Caching System

Layers:
1. OCR Results (Redis + S3)
2. Embeddings (Redis, no expiry)
3. Vector Search (Redis, 1h TTL)
4. RAG Responses (Redis, 24h TTL)
5. Document Metadata (Redis, 1h TTL)
6. Web Fetch Results (Redis + S3)

Features:
- Automatic fallback to in-memory if Redis unavailable
- Decorator for easy caching
- Metrics tracking
- Pattern-based invalidation
"""
import os
import json
import time
import hashlib
import logging
import functools
from typing import Optional, Any, Dict, Callable, List
from dataclasses import dataclass, asdict, field
from datetime import datetime

logger = logging.getLogger(__name__)


# ============================================================================
# Cache Metrics
# ============================================================================

@dataclass
class CacheMetrics:
    """Track cache performance metrics"""
    hits: int = 0
    misses: int = 0
    errors: int = 0
    avg_get_time_ms: float = 0.0
    avg_set_time_ms: float = 0.0
    last_reset: datetime = field(default_factory=datetime.now)
    
    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return (self.hits / total * 100) if total > 0 else 0.0


# ============================================================================
# Cache Key Builders
# ============================================================================

class CacheKeys:
    """Cache key generation utilities"""
    PREFIX = "ensure"
    
    @staticmethod
    def _hash(content: str, length: int = 12) -> str:
        return hashlib.sha256(content.encode()).hexdigest()[:length]
    
    @classmethod
    def ocr(cls, image_bytes: bytes) -> str:
        """OCR result cache key"""
        hash_val = hashlib.sha256(image_bytes).hexdigest()[:16]
        return f"{cls.PREFIX}:ocr:{hash_val}"
    
    @classmethod
    def embedding(cls, text: str, model: str = "minilm") -> str:
        """Embedding cache key"""
        hash_val = cls._hash(text)
        return f"{cls.PREFIX}:emb:{model}:{hash_val}"
    
    @classmethod
    def search(cls, query_hash: str, classroom_id: str, top_k: int) -> str:
        """Vector search results cache key"""
        return f"{cls.PREFIX}:search:{query_hash}:{classroom_id}:{top_k}"
    
    @classmethod
    def rag(cls, question: str, classroom_id: str) -> str:
        """RAG response cache key"""
        q_hash = cls._hash(question)
        return f"{cls.PREFIX}:rag:{q_hash}:{classroom_id}"
    
    @classmethod
    def document(cls, document_id: str) -> str:
        """Document metadata cache key"""
        return f"{cls.PREFIX}:doc:{document_id}"
    
    @classmethod
    def chunk(cls, chunk_id: str) -> str:
        """Chunk metadata cache key"""
        return f"{cls.PREFIX}:chunk:{chunk_id}"
    
    @classmethod
    def web(cls, source_type: str, url: str) -> str:
        """Web fetch results cache key"""
        url_hash = cls._hash(url)
        return f"{cls.PREFIX}:web:{source_type}:{url_hash}"


# ============================================================================
# TTL Constants
# ============================================================================

class CacheTTL:
    """Cache TTL values in seconds"""
    OCR = 7 * 24 * 3600          # 7 days
    EMBEDDING = None             # No expiry (deterministic)
    SEARCH = 3600                # 1 hour
    RAG = 24 * 3600              # 24 hours
    DOCUMENT_META = 3600         # 1 hour
    WEB = 7 * 24 * 3600          # 7 days


# ============================================================================
# Unified Cache Service
# ============================================================================

class UnifiedCacheService:
    """
    Unified 6-layer cache service with Redis backend.
    
    Features:
    - Graceful degradation if Redis unavailable
    - In-memory fallback
    - Metrics tracking
    - Pattern-based invalidation
    
    Usage:
        cache = UnifiedCacheService()
        
        # OCR caching
        result = cache.get_ocr(image_bytes)
        cache.set_ocr(image_bytes, ocr_result)
        
        # Embedding caching
        embedding = cache.get_embedding(text, model)
        cache.set_embedding(text, model, embedding_vector)
    """
    
    def __init__(self, redis_url: str = None):
        self.enabled = os.getenv("CACHE_ENABLED", "true").lower() == "true"
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379")
        self.redis_client = None
        self._memory_cache: Dict[str, tuple] = {}  # Fallback cache
        self._metrics: Dict[str, CacheMetrics] = {
            "ocr": CacheMetrics(),
            "embedding": CacheMetrics(),
            "search": CacheMetrics(),
            "rag": CacheMetrics(),
            "document": CacheMetrics(),
            "web": CacheMetrics()
        }
        
        self._initialize_redis()
    
    def _initialize_redis(self):
        """Initialize Redis connection with error handling"""
        if not self.enabled:
            logger.info("[UnifiedCache] Caching disabled via CACHE_ENABLED=false")
            return
        
        try:
            import redis
            self.redis_client = redis.from_url(
                self.redis_url,
                decode_responses=True,
                socket_timeout=5,
                socket_connect_timeout=5
            )
            self.redis_client.ping()
            logger.info("[UnifiedCache] Redis connected successfully")
        except ImportError:
            logger.warning("[UnifiedCache] redis package not installed")
            self.redis_client = None
        except Exception as e:
            logger.warning(f"[UnifiedCache] Redis connection failed: {e}")
            self.redis_client = None
    
    # =========================================================================
    # Core Operations
    # =========================================================================
    
    def _get(self, key: str, layer: str) -> Optional[Any]:
        """Get value from cache with metrics tracking"""
        start = time.time()
        
        try:
            if self.redis_client:
                data = self.redis_client.get(key)
                if data:
                    self._metrics[layer].hits += 1
                    return json.loads(data)
            else:
                # In-memory fallback
                if key in self._memory_cache:
                    value, expiry = self._memory_cache[key]
                    if expiry is None or time.time() < expiry:
                        self._metrics[layer].hits += 1
                        return value
                    del self._memory_cache[key]
            
            self._metrics[layer].misses += 1
            return None
            
        except Exception as e:
            self._metrics[layer].errors += 1
            logger.warning(f"[UnifiedCache] Get error: {e}")
            return None
        finally:
            elapsed = (time.time() - start) * 1000
            metrics = self._metrics[layer]
            total = metrics.hits + metrics.misses
            metrics.avg_get_time_ms = (metrics.avg_get_time_ms * (total - 1) + elapsed) / total if total > 0 else elapsed
    
    def _set(self, key: str, value: Any, layer: str, ttl: Optional[int] = None) -> bool:
        """Set value in cache with metrics tracking"""
        start = time.time()
        
        try:
            data = json.dumps(value)
            
            if self.redis_client:
                if ttl:
                    self.redis_client.setex(key, ttl, data)
                else:
                    self.redis_client.set(key, data)
                return True
            else:
                # In-memory fallback
                expiry = time.time() + ttl if ttl else None
                self._memory_cache[key] = (value, expiry)
                return True
                
        except Exception as e:
            self._metrics[layer].errors += 1
            logger.warning(f"[UnifiedCache] Set error: {e}")
            return False
        finally:
            elapsed = (time.time() - start) * 1000
            metrics = self._metrics[layer]
            metrics.avg_set_time_ms = (metrics.avg_set_time_ms + elapsed) / 2
    
    def _delete(self, key: str) -> bool:
        """Delete key from cache"""
        try:
            if self.redis_client:
                return self.redis_client.delete(key) > 0
            elif key in self._memory_cache:
                del self._memory_cache[key]
                return True
        except Exception as e:
            logger.warning(f"[UnifiedCache] Delete error: {e}")
        return False
    
    def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate all keys matching pattern"""
        count = 0
        try:
            if self.redis_client:
                keys = list(self.redis_client.scan_iter(match=pattern))
                if keys:
                    count = self.redis_client.delete(*keys)
                    logger.info(f"[UnifiedCache] Invalidated {count} keys matching '{pattern}'")
            else:
                # In-memory fallback with simple pattern matching
                import fnmatch
                to_delete = [k for k in self._memory_cache if fnmatch.fnmatch(k, pattern)]
                for k in to_delete:
                    del self._memory_cache[k]
                count = len(to_delete)
        except Exception as e:
            logger.warning(f"[UnifiedCache] Invalidate error: {e}")
        return count
    
    # =========================================================================
    # Layer 1: OCR Results
    # =========================================================================
    
    def get_ocr(self, image_bytes: bytes) -> Optional[Dict]:
        """Get cached OCR result"""
        key = CacheKeys.ocr(image_bytes)
        return self._get(key, "ocr")
    
    def set_ocr(self, image_bytes: bytes, result: Dict) -> bool:
        """Cache OCR result (7 days)"""
        key = CacheKeys.ocr(image_bytes)
        return self._set(key, result, "ocr", CacheTTL.OCR)
    
    # =========================================================================
    # Layer 2: Embeddings
    # =========================================================================
    
    def get_embedding(self, text: str, model: str = "minilm") -> Optional[List[float]]:
        """Get cached embedding vector"""
        key = CacheKeys.embedding(text, model)
        return self._get(key, "embedding")
    
    def set_embedding(self, text: str, model: str, embedding: List[float]) -> bool:
        """Cache embedding (no expiry - deterministic)"""
        key = CacheKeys.embedding(text, model)
        return self._set(key, embedding, "embedding", ttl=None)
    
    # =========================================================================
    # Layer 3: Vector Search Results
    # =========================================================================
    
    def get_search(self, query_hash: str, classroom_id: str, top_k: int) -> Optional[List]:
        """Get cached search results"""
        key = CacheKeys.search(query_hash, classroom_id, top_k)
        return self._get(key, "search")
    
    def set_search(self, query_hash: str, classroom_id: str, top_k: int, results: List) -> bool:
        """Cache search results (1 hour)"""
        key = CacheKeys.search(query_hash, classroom_id, top_k)
        return self._set(key, results, "search", CacheTTL.SEARCH)
    
    # =========================================================================
    # Layer 4: RAG Responses
    # =========================================================================
    
    def get_rag(self, question: str, classroom_id: str) -> Optional[Dict]:
        """Get cached RAG response"""
        key = CacheKeys.rag(question, classroom_id)
        return self._get(key, "rag")
    
    def set_rag(self, question: str, classroom_id: str, response: Dict) -> bool:
        """Cache RAG response (24 hours)"""
        key = CacheKeys.rag(question, classroom_id)
        return self._set(key, response, "rag", CacheTTL.RAG)
    
    # =========================================================================
    # Layer 5: Document Metadata
    # =========================================================================
    
    def get_document(self, document_id: str) -> Optional[Dict]:
        """Get cached document metadata"""
        key = CacheKeys.document(document_id)
        return self._get(key, "document")
    
    def set_document(self, document_id: str, metadata: Dict) -> bool:
        """Cache document metadata (1 hour)"""
        key = CacheKeys.document(document_id)
        return self._set(key, metadata, "document", CacheTTL.DOCUMENT_META)
    
    def get_chunk(self, chunk_id: str) -> Optional[Dict]:
        """Get cached chunk metadata"""
        key = CacheKeys.chunk(chunk_id)
        return self._get(key, "document")
    
    def set_chunk(self, chunk_id: str, metadata: Dict) -> bool:
        """Cache chunk metadata (1 hour)"""
        key = CacheKeys.chunk(chunk_id)
        return self._set(key, metadata, "document", CacheTTL.DOCUMENT_META)
    
    # =========================================================================
    # Layer 6: Web Fetch Results
    # =========================================================================
    
    def get_web(self, source_type: str, url: str) -> Optional[Dict]:
        """Get cached web content"""
        key = CacheKeys.web(source_type, url)
        return self._get(key, "web")
    
    def set_web(self, source_type: str, url: str, content: Dict) -> bool:
        """Cache web content (7 days)"""
        key = CacheKeys.web(source_type, url)
        return self._set(key, content, "web", CacheTTL.WEB)
    
    # =========================================================================
    # Invalidation Methods
    # =========================================================================
    
    def invalidate_classroom(self, classroom_id: str) -> int:
        """Invalidate all cache entries for a classroom"""
        count = 0
        count += self.invalidate_pattern(f"ensure:search:*:{classroom_id}:*")
        count += self.invalidate_pattern(f"ensure:rag:*:{classroom_id}")
        logger.info(f"[UnifiedCache] Invalidated {count} keys for classroom {classroom_id}")
        return count
    
    def invalidate_document(self, document_id: str) -> int:
        """Invalidate all cache entries for a document"""
        count = 0
        count += self._delete(CacheKeys.document(document_id))
        logger.info(f"[UnifiedCache] Invalidated document {document_id}")
        return count
    
    def invalidate_embedding_model(self, old_model: str) -> int:
        """Invalidate embeddings for old model version"""
        return self.invalidate_pattern(f"ensure:emb:{old_model}:*")
    
    # =========================================================================
    # Metrics & Stats
    # =========================================================================
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        stats = {"enabled": self.enabled, "backend": "redis" if self.redis_client else "memory"}
        
        for layer, metrics in self._metrics.items():
            stats[layer] = {
                "hits": metrics.hits,
                "misses": metrics.misses,
                "hit_rate": round(metrics.hit_rate, 2),
                "errors": metrics.errors,
                "avg_get_ms": round(metrics.avg_get_time_ms, 2),
                "avg_set_ms": round(metrics.avg_set_time_ms, 2)
            }
        
        if self.redis_client:
            try:
                info = self.redis_client.info("memory")
                stats["redis_memory_used"] = info.get("used_memory_human", "unknown")
            except Exception:
                pass
        
        return stats
    
    def reset_metrics(self):
        """Reset all cache metrics"""
        for layer in self._metrics:
            self._metrics[layer] = CacheMetrics()


# ============================================================================
# Cache Decorator
# ============================================================================

def cache_result(
    key_builder: Callable[..., str],
    ttl: Optional[int] = 3600,
    layer: str = "document"
):
    """
    Decorator for caching function results.
    
    Usage:
        @cache_result(
            key_builder=lambda doc_id: f"ensure:doc:{doc_id}",
            ttl=3600,
            layer="document"
        )
        def get_document_metadata(doc_id: str) -> dict:
            # expensive operation
            return metadata
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            cache = get_unified_cache()
            key = key_builder(*args, **kwargs)
            
            # Try cache
            cached = cache._get(key, layer)
            if cached is not None:
                logger.debug(f"[CacheDecorator] Hit: {key}")
                return cached
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Cache result
            if result is not None:
                cache._set(key, result, layer, ttl)
                logger.debug(f"[CacheDecorator] Set: {key}")
            
            return result
        return wrapper
    return decorator


# ============================================================================
# Singleton
# ============================================================================

_unified_cache: Optional[UnifiedCacheService] = None


def get_unified_cache() -> UnifiedCacheService:
    """Get or create unified cache singleton"""
    global _unified_cache
    if _unified_cache is None:
        _unified_cache = UnifiedCacheService()
    return _unified_cache
