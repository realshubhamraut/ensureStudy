"""
Web Content Cache Service

Uses Qdrant to cache web-crawled content for fast retrieval.
Cache-first logic: check cache before crawling, only crawl on cache miss.

FEATURES:
- Persistent file-based Qdrant (no Docker needed)
- Semantic similarity search
- Configurable confidence threshold (default 0.85)
- Auto-expiry support via timestamps
"""
import os
import uuid
import hashlib
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

# Cache settings
CACHE_COLLECTION = "web_content_cache"
CACHE_THRESHOLD = 0.85  # Minimum similarity to use cached result
EMBEDDING_DIM = 384  # MiniLM dimension


@dataclass
class CacheHit:
    """Result from cache lookup."""
    query: str
    answer: str
    sources: List[str]
    confidence: float
    cached_at: str
    similarity: float


# Global clients
_cache_client: Optional[QdrantClient] = None
_embedding_model: Optional[SentenceTransformer] = None


def get_cache_client() -> QdrantClient:
    """Get or create persistent Qdrant client."""
    global _cache_client
    
    if _cache_client is None:
        # Use local file-based storage (persists across restarts)
        cache_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "data",
            "qdrant_cache"
        )
        os.makedirs(cache_path, exist_ok=True)
        
        print(f"[CACHE] Initializing Qdrant at: {cache_path}")
        _cache_client = QdrantClient(path=cache_path)
        
        # Ensure collection exists
        try:
            collections = _cache_client.get_collections().collections
            names = [c.name for c in collections]
            
            if CACHE_COLLECTION not in names:
                print(f"[CACHE] Creating collection: {CACHE_COLLECTION}")
                _cache_client.create_collection(
                    collection_name=CACHE_COLLECTION,
                    vectors_config=VectorParams(
                        size=EMBEDDING_DIM, 
                        distance=Distance.COSINE
                    )
                )
        except Exception as e:
            print(f"[CACHE] Collection check error: {e}")
    
    return _cache_client


def get_embedding_model() -> SentenceTransformer:
    """Get or load embedding model."""
    global _embedding_model
    
    if _embedding_model is None:
        print("[CACHE] Loading embedding model...")
        _embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    return _embedding_model


def embed_query(query: str) -> List[float]:
    """Generate embedding for a query."""
    model = get_embedding_model()
    embedding = model.encode(query, normalize_embeddings=True)
    return embedding.tolist()


def search_cache(query: str, threshold: float = CACHE_THRESHOLD) -> Optional[CacheHit]:
    """
    Search cache for similar queries.
    
    Returns CacheHit if similarity >= threshold, else None.
    """
    print(f"[CACHE] Searching for: '{query[:50]}...'")
    
    try:
        client = get_cache_client()
        query_embedding = embed_query(query)
        
        results = client.query_points(
            collection_name=CACHE_COLLECTION,
            query=query_embedding,
            limit=1
        )
        
        if results.points:
            best = results.points[0]
            similarity = best.score
            
            print(f"[CACHE] Best match similarity: {similarity:.3f} (threshold: {threshold})")
            
            if similarity >= threshold:
                print(f"[CACHE] ✅ CACHE HIT!")
                return CacheHit(
                    query=best.payload.get('query', ''),
                    answer=best.payload.get('answer', ''),
                    sources=best.payload.get('sources', []),
                    confidence=best.payload.get('confidence', 0.0),
                    cached_at=best.payload.get('cached_at', ''),
                    similarity=similarity
                )
            else:
                print(f"[CACHE] ❌ Below threshold, will crawl fresh")
        else:
            print(f"[CACHE] ❌ No cached entries found")
            
    except Exception as e:
        print(f"[CACHE] Search error: {e}")
    
    return None


def store_in_cache(
    query: str,
    answer: str,
    sources: List[str],
    confidence: float = 0.9
) -> bool:
    """
    Store a query-answer pair in cache.
    
    Returns True if successful.
    """
    print(f"[CACHE] Storing: '{query[:50]}...'")
    
    try:
        client = get_cache_client()
        query_embedding = embed_query(query)
        
        # Generate UUID from query hash (deterministic for same queries)
        query_hash = hashlib.md5(query.lower().strip().encode()).hexdigest()
        # Convert hash to valid UUID format
        point_id = str(uuid.UUID(query_hash))
        
        client.upsert(
            collection_name=CACHE_COLLECTION,
            points=[
                PointStruct(
                    id=point_id,
                    vector=query_embedding,
                    payload={
                        "query": query,
                        "answer": answer,
                        "sources": sources,
                        "confidence": confidence,
                        "cached_at": datetime.now().isoformat()
                    }
                )
            ]
        )
        
        print(f"[CACHE] ✅ Cached successfully (id: {point_id})")
        return True
        
    except Exception as e:
        print(f"[CACHE] Store error: {e}")
        return False


def get_cache_stats() -> Dict[str, Any]:
    """Get cache statistics."""
    try:
        client = get_cache_client()
        info = client.get_collection(CACHE_COLLECTION)
        
        return {
            "collection": CACHE_COLLECTION,
            "points_count": info.points_count,
            "vectors_count": info.vectors_count,
            "status": "healthy"
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


def clear_cache() -> bool:
    """Clear all cached entries."""
    try:
        client = get_cache_client()
        client.delete_collection(CACHE_COLLECTION)
        client.create_collection(
            collection_name=CACHE_COLLECTION,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE)
        )
        print("[CACHE] ✅ Cache cleared")
        return True
    except Exception as e:
        print(f"[CACHE] Clear error: {e}")
        return False


# ============================================================================
# Cache-First Web Ingest
# ============================================================================

async def ingest_with_cache(
    query: str,
    threshold: float = CACHE_THRESHOLD,
    force_refresh: bool = False
) -> Tuple[Dict[str, Any], bool]:
    """
    Cache-first web ingestion.
    
    1. Check cache for similar query
    2. If cache hit with high confidence -> return cached
    3. If cache miss -> crawl web, cache result, return
    
    Returns: (result_dict, was_cached)
    """
    # Step 1: Check cache (unless force refresh)
    if not force_refresh:
        cache_hit = search_cache(query, threshold)
        
        if cache_hit:
            return {
                "success": True,
                "answer": cache_hit.answer,
                "sources": cache_hit.sources,
                "confidence": cache_hit.confidence,
                "cached": True,
                "similarity": cache_hit.similarity,
                "cached_at": cache_hit.cached_at
            }, True
    
    # Step 2: Cache miss - crawl web
    print(f"[CACHE] Cache miss, crawling web...")
    
    from .web_ingest_service import ingest_web_resources
    
    web_result = await ingest_web_resources(
        query=query,
        max_sources=2
    )
    
    if web_result.success and web_result.resources:
        # Build answer from crawled content
        combined_content = "\n\n".join([
            r.clean_content[:2000] for r in web_result.resources 
            if r.clean_content
        ])
        
        sources = [r.url for r in web_result.resources]
        
        # Step 3: Store in cache
        store_in_cache(
            query=query,
            answer=combined_content,
            sources=sources,
            confidence=0.9
        )
        
        return {
            "success": True,
            "answer": combined_content,
            "sources": sources,
            "confidence": 0.9,
            "cached": False,
            "resources": web_result.resources
        }, False
    
    return {
        "success": False,
        "error": "No content found"
    }, False
