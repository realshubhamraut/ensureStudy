"""
Qdrant Service - Vector Database Layer for Semantic Search

Features:
- Collection management with proper schema
- Batch indexing with payload indices
- Semantic search with filters and reranking
- Redis caching integration
- Confidence scoring algorithm
- Document-level operations

Based on approved Qdrant design specification.
"""
import os
import time
import hashlib
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum

from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, PointStruct, Filter, FieldCondition,
    MatchValue, Range, PayloadSchemaType, CreateAliasOperation,
    OptimizersConfigDiff, HnswConfigDiff, ScalarQuantization,
    ScalarQuantizationConfig, ScalarType, QuantizationSearchParams
)

logger = logging.getLogger(__name__)


# ============================================================================
# Data Classes
# ============================================================================

class SourceType(str, Enum):
    """Source types for trust scoring"""
    STUDENT_UPLOAD = "student_upload"
    TEACHER_MATERIAL = "teacher_material"
    WEB_FETCHED = "web_fetched"
    VIDEO_TRANSCRIPT = "video_transcript"


@dataclass
class ChunkMetadata:
    """Metadata for indexed chunks"""
    document_id: str
    chunk_id: str
    chunk_index: int
    chunk_text: str  # First 200 chars for preview
    source_type: str
    source_confidence: float
    student_id: str
    classroom_id: str
    page_number: int
    section_heading: Optional[str] = None
    title: Optional[str] = None
    subject: Optional[str] = None
    contains_formula: bool = False
    formula_latex: Optional[str] = None
    url: Optional[str] = None
    created_at: Optional[str] = None
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_model_version: str = "v1"
    language: str = "en"


@dataclass
class SearchResult:
    """Search result with scores"""
    point_id: str
    score: float  # Vector similarity
    final_score: float  # After reranking
    payload: Dict[str, Any]


@dataclass
class CollectionInfo:
    """Collection statistics"""
    name: str
    vectors_count: int
    points_count: int
    indexed_vectors_count: int
    status: str


# Trust scores for different source types
SOURCE_TRUST_SCORES = {
    SourceType.TEACHER_MATERIAL.value: 0.95,
    SourceType.STUDENT_UPLOAD.value: 0.75,
    SourceType.WEB_FETCHED.value: 0.65,
    SourceType.VIDEO_TRANSCRIPT.value: 0.60,
}


# ============================================================================
# Qdrant Service
# ============================================================================

class QdrantService:
    """
    Qdrant vector database service for semantic search.
    
    Features:
    - 384-dim vectors with cosine similarity
    - Payload indexing for fast filtering
    - Redis caching for search results
    - Confidence-based reranking
    
    Environment Variables:
    - QDRANT_HOST: Qdrant server host (default: localhost)
    - QDRANT_PORT: Qdrant server port (default: 6333)
    - QDRANT_API_KEY: API key for Qdrant Cloud (optional)
    - QDRANT_COLLECTION: Collection name (default: classroom_documents)
    """
    
    VECTOR_SIZE = 384  # all-MiniLM-L6-v2
    
    def __init__(self, collection_name: str = None):
        self.collection_name = collection_name or os.getenv(
            "QDRANT_COLLECTION", "classroom_documents"
        )
        
        # Initialize Qdrant client
        host = os.getenv("QDRANT_HOST", "localhost")
        port = int(os.getenv("QDRANT_PORT", "6333"))
        api_key = os.getenv("QDRANT_API_KEY")
        
        if api_key:
            # Qdrant Cloud
            self.client = QdrantClient(
                url=f"https://{host}",
                api_key=api_key
            )
        else:
            # Local Qdrant
            self.client = QdrantClient(host=host, port=port)
        
        # Initialize cache
        self.cache = None
        self._init_cache()
        
        # Ensure collection exists
        self._ensure_collection()
        
        logger.info(f"[Qdrant] Initialized with collection: {self.collection_name}")
    
    def _init_cache(self):
        """Initialize Redis cache"""
        try:
            from app.services.response_cache import get_response_cache
            self.cache = get_response_cache()
        except Exception as e:
            logger.warning(f"[Qdrant] Cache not available: {e}")
    
    def _ensure_collection(self):
        """Create collection if it doesn't exist"""
        try:
            collections = self.client.get_collections().collections
            exists = any(c.name == self.collection_name for c in collections)
            
            if not exists:
                self._create_collection()
                self._create_indices()
                
        except Exception as e:
            logger.error(f"[Qdrant] Error checking collection: {e}")
    
    def _create_collection(self):
        """Create collection with optimized settings"""
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.VECTOR_SIZE,
                distance=Distance.COSINE
            ),
            optimizers_config=OptimizersConfigDiff(
                memmap_threshold=20000,
                indexing_threshold=10000
            ),
            hnsw_config=HnswConfigDiff(
                m=16,
                ef_construct=100,
                full_scan_threshold=10000
            ),
            # Optional: INT8 quantization for memory savings
            quantization_config=ScalarQuantization(
                scalar=ScalarQuantizationConfig(
                    type=ScalarType.INT8,
                    quantile=0.99,
                    always_ram=True
                )
            )
        )
        logger.info(f"[Qdrant] Created collection: {self.collection_name}")
    
    def _create_indices(self):
        """Create payload indices for filtering"""
        indices = [
            ("classroom_id", PayloadSchemaType.KEYWORD),
            ("student_id", PayloadSchemaType.KEYWORD),
            ("document_id", PayloadSchemaType.KEYWORD),
            ("source_type", PayloadSchemaType.KEYWORD),
            ("subject", PayloadSchemaType.KEYWORD),
            ("contains_formula", PayloadSchemaType.BOOL),
        ]
        
        for field_name, field_type in indices:
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema=field_type
                )
            except Exception as e:
                logger.warning(f"[Qdrant] Index creation warning for {field_name}: {e}")
        
        logger.info(f"[Qdrant] Created {len(indices)} payload indices")
    
    # ========================================================================
    # Indexing Operations
    # ========================================================================
    
    def index_chunk(
        self,
        embedding: List[float],
        metadata: ChunkMetadata
    ) -> str:
        """
        Index a single chunk with embedding and metadata.
        
        Args:
            embedding: 384-dim vector
            metadata: ChunkMetadata object
            
        Returns:
            Point ID (UUID string)
        """
        import uuid
        
        point_id = str(uuid.uuid4())
        
        # Prepare payload
        payload = asdict(metadata)
        if not payload.get("created_at"):
            payload["created_at"] = datetime.utcnow().isoformat()
        
        # Truncate chunk_text to 200 chars
        if len(payload.get("chunk_text", "")) > 200:
            payload["chunk_text"] = payload["chunk_text"][:200]
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=payload
                )
            ]
        )
        
        logger.debug(f"[Qdrant] Indexed chunk: {point_id}")
        return point_id
    
    def index_batch(
        self,
        chunks: List[Tuple[List[float], ChunkMetadata]]
    ) -> List[str]:
        """
        Batch index multiple chunks.
        
        Args:
            chunks: List of (embedding, metadata) tuples
            
        Returns:
            List of point IDs
        """
        import uuid
        
        points = []
        point_ids = []
        
        for embedding, metadata in chunks:
            point_id = str(uuid.uuid4())
            point_ids.append(point_id)
            
            payload = asdict(metadata)
            if not payload.get("created_at"):
                payload["created_at"] = datetime.utcnow().isoformat()
            
            if len(payload.get("chunk_text", "")) > 200:
                payload["chunk_text"] = payload["chunk_text"][:200]
            
            points.append(PointStruct(
                id=point_id,
                vector=embedding,
                payload=payload
            ))
        
        # Batch upsert (max 100 at a time)
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch
            )
        
        logger.info(f"[Qdrant] Batch indexed {len(points)} chunks")
        return point_ids
    
    # ========================================================================
    # Search Operations
    # ========================================================================
    
    def search_semantic(
        self,
        query_embedding: List[float],
        classroom_id: str,
        top_k: int = 10,
        filters: Optional[Dict] = None,
        score_threshold: float = 0.3,
        use_cache: bool = True
    ) -> List[SearchResult]:
        """
        Semantic search with filters and reranking.
        
        Args:
            query_embedding: 384-dim query vector
            classroom_id: Classroom to search within
            top_k: Number of results to return
            filters: Additional filters (student_id, source_type, etc.)
            score_threshold: Minimum similarity score
            use_cache: Whether to use Redis cache
            
        Returns:
            List of SearchResult objects with reranked scores
        """
        # Check cache
        cache_key = self._generate_cache_key(query_embedding, classroom_id, top_k, filters)
        if use_cache and self.cache:
            cached = self._get_from_cache(cache_key)
            if cached:
                logger.debug("[Qdrant] Cache hit for semantic search")
                return cached
        
        # Build filter
        must_conditions = [
            FieldCondition(
                key="classroom_id",
                match=MatchValue(value=classroom_id)
            )
        ]
        
        if filters:
            if filters.get("student_id"):
                must_conditions.append(FieldCondition(
                    key="student_id",
                    match=MatchValue(value=filters["student_id"])
                ))
            if filters.get("source_type"):
                must_conditions.append(FieldCondition(
                    key="source_type",
                    match=MatchValue(value=filters["source_type"])
                ))
            if filters.get("contains_formula") is not None:
                must_conditions.append(FieldCondition(
                    key="contains_formula",
                    match=MatchValue(value=filters["contains_formula"])
                ))
            if filters.get("subject"):
                must_conditions.append(FieldCondition(
                    key="subject",
                    match=MatchValue(value=filters["subject"])
                ))
        
        query_filter = Filter(must=must_conditions)
        
        # Execute search
        start_time = time.time()
        
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            query_filter=query_filter,
            limit=top_k * 2,  # Get more for reranking
            score_threshold=score_threshold,
            with_payload=True,
            with_vectors=False
        )
        
        search_time = int((time.time() - start_time) * 1000)
        logger.debug(f"[Qdrant] Search completed in {search_time}ms, found {len(results)} results")
        
        # Apply reranking
        reranked = self._rerank_results(results)[:top_k]
        
        # Convert to SearchResult objects
        search_results = [
            SearchResult(
                point_id=str(r["point_id"]),
                score=r["vector_score"],
                final_score=r["final_score"],
                payload=r["payload"]
            )
            for r in reranked
        ]
        
        # Cache results
        if use_cache and self.cache:
            self._set_cache(cache_key, search_results)
        
        return search_results
    
    def search_by_document(self, document_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve all chunks for a document.
        
        Args:
            document_id: Document UUID
            
        Returns:
            List of chunks with payloads
        """
        results = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id)
                    )
                ]
            ),
            limit=1000,
            with_payload=True,
            with_vectors=False
        )
        
        return [
            {
                "point_id": str(point.id),
                "payload": point.payload
            }
            for point in results[0]
        ]
    
    def search_formula(
        self,
        query_embedding: List[float],
        classroom_id: str,
        top_k: int = 8
    ) -> List[SearchResult]:
        """
        Search specifically for formula-containing chunks.
        
        Args:
            query_embedding: Embedding of formula query
            classroom_id: Classroom to search
            top_k: Number of results
            
        Returns:
            Formula chunks with LaTeX
        """
        return self.search_semantic(
            query_embedding=query_embedding,
            classroom_id=classroom_id,
            top_k=top_k,
            filters={"contains_formula": True}
        )
    
    # ========================================================================
    # Management Operations
    # ========================================================================
    
    def delete_document(self, document_id: str) -> int:
        """
        Delete all chunks for a document.
        
        Args:
            document_id: Document UUID
            
        Returns:
            Number of deleted points
        """
        # Get all points for document
        points = self.search_by_document(document_id)
        point_ids = [p["point_id"] for p in points]
        
        if point_ids:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=point_ids
            )
        
        # Invalidate cache for this classroom
        if points and self.cache:
            classroom_id = points[0]["payload"].get("classroom_id")
            if classroom_id:
                self._invalidate_classroom_cache(classroom_id)
        
        logger.info(f"[Qdrant] Deleted {len(point_ids)} chunks for document {document_id}")
        return len(point_ids)
    
    def update_metadata(
        self,
        point_id: str,
        metadata: Dict[str, Any]
    ) -> bool:
        """
        Update chunk metadata.
        
        Args:
            point_id: Point ID to update
            metadata: Fields to update
            
        Returns:
            True if successful
        """
        try:
            self.client.set_payload(
                collection_name=self.collection_name,
                payload=metadata,
                points=[point_id]
            )
            return True
        except Exception as e:
            logger.error(f"[Qdrant] Failed to update metadata: {e}")
            return False
    
    def get_collection_info(self) -> CollectionInfo:
        """Get collection statistics."""
        info = self.client.get_collection(self.collection_name)
        
        return CollectionInfo(
            name=self.collection_name,
            vectors_count=info.vectors_count,
            points_count=info.points_count,
            indexed_vectors_count=info.indexed_vectors_count or 0,
            status=info.status.value
        )
    
    def health_check(self) -> bool:
        """Check Qdrant connectivity."""
        try:
            self.client.get_collections()
            return True
        except Exception as e:
            logger.error(f"[Qdrant] Health check failed: {e}")
            return False
    
    # ========================================================================
    # Reranking & Scoring
    # ========================================================================
    
    def _rerank_results(self, results: List) -> List[Dict]:
        """
        Apply confidence-based reranking.
        
        Score formula:
        final_score = 0.5×similarity + 0.3×source_confidence + 0.2×trust_score
        """
        reranked = []
        
        for r in results:
            payload = r.payload
            vector_score = r.score
            
            # Get source confidence
            source_confidence = payload.get("source_confidence", 0.7)
            
            # Get trust score based on source type
            source_type = payload.get("source_type", "student_upload")
            trust_score = SOURCE_TRUST_SCORES.get(source_type, 0.7)
            
            # Calculate final score
            final_score = (
                0.50 * vector_score +
                0.30 * source_confidence +
                0.20 * trust_score
            )
            
            # Apply recency boost
            created_at = payload.get("created_at")
            if created_at:
                recency_boost = self._calculate_recency_boost(created_at)
                final_score *= recency_boost
            
            reranked.append({
                "point_id": r.id,
                "vector_score": vector_score,
                "final_score": final_score,
                "payload": payload
            })
        
        # Sort by final score
        reranked.sort(key=lambda x: x["final_score"], reverse=True)
        
        return reranked
    
    def _calculate_recency_boost(self, created_at: str) -> float:
        """Calculate recency boost multiplier."""
        try:
            created = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            age_days = (datetime.utcnow() - created.replace(tzinfo=None)).days
            
            if age_days < 7:
                return 1.20
            elif age_days < 30:
                return 1.10
            elif age_days < 90:
                return 1.00
            else:
                return 0.95
                
        except Exception:
            return 1.0
    
    # ========================================================================
    # Caching
    # ========================================================================
    
    def _generate_cache_key(
        self,
        embedding: List[float],
        classroom_id: str,
        top_k: int,
        filters: Optional[Dict]
    ) -> str:
        """Generate cache key from search parameters."""
        # Hash the embedding (first and last 10 values for speed)
        embed_sample = embedding[:10] + embedding[-10:]
        embed_hash = hashlib.md5(str(embed_sample).encode()).hexdigest()[:8]
        
        filter_hash = ""
        if filters:
            filter_hash = hashlib.md5(json.dumps(filters, sort_keys=True).encode()).hexdigest()[:6]
        
        return f"qdrant:search:{embed_hash}:cls:{classroom_id}:k:{top_k}:f:{filter_hash}"
    
    def _get_from_cache(self, cache_key: str) -> Optional[List[SearchResult]]:
        """Get search results from cache."""
        if not self.cache:
            return None
        
        try:
            if hasattr(self.cache, 'redis_client') and self.cache.redis_client:
                data = self.cache.redis_client.get(cache_key)
                if data:
                    parsed = json.loads(data)
                    return [SearchResult(**r) for r in parsed]
        except Exception as e:
            logger.warning(f"[Qdrant] Cache read error: {e}")
        
        return None
    
    def _set_cache(self, cache_key: str, results: List[SearchResult]):
        """Cache search results."""
        if not self.cache:
            return
        
        try:
            data = [asdict(r) for r in results]
            if hasattr(self.cache, 'redis_client') and self.cache.redis_client:
                self.cache.redis_client.setex(
                    cache_key,
                    86400,  # 24 hours
                    json.dumps(data)
                )
        except Exception as e:
            logger.warning(f"[Qdrant] Cache write error: {e}")
    
    def _invalidate_classroom_cache(self, classroom_id: str):
        """Invalidate all cache entries for a classroom."""
        if not self.cache:
            return
        
        try:
            if hasattr(self.cache, 'redis_client') and self.cache.redis_client:
                pattern = f"qdrant:search:*:cls:{classroom_id}:*"
                keys = list(self.cache.redis_client.scan_iter(match=pattern))
                if keys:
                    self.cache.redis_client.delete(*keys)
                    logger.debug(f"[Qdrant] Invalidated {len(keys)} cache entries")
        except Exception as e:
            logger.warning(f"[Qdrant] Cache invalidation error: {e}")


# ============================================================================
# Singleton & Convenience Functions
# ============================================================================

_qdrant_service: Optional[QdrantService] = None


def get_qdrant_service() -> QdrantService:
    """Get or create Qdrant service singleton."""
    global _qdrant_service
    if _qdrant_service is None:
        _qdrant_service = QdrantService()
    return _qdrant_service


async def semantic_search(
    query_embedding: List[float],
    classroom_id: str,
    top_k: int = 10,
    filters: Optional[Dict] = None
) -> List[SearchResult]:
    """
    Convenience function for semantic search.
    
    Args:
        query_embedding: Query vector
        classroom_id: Classroom ID
        top_k: Number of results
        filters: Additional filters
        
    Returns:
        List of SearchResult objects
    """
    service = get_qdrant_service()
    return service.search_semantic(
        query_embedding=query_embedding,
        classroom_id=classroom_id,
        top_k=top_k,
        filters=filters
    )
