"""
Session Service - Core logic for session-based resource chaining

Implements:
- Session creation and management
- Turn storage with embedding
- Relatedness detection using cosine similarity
- Resource deduplication by URL/hash/vector similarity
- LRU eviction for bounded resource list
"""
import os
import uuid
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple, Any
from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class SessionData:
    """Session information"""
    session_id: str
    user_id: str
    classroom_id: Optional[str]
    created_at: str
    last_active_at: str
    turn_count: int
    resource_count: int
    config: dict


@dataclass
class TurnData:
    """Turn information"""
    turn_number: int
    question: str
    related: bool
    relatedness_score: Optional[float]
    timestamp: str


@dataclass
class ResourceData:
    """Resource information"""
    resource_id: str
    resource_type: str
    source: str
    url: Optional[str]
    title: str
    preview_summary: Optional[str]
    inline_render: bool
    inserted_at: str
    last_referenced_at: str
    content_hash: Optional[str]


@dataclass
class AppendResult:
    """Result of append_resource operation"""
    inserted: bool
    resource_id: str
    reason: str  # "new", "duplicate_url", "duplicate_hash", "duplicate_vector"


# ============================================================================
# Configuration
# ============================================================================

DEFAULT_CONFIG = {
    "ttl_hours": int(os.getenv("SESSION_TTL_HOURS", "24")),
    "max_resources": int(os.getenv("SESSION_MAX_RESOURCES", "25")),
    "relatedness_threshold": float(os.getenv("SESSION_RELATEDNESS_THRESHOLD", "0.65")),
    "relatedness_lookback": int(os.getenv("SESSION_RELATEDNESS_LOOKBACK", "3")),
    "dedup_vector_threshold": float(os.getenv("DEDUP_VECTOR_SIMILARITY_THRESHOLD", "0.95"))
}


# ============================================================================
# Session Service
# ============================================================================

class SessionService:
    """
    Manages tutoring sessions for resource chaining.
    
    Features:
    - Session lifecycle management
    - Turn tracking with embeddings
    - Relatedness detection via cosine similarity
    - Resource deduplication by URL, hash, or vector similarity
    - LRU eviction for bounded resource lists
    - PostgreSQL persistence (optional)
    - Redis caching (optional)
    - Telemetry logging
    """
    
    def __init__(
        self, 
        embedding_model: SentenceTransformer = None,
        persist_to_db: bool = None,
        use_cache: bool = None
    ):
        """
        Initialize session service.
        
        Args:
            embedding_model: Sentence transformer for embeddings (lazy loaded if None)
            persist_to_db: Whether to persist to PostgreSQL (default: from env)
            use_cache: Whether to use Redis cache (default: from env)
        """
        self._embedding_model = embedding_model
        self._sessions: Dict[str, dict] = {}  # In-memory cache
        
        # Persistence settings
        self._persist_to_db = persist_to_db if persist_to_db is not None else \
            os.getenv("SESSION_PERSIST_TO_DB", "false").lower() == "true"
        self._use_cache = use_cache if use_cache is not None else \
            os.getenv("SESSION_USE_CACHE", "true").lower() == "true"
        
        # Lazy-loaded dependencies
        self._repository = None
        self._cache = None
        self._telemetry = None
        
        logger.info(f"[SESSION] Initialized persist_to_db={self._persist_to_db} use_cache={self._use_cache}")
        
    @property
    def embedding_model(self) -> SentenceTransformer:
        """Lazy load embedding model"""
        if self._embedding_model is None:
            model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
            logger.info(f"[SESSION] Loading embedding model: {model_name}")
            self._embedding_model = SentenceTransformer(model_name)
        return self._embedding_model
    
    @property
    def repository(self):
        """Lazy load repository"""
        if self._repository is None and self._persist_to_db:
            from .session_repository import get_session_repository
            self._repository = get_session_repository()
        return self._repository
    
    @property
    def cache(self):
        """Lazy load cache"""
        if self._cache is None and self._use_cache:
            from .session_cache import get_session_cache
            self._cache = get_session_cache()
        return self._cache
    
    @property
    def telemetry(self):
        """Lazy load telemetry"""
        if self._telemetry is None:
            from .session_telemetry import get_session_telemetry
            self._telemetry = get_session_telemetry()
        return self._telemetry
    
    # ========================================================================
    # Session Management
    # ========================================================================
    
    def create_session(
        self, 
        user_id: str, 
        classroom_id: Optional[str] = None,
        config: Optional[dict] = None
    ) -> SessionData:
        """
        Create a new tutoring session.
        
        Args:
            user_id: User creating the session
            classroom_id: Optional classroom context
            config: Optional custom configuration
            
        Returns:
            SessionData with session info
        """
        session_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()
        
        session_config = {**DEFAULT_CONFIG, **(config or {})}
        
        session = {
            "session_id": session_id,
            "user_id": user_id,
            "classroom_id": classroom_id,
            "created_at": now,
            "last_active_at": now,
            "config": session_config,
            "turns": [],
            "resources": [],
            "turn_embeddings": [],
            "is_active": True,
            # Session Intelligence fields
            "last_topic_vector": None,
            "last_decision": "new_topic",
            "consecutive_borderline": 0,
            "topic_segments": []  # Track topic boundaries
        }
        
        # Store in memory
        self._sessions[session_id] = session
        
        # Persist to cache
        if self.cache and self.cache.is_available:
            self.cache.set(session_id, session)
        
        # Persist to database
        if self.repository:
            self.repository.save_session(session)
        
        # Log telemetry
        self.telemetry.log_session_created(session_id, user_id, classroom_id)
        
        logger.info(f"[SESSION] Created session_id={session_id} user={user_id} classroom={classroom_id}")
        
        return SessionData(
            session_id=session_id,
            user_id=user_id,
            classroom_id=classroom_id,
            created_at=now,
            last_active_at=now,
            turn_count=0,
            resource_count=0,
            config=session_config
        )
    
    def get_session(self, session_id: str) -> Optional[SessionData]:
        """
        Get session by ID.
        
        Lookup order: memory -> cache -> database
        
        Args:
            session_id: Session UUID
            
        Returns:
            SessionData or None if not found/expired
        """
        session = self._sessions.get(session_id)
        
        # Try cache if not in memory
        if not session and self.cache and self.cache.is_available:
            cached = self.cache.get(session_id)
            if cached:
                self.telemetry.log_cache_hit(session_id)
                self._sessions[session_id] = cached
                session = cached
            else:
                self.telemetry.log_cache_miss(session_id)
        
        # Try database if not in cache
        if not session and self.repository:
            db_session = self.repository.load_full_session(session_id)
            if db_session:
                self.telemetry.log_db_fallback(session_id)
                self.telemetry.log_session_loaded(session_id, "database")
                self._sessions[session_id] = db_session
                # Warm cache
                if self.cache and self.cache.is_available:
                    self.cache.set(session_id, db_session)
                session = db_session
        
        if not session:
            logger.warning(f"[SESSION] Session not found: {session_id}")
            return None
        
        # Check expiration
        ttl_hours = session["config"].get("ttl_hours", 24)
        last_active = datetime.fromisoformat(session["last_active_at"])
        if datetime.utcnow() > last_active + timedelta(hours=ttl_hours):
            duration = (datetime.utcnow() - datetime.fromisoformat(session["created_at"])).total_seconds() / 3600
            self.telemetry.log_session_expired(session_id, duration)
            logger.info(f"[SESSION] Session expired: {session_id}")
            del self._sessions[session_id]
            if self.cache:
                self.cache.delete(session_id)
            return None
        
        return SessionData(
            session_id=session["session_id"],
            user_id=session["user_id"],
            classroom_id=session["classroom_id"],
            created_at=session["created_at"],
            last_active_at=session["last_active_at"],
            turn_count=len(session["turns"]),
            resource_count=len(session["resources"]),
            config=session["config"]
        )
    
    def touch_session(self, session_id: str) -> bool:
        """Update last_active_at timestamp"""
        if session_id in self._sessions:
            self._sessions[session_id]["last_active_at"] = datetime.utcnow().isoformat()
            return True
        return False
    
    # ========================================================================
    # Turn Management
    # ========================================================================
    
    def add_turn(
        self, 
        session_id: str, 
        question: str,
        embedding: Optional[List[float]] = None
    ) -> Optional[TurnData]:
        """
        Add a new query turn to session.
        
        Args:
            session_id: Session UUID
            question: User's question
            embedding: Pre-computed embedding (computed if None)
            
        Returns:
            TurnData with relatedness info
        """
        session = self._sessions.get(session_id)
        if not session:
            logger.error(f"[SESSION] Cannot add turn - session not found: {session_id}")
            return None
        
        # Compute embedding if not provided
        if embedding is None:
            embedding = self.embedding_model.encode(question).tolist()
        
        # Compute relatedness to previous turns
        related, relatedness_score = self.compute_relatedness(
            current_embedding=embedding,
            session_id=session_id
        )
        
        # Create turn
        turn_number = len(session["turns"]) + 1
        now = datetime.utcnow().isoformat()
        
        turn = {
            "turn_number": turn_number,
            "question": question,
            "question_hash": hashlib.sha256(question.encode()).hexdigest(),
            "related": related,
            "relatedness_score": relatedness_score,
            "timestamp": now
        }
        
        session["turns"].append(turn)
        session["turn_embeddings"].append(embedding)
        session["last_active_at"] = now
        
        # Persist turn to database
        if self.repository:
            turn_with_embedding = {**turn, "embedding": embedding}
            self.repository.save_turn(session_id, turn_with_embedding)
        
        # Update cache
        if self.cache and self.cache.is_available:
            self.cache.set(session_id, session)
        
        # Log telemetry
        self.telemetry.log_turn_added(session_id, turn_number, related, relatedness_score)
        
        logger.info(
            f"[SESSION] session_id={session_id} turn={turn_number} "
            f"query=\"{question[:50]}...\" related={related} sim={relatedness_score if relatedness_score is not None else 0:.2f}"
        )
        
        return TurnData(
            turn_number=turn_number,
            question=question,
            related=related,
            relatedness_score=relatedness_score,
            timestamp=now
        )
    
    def compute_relatedness(
        self, 
        current_embedding: List[float], 
        session_id: str
    ) -> Tuple[bool, Optional[float]]:
        """
        Compute if current query is related to previous turns.
        
        Args:
            current_embedding: Embedding of current question
            session_id: Session to compare against
            
        Returns:
            (related: bool, max_similarity: float)
        """
        session = self._sessions.get(session_id)
        if not session or not session["turn_embeddings"]:
            return False, None
        
        config = session["config"]
        threshold = config.get("relatedness_threshold", 0.65)
        lookback = config.get("relatedness_lookback", 3)
        
        # Get last N turn embeddings
        prev_embeddings = session["turn_embeddings"][-lookback:]
        if not prev_embeddings:
            return False, None
        
        # Compute cosine similarities
        current_vec = np.array(current_embedding)
        max_similarity = 0.0
        
        for prev_emb in prev_embeddings:
            prev_vec = np.array(prev_emb)
            similarity = np.dot(current_vec, prev_vec) / (
                np.linalg.norm(current_vec) * np.linalg.norm(prev_vec)
            )
            max_similarity = max(max_similarity, similarity)
        
        related = max_similarity >= threshold
        return related, float(max_similarity)
    
    # ========================================================================
    # Resource Management
    # ========================================================================
    
    def append_resource(
        self,
        session_id: str,
        resource_type: str,
        source: str,
        url: Optional[str] = None,
        title: str = "",
        preview_summary: Optional[str] = None,
        content_hash: Optional[str] = None,
        inline_render: bool = False,
        inline_html: Optional[str] = None,
        qdrant_collection: Optional[str] = None,
        content_embedding: Optional[List[float]] = None
    ) -> AppendResult:
        """
        Append a resource to session with deduplication.
        
        Dedup checks (in order):
        1. Canonical URL match
        2. Content hash match
        3. Vector similarity > 0.95
        
        Args:
            session_id: Session UUID
            resource_type: text, pdf, image, video, flowchart, wikipedia
            source: classroom, wikipedia, web, youtube
            url: Resource URL
            title: Resource title
            preview_summary: 180-300 char preview
            content_hash: SHA256 of content
            inline_render: Whether to render inline (Wikipedia)
            inline_html: Sanitized HTML for inline display
            qdrant_collection: Collection where chunks stored
            content_embedding: For vector dedup
            
        Returns:
            AppendResult with inserted flag and reason
        """
        session = self._sessions.get(session_id)
        if not session:
            logger.error(f"[SESSION] Cannot append resource - session not found: {session_id}")
            return AppendResult(inserted=False, resource_id="", reason="session_not_found")
        
        # Normalize URL for dedup
        canonical_url = self._normalize_url(url) if url else None
        
        # Check for duplicates
        for existing in session["resources"]:
            # Check canonical URL
            if canonical_url and existing.get("canonical_url") == canonical_url:
                existing["last_referenced_at"] = datetime.utcnow().isoformat()
                logger.info(f"[APPEND] url={url} inserted=false reason=duplicate_url hash={content_hash}")
                return AppendResult(
                    inserted=False, 
                    resource_id=existing["resource_id"], 
                    reason="duplicate_url"
                )
            
            # Check content hash
            if content_hash and existing.get("content_hash") == content_hash:
                existing["last_referenced_at"] = datetime.utcnow().isoformat()
                logger.info(f"[APPEND] url={url} inserted=false reason=duplicate_hash hash={content_hash}")
                return AppendResult(
                    inserted=False, 
                    resource_id=existing["resource_id"], 
                    reason="duplicate_hash"
                )
        
        # Check vector similarity if embedding provided
        if content_embedding:
            dedup_threshold = session["config"].get("dedup_vector_threshold", 0.95)
            for existing in session["resources"]:
                if existing.get("embedding"):
                    similarity = self._cosine_similarity(content_embedding, existing["embedding"])
                    if similarity >= dedup_threshold:
                        existing["last_referenced_at"] = datetime.utcnow().isoformat()
                        logger.info(f"[APPEND] url={url} inserted=false reason=duplicate_vector sim={similarity:.3f}")
                        return AppendResult(
                            inserted=False, 
                            resource_id=existing["resource_id"], 
                            reason="duplicate_vector"
                        )
        
        # Evict oldest if at capacity
        max_resources = session["config"].get("max_resources", 25)
        if len(session["resources"]) >= max_resources:
            self._evict_oldest_resource(session)
        
        # Insert new resource
        resource_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()
        
        resource = {
            "resource_id": resource_id,
            "resource_type": resource_type,
            "source": source,
            "url": url,
            "canonical_url": canonical_url,
            "title": title,
            "preview_summary": preview_summary[:300] if preview_summary else None,
            "content_hash": content_hash,
            "inline_render": inline_render,
            "inline_html": inline_html,
            "qdrant_collection": qdrant_collection,
            "embedding": content_embedding,
            "inserted_at": now,
            "last_referenced_at": now
        }
        
        session["resources"].append(resource)
        session["last_active_at"] = now
        
        logger.info(f"[APPEND] url={url} inserted=true reason=new hash={content_hash}")
        
        return AppendResult(inserted=True, resource_id=resource_id, reason="new")
    
    def get_resource_list(self, session_id: str) -> List[ResourceData]:
        """
        Get all resources in session.
        
        Args:
            session_id: Session UUID
            
        Returns:
            List of ResourceData, sorted by last_referenced_at desc
        """
        session = self._sessions.get(session_id)
        if not session:
            return []
        
        resources = sorted(
            session["resources"],
            key=lambda r: r["last_referenced_at"],
            reverse=True
        )
        
        return [
            ResourceData(
                resource_id=r["resource_id"],
                resource_type=r["resource_type"],
                source=r["source"],
                url=r["url"],
                title=r["title"],
                preview_summary=r["preview_summary"],
                inline_render=r["inline_render"],
                inserted_at=r["inserted_at"],
                last_referenced_at=r["last_referenced_at"],
                content_hash=r["content_hash"]
            )
            for r in resources
        ]
    
    def get_resource_ids(self, session_id: str) -> List[str]:
        """Get list of resource IDs for Qdrant filtering"""
        session = self._sessions.get(session_id)
        if not session:
            return []
        return [r["resource_id"] for r in session["resources"]]
    
    # ========================================================================
    # Helpers
    # ========================================================================
    
    def _normalize_url(self, url: str) -> str:
        """Normalize URL for deduplication"""
        if not url:
            return ""
        
        # Remove protocol
        url = url.replace("https://", "").replace("http://", "")
        
        # Remove www
        url = url.replace("www.", "")
        
        # Remove trailing slash
        url = url.rstrip("/")
        
        # Remove common tracking params
        if "?" in url:
            base, params = url.split("?", 1)
            # Keep only essential params
            url = base
        
        return url.lower()
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Compute cosine similarity between two vectors"""
        a_vec = np.array(a)
        b_vec = np.array(b)
        return float(np.dot(a_vec, b_vec) / (np.linalg.norm(a_vec) * np.linalg.norm(b_vec)))
    
    def _evict_oldest_resource(self, session: dict) -> None:
        """Remove oldest resource by last_referenced_at"""
        if not session["resources"]:
            return
        
        # Find oldest
        oldest_idx = 0
        oldest_time = session["resources"][0]["last_referenced_at"]
        
        for i, r in enumerate(session["resources"]):
            if r["last_referenced_at"] < oldest_time:
                oldest_time = r["last_referenced_at"]
                oldest_idx = i
        
        evicted = session["resources"].pop(oldest_idx)
        logger.info(f"[SESSION] Evicted resource: {evicted['title']} (LRU)")
    
    def export_session_json(self, session_id: str) -> Optional[dict]:
        """
        Export session as JSON for debugging/audit.
        
        Returns full session state including query chain and resources.
        """
        session = self._sessions.get(session_id)
        if not session:
            return None
        
        return {
            "session_id": session["session_id"],
            "user_id": session["user_id"],
            "classroom_id": session["classroom_id"],
            "created_at": session["created_at"],
            "last_active_at": session["last_active_at"],
            "config": session["config"],
            "query_chain": [
                {
                    "turn": t["turn_number"],
                    "question": t["question"],
                    "related": t["related"],
                    "timestamp": t["timestamp"]
                }
                for t in session["turns"]
            ],
            "resource_list": [
                {
                    "resource_id": r["resource_id"],
                    "source": r["source"],
                    "url": r["url"],
                    "title": r["title"],
                    "inserted_at": r["inserted_at"],
                    "hash": r["content_hash"],
                    "qdrant_collection": r["qdrant_collection"]
                }
                for r in session["resources"]
            ]
        }


# ============================================================================
# Singleton
# ============================================================================

_session_service: Optional[SessionService] = None


def get_session_service() -> SessionService:
    """Get singleton SessionService instance"""
    global _session_service
    if _session_service is None:
        _session_service = SessionService()
    return _session_service
