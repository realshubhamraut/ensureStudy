"""
Topic Anchor Layer (TAL) Service

Creates and manages topic anchors that:
- Anchor conversations to a specific topic
- Enforce retrieval filtering within topic scope
- Provide LLM system prompt constraints
- Track topic history for audit

Integrates with ABCR for relatedness detection.
"""
import os
import uuid
import hashlib
import logging
import time
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

FEATURE_TOPIC_ANCHOR = os.getenv("FEATURE_TOPIC_ANCHOR", "true").lower() == "true"
TOPIC_ANCHOR_TTL_HOURS = int(os.getenv("TOPIC_ANCHOR_TTL_HOURS", "24"))
ANCHOR_BOOST = float(os.getenv("ANCHOR_BOOST", "1.15"))
ANCHOR_MIN_HITS = int(os.getenv("ANCHOR_MIN_HITS", "1"))
ANCHOR_PRIORITY_MIN_SIM = float(os.getenv("ANCHOR_PRIORITY_MIN_SIM", "0.40"))
MAX_TOPIC_ENTITIES = int(os.getenv("MAX_TOPIC_ENTITIES", "30"))


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class TopicAnchor:
    """
    A topic anchor that constrains conversation scope.
    
    Created when a new topic is started, persisted until:
    - User resets topic
    - ABCR detects new unrelated topic (without override)
    - TTL expires
    """
    id: str
    canonical_title: str
    topic_embedding: Optional[List[float]] = None
    subject_scope: List[str] = field(default_factory=list)
    locked_entities: List[str] = field(default_factory=list)
    source: str = "user_query"  # user_query | document | system
    created_by_session: Optional[str] = None
    created_by_turn: int = 0
    created_at: str = ""
    expires_at: str = ""
    
    def __post_init__(self):
        if not self.id:
            self.id = f"anchor_{uuid.uuid4().hex[:12]}"
        if not self.created_at:
            self.created_at = datetime.utcnow().isoformat()
        if not self.expires_at:
            expires = datetime.utcnow() + timedelta(hours=TOPIC_ANCHOR_TTL_HOURS)
            self.expires_at = expires.isoformat()
    
    def is_expired(self) -> bool:
        """Check if anchor has expired."""
        try:
            expires = datetime.fromisoformat(self.expires_at)
            return datetime.utcnow() > expires
        except:
            return False
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "canonical_title": self.canonical_title,
            "subject_scope": self.subject_scope,
            "locked_entities": self.locked_entities[:10],  # Limit for response
            "source": self.source,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
        }
    
    def to_prompt_fragment(self) -> str:
        """
        Generate LLM system prompt fragment for anchor enforcement.
        
        This constrains the LLM to stay within topic scope.
        """
        scope_str = ", ".join(self.subject_scope[:5]) if self.subject_scope else "general"
        entities_str = ", ".join(self.locked_entities[:10]) if self.locked_entities else "none"
        
        return f"""Active Topic Anchor:
- topic_id: {hashlib.sha256(self.id.encode()).hexdigest()[:16]}
- canonical_name: "{self.canonical_title}"
- scope: [{scope_str}]
- key_entities: [{entities_str}]

Instruction: You must only use the provided CONTEXT and stay within the anchor scope to answer. Do not generalize outside this topic. If the context is insufficient, respond: "I don't have enough information about {self.canonical_title} to answer this specific question. Could you provide more context or ask a related question?"""


@dataclass
class TopicHistoryEntry:
    """Entry in topic history for audit."""
    anchor_id: str
    canonical_title: str
    started_at: str
    ended_at: str
    end_reason: str  # user_reset | new_topic | ttl_expired
    turns_count: int


# ============================================================================
# Topic Anchor Service
# ============================================================================

class TopicAnchorService:
    """
    Service for managing topic anchors.
    
    Features:
    - Create anchors from user queries or documents
    - Get active anchor for session
    - Clear/reset anchors
    - Track topic history
    - Compute topic embeddings
    """
    
    def __init__(self, embedding_model=None):
        self._embedding_model = embedding_model
        self._session_anchors: Dict[str, TopicAnchor] = {}
        self._session_history: Dict[str, List[TopicHistoryEntry]] = {}
        
        logger.info(
            f"[TAL] Service initialized: enabled={FEATURE_TOPIC_ANCHOR} "
            f"ttl={TOPIC_ANCHOR_TTL_HOURS}h boost={ANCHOR_BOOST}"
        )
    
    @property
    def embedding_model(self):
        """Lazy-load embedding model."""
        if self._embedding_model is None:
            from sentence_transformers import SentenceTransformer
            self._embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        return self._embedding_model
    
    def create_anchor(
        self,
        session_id: str,
        request_id: str,
        canonical_title: str,
        context_chunks: List[str] = None,
        source: str = "user_query",
        turn_index: int = 0
    ) -> TopicAnchor:
        """
        Create a new topic anchor for a session.
        
        Args:
            session_id: Session identifier
            request_id: Request ID for logging
            canonical_title: Canonical topic name (e.g., "French Revolution")
            context_chunks: Retrieved context for entity extraction
            source: Source of anchor (user_query, document, system)
            turn_index: Turn number that created this anchor
            
        Returns:
            Created TopicAnchor
        """
        # Clear any existing anchor
        if session_id in self._session_anchors:
            self.clear_anchor(session_id, reason="replaced")
        
        # Compute topic embedding
        topic_embedding = self._compute_embedding(canonical_title)
        
        # Extract entities from context
        locked_entities = self._extract_entities(
            canonical_title, 
            context_chunks or []
        )
        
        # Generate subject scope
        subject_scope = self._generate_scope(canonical_title, context_chunks)
        
        # Create anchor
        anchor = TopicAnchor(
            id=f"anchor_{uuid.uuid4().hex[:12]}",
            canonical_title=canonical_title,
            topic_embedding=topic_embedding,
            subject_scope=subject_scope,
            locked_entities=locked_entities,
            source=source,
            created_by_session=session_id,
            created_by_turn=turn_index
        )
        
        # Store
        self._session_anchors[session_id] = anchor
        
        logger.info(
            f"[TAL] Created anchor: session={session_id[:8]}... "
            f"title=\"{canonical_title}\" entities={len(locked_entities)} "
            f"scope={subject_scope[:3]}"
        )
        
        return anchor
    
    def get_anchor(self, session_id: str) -> Optional[TopicAnchor]:
        """
        Get active topic anchor for a session.
        
        Returns None if no anchor or anchor expired.
        """
        anchor = self._session_anchors.get(session_id)
        
        if anchor and anchor.is_expired():
            self.clear_anchor(session_id, reason="ttl_expired")
            return None
        
        return anchor
    
    def update_anchor(
        self,
        session_id: str,
        additional_entities: List[str] = None,
        additional_scope: List[str] = None
    ) -> Optional[TopicAnchor]:
        """
        Update an existing anchor with new entities or scope.
        """
        anchor = self.get_anchor(session_id)
        if not anchor:
            return None
        
        if additional_entities:
            anchor.locked_entities = list(set(
                anchor.locked_entities + additional_entities
            ))[:MAX_TOPIC_ENTITIES]
        
        if additional_scope:
            anchor.subject_scope = list(set(
                anchor.subject_scope + additional_scope
            ))
        
        return anchor
    
    def clear_anchor(
        self,
        session_id: str,
        reason: str = "user_reset"
    ) -> bool:
        """
        Clear the active anchor for a session.
        
        Args:
            session_id: Session identifier
            reason: Reason for clearing (user_reset, new_topic, ttl_expired, replaced)
            
        Returns:
            True if anchor was cleared
        """
        anchor = self._session_anchors.get(session_id)
        
        if not anchor:
            return False
        
        # Add to history
        if session_id not in self._session_history:
            self._session_history[session_id] = []
        
        self._session_history[session_id].append(TopicHistoryEntry(
            anchor_id=anchor.id,
            canonical_title=anchor.canonical_title,
            started_at=anchor.created_at,
            ended_at=datetime.utcnow().isoformat(),
            end_reason=reason,
            turns_count=0  # TODO: Track actual turns
        ))
        
        # Remove anchor
        del self._session_anchors[session_id]
        
        logger.info(
            f"[TAL] Cleared anchor: session={session_id[:8]}... "
            f"title=\"{anchor.canonical_title}\" reason={reason}"
        )
        
        return True
    
    def get_topic_history(self, session_id: str) -> List[TopicHistoryEntry]:
        """Get topic history for a session."""
        return self._session_history.get(session_id, [])
    
    def enforce_anchor(
        self,
        session_id: str,
        request_id: str
    ) -> Tuple[Optional[dict], Optional[str]]:
        """
        Get enforcement constraints for an anchored query.
        
        Returns:
            (qdrant_filter, llm_prompt_fragment) or (None, None) if no anchor
        """
        anchor = self.get_anchor(session_id)
        
        if not anchor:
            return None, None
        
        # Build Qdrant filter
        qdrant_filter = {
            "should": [
                # Match by topic anchor (if chunks are tagged)
                # {"key": "topic_anchor_id", "match": {"value": anchor.id}},
                # Or match by subject scope
                {"key": "subject", "match": {"any": anchor.subject_scope}} if anchor.subject_scope else None
            ]
        }
        # Remove None values
        qdrant_filter["should"] = [f for f in qdrant_filter["should"] if f]
        
        # LLM prompt fragment
        prompt_fragment = anchor.to_prompt_fragment()
        
        logger.debug(f"[TAL] Enforcing anchor: {anchor.canonical_title}")
        
        return qdrant_filter, prompt_fragment
    
    def _compute_embedding(self, text: str) -> List[float]:
        """Compute embedding for text."""
        try:
            embedding = self.embedding_model.encode(text)
            return embedding.tolist()
        except Exception as e:
            logger.warning(f"[TAL] Embedding failed: {e}")
            return []
    
    def _extract_entities(
        self,
        title: str,
        context_chunks: List[str]
    ) -> List[str]:
        """
        Extract key entities from title and context.
        
        Uses simple pattern matching (no spaCy dependency).
        """
        entities = set()
        
        # Add title words as entities
        title_words = title.split()
        for word in title_words:
            if len(word) > 2 and word[0].isupper():
                entities.add(word)
        
        # Add multi-word title as entity
        if len(title_words) > 1:
            entities.add(title)
        
        # Extract from context using patterns
        for chunk in context_chunks[:5]:  # Limit chunks
            words = chunk.split()
            for i, word in enumerate(words):
                # Capitalized words (potential proper nouns)
                if len(word) > 2 and word[0].isupper() and word.isalpha():
                    entities.add(word)
                
                # Years (4 digits)
                if word.isdigit() and len(word) == 4:
                    entities.add(word)
                
                # Look for two-word proper nouns
                if i < len(words) - 1:
                    next_word = words[i + 1]
                    if (word[0].isupper() and next_word[0].isupper() and
                        word.isalpha() and next_word.isalpha()):
                        entities.add(f"{word} {next_word}")
        
        return list(entities)[:MAX_TOPIC_ENTITIES]
    
    def _generate_scope(
        self,
        title: str,
        context_chunks: List[str]
    ) -> List[str]:
        """
        Generate subject scope based on title and context.
        
        Returns list of scope keywords.
        """
        # Default scope categories based on common patterns
        scope_keywords = {
            "revolution": ["causes", "events", "consequences", "key figures", "timeline"],
            "war": ["causes", "battles", "consequences", "leaders", "timeline"],
            "theorem": ["definition", "proof", "applications", "examples", "history"],
            "law": ["definition", "explanation", "applications", "examples", "formula"],
            "theory": ["concepts", "explanation", "evidence", "applications", "history"],
            "process": ["steps", "mechanism", "examples", "applications", "importance"],
        }
        
        title_lower = title.lower()
        scope = []
        
        # Match known patterns
        for keyword, scopes in scope_keywords.items():
            if keyword in title_lower:
                scope.extend(scopes)
                break
        
        # Add generic scope if none matched
        if not scope:
            scope = ["explanation", "examples", "applications", "key points"]
        
        return scope


# ============================================================================
# Singleton
# ============================================================================

_tal_service: Optional[TopicAnchorService] = None


def get_topic_anchor_service() -> TopicAnchorService:
    """Get singleton TopicAnchorService instance."""
    global _tal_service
    if _tal_service is None:
        _tal_service = TopicAnchorService()
    return _tal_service


# ============================================================================
# Utility Functions
# ============================================================================

def extract_canonical_title(user_question: str) -> str:
    """
    Extract canonical topic title from user question.
    
    E.g., "tell me about the French Revolution" -> "French Revolution"
    """
    prefixes = [
        "tell me about", "explain", "what is", "what are",
        "describe", "summarize", "define", "how does", "how do",
        "why is", "why does", "who is", "who was", "when did",
        "can you explain", "please explain", "help me understand",
        "i want to know about", "teach me about", "show me"
    ]
    
    text = user_question.strip()
    text_lower = text.lower()
    
    # Remove prefixes
    for prefix in sorted(prefixes, key=len, reverse=True):
        if text_lower.startswith(prefix):
            text = text[len(prefix):].strip()
            break
    
    # Remove trailing punctuation
    text = text.rstrip('?!.,')
    
    # Remove "the" at start
    if text.lower().startswith("the "):
        text = text[4:]
    
    # Title case
    words = text.split()
    if words:
        text = " ".join(w.capitalize() if w.islower() else w for w in words)
    
    return text.strip()
