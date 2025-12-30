"""
Session Intelligence - Decision algorithm for context routing

Implements the core logic to determine whether a query is:
- "related": Continue previous topic, prioritize session resources
- "new_topic": Fresh retrieval, don't prioritize session resources

Features:
- Cosine similarity with last N turns and topic centroid
- Hysteresis to prevent flip-flop on borderline queries
- Configurable thresholds via environment variables
- Comprehensive logging for auditability
"""
import os
import uuid
import hashlib
import logging
from typing import Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

SESSION_RELATED_SIM = float(os.getenv("SESSION_RELATED_SIM", "0.65"))
SESSION_FORGET_SIM = float(os.getenv("SESSION_FORGET_SIM", "0.45"))
RELATED_WINDOW = int(os.getenv("RELATED_WINDOW", "3"))
HYSTERESIS_TURNS = int(os.getenv("HYSTERESIS_TURNS", "2"))


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class SessionDecision:
    """Result of session intelligence decision."""
    decision: str  # "related" or "new_topic"
    max_similarity: float
    most_similar_turn_index: Optional[int]
    all_similarities: List[float]
    centroid_similarity: Optional[float]
    hysteresis_applied: bool
    reasoning: str
    request_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])


@dataclass  
class EmbeddingInfo:
    """Embedding metadata for logging."""
    embedding_hash: str
    embedding_dim: int


# ============================================================================
# Session Intelligence
# ============================================================================

class SessionIntelligence:
    """
    Computes context routing decisions for session-based queries.
    
    Algorithm:
    1. Compute cosine similarity with last N turn embeddings
    2. Compute similarity with topic centroid (sliding window average)
    3. Apply thresholds with hysteresis for borderline cases
    4. Return decision with full audit trail
    """
    
    def __init__(
        self,
        related_threshold: float = None,
        forget_threshold: float = None,
        related_window: int = None,
        hysteresis_turns: int = None
    ):
        self.related_threshold = related_threshold or SESSION_RELATED_SIM
        self.forget_threshold = forget_threshold or SESSION_FORGET_SIM
        self.related_window = related_window or RELATED_WINDOW
        self.hysteresis_turns = hysteresis_turns or HYSTERESIS_TURNS
        
        logger.info(
            f"[SESSION_INTEL] Initialized: related_threshold={self.related_threshold} "
            f"forget_threshold={self.forget_threshold} window={self.related_window} "
            f"hysteresis={self.hysteresis_turns}"
        )
    
    def compute_embedding_hash(self, embedding: List[float]) -> str:
        """Compute SHA256 hash of embedding for audit trail."""
        emb_bytes = np.array(embedding, dtype=np.float32).tobytes()
        return f"sha256:{hashlib.sha256(emb_bytes).hexdigest()[:16]}"
    
    def compute_centroid(self, embeddings: List[List[float]]) -> List[float]:
        """
        Compute centroid (mean) of embedding vectors.
        
        Returns normalized centroid for cosine similarity.
        """
        if not embeddings:
            return []
        
        arr = np.array(embeddings)
        centroid = np.mean(arr, axis=0)
        
        # Normalize for cosine similarity
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
        
        return centroid.tolist()
    
    def cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if not a or not b:
            return 0.0
        
        a_vec = np.array(a)
        b_vec = np.array(b)
        
        dot = np.dot(a_vec, b_vec)
        norm_a = np.linalg.norm(a_vec)
        norm_b = np.linalg.norm(b_vec)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return float(dot / (norm_a * norm_b))
    
    def compute_decision(
        self,
        query_embedding: List[float],
        turn_embeddings: List[List[float]],
        last_topic_vector: Optional[List[float]] = None,
        last_decision: str = "new_topic",
        consecutive_borderline: int = 0,
        session_id: str = "",
        turn_index: int = 0,
        query_text: str = ""
    ) -> Tuple[SessionDecision, dict]:
        """
        Compute session context decision.
        
        Args:
            query_embedding: Embedding of current query
            turn_embeddings: Embeddings of last N turns
            last_topic_vector: Centroid of previous related turns
            last_decision: Previous decision ("related" or "new_topic")
            consecutive_borderline: Count of consecutive borderline decisions
            session_id: For logging
            turn_index: Current turn number
            query_text: For logging (truncated)
            
        Returns:
            (SessionDecision, updated_session_state)
        """
        request_id = str(uuid.uuid4())[:8]
        
        # Log session context
        logger.info(
            f"[SESSION] session_id={session_id[:8] if session_id else 'none'}... "
            f"request_id={request_id} turn_index={turn_index} "
            f"query=\"{query_text[:50]}...\""
        )
        
        # Compute embedding hash
        emb_hash = self.compute_embedding_hash(query_embedding)
        logger.info(f"[EMB] emb_hash={emb_hash} emb_dim={len(query_embedding)}")
        
        # Handle case with no previous turns
        if not turn_embeddings:
            decision = SessionDecision(
                decision="new_topic",
                max_similarity=0.0,
                most_similar_turn_index=None,
                all_similarities=[],
                centroid_similarity=None,
                hysteresis_applied=False,
                reasoning="First query in session",
                request_id=request_id
            )
            
            logger.info(
                f"[SIM] sims=[] max_sim=0.00"
            )
            logger.info(
                f"[DECISION] decision=new_topic threshold_related={self.related_threshold} "
                f"threshold_forget={self.forget_threshold} last_decision={last_decision} "
                f"hysteresis_active=false reason=first_query"
            )
            
            return decision, {
                "last_decision": "new_topic",
                "consecutive_borderline": 0,
                "last_topic_vector": query_embedding  # Start new centroid
            }
        
        # Get last N embeddings
        recent_embeddings = turn_embeddings[-self.related_window:]
        
        # Compute similarities with each recent turn
        similarities = []
        for emb in recent_embeddings:
            sim = self.cosine_similarity(query_embedding, emb)
            similarities.append(sim)
        
        # Find max similarity and index
        max_sim = max(similarities) if similarities else 0.0
        most_similar_idx = similarities.index(max_sim) if similarities else None
        
        # Adjust index to actual turn index
        if most_similar_idx is not None:
            actual_turn_idx = len(turn_embeddings) - len(recent_embeddings) + most_similar_idx + 1
        else:
            actual_turn_idx = None
        
        # Compute centroid similarity if available
        centroid_sim = None
        if last_topic_vector:
            centroid_sim = self.cosine_similarity(query_embedding, last_topic_vector)
            # Include centroid in max calculation
            max_sim = max(max_sim, centroid_sim)
        
        # Log similarities
        sim_strs = [f"{s:.2f}" for s in similarities]
        centroid_sim_str = f"{centroid_sim:.2f}" if centroid_sim is not None else "none"
        logger.info(
            f"[SIM] sims=[{','.join(sim_strs)}] max_sim={max_sim:.2f} "
            f"most_similar_turn={actual_turn_idx} centroid_sim={centroid_sim_str}"
        )
        
        # Apply decision logic
        decision_str = ""
        hysteresis_applied = False
        reasoning = ""
        new_consecutive_borderline = 0
        
        if max_sim >= self.related_threshold:
            # Clear related - above threshold
            decision_str = "related"
            reasoning = f"max_sim {max_sim:.2f} >= threshold {self.related_threshold}"
            new_consecutive_borderline = 0
            
        elif max_sim <= self.forget_threshold:
            # Clear new topic - below threshold
            decision_str = "new_topic"
            reasoning = f"max_sim {max_sim:.2f} <= threshold {self.forget_threshold}"
            new_consecutive_borderline = 0
            
        else:
            # Borderline - apply hysteresis
            new_consecutive_borderline = consecutive_borderline + 1
            
            if new_consecutive_borderline < self.hysteresis_turns:
                # Haven't hit hysteresis limit - keep previous decision
                decision_str = last_decision
                hysteresis_applied = True
                reasoning = (
                    f"borderline {max_sim:.2f}, hysteresis keeping {last_decision} "
                    f"(count {new_consecutive_borderline}/{self.hysteresis_turns})"
                )
            else:
                # Exceeded hysteresis limit - default to new_topic (safe)
                decision_str = "new_topic"
                hysteresis_applied = True
                reasoning = (
                    f"borderline {max_sim:.2f}, hysteresis exceeded "
                    f"({new_consecutive_borderline}>={self.hysteresis_turns}), defaulting to new_topic"
                )
                new_consecutive_borderline = 0
        
        # Log decision
        logger.info(
            f"[DECISION] decision={decision_str} threshold_related={self.related_threshold} "
            f"threshold_forget={self.forget_threshold} last_decision={last_decision} "
            f"hysteresis_active={hysteresis_applied} reason={reasoning}"
        )
        
        # Compute updated topic vector
        new_topic_vector = last_topic_vector
        if decision_str == "related":
            # Update centroid with new embedding (sliding window)
            window_embeddings = recent_embeddings + [query_embedding]
            window_embeddings = window_embeddings[-self.related_window:]
            new_topic_vector = self.compute_centroid(window_embeddings)
        else:
            # New topic - start fresh centroid with this embedding
            new_topic_vector = query_embedding
        
        decision = SessionDecision(
            decision=decision_str,
            max_similarity=max_sim,
            most_similar_turn_index=actual_turn_idx,
            all_similarities=similarities,
            centroid_similarity=centroid_sim,
            hysteresis_applied=hysteresis_applied,
            reasoning=reasoning,
            request_id=request_id
        )
        
        updated_state = {
            "last_decision": decision_str,
            "consecutive_borderline": new_consecutive_borderline,
            "last_topic_vector": new_topic_vector
        }
        
        return decision, updated_state
    
    def get_retrieval_order(
        self, 
        decision: str, 
        force_session_prioritize: bool = False
    ) -> List[str]:
        """
        Get retrieval priority order based on decision.
        
        Args:
            decision: "related" or "new_topic"
            force_session_prioritize: Override to force session priority
            
        Returns:
            List of sources in priority order
        """
        if decision == "related" or force_session_prioritize:
            order = ["session", "classroom", "global", "web"]
        else:
            # New topic - skip session priority
            order = ["classroom", "global", "web"]
        
        logger.info(f"[MCP] context_order={order}")
        return order


# ============================================================================
# Singleton
# ============================================================================

_session_intelligence: Optional[SessionIntelligence] = None


def get_session_intelligence() -> SessionIntelligence:
    """Get singleton SessionIntelligence instance."""
    global _session_intelligence
    if _session_intelligence is None:
        _session_intelligence = SessionIntelligence()
    return _session_intelligence
