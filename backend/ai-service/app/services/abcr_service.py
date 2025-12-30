"""
ABCR Service - Attention-Based Context Routing

Combines token-level attention (DistilBERT) with sentence-level cosine similarity
to provide robust context routing for follow-up questions.

Algorithm:
1. Compute query token embeddings using DistilBERT
2. Compute cross-attention with cached turn token embeddings
3. Combine attention score with cosine similarity
4. Apply threshold + hysteresis for decision

Features:
- Token-level attention for fine-grained similarity
- Cosine similarity for sentence-level matching
- Hysteresis to prevent flip-flopping
- Fallback to cosine-only on failure
- Comprehensive logging
"""
import os
import time
import logging
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

ABCR_ENABLED = os.getenv("ABCR_ENABLED", "true").lower() == "true"
ABCR_MODEL = os.getenv("ABCR_MODEL", "distilbert-base-uncased")
ABCR_LAST_N_TURNS = int(os.getenv("ABCR_LAST_N_TURNS", "3"))
ABCR_MAX_TOKENS = int(os.getenv("ABCR_MAX_TOKENS", "128"))
ABCR_W_ATT = float(os.getenv("ABCR_W_ATT", "0.7"))
ABCR_W_EMB = float(os.getenv("ABCR_W_EMB", "0.3"))
ABCR_RELATED_THRESHOLD = float(os.getenv("ABCR_RELATED_THRESHOLD", "0.65"))
ABCR_FORGET_THRESHOLD = float(os.getenv("ABCR_FORGET_THRESHOLD", "0.45"))
ABCR_HYSTERESIS = int(os.getenv("ABCR_HYSTERESIS", "2"))
ABCR_FALLBACK_TIMEOUT_MS = int(os.getenv("ABCR_FALLBACK_TIMEOUT_MS", "300"))


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class TurnScore:
    """Score for a single turn."""
    turn_index: int
    attention_score: float
    cosine_score: float
    combined_score: float
    turn_excerpt: str = ""


@dataclass
class RelatednessResult:
    """Result of ABCR computation."""
    request_id: str
    decision: str  # "related" or "new_topic"
    max_relatedness: float
    matched_turn_index: Optional[int]
    matched_turn_excerpt: str
    turn_scores: List[TurnScore]
    fallback_used: bool
    computation_time_ms: int
    
    def to_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "decision": self.decision,
            "max_relatedness": round(self.max_relatedness, 3),
            "matched_turn_index": self.matched_turn_index,
            "matched_turn_excerpt": self.matched_turn_excerpt[:50] if self.matched_turn_excerpt else "",
            "turn_scores": [
                {
                    "turn": s.turn_index,
                    "att": round(s.attention_score, 3),
                    "cos": round(s.cosine_score, 3),
                    "combined": round(s.combined_score, 3)
                }
                for s in self.turn_scores
            ],
            "fallback_used": self.fallback_used,
            "computation_time_ms": self.computation_time_ms
        }


# ============================================================================
# ABCR Service
# ============================================================================

class ABCRService:
    """
    Attention-Based Context Routing Service.
    
    Computes relatedness between query and session turns using:
    - Token-level attention (DistilBERT)
    - Sentence-level cosine similarity (MiniLM)
    """
    
    def __init__(self):
        self._tokenizer = None
        self._token_model = None
        self._sentence_model = None
        self._device = None
        self._initialized = False
        
        logger.info(
            f"[ABCR] Service created: enabled={ABCR_ENABLED} model={ABCR_MODEL} "
            f"thresholds=[{ABCR_RELATED_THRESHOLD},{ABCR_FORGET_THRESHOLD}] "
            f"weights=[att={ABCR_W_ATT},emb={ABCR_W_EMB}]"
        )
    
    def _ensure_initialized(self):
        """Lazy initialization of models."""
        if self._initialized:
            return
        
        try:
            import torch
            from transformers import AutoTokenizer, AutoModel
            from sentence_transformers import SentenceTransformer
            
            # Device selection
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Token-level model (DistilBERT)
            logger.info(f"[ABCR] Loading token model: {ABCR_MODEL}")
            self._tokenizer = AutoTokenizer.from_pretrained(ABCR_MODEL)
            self._token_model = AutoModel.from_pretrained(ABCR_MODEL)
            self._token_model.to(self._device)
            self._token_model.eval()
            
            # Sentence-level model (MiniLM)
            logger.info("[ABCR] Loading sentence model: all-MiniLM-L6-v2")
            self._sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            
            self._initialized = True
            logger.info(f"[ABCR] Models loaded successfully on {self._device}")
            
        except Exception as e:
            logger.error(f"[ABCR] Model initialization failed: {e}")
            raise
    
    def compute_token_embeddings(self, text: str) -> np.ndarray:
        """
        Compute token-level embeddings for text.
        
        Returns:
            Token embeddings [seq_len, hidden_dim]
        """
        import torch
        
        self._ensure_initialized()
        
        # Tokenize
        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            max_length=ABCR_MAX_TOKENS,
            truncation=True,
            padding=True
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        
        # Forward pass
        with torch.no_grad():
            outputs = self._token_model(**inputs)
            hidden_states = outputs.last_hidden_state  # [1, seq_len, hidden_dim]
        
        # Normalize to unit vectors
        embeddings = hidden_states.squeeze(0)  # [seq_len, hidden_dim]
        embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
        
        return embeddings.cpu().numpy()
    
    def compute_sentence_embedding(self, text: str) -> np.ndarray:
        """Compute sentence-level embedding."""
        self._ensure_initialized()
        embedding = self._sentence_model.encode(text)
        return embedding / np.linalg.norm(embedding)
    
    def compute_attention_score(
        self,
        query_emb: np.ndarray,
        turn_emb: np.ndarray
    ) -> float:
        """
        Compute attention-based similarity score.
        
        Uses cross-attention: for each query token, find max attention to turn tokens.
        Final score is mean of these max attentions.
        
        Args:
            query_emb: [q_len, hidden_dim]
            turn_emb: [t_len, hidden_dim]
            
        Returns:
            Attention score in range [0, 1]
        """
        # Cross-attention matrix: query tokens attend to turn tokens
        # A[i,j] = similarity between query token i and turn token j
        A = query_emb @ turn_emb.T  # [q_len, t_len]
        
        # For each query token, get max attention to any turn token
        token_focus = A.max(axis=1)  # [q_len]
        
        # Mean across query tokens
        raw_score = token_focus.mean()
        
        # Normalize to [0, 1] using sigmoid
        # Adjust k and offset based on expected score distribution
        k = 5.0
        offset = 0.5
        normalized = 1.0 / (1.0 + np.exp(-k * (raw_score - offset)))
        
        return float(normalized)
    
    def compute_cosine_similarity(
        self,
        query_emb: np.ndarray,
        turn_emb: np.ndarray
    ) -> float:
        """
        Compute cosine similarity between sentence embeddings.
        
        Returns:
            Cosine similarity normalized to [0, 1]
        """
        # Dot product of normalized vectors = cosine similarity
        sim = float(np.dot(query_emb, turn_emb))
        
        # Normalize from [-1, 1] to [0, 1]
        return (sim + 1.0) / 2.0
    
    def compute_relatedness(
        self,
        session_id: str,
        query_text: str,
        request_id: str = "",
        turn_texts: List[str] = None,
        turn_embeddings: Dict[int, np.ndarray] = None,
        last_decision: str = "new_topic",
        consecutive_borderline: int = 0,
        clicked_suggestion: bool = False
    ) -> Tuple[RelatednessResult, dict]:
        """
        Compute relatedness between query and session turns.
        
        Args:
            session_id: Session identifier
            query_text: Current query text
            request_id: Request ID for logging
            turn_texts: List of previous turn texts
            turn_embeddings: Cached token embeddings per turn
            last_decision: Previous decision (for hysteresis)
            consecutive_borderline: Counter for borderline decisions
            clicked_suggestion: Whether this is a clicked suggestion
            
        Returns:
            (RelatednessResult, updated_state)
        """
        start_time = time.time()
        
        # Default result for empty/disabled cases
        if not ABCR_ENABLED or not turn_texts:
            return self._fallback_result(
                request_id=request_id,
                reason="disabled" if not ABCR_ENABLED else "no_turns"
            ), {"last_decision": "new_topic", "consecutive_borderline": 0}
        
        turn_scores = []
        fallback_used = False
        
        try:
            # Log start
            logger.info(
                f"[ABCR] session_id={session_id[:8]}... request_id={request_id} "
                f"query=\"{query_text[:40]}...\" clicked={clicked_suggestion}"
            )
            
            # Compute query embeddings
            query_token_emb = self.compute_token_embeddings(query_text)
            query_sentence_emb = self.compute_sentence_embedding(query_text)
            
            logger.info(f"[ABCR] query_tokens={query_token_emb.shape[0]} last_n_turns={len(turn_texts)}")
            
            # Score each turn
            for i, turn_text in enumerate(turn_texts[-ABCR_LAST_N_TURNS:]):
                turn_index = len(turn_texts) - ABCR_LAST_N_TURNS + i
                if turn_index < 0:
                    turn_index = i
                
                # Get or compute turn token embeddings
                if turn_embeddings and turn_index in turn_embeddings:
                    turn_token_emb = turn_embeddings[turn_index]
                else:
                    turn_token_emb = self.compute_token_embeddings(turn_text)
                
                # Compute turn sentence embedding
                turn_sentence_emb = self.compute_sentence_embedding(turn_text)
                
                # Attention score
                att_score = self.compute_attention_score(query_token_emb, turn_token_emb)
                
                # Cosine score
                cos_score = self.compute_cosine_similarity(query_sentence_emb, turn_sentence_emb)
                
                # Combined score
                combined = ABCR_W_ATT * att_score + ABCR_W_EMB * cos_score
                
                turn_scores.append(TurnScore(
                    turn_index=turn_index,
                    attention_score=att_score,
                    cosine_score=cos_score,
                    combined_score=combined,
                    turn_excerpt=turn_text[:50]
                ))
            
            # Log turn scores
            scores_log = [
                {"turn": s.turn_index, "att": round(s.attention_score, 2), 
                 "cos": round(s.cosine_score, 2), "combined": round(s.combined_score, 2)}
                for s in turn_scores
            ]
            logger.info(f"[ABCR] turn_scores={scores_log}")
            
        except Exception as e:
            logger.warning(f"[ABCR] Computation failed, using fallback: {e}")
            fallback_used = True
            
            # Fallback: use cosine-only
            try:
                query_emb = self.compute_sentence_embedding(query_text)
                for i, turn_text in enumerate(turn_texts[-ABCR_LAST_N_TURNS:]):
                    turn_emb = self.compute_sentence_embedding(turn_text)
                    cos_score = self.compute_cosine_similarity(query_emb, turn_emb)
                    
                    turn_scores.append(TurnScore(
                        turn_index=i,
                        attention_score=0.0,
                        cosine_score=cos_score,
                        combined_score=cos_score,  # Cosine only
                        turn_excerpt=turn_text[:50]
                    ))
            except Exception as e2:
                logger.error(f"[ABCR] Fallback also failed: {e2}")
        
        # Find max relatedness
        if turn_scores:
            best_turn = max(turn_scores, key=lambda s: s.combined_score)
            max_relatedness = best_turn.combined_score
            matched_turn_index = best_turn.turn_index
            matched_turn_excerpt = best_turn.turn_excerpt
        else:
            max_relatedness = 0.0
            matched_turn_index = None
            matched_turn_excerpt = ""
        
        # Decision with hysteresis
        decision, new_consecutive = self._make_decision(
            max_relatedness=max_relatedness,
            last_decision=last_decision,
            consecutive_borderline=consecutive_borderline,
            clicked_suggestion=clicked_suggestion
        )
        
        computation_time = int((time.time() - start_time) * 1000)
        
        # Log decision
        logger.info(
            f"[ABCR] decision={decision} max_relatedness={max_relatedness:.3f} "
            f"matched_turn={matched_turn_index} fallback={fallback_used} "
            f"clicked={clicked_suggestion} time_ms={computation_time}"
        )
        
        result = RelatednessResult(
            request_id=request_id,
            decision=decision,
            max_relatedness=max_relatedness,
            matched_turn_index=matched_turn_index,
            matched_turn_excerpt=matched_turn_excerpt,
            turn_scores=turn_scores,
            fallback_used=fallback_used,
            computation_time_ms=computation_time
        )
        
        updated_state = {
            "last_decision": decision,
            "consecutive_borderline": new_consecutive
        }
        
        return result, updated_state
    
    def _make_decision(
        self,
        max_relatedness: float,
        last_decision: str,
        consecutive_borderline: int,
        clicked_suggestion: bool
    ) -> Tuple[str, int]:
        """
        Make related/new_topic decision with hysteresis.
        
        Special handling for clicked suggestions: more lenient threshold.
        """
        # For clicked suggestions, use lower threshold (benefit of doubt)
        related_threshold = ABCR_RELATED_THRESHOLD
        if clicked_suggestion:
            related_threshold = ABCR_FORGET_THRESHOLD  # More lenient
        
        # Clear decision
        if max_relatedness >= related_threshold:
            return "related", 0
        
        if max_relatedness <= ABCR_FORGET_THRESHOLD:
            return "new_topic", 0
        
        # Borderline: apply hysteresis
        consecutive_borderline += 1
        
        if consecutive_borderline >= ABCR_HYSTERESIS:
            # Flip after enough borderline cases
            new_decision = "new_topic" if last_decision == "related" else "related"
            return new_decision, 0
        
        # Stay with previous decision
        return last_decision, consecutive_borderline
    
    def _fallback_result(self, request_id: str, reason: str) -> RelatednessResult:
        """Create fallback result when ABCR can't run."""
        logger.info(f"[ABCR] Fallback result: reason={reason}")
        return RelatednessResult(
            request_id=request_id,
            decision="new_topic",
            max_relatedness=0.0,
            matched_turn_index=None,
            matched_turn_excerpt="",
            turn_scores=[],
            fallback_used=True,
            computation_time_ms=0
        )


# ============================================================================
# Singleton
# ============================================================================

_abcr_service: Optional[ABCRService] = None


def get_abcr_service() -> ABCRService:
    """Get singleton ABCRService instance."""
    global _abcr_service
    if _abcr_service is None:
        _abcr_service = ABCRService()
    return _abcr_service
