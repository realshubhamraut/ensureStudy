"""
Suggestion Engine - Dynamic "Students Also Ask" Generation

Generates context-aware, diverse, non-repetitive follow-up question suggestions.

Pipeline:
1. Extract key phrases from context chunks
2. Generate candidates from templates
3. Filter session duplicates
4. Score & rank by semantic similarity
5. Apply diversity filter
6. Return top K suggestions

All decisions are logged and auditable.
"""
import os
import uuid
import hashlib
import logging
import time
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from .phrase_extractor import extract_key_phrases, ExtractedPhrase
from .suggestion_templates import (
    TEMPLATES, 
    get_diverse_templates, 
    get_generic_fallbacks,
    instantiate_template
)

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

SUGGEST_K = int(os.getenv("SUGGEST_K", "6"))
DIVERSITY_SIM = float(os.getenv("DIVERSITY_SIM", "0.8"))
SESSION_SIM_WEIGHT = float(os.getenv("SESSION_SIM_WEIGHT", "0.5"))
CHUNK_SIM_WEIGHT = float(os.getenv("CHUNK_SIM_WEIGHT", "0.4"))
SESSION_RECENCY_WEIGHT = float(os.getenv("SESSION_RECENCY_WEIGHT", "0.1"))
SUGGEST_MAX_PHRASES = int(os.getenv("SUGGEST_MAX_PHRASES", "8"))
SUGGEST_HISTORY_LIMIT = int(os.getenv("SUGGEST_HISTORY_LIMIT", "50"))
USE_LLM_PARAPHRASE = os.getenv("USE_LLM_PARAPHRASE", "false").lower() == "true"


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class SuggestedQuestion:
    """A generated suggestion with metadata."""
    id: str
    text: str
    intent: str
    score: float
    novel: bool
    source_phrases: List[str]
    action: str = "query"  # or "open_resource"
    embedding: Optional[List[float]] = None
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "text": self.text,
            "intent": self.intent,
            "score": round(self.score, 3),
            "novel": self.novel,
            "source_phrases": self.source_phrases,
            "action": self.action,
        }


@dataclass
class SuggestionCandidate:
    """Internal candidate before ranking."""
    text: str
    intent: str
    source_phrases: List[str]
    hash: str = field(default="")
    score: float = 0.0
    embedding: Optional[List[float]] = None
    
    def __post_init__(self):
        if not self.hash:
            self.hash = hashlib.sha256(self.text.lower().encode()).hexdigest()[:16]


@dataclass
class SuggestionHistory:
    """Tracks previously shown suggestions."""
    hash: str
    text: str
    shown_at: str


# ============================================================================
# Suggestion Engine
# ============================================================================

class SuggestionEngine:
    """
    Generates dynamic, context-aware follow-up question suggestions.
    
    Features:
    - Phrase extraction from context chunks
    - Template-based candidate generation
    - Session-aware novelty filtering
    - Semantic scoring and ranking
    - Diversity filtering
    """
    
    def __init__(self, embedding_model=None):
        """
        Initialize the suggestion engine.
        
        Args:
            embedding_model: SentenceTransformer model for embeddings
        """
        self._embedding_model = embedding_model
        logger.info(
            f"[SUGGEST_ENGINE] Initialized: K={SUGGEST_K} diversity={DIVERSITY_SIM} "
            f"weights=[{SESSION_SIM_WEIGHT},{CHUNK_SIM_WEIGHT},{SESSION_RECENCY_WEIGHT}]"
        )
    
    @property
    def embedding_model(self):
        """Lazy-load embedding model."""
        if self._embedding_model is None:
            from sentence_transformers import SentenceTransformer
            self._embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        return self._embedding_model
    
    def generate_suggestions(
        self,
        request_id: str,
        session_id: str,
        user_question: str,
        answer: str,
        context_chunks: List[dict],
        session_history: List[str] = None,  # Previously shown hashes
        session_resources: List[str] = None,  # Session resource phrases
        canonical_seed: str = None,  # CRITICAL: Immutable topic seed from TopicAnchor
        k: int = None
    ) -> Tuple[List[SuggestedQuestion], dict]:
        """
        Generate K diverse follow-up question suggestions.
        
        CRITICAL: canonical_seed is the ONLY allowed topic seed.
        This prevents semantic recursion where suggestions become nested:
        "What were the causes of What Were The Causes Of French Revolution"
        
        Args:
            request_id: Request ID for logging
            session_id: Session ID for novelty tracking
            user_question: Original user question (NOT used as seed)
            answer: Generated answer text
            context_chunks: MCP-selected context chunks
            session_history: Hashes of previously shown suggestions
            session_resources: Phrases from session resources for boosting
            canonical_seed: IMMUTABLE topic seed from TopicAnchor.canonical_seed
            k: Number of suggestions to return
            
        Returns:
            (suggestions, debug_info)
        """
        start_time = time.time()
        k = k or SUGGEST_K
        session_history = session_history or []
        session_resources = session_resources or []
        
        # Log start
        logger.info(
            f"[SUGGEST] session_id={session_id[:8] if session_id else 'none'}... "
            f"request_id={request_id} user_q=\"{user_question[:50]}...\""
        )
        
        # Extract chunk texts
        chunk_texts = self._extract_chunk_texts(context_chunks)
        
        # ========================================
        # CANONICAL SEED ENFORCEMENT (CSE)
        # ========================================
        # CRITICAL: Use canonical_seed if provided, NEVER extract from user_question
        # This prevents semantic recursion poisoning
        if canonical_seed:
            main_topic = canonical_seed
            logger.info(f"[CSE] Using canonical_seed=\"{main_topic}\" (immutable)")
        else:
            # Fallback: extract from user question ONLY if no canonical_seed
            # This should only happen on first turn
            main_topic = self._extract_main_topic(user_question)
            logger.warning(f"[CSE] No canonical_seed provided, extracted=\"{main_topic}\"")
        
        logger.info(f"[TOPIC] main_topic=\"{main_topic}\"")
        
        # Step 2: Extract key phrases from context
        phrases = extract_key_phrases(
            chunk_texts + [answer],  # Include answer for phrases
            max_phrases=SUGGEST_MAX_PHRASES
        )
        
        # Add main topic as highest priority phrase if not already present
        phrase_texts = [p.text.lower() for p in phrases]
        if main_topic and main_topic.lower() not in phrase_texts:
            from .phrase_extractor import ExtractedPhrase
            phrases.insert(0, ExtractedPhrase(
                text=main_topic,
                source="canonical_seed",  # Mark as canonical
                importance_score=2.0  # Highest priority
            ))
        
        phrase_texts = [p.text for p in phrases]
        logger.info(f"[EXTRACT] phrases={phrase_texts[:5]}...")
        
        # Step 3: Generate candidates with main_topic always included
        candidates = self._generate_candidates(phrases, user_question, answer, main_topic)
        logger.info(f"[CANDIDATES] cand_count={len(candidates)}")
        
        # Step 3: Filter session duplicates
        novel_candidates, filtered_count = self._filter_duplicates(
            candidates, 
            session_history
        )
        logger.info(f"[DUP_CHECK] duplicates_filtered={filtered_count} new_suggestions={len(novel_candidates)}")
        
        # Step 4: Score and rank candidates
        scored_candidates = self._score_candidates(
            novel_candidates,
            user_question,
            chunk_texts,
            session_resources
        )
        
        # Sort by score
        scored_candidates.sort(key=lambda c: -c.score)
        
        # Log top candidates
        top_cands = [{"text": c.text[:40], "score": round(c.score, 2)} for c in scored_candidates[:5]]
        logger.debug(f"[CANDIDATES] top_candidates={top_cands}")
        
        # Step 5: Apply diversity filter
        diverse_candidates = self._apply_diversity_filter(scored_candidates, k)
        
        # Step 6: Build final suggestions
        suggestions = []
        for i, cand in enumerate(diverse_candidates):
            sugg = SuggestedQuestion(
                id=f"s{i+1}_{request_id[:6]}",
                text=cand.text,
                intent=cand.intent,
                score=cand.score,
                novel=True,  # All passed duplicate filter
                source_phrases=cand.source_phrases[:3],
                embedding=cand.embedding
            )
            suggestions.append(sugg)
        
        # If not enough, add generic fallbacks
        if len(suggestions) < k:
            generics = get_generic_fallbacks()
            for i, text in enumerate(generics):
                if len(suggestions) >= k:
                    break
                ghash = hashlib.sha256(text.lower().encode()).hexdigest()[:16]
                if ghash not in session_history:
                    suggestions.append(SuggestedQuestion(
                        id=f"g{i+1}_{request_id[:6]}",
                        text=text,
                        intent="general",
                        score=0.5,
                        novel=True,
                        source_phrases=[]
                    ))
        
        elapsed_ms = int((time.time() - start_time) * 1000)
        
        # Log output
        output_log = [{"id": s.id, "text": s.text[:30], "intent": s.intent, "score": s.score} 
                      for s in suggestions]
        logger.info(f"[OUTPUT] suggestions={len(suggestions)} elapsed_ms={elapsed_ms}")
        
        debug_info = {
            "request_id": request_id,
            "phrases_extracted": len(phrases),
            "candidates_generated": len(candidates),
            "duplicates_filtered": filtered_count,
            "diverse_selected": len(diverse_candidates),
            "elapsed_ms": elapsed_ms,
            "phrases": phrase_texts,
        }
        
        return suggestions, debug_info
    
    def _extract_chunk_texts(self, context_chunks: List[dict]) -> List[str]:
        """Extract text content from context chunks."""
        texts = []
        for chunk in context_chunks:
            if isinstance(chunk, dict):
                text = chunk.get("text") or chunk.get("content") or ""
            else:
                text = str(chunk)
            if text:
                texts.append(text)
        return texts
    
    def _extract_main_topic(self, user_question: str) -> str:
        """
        Extract the main topic from the user's question.
        
        This is critical for ensuring follow-up suggestions maintain context.
        E.g., "tell me about French Revolution" -> "French Revolution"
        """
        # Remove common question prefixes
        prefixes_to_remove = [
            "tell me about", "explain", "what is", "what are", 
            "describe", "summarize", "define", "how does", "how do",
            "why is", "why does", "who is", "who was", "when did",
            "can you explain", "please explain", "help me understand",
            "i want to know about", "i need to learn about",
            "teach me about", "show me"
        ]
        
        text = user_question.strip()
        text_lower = text.lower()
        
        # Remove prefixes
        for prefix in sorted(prefixes_to_remove, key=len, reverse=True):
            if text_lower.startswith(prefix):
                text = text[len(prefix):].strip()
                text_lower = text.lower()
                break
        
        # Remove trailing punctuation
        text = text.rstrip('?!.,')
        
        # Remove "the" at start
        if text_lower.startswith("the "):
            text = text[4:]
        
        # Capitalize first letter of each word for proper nouns
        words = text.split()
        if words:
            # Keep original capitalization if it looks intentional
            if not any(w[0].isupper() for w in words if w):
                # Title case if all lowercase
                text = " ".join(w.capitalize() for w in words)
            else:
                text = " ".join(words)
        
        return text.strip() if len(text) > 2 else ""
    
    def _generate_candidates(
        self, 
        phrases: List[ExtractedPhrase],
        user_question: str,
        answer: str,
        main_topic: str = ""
    ) -> List[SuggestionCandidate]:
        """
        Generate candidate questions from phrases and templates.
        
        IMPORTANT: The main_topic from user question gets priority and is
        used for most suggestions to maintain context.
        """
        candidates = []
        seen_hashes = set()
        
        # Avoid suggesting exact phrases from the answer
        answer_lower = answer.lower()
        
        # Generate suggestions using MAIN TOPIC first (for context)
        if main_topic and len(main_topic) > 3:
            for intent, templates in TEMPLATES.items():
                # Use first template for each intent with main topic
                for template in templates[:1]:  # Just one per intent
                    question = instantiate_template(template, main_topic)
                    
                    cand = SuggestionCandidate(
                        text=question,
                        intent=intent,
                        source_phrases=[main_topic]
                    )
                    
                    if cand.hash not in seen_hashes:
                        seen_hashes.add(cand.hash)
                        candidates.append(cand)
        
        # Then add candidates from extracted phrases (for variety)
        for phrase in phrases:
            topic = phrase.text
            
            # Skip if topic is main topic (already handled) or too generic
            if len(topic) < 4 or topic.lower() == main_topic.lower():
                continue
            
            # Skip phrases that appear exactly in answer (redundant)
            if topic.lower() in answer_lower[:200]:
                continue
            
            # Generate candidates for each intent
            for intent, templates in TEMPLATES.items():
                # Pick just 1 template per intent per phrase
                for template in templates[:1]:
                    question = instantiate_template(template, topic)
                    
                    cand = SuggestionCandidate(
                        text=question,
                        intent=intent,
                        source_phrases=[topic]
                    )
                    
                    if cand.hash not in seen_hashes:
                        seen_hashes.add(cand.hash)
                        candidates.append(cand)
        
        return candidates
    
    def _filter_duplicates(
        self, 
        candidates: List[SuggestionCandidate],
        session_history: List[str]
    ) -> Tuple[List[SuggestionCandidate], int]:
        """
        Filter out candidates already shown in session AND detect semantic recursion.
        
        CRITICAL: This prevents semantic recursion where suggestions become nested:
        "What were the causes of What Were The Causes Of French Revolution"
        """
        import re
        
        history_set = set(session_history)
        novel = []
        filtered = 0
        
        # Anti-recursion patterns - REJECT any candidate matching these
        RECURSION_PATTERNS = [
            # Nested "of What" patterns
            r'of\s+What\s+',
            r'of\s+How\s+',
            r'of\s+Why\s+',
            # Double template patterns
            r'What\s+Were\s+The\s+Main\s+Causes\s+Of\s+What',
            r'What\s+Are\s+The\s+Key\s+.*\s+Of\s+What',
            r'How\s+Did\s+.*\s+Of\s+How',
            # Multiple "of" chains (more than 2)
            r'of\s+[^?]+\s+of\s+[^?]+\s+of\s+',
            # Template markers appearing in topic
            r'\{topic\}',
            r'\[topic\]',
        ]
        
        compiled_patterns = [re.compile(p, re.IGNORECASE) for p in RECURSION_PATTERNS]
        
        for cand in candidates:
            # Check session history
            if cand.hash in history_set:
                filtered += 1
                continue
            
            # Check for recursion patterns (CRITICAL)
            is_recursive = False
            for pattern in compiled_patterns:
                if pattern.search(cand.text):
                    logger.warning(f"[CSE-REJECT] Recursive pattern detected: \"{cand.text[:50]}...\"")
                    is_recursive = True
                    filtered += 1
                    break
            
            if not is_recursive:
                novel.append(cand)
        
        return novel, filtered
    
    def _score_candidates(
        self,
        candidates: List[SuggestionCandidate],
        user_question: str,
        chunk_texts: List[str],
        session_resources: List[str]
    ) -> List[SuggestionCandidate]:
        """Score candidates using semantic similarity."""
        if not candidates:
            return []
        
        # Get embeddings
        question_emb = self.embedding_model.encode(user_question)
        question_emb = question_emb / np.linalg.norm(question_emb)
        
        # Chunk centroid
        if chunk_texts:
            chunk_embs = self.embedding_model.encode(chunk_texts)
            chunk_centroid = np.mean(chunk_embs, axis=0)
            chunk_centroid = chunk_centroid / np.linalg.norm(chunk_centroid)
        else:
            chunk_centroid = None
        
        # Session resource phrases (for recency boost)
        session_set = set(r.lower() for r in session_resources)
        
        # Encode all candidate texts
        cand_texts = [c.text for c in candidates]
        cand_embs = self.embedding_model.encode(cand_texts)
        
        for i, cand in enumerate(candidates):
            cand_emb = cand_embs[i]
            cand_emb_norm = cand_emb / np.linalg.norm(cand_emb)
            cand.embedding = cand_emb_norm.tolist()
            
            # Question similarity
            q_sim = float(np.dot(cand_emb_norm, question_emb))
            
            # Chunk similarity
            c_sim = 0.0
            if chunk_centroid is not None:
                c_sim = float(np.dot(cand_emb_norm, chunk_centroid))
            
            # Session recency boost
            session_boost = 0.0
            for phrase in cand.source_phrases:
                if phrase.lower() in session_set:
                    session_boost = 1.0
                    break
            
            # Weighted score
            cand.score = (
                SESSION_SIM_WEIGHT * q_sim +
                CHUNK_SIM_WEIGHT * c_sim +
                SESSION_RECENCY_WEIGHT * session_boost
            )
        
        return candidates
    
    def _apply_diversity_filter(
        self,
        candidates: List[SuggestionCandidate],
        k: int
    ) -> List[SuggestionCandidate]:
        """
        Apply greedy diversity filter.
        
        Rejects candidates too similar to already selected ones.
        """
        if not candidates:
            return []
        
        selected = []
        rejected_count = 0
        
        for cand in candidates:
            if not cand.embedding:
                continue
            
            cand_emb = np.array(cand.embedding)
            
            # Check similarity with already selected
            is_diverse = True
            for sel in selected:
                if sel.embedding:
                    sel_emb = np.array(sel.embedding)
                    sim = float(np.dot(cand_emb, sel_emb))
                    if sim > DIVERSITY_SIM:
                        is_diverse = False
                        rejected_count += 1
                        break
            
            if is_diverse:
                selected.append(cand)
            
            if len(selected) >= k:
                break
        
        logger.info(f"[DIVERSITY] selected={len(selected)} rejected_for_similarity={rejected_count}")
        
        return selected
    
    def update_session_history(
        self,
        session_history: List[SuggestionHistory],
        new_suggestions: List[SuggestedQuestion]
    ) -> List[SuggestionHistory]:
        """
        Update session suggestion history with new suggestions.
        
        Applies LRU eviction if history exceeds limit.
        """
        now = datetime.utcnow().isoformat()
        
        for sugg in new_suggestions:
            session_history.append(SuggestionHistory(
                hash=hashlib.sha256(sugg.text.lower().encode()).hexdigest()[:16],
                text=sugg.text,
                shown_at=now
            ))
        
        # LRU eviction
        if len(session_history) > SUGGEST_HISTORY_LIMIT:
            session_history = session_history[-SUGGEST_HISTORY_LIMIT:]
        
        return session_history


# ============================================================================
# Singleton
# ============================================================================

_suggestion_engine: Optional[SuggestionEngine] = None


def get_suggestion_engine() -> SuggestionEngine:
    """Get singleton SuggestionEngine instance."""
    global _suggestion_engine
    if _suggestion_engine is None:
        _suggestion_engine = SuggestionEngine()
    return _suggestion_engine
