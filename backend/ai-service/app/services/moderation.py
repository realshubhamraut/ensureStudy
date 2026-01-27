"""
Academic Moderation Gate - HuggingFace Inference API (Cloud BART-MNLI)

Uses HuggingFace Inference API for zero-shot classification.
No local model loading - calls the cloud API instead.

Philosophy:
- ALLOW: Any question that seeks knowledge/learning
- BLOCK: Only genuinely irrelevant or harmful content
"""
import re
import logging
import os
from typing import Tuple
from functools import lru_cache

import httpx

from ..api.schemas.tutor import ModerationResult

logger = logging.getLogger(__name__)


# ============================================================================
# HuggingFace Inference API Configuration
# ============================================================================

# Use facebook/bart-large-mnli - reliable and well-maintained
HF_API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"

# Classification labels (simple, clear)
CANDIDATE_LABELS = [
    "educational question about learning or knowledge",
    "casual chat or entertainment request"
]

# Educational keywords for fallback classification
EDUCATIONAL_KEYWORDS = [
    # Question words
    'what', 'why', 'how', 'when', 'where', 'who', 'which', 'explain', 'describe',
    'define', 'tell me about', 'what is', 'what are', 'what was', 'what were',
    
    # Academic subjects
    'science', 'physics', 'chemistry', 'biology', 'math', 'mathematics',
    'history', 'geography', 'literature', 'economics', 'psychology',
    'philosophy', 'sociology', 'anthropology', 'astronomy', 'geology',
    
    # Technical topics
    'programming', 'algorithm', 'computer', 'software', 'technology',
    'engineering', 'machine learning', 'artificial intelligence', 'data',
    
    # Learning words
    'learn', 'study', 'understand', 'concept', 'theory', 'principle',
    'example', 'difference between', 'compare', 'contrast', 'analyze',
    
    # Natural phenomena
    'photosynthesis', 'evolution', 'gravity', 'electricity', 'magnetism',
    'refraction', 'reflection', 'diffraction', 'thermodynamics', 'quantum',
    
    # Historical/Cultural
    'revolution', 'war', 'civilization', 'culture', 'religion', 'art',
    'music', 'architecture', 'invention', 'discovery'
]


# ============================================================================
# Minimal Safety Filters (Regex - very fast)
# ============================================================================

HARMFUL_PATTERNS = [
    r'\b(how to|ways to)\s+(kill|murder|harm|attack|hurt)\s+\w+',
    r'\b(bomb|explosive|weapon)\s+(make|build|create)',
    r'\b(suicide|self.?harm)\s+(method|how)',
    r'\b(hack|crack)\s+(into|password|account)',
    r'\b(drug|narcotic).*(make|cook|synthesize)',
    r'\b(porn|nude|nsfw|xxx)\b',
]

PURE_CHITCHAT = [
    r'^(hi+|hello+|hey+|sup|yo)[\s!?.]*$',
    r'^how are you[\s?!]*$',
    r'^what\'?s up[\s?!]*$',
    r'^(good morning|good night|bye|goodbye)[\s!.]*$',
]


def _matches_any_pattern(text: str, patterns: list) -> bool:
    """Check if text matches any regex pattern."""
    text_lower = text.lower().strip()
    for pattern in patterns:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return True
    return False


# ============================================================================
# Keyword-Based Classification (Primary - Fast & Reliable)
# ============================================================================

def _classify_with_keywords(question: str) -> Tuple[str, float]:
    """
    Classify using keyword matching - fast and reliable.
    
    Returns: (intent, confidence)
    - intent: "educational" or "non_educational"
    - confidence: 0.0 to 1.0
    """
    question_lower = question.lower().strip()
    
    # Count educational keyword matches
    matches = 0
    for keyword in EDUCATIONAL_KEYWORDS:
        if keyword in question_lower:
            matches += 1
    
    # Calculate confidence based on matches
    if matches >= 3:
        confidence = 0.95
    elif matches >= 2:
        confidence = 0.90
    elif matches >= 1:
        confidence = 0.85
    else:
        # Check if it looks like a question (starts with question word or ends with ?)
        if question_lower.endswith('?') or any(question_lower.startswith(w) for w in ['what', 'why', 'how', 'when', 'where', 'who', 'which', 'can', 'could', 'would', 'is', 'are', 'was', 'were', 'does', 'do', 'did']):
            confidence = 0.80
            matches = 1  # Treat as educational
        else:
            confidence = 0.70
    
    if matches > 0:
        logger.debug(f"[MODERATION] Keyword match: {matches} keywords, confidence {confidence:.0%}")
        return ("educational", confidence)
    else:
        return ("non_educational", confidence)


def _get_hf_token() -> str:
    """Get HuggingFace API token from environment."""
    return os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN") or ""


@lru_cache(maxsize=1000)
def _classify_with_hf_api(question: str) -> Tuple[str, float]:
    """
    Classify using HuggingFace Inference API (BART-MNLI).
    Falls back to keyword classification if API fails.
    
    Returns: (intent, confidence)
    - intent: "educational" or "non_educational"
    - confidence: 0.0 to 1.0
    """
    # First try keyword classification (fast, reliable)
    keyword_result = _classify_with_keywords(question)
    
    # If keyword classification is confident, use it directly
    if keyword_result[1] >= 0.85:
        logger.info(f"[MODERATION] Using keyword classifier: {keyword_result[0]} ({keyword_result[1]:.0%})")
        return keyword_result
    
    # Try HuggingFace API for edge cases
    hf_token = _get_hf_token()
    
    headers = {"Content-Type": "application/json"}
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"
    
    payload = {
        "inputs": question,
        "parameters": {
            "candidate_labels": CANDIDATE_LABELS,
            "multi_label": False
        }
    }
    
    try:
        logger.debug(f"[MODERATION] 🔄 Trying HF API for: {question[:40]}...")
        
        response = httpx.post(
            HF_API_URL,
            headers=headers,
            json=payload,
            timeout=5.0  # 5 second timeout
        )
        
        if response.status_code == 200:
            result = response.json()
            
            if isinstance(result, dict) and "labels" in result and "scores" in result:
                top_label = result["labels"][0]
                top_score = result["scores"][0]
                
                # Determine intent based on top label
                if "educational" in top_label.lower():
                    logger.info(f"[MODERATION] HF API: educational ({top_score:.0%})")
                    return ("educational", top_score)
                else:
                    logger.info(f"[MODERATION] HF API: non-educational ({top_score:.0%})")
                    return ("non_educational", top_score)
        
        # API failed - use keyword fallback
        logger.warning(f"[MODERATION] HF API error {response.status_code}, using keyword fallback")
    
    except Exception as e:
        logger.warning(f"[MODERATION] HF API exception: {e}, using keyword fallback")
    
    # Return keyword result as fallback
    return keyword_result


# ============================================================================
# Main Moderation Function
# ============================================================================

def moderate_query(user_id: str, question: str) -> ModerationResult:
    """
    Moderate using HuggingFace Inference API (cloud BART-MNLI).
    
    Flow:
    1. Block harmful content (regex - instant)
    2. Block pure chitchat (regex - instant)
    3. Use HF API for zero-shot classification
    4. Allow if educational, block if clearly non-educational
    
    Philosophy: Be permissive. Only block clearly irrelevant content.
    """
    question = question.strip()
    
    # Empty check
    if not question or len(question) < 2:
        return ModerationResult(
            decision="block",
            confidence=1.0,
            category="empty",
            reason="Please enter a question!"
        )
    
    # 1. Block harmful content (instant, no API needed)
    if _matches_any_pattern(question, HARMFUL_PATTERNS):
        logger.warning(f"[MODERATION] 🚫 Harmful content blocked")
        return ModerationResult(
            decision="block",
            confidence=0.95,
            category="harmful",
            reason="I can't help with that. Let's focus on learning! 📚"
        )
    
    # 2. Block pure chitchat (instant, no API needed)
    if _matches_any_pattern(question, PURE_CHITCHAT):
        logger.info(f"[MODERATION] ❌ Pure chitchat blocked: {question}")
        return ModerationResult(
            decision="block",
            confidence=0.9,
            category="chitchat",
            reason="I'm your learning assistant! Ask me anything educational - history, science, math, technology, or any topic you want to learn about. 🎓"
        )
    
    # 3. Use HuggingFace API for classification
    intent, confidence = _classify_with_hf_api(question)
    
    logger.info(f"[MODERATION] {'✅' if intent == 'educational' else '❌'} {intent} ({confidence:.0%}): {question[:40]}...")
    
    if intent == "educational":
        return ModerationResult(
            decision="allow",
            confidence=confidence,
            category="academic",
            reason=None
        )
    
    # Non-educational - but only block if high confidence
    if confidence >= 0.75:
        return ModerationResult(
            decision="block",
            confidence=confidence,
            category="non_educational",
            reason="I'm here to help you learn! Ask me about any topic - history, science, technology, culture, or anything you're curious about. 🌟"
        )
    
    # Low confidence non-educational - give benefit of doubt, allow
    logger.info(f"[MODERATION] ⚠️ Low confidence ({confidence:.0%}), allowing anyway")
    return ModerationResult(
        decision="allow",
        confidence=confidence,
        category="uncertain",
        reason=None
    )


# ============================================================================
# Utility
# ============================================================================

def preload_classifier():
    """Warm up the HF API by making a test call."""
    logger.info("[MODERATION] Warming up HuggingFace API...")
    try:
        _classify_with_hf_api("What is photosynthesis?")
        logger.info("[MODERATION] ✅ HuggingFace API ready")
    except Exception as e:
        logger.warning(f"[MODERATION] API warmup failed: {e}")


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    test_cases = [
        # Should ALLOW (educational)
        "Who was Adolf Hitler?",
        "What is photosynthesis?",
        "Explain machine learning",
        "How do vaccines work?",
        "Tell me about the French Revolution",
        "What causes earthquakes?",
        "History of jazz music",
        "What is blockchain technology?",
        
        # Should BLOCK (non-educational)
        "Hi how are you?",
        "Hey!",
        "What's up",
        "Recommend me a Netflix show",
    ]
    
    print("=" * 60)
    print("HUGGINGFACE API MODERATION TEST")
    print("=" * 60)
    print(f"HF Token: {'✅ Set' if _get_hf_token() else '❌ Not set'}")
    print("=" * 60)
    
    for q in test_cases:
        result = moderate_query("test", q)
        status = "✅" if result.decision == "allow" else "❌"
        print(f"{status} [{result.confidence:.0%}] {q[:45]}")
    
    print("\n" + "=" * 60)
