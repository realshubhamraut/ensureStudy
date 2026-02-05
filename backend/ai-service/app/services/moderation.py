"""
Academic Moderation Gate - LLM-Based Classification

Uses Groq LLM for intelligent query classification.
No brittle regex patterns - pure LLM reasoning.

Philosophy:
- ALLOW: Educational questions, learning-focused queries
- BLOCK: Shopping, entertainment, casual chat, harmful content
"""
import logging
import os
import re
from typing import Tuple
from functools import lru_cache

from groq import Groq

from ..api.schemas.tutor import ModerationResult

logger = logging.getLogger(__name__)


# ============================================================================
# Minimal Safety Filter (Regex only for truly harmful content)
# ============================================================================

HARMFUL_PATTERNS = [
    r'\b(how to|ways to)\s+(kill|murder|harm|attack|hurt)\s+\w+',
    r'\b(bomb|explosive|weapon)\s+(make|build|create)',
    r'\b(suicide|self.?harm)\s+(method|how)',
    r'\b(hack|crack)\s+(into|password|account)',
    r'\b(drug|narcotic).*(make|cook|synthesize)',
    r'\b(porn|nude|nsfw|xxx)\b',
]


def _is_harmful(text: str) -> bool:
    """Check for genuinely harmful content (safety filter)."""
    text_lower = text.lower().strip()
    for pattern in HARMFUL_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return True
    return False


# ============================================================================
# LLM-Based Classification using Groq
# ============================================================================

CLASSIFICATION_PROMPT = """You are a moderation classifier for an educational tutoring platform.

Classify the following user query into ONE of these categories:

1. EDUCATIONAL - Questions seeking knowledge, learning, or understanding about any academic topic (science, history, technology, how things work, etc.)

2. NON_EDUCATIONAL - Queries that are NOT about learning, including:
   - Product prices, shopping queries ("what is the price of...")
   - Entertainment recommendations (movies, shows, games)
   - Sports scores/results
   - Weather updates
   - Social media/trending topics
   - Casual chat/greetings
   - Personal advice unrelated to academics

Rules:
- Questions about HOW something works = EDUCATIONAL
- Questions about understanding concepts = EDUCATIONAL  
- Questions asking for prices, recommendations, or current events = NON_EDUCATIONAL
- When in doubt, lean towards EDUCATIONAL

User Query: "{query}"

Respond with ONLY one word: EDUCATIONAL or NON_EDUCATIONAL"""


def _get_groq_client() -> Groq:
    """Get Groq client instance."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set")
    return Groq(api_key=api_key)


@lru_cache(maxsize=500)
def _classify_with_llm(question: str) -> Tuple[str, float]:
    """
    Classify query using Groq LLM.
    
    Returns: (category, confidence)
    - category: "educational" or "non_educational"
    - confidence: 0.0 to 1.0
    """
    try:
        client = _get_groq_client()
        
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",  # Fast, cheap model for classification
            messages=[
                {
                    "role": "system",
                    "content": "You are a query classifier. Respond with only one word."
                },
                {
                    "role": "user", 
                    "content": CLASSIFICATION_PROMPT.format(query=question)
                }
            ],
            max_tokens=10,
            temperature=0.0  # Deterministic output
        )
        
        result = response.choices[0].message.content.strip().upper()
        
        if "NON" in result or "NON_EDUCATIONAL" in result:
            logger.info(f"[MODERATION] LLM classified as NON_EDUCATIONAL: {question[:50]}...")
            return ("non_educational", 0.92)
        else:
            logger.info(f"[MODERATION] LLM classified as EDUCATIONAL: {question[:50]}...")
            return ("educational", 0.92)
            
    except Exception as e:
        logger.warning(f"[MODERATION] LLM classification failed: {e}")
        # Fallback: allow on error (fail-open for better UX)
        return ("educational", 0.5)


# ============================================================================
# Main Moderation Function
# ============================================================================

def moderate_query(user_id: str, question: str) -> ModerationResult:
    """
    Moderate query using LLM-based classification.
    
    Flow:
    1. Block harmful content (regex - instant safety check)
    2. Use LLM for intelligent classification
    3. Return appropriate decision
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
    
    # 1. Block harmful content (safety filter - instant)
    if _is_harmful(question):
        logger.warning(f"[MODERATION] 🚫 Harmful content blocked")
        return ModerationResult(
            decision="block",
            confidence=0.99,
            category="harmful",
            reason="I can't help with that. Let's focus on learning! 📚"
        )
    
    # 2. LLM-based classification
    category, confidence = _classify_with_llm(question)
    
    if category == "educational":
        logger.info(f"[MODERATION] ✅ Allowed: {question[:40]}...")
        return ModerationResult(
            decision="allow",
            confidence=confidence,
            category="academic",
            reason=None
        )
    else:
        logger.info(f"[MODERATION] ❌ Blocked (non-educational): {question[:40]}...")
        return ModerationResult(
            decision="block",
            confidence=confidence,
            category="non_educational",
            reason="I'm your learning assistant! I can help with educational topics like science, history, math, technology, and more. For shopping or entertainment, try other services! 📚"
        )


# ============================================================================
# Utility
# ============================================================================

def preload_classifier():
    """Warm up the LLM by making a test call."""
    logger.info("[MODERATION] Warming up LLM classifier...")
    try:
        _classify_with_llm("What is photosynthesis?")
        logger.info("[MODERATION] ✅ LLM classifier ready")
    except Exception as e:
        logger.warning(f"[MODERATION] Warmup failed: {e}")


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    test_cases = [
        # Should ALLOW (educational)
        ("What is photosynthesis?", True),
        ("Explain machine learning", True),
        ("How do vaccines work?", True),
        ("Who was Napoleon Bonaparte?", True),
        ("What causes earthquakes?", True),
        ("How does WiFi work?", True),
        ("What is the structure of an atom?", True),
        
        # Should BLOCK (non-educational)
        ("What is the price of Samsung Galaxy S23?", False),
        ("Recommend me a Netflix show", False),
        ("Hi how are you?", False),
        ("What's the weather in Mumbai?", False),
        ("Who won the IPL match yesterday?", False),
        ("Best games to play on PS5", False),
    ]
    
    print("=" * 60)
    print("LLM-BASED MODERATION TEST")
    print("=" * 60)
    
    correct = 0
    for q, expected_allow in test_cases:
        result = moderate_query("test", q)
        actual_allow = result.decision == "allow"
        status = "✅" if actual_allow == expected_allow else "❌"
        if actual_allow == expected_allow:
            correct += 1
        print(f"{status} [{result.decision:5s}] {q[:50]}")
    
    print(f"\nAccuracy: {correct}/{len(test_cases)} ({100*correct/len(test_cases):.0f}%)")
    print("=" * 60)
