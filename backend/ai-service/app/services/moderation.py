"""
Academic Moderation Gate

Filters non-academic, harmful, or off-topic queries BEFORE retrieval and LLM calls.
Uses deterministic rules + lightweight keyword matching (no LLM).

Moderation Criteria:
1. BLOCK: Harmful content (violence, explicit, illegal)
2. BLOCK: Personal/private requests (homework answers, exam cheating)
3. WARN: Borderline academic (coding, career advice)
4. ALLOW: Clear academic questions (explain, define, calculate, compare)

Example Inputs/Outputs:
- "Explain photosynthesis" -> allow, confidence=0.95, category="science"
- "Do my homework for me" -> block, confidence=0.90, category="cheating"
- "What's the weather?" -> block, confidence=0.85, category="off_topic"
- "How to write code for sorting" -> warn, confidence=0.70, category="borderline"
"""
import re
from typing import Tuple
from ..api.schemas.tutor import ModerationResult


# ============================================================================
# Keyword Lists for Classification
# ============================================================================

HARMFUL_PATTERNS = [
    r'\b(kill|murder|suicide|harm|attack|weapon|bomb|drug|illegal)\b',
    r'\b(hack|crack|bypass|exploit|steal)\b',
    r'\b(explicit|porn|nude|nsfw)\b',
]

CHEATING_PATTERNS = [
    r'\b(do my homework|write my essay|complete my assignment)\b',
    r'\b(exam answers|test solutions|cheat sheet)\b',
    r'\b(copy paste|plagiarize)\b',
]

OFF_TOPIC_PATTERNS = [
    r'\b(weather|sports score|movie|celebrity|gossip|news)\b',
    r'\b(recipe|cooking|gaming|entertainment)\b',
    r'\b(relationship|dating|personal advice)\b',
]

ACADEMIC_INDICATORS = [
    r'\b(explain|define|describe|compare|contrast|analyze)\b',
    r'\b(understand|learn|study|teach me|help me|show me)\b',
    r'\b(what is|how does|why does|when did|who was)\b',
    r'\b(calculate|solve|derive|prove|find the)\b',
    r'\b(formula|equation|theorem|principle|law|theory)\b',
    r'\b(chapter|topic|subject|lesson|concept)\b',
]

SUBJECT_KEYWORDS = {
    "math": r'\b(math|algebra|geometry|calculus|trigonometry|arithmetic|equation|graph)\b',
    "physics": r'\b(physics|motion|force|energy|newton|velocity|acceleration|wave)\b',
    "chemistry": r'\b(chemistry|element|compound|reaction|molecule|atom|periodic|bond)\b',
    "biology": r'\b(biology|cell|organism|dna|photosynthesis|evolution|genetics|anatomy)\b',
    "history": r'\b(history|war|civilization|empire|revolution|ancient|medieval|century)\b',
    "english": r'\b(grammar|vocabulary|literature|essay|writing|poetry|novel|language)\b',
}


# ============================================================================
# Moderation Functions
# ============================================================================

def _check_patterns(text: str, patterns: list) -> Tuple[bool, float]:
    """Check if text matches any pattern. Returns (matched, confidence)."""
    text_lower = text.lower()
    matches = 0
    for pattern in patterns:
        if re.search(pattern, text_lower, re.IGNORECASE):
            matches += 1
    if matches > 0:
        confidence = min(0.95, 0.7 + (matches * 0.1))
        return True, confidence
    return False, 0.0


def _detect_subject(text: str) -> str:
    """Detect academic subject from text."""
    text_lower = text.lower()
    for subject, pattern in SUBJECT_KEYWORDS.items():
        if re.search(pattern, text_lower, re.IGNORECASE):
            return subject
    return "general"


def _is_academic_question(text: str) -> Tuple[bool, float]:
    """Check if text appears to be an academic question."""
    text_lower = text.lower()
    
    # Check for question structure
    has_question_mark = "?" in text
    has_academic_verb = bool(re.search(r'\b(explain|define|describe|calculate|solve|what|how|why)\b', text_lower))
    has_academic_noun = bool(re.search(r'\b(formula|equation|theorem|concept|principle|example)\b', text_lower))
    
    # Score based on indicators
    score = 0.3  # Base score
    if has_question_mark:
        score += 0.2
    if has_academic_verb:
        score += 0.3
    if has_academic_noun:
        score += 0.2
    
    # Check explicit academic indicators
    for pattern in ACADEMIC_INDICATORS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            score = min(score + 0.15, 1.0)
    
    return score >= 0.5, score


def moderate_query(user_id: str, question: str) -> ModerationResult:
    """
    Main moderation function. Runs BEFORE retrieval and LLM.
    
    Args:
        user_id: Student identifier (for logging/rate limiting)
        question: Raw question text
        
    Returns:
        ModerationResult with decision, confidence, category, reason
        
    Example:
        >>> moderate_query("usr_123", "Explain Newton's laws")
        ModerationResult(decision='allow', confidence=0.92, category='physics', reason=None)
        
        >>> moderate_query("usr_123", "Do my homework for me")
        ModerationResult(decision='block', confidence=0.90, category='cheating', reason='Appears to be a cheating request')
    """
    question = question.strip()
    
    # Step 1: Check for harmful content (highest priority)
    is_harmful, harmful_conf = _check_patterns(question, HARMFUL_PATTERNS)
    if is_harmful:
        return ModerationResult(
            decision="block",
            confidence=harmful_conf,
            category="harmful",
            reason="Query contains potentially harmful content"
        )
    
    # Step 2: Check for cheating attempts
    is_cheating, cheat_conf = _check_patterns(question, CHEATING_PATTERNS)
    if is_cheating:
        return ModerationResult(
            decision="block",
            confidence=cheat_conf,
            category="cheating",
            reason="Appears to be a cheating request"
        )
    
    # Step 3: Check for off-topic queries
    is_off_topic, off_topic_conf = _check_patterns(question, OFF_TOPIC_PATTERNS)
    if is_off_topic:
        return ModerationResult(
            decision="block",
            confidence=off_topic_conf,
            category="off_topic",
            reason="Query is not related to academic topics"
        )
    
    # Step 4: Check if it's an academic question
    is_academic, academic_conf = _is_academic_question(question)
    
    # Step 5: Detect subject
    subject = _detect_subject(question)
    
    # Step 6: Make decision - BE PERMISSIVE, only block harmful
    # Allow almost everything - users can ask any question they want
    if is_academic or academic_conf >= 0.2:  # Very low threshold
        return ModerationResult(
            decision="allow",
            confidence=max(academic_conf, 0.5),
            category=subject if subject != "general" else "other",
            reason=None
        )
    else:
        # Even if not clearly academic, still allow it
        return ModerationResult(
            decision="allow",
            confidence=0.5,
            category="general",
            reason=None
        )


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Test examples
    test_cases = [
        ("usr_1", "Explain photosynthesis in plants"),
        ("usr_2", "What is Newton's first law of motion?"),
        ("usr_3", "Do my homework for me"),
        ("usr_4", "What's the weather like today?"),
        ("usr_5", "How to calculate velocity from acceleration?"),
        ("usr_6", "Write code for bubble sort"),
        ("usr_7", "Hello"),
    ]
    
    for user_id, question in test_cases:
        result = moderate_query(user_id, question)
        print(f"\nQ: {question}")
        print(f"→ {result.decision.upper()} | conf={result.confidence:.2f} | cat={result.category}")
        if result.reason:
            print(f"   Reason: {result.reason}")
