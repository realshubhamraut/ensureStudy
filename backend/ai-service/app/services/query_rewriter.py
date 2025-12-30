"""
Query Rewriter & Ambiguous Query Handler

Detects short/ambiguous follow-up queries and rewrites them with TopicAnchor context.
Prevents unrelated web results by enforcing topic context.

Example:
    Query: "how many people were killed"
    Anchor: "French Revolution"
    Rewritten: "How many people were killed during the French Revolution (1789-1799)?"
"""
import os
import re
import logging
from typing import Optional, Tuple, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

MIN_WORDS = int(os.getenv("QUERY_MIN_WORDS", "6"))
NUMERIC_QUERY_PATTERNS = [
    r'\bhow many\b',
    r'\bnumber of\b',
    r'\bcount of\b',
    r'\bestimated deaths?\b',
    r'\bcasualties\b',
    r'\bkilled\b',
    r'\bdied\b',
    r'\bdeaths?\b',
    r'\bvictims?\b',
    r'\btotal\b',
    r'\bstatistics?\b',
]

# Ambiguous pronouns/terms that indicate missing context
AMBIGUOUS_TERMS = [
    'this', 'that', 'these', 'those', 'it', 'they', 'them',
    'what is the number', 'how many people', 'how many were',
    'what happened', 'who was', 'when did', 'where did',
]


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class RewriteResult:
    """Result of query rewriting."""
    original_query: str
    rewritten_query: str
    was_rewritten: bool
    is_numeric_query: bool
    anchor_title: Optional[str] = None
    anchor_time_range: Optional[str] = None
    reason: str = ""
    
    def to_dict(self) -> dict:
        return {
            "original_query": self.original_query,
            "rewritten_query": self.rewritten_query,
            "was_rewritten": self.was_rewritten,
            "is_numeric_query": self.is_numeric_query,
            "anchor_title": self.anchor_title,
            "reason": self.reason,
        }


# ============================================================================
# Query Analysis Functions
# ============================================================================

def is_ambiguous_short_query(query_text: str, has_active_anchor: bool = False) -> bool:
    """
    Detect if a query is ambiguous or too short to stand alone.
    
    Returns True if:
    - Query has fewer than MIN_WORDS words, OR
    - Query contains pronouns/generic terms, OR
    - Query lacks explicit topic terms and anchor exists
    """
    query_lower = query_text.lower().strip()
    words = query_lower.split()
    
    # Check word count
    if len(words) < MIN_WORDS:
        logger.debug(f"[REWRITE] Short query: {len(words)} words < {MIN_WORDS}")
        return True
    
    # Check for ambiguous terms
    for term in AMBIGUOUS_TERMS:
        if term in query_lower:
            logger.debug(f"[REWRITE] Ambiguous term found: '{term}'")
            return True
    
    # If anchor exists and query doesn't contain proper nouns, it's likely ambiguous
    if has_active_anchor:
        # Check if query has any capitalized words (proper nouns)
        has_proper_nouns = any(
            word[0].isupper() and len(word) > 2 
            for word in query_text.split() 
            if word and word[0].isalpha()
        )
        if not has_proper_nouns:
            logger.debug("[REWRITE] No proper nouns and anchor exists")
            return True
    
    return False


def is_numeric_query(query_text: str) -> bool:
    """
    Detect if query is asking for a numeric fact/count.
    """
    query_lower = query_text.lower()
    
    for pattern in NUMERIC_QUERY_PATTERNS:
        if re.search(pattern, query_lower):
            return True
    
    return False


def get_topic_time_range(topic_title: str) -> Optional[str]:
    """
    Extract or lookup time range for known historical topics.
    """
    # Known historical events with dates
    TIME_RANGES = {
        "french revolution": "1789-1799",
        "american revolution": "1765-1783",
        "world war 1": "1914-1918",
        "world war 2": "1939-1945",
        "world war i": "1914-1918",
        "world war ii": "1939-1945",
        "civil war": "1861-1865",
        "renaissance": "14th-17th century",
        "industrial revolution": "1760-1840",
        "cold war": "1947-1991",
    }
    
    topic_lower = topic_title.lower()
    for topic, date_range in TIME_RANGES.items():
        if topic in topic_lower:
            return date_range
    
    return None


# ============================================================================
# Query Rewriting
# ============================================================================

def rewrite_with_anchor(
    query_text: str,
    anchor_title: str,
    anchor_time_range: Optional[str] = None
) -> RewriteResult:
    """
    Rewrite an ambiguous query by injecting the topic anchor context.
    
    Args:
        query_text: Original user query
        anchor_title: Canonical topic title from TopicAnchor
        anchor_time_range: Optional time range for historical topics
        
    Returns:
        RewriteResult with original and rewritten query
    """
    original = query_text.strip()
    original_lower = original.lower()
    
    # Check if query already contains the topic
    if anchor_title.lower() in original_lower:
        logger.info(f"[REWRITE] Query already contains anchor: '{anchor_title}'")
        return RewriteResult(
            original_query=original,
            rewritten_query=original,
            was_rewritten=False,
            is_numeric_query=is_numeric_query(original),
            anchor_title=anchor_title,
            reason="Query already contains topic"
        )
    
    # Get time range if not provided
    if not anchor_time_range:
        anchor_time_range = get_topic_time_range(anchor_title)
    
    # Build contextual suffix
    if anchor_time_range:
        context_suffix = f" during the {anchor_title} ({anchor_time_range})"
    else:
        context_suffix = f" during the {anchor_title}"
    
    # Clean up query for rewriting
    rewritten = original.rstrip('?.,!')
    
    # Add context
    rewritten = f"{rewritten}{context_suffix}?"
    
    # Capitalize first letter
    rewritten = rewritten[0].upper() + rewritten[1:]
    
    logger.info(f"[REWRITE] '{original}' → '{rewritten}'")
    
    return RewriteResult(
        original_query=original,
        rewritten_query=rewritten,
        was_rewritten=True,
        is_numeric_query=is_numeric_query(rewritten),
        anchor_title=anchor_title,
        anchor_time_range=anchor_time_range,
        reason="Injected topic anchor context"
    )


def process_query(
    query_text: str,
    active_anchor: Optional[object] = None  # TopicAnchor
) -> RewriteResult:
    """
    Main entry point: analyze query and rewrite if needed.
    
    Args:
        query_text: User's original query
        active_anchor: Active TopicAnchor from session (if any)
        
    Returns:
        RewriteResult with rewriting decision
    """
    has_anchor = active_anchor is not None
    
    # Check if query needs rewriting
    if is_ambiguous_short_query(query_text, has_anchor) and active_anchor:
        anchor_title = getattr(active_anchor, 'canonical_title', str(active_anchor))
        return rewrite_with_anchor(query_text, anchor_title)
    
    # No rewriting needed
    return RewriteResult(
        original_query=query_text,
        rewritten_query=query_text,
        was_rewritten=False,
        is_numeric_query=is_numeric_query(query_text),
        anchor_title=getattr(active_anchor, 'canonical_title', None) if active_anchor else None,
        reason="Query not ambiguous or no anchor"
    )


# ============================================================================
# Web Result Filtering
# ============================================================================

def validate_web_result_for_anchor(
    result_text: str,
    result_url: str,
    anchor_title: str,
    query_is_numeric: bool = False
) -> Tuple[bool, str]:
    """
    Validate if a web result is relevant to the active anchor topic.
    
    CRITICAL: Prevents unrelated contemporary statistics from appearing.
    
    Returns:
        (is_valid, reason)
    """
    if not anchor_title:
        return True, "No anchor to validate against"
    
    text_lower = result_text.lower()
    anchor_lower = anchor_title.lower()
    
    # Check if anchor topic appears in the result
    if anchor_lower not in text_lower:
        # For numeric queries, we're strict
        if query_is_numeric:
            return False, f"Numeric query result missing anchor topic '{anchor_title}'"
        # For non-numeric, be more lenient but still flag
        logger.warning(f"[FILTER] Result missing anchor topic: {result_url[:50]}")
    
    # Check for unrelated contemporary topics
    UNRELATED_CONTEMPORARY = [
        'transgender', 'lgbtq', 'covid', 'pandemic 2020',
        'trump', 'biden', 'climate change', 'cryptocurrency',
        'tiktok', 'twitter', 'facebook', 'instagram',
    ]
    
    for term in UNRELATED_CONTEMPORARY:
        if term in text_lower and anchor_lower not in text_lower:
            return False, f"Unrelated contemporary content: '{term}'"
    
    # For numeric queries, require anchor + number in same paragraph
    if query_is_numeric:
        # Check if anchor and a number appear close together
        paragraphs = result_text.split('\n\n')
        for para in paragraphs:
            para_lower = para.lower()
            if anchor_lower in para_lower and re.search(r'\d{3,}', para):
                return True, "Anchor and numeric in same paragraph"
        
        # No paragraph with both anchor and number
        logger.debug(f"[FILTER] No paragraph with anchor+number: {result_url[:50]}")
        # Still allow if anchor is present anywhere
        if anchor_lower in text_lower:
            return True, "Anchor present in text"
        return False, "Numeric query but no anchor+number context"
    
    return True, "Passed validation"


# ============================================================================
# Numeric Extraction
# ============================================================================

@dataclass
class NumericCandidate:
    """A numeric value extracted from a source."""
    value: int
    range_min: Optional[int] = None
    range_max: Optional[int] = None
    unit: str = ""
    source_url: str = ""
    source_type: str = ""  # wikipedia, classroom, academic, web
    snippet: str = ""
    extraction_confidence: float = 0.5
    
    def to_dict(self) -> dict:
        result = {
            "value": self.value,
            "unit": self.unit,
            "source_url": self.source_url,
            "source_type": self.source_type,
            "snippet": self.snippet[:200] if self.snippet else "",
            "extraction_confidence": round(self.extraction_confidence, 2),
        }
        if self.range_min and self.range_max:
            result["range"] = [self.range_min, self.range_max]
        return result


def extract_numeric_from_text(
    text: str,
    source_url: str = "",
    source_type: str = "web"
) -> List[NumericCandidate]:
    """
    Extract numeric values from text.
    
    Handles:
    - Plain numbers: 40,000
    - Ranges: 25,000 to 40,000
    - Approximations: ~30,000, circa 30,000, approximately 30,000
    """
    candidates = []
    
    # Pattern for numbers with optional commas
    NUMBER = r'[\d,]+(?:\.\d+)?'
    
    patterns = [
        # Range patterns
        (rf'(?:between\s+)?({NUMBER})\s*(?:to|and|-|–)\s*({NUMBER})\s*(people|deaths|casualties|killed)?', 'range'),
        # Approximation patterns  
        (rf'(?:approximately|about|around|roughly|circa|c\.?|~|est\.?)\s*({NUMBER})\s*(people|deaths|casualties|killed)?', 'approx'),
        # Plain number with context
        (rf'({NUMBER})\s*(people|deaths|casualties|killed|dead|victims)', 'plain'),
    ]
    
    for pattern, pattern_type in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            try:
                if pattern_type == 'range':
                    min_val = int(match.group(1).replace(',', ''))
                    max_val = int(match.group(2).replace(',', ''))
                    mid_val = (min_val + max_val) // 2
                    unit = match.group(3) or 'people'
                    
                    # Get context around match
                    start = max(0, match.start() - 50)
                    end = min(len(text), match.end() + 50)
                    snippet = text[start:end]
                    
                    candidates.append(NumericCandidate(
                        value=mid_val,
                        range_min=min_val,
                        range_max=max_val,
                        unit=unit.lower() if unit else 'people',
                        source_url=source_url,
                        source_type=source_type,
                        snippet=snippet,
                        extraction_confidence=0.85
                    ))
                    
                elif pattern_type == 'approx':
                    val = int(match.group(1).replace(',', ''))
                    unit = match.group(2) or 'people'
                    
                    start = max(0, match.start() - 50)
                    end = min(len(text), match.end() + 50)
                    snippet = text[start:end]
                    
                    candidates.append(NumericCandidate(
                        value=val,
                        unit=unit.lower() if unit else 'people',
                        source_url=source_url,
                        source_type=source_type,
                        snippet=snippet,
                        extraction_confidence=0.75
                    ))
                    
                elif pattern_type == 'plain':
                    val = int(match.group(1).replace(',', ''))
                    unit = match.group(2) or ''
                    
                    start = max(0, match.start() - 50)
                    end = min(len(text), match.end() + 50)
                    snippet = text[start:end]
                    
                    candidates.append(NumericCandidate(
                        value=val,
                        unit=unit.lower() if unit else 'people',
                        source_url=source_url,
                        source_type=source_type,
                        snippet=snippet,
                        extraction_confidence=0.65
                    ))
                    
            except (ValueError, IndexError) as e:
                logger.debug(f"[NUMERIC] Parse error: {e}")
                continue
    
    return candidates
