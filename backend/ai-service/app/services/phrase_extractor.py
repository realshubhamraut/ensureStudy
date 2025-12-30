"""
Phrase Extractor - Key phrase extraction for suggestion generation

Extracts meaningful phrases from context chunks using:
1. Noun phrase patterns (regex-based, no spaCy dependency)
2. TF-IDF top terms
3. Named entity recognition (regex patterns)
4. Formula detection (LaTeX patterns)

All methods are deterministic and CPU-friendly.
"""
import re
import hashlib
import logging
from typing import List, Set, Tuple
from dataclasses import dataclass
from collections import Counter

logger = logging.getLogger(__name__)


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class ExtractedPhrase:
    """An extracted key phrase with metadata."""
    text: str
    source: str  # "noun_phrase", "tfidf", "entity", "formula"
    frequency: int = 1
    importance_score: float = 1.0
    
    @property
    def hash(self) -> str:
        """Compute unique hash for deduplication."""
        return hashlib.sha256(self.text.lower().encode()).hexdigest()[:16]


# ============================================================================
# Simple Tokenizer
# ============================================================================

def _tokenize(text: str) -> List[str]:
    """Simple word tokenization."""
    # Remove special characters except apostrophes
    text = re.sub(r"[^\w\s'-]", " ", text)
    # Split on whitespace
    tokens = text.lower().split()
    # Filter very short tokens
    return [t for t in tokens if len(t) > 2]


def _get_stopwords() -> Set[str]:
    """Return a set of common English stopwords."""
    return {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "as", "is", "was", "are", "were", "been",
        "be", "have", "has", "had", "do", "does", "did", "will", "would", 
        "could", "should", "may", "might", "must", "shall", "can", "need",
        "this", "that", "these", "those", "it", "its", "they", "them", "their",
        "we", "you", "he", "she", "him", "her", "his", "hers", "our", "your",
        "what", "which", "who", "whom", "whose", "when", "where", "why", "how",
        "all", "each", "every", "both", "few", "more", "most", "other", "some",
        "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too",
        "very", "just", "also", "then", "now", "here", "there", "about", "into",
        "through", "during", "before", "after", "above", "below", "between",
        "under", "again", "further", "once", "any", "because", "if", "while",
        "being", "having", "doing", "until", "against", "among", "over", "out",
        "etc", "example", "examples", "using", "used", "like", "called", "known",
        "given", "based", "include", "includes", "including", "one", "two", "first"
    }


# ============================================================================
# Noun Phrase Extraction (Regex-based)
# ============================================================================

# Simple noun phrase patterns (adjective* noun+)
NOUN_PHRASE_PATTERNS = [
    r'\b(?:[A-Z][a-z]+\s+)+(?:theorem|law|principle|equation|formula|rule|concept|method)\b',  # Named theorems
    r'\b(?:the\s+)?(?:\w+\s+){1,2}(?:of|for)\s+(?:\w+\s*)+\b',  # X of Y patterns
    r'\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',  # Proper noun sequences
    r'\b(?:\w+(?:ed|ing|tion|sion|ment|ness|ity))\s+\w+\b',  # Complex noun phrases
]


def extract_noun_phrases(text: str, max_phrases: int = 10) -> List[str]:
    """
    Extract noun-like phrases using regex patterns.
    
    This is a lightweight alternative to spaCy noun_chunks.
    """
    phrases = []
    stopwords = _get_stopwords()
    
    # Pattern 1: Capitalized sequences (proper nouns, theorems)
    caps_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+(?:theorem|law|principle|equation|formula|rule))?\b)'
    for match in re.finditer(caps_pattern, text):
        phrase = match.group(1).strip()
        if len(phrase) > 3 and phrase.lower() not in stopwords:
            phrases.append(phrase)
    
    # Pattern 2: "the X of Y" structures
    of_pattern = r'\bthe\s+(\w+(?:\s+\w+)?)\s+of\s+(\w+(?:\s+\w+)?)\b'
    for match in re.finditer(of_pattern, text.lower()):
        combined = f"{match.group(1)} of {match.group(2)}"
        if len(combined) > 5:
            phrases.append(combined)
    
    # Pattern 3: Technical term patterns
    tech_pattern = r'\b(\w+(?:tion|sion|ment|ity|ness|ance|ence|ism|ist|ous|ive)\s+\w+)\b'
    for match in re.finditer(tech_pattern, text.lower()):
        phrase = match.group(1)
        if len(phrase) > 5:
            phrases.append(phrase)
    
    # Deduplicate and limit
    seen = set()
    result = []
    for p in phrases:
        key = p.lower().strip()
        if key not in seen and len(key) > 3:
            seen.add(key)
            result.append(p.strip())
            if len(result) >= max_phrases:
                break
    
    return result


# ============================================================================
# TF-IDF Top Terms
# ============================================================================

def extract_tfidf_terms(
    texts: List[str], 
    max_terms: int = 8,
    min_df: int = 1
) -> List[Tuple[str, float]]:
    """
    Extract top terms using simple TF-IDF scoring.
    
    This is a lightweight implementation without sklearn.
    """
    stopwords = _get_stopwords()
    
    # Tokenize all texts
    all_tokens = []
    doc_tokens = []
    for text in texts:
        tokens = _tokenize(text)
        filtered = [t for t in tokens if t not in stopwords and len(t) > 3]
        doc_tokens.append(set(filtered))
        all_tokens.extend(filtered)
    
    if not all_tokens:
        return []
    
    # Term frequency
    tf = Counter(all_tokens)
    
    # Document frequency
    df = Counter()
    for doc in doc_tokens:
        for term in doc:
            df[term] += 1
    
    # TF-IDF (simplified)
    num_docs = len(texts)
    import math
    tfidf_scores = {}
    for term, freq in tf.items():
        if df[term] >= min_df:
            idf = math.log((num_docs + 1) / (df[term] + 1)) + 1
            tfidf_scores[term] = freq * idf
    
    # Sort by score
    sorted_terms = sorted(tfidf_scores.items(), key=lambda x: -x[1])
    return sorted_terms[:max_terms]


# ============================================================================
# Named Entity Patterns
# ============================================================================

ENTITY_PATTERNS = {
    "PERSON": r'\b([A-Z][a-z]+(?:\s+[A-Z]\.?\s*)?[A-Z][a-z]+)\b',  # John Smith, J. Smith
    "THEOREM": r'\b([A-Z][a-z]+(?:\'s)?\s+(?:theorem|law|principle|equation|conjecture|lemma))\b',
    "FORMULA": r'\b([A-Z][a-z]+(?:\'s)?\s+(?:formula|rule|identity|inequality))\b',
    "CONCEPT": r'\b([A-Z][a-z]+\s+(?:number|function|constant|series|sequence|space|group|ring|field)s?)\b',
}


def extract_named_entities(text: str) -> List[Tuple[str, str]]:
    """
    Extract named entities using regex patterns.
    
    Returns list of (entity_text, entity_type) tuples.
    """
    entities = []
    
    for entity_type, pattern in ENTITY_PATTERNS.items():
        for match in re.finditer(pattern, text):
            entity = match.group(1).strip()
            if len(entity) > 3:
                entities.append((entity, entity_type))
    
    return entities


# ============================================================================
# Formula Detection
# ============================================================================

FORMULA_PATTERNS = [
    r'\$([^$]+)\$',  # Inline LaTeX
    r'\\\(([^)]+)\\\)',  # LaTeX math
    r'\b([a-z]\s*[²³⁴⁵⁶⁷⁸⁹])\b',  # Superscript notation
    r'\b([a-z]\^[0-9]+)\b',  # a^2 notation
    r'\b(\w+\s*=\s*[^.]+?)(?:\.|,|$)',  # Equation patterns like "E = mc²"
]


def extract_formulas(text: str) -> List[str]:
    """Extract mathematical formulas from text."""
    formulas = []
    
    for pattern in FORMULA_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            formula = match.group(1).strip()
            if len(formula) > 2 and len(formula) < 50:
                formulas.append(formula)
    
    return list(set(formulas))


# ============================================================================
# Main Extraction Function
# ============================================================================

def extract_key_phrases(
    context_chunks: List[str],
    max_phrases: int = 8,
    include_formulas: bool = True
) -> List[ExtractedPhrase]:
    """
    Extract key phrases from context chunks.
    
    Combines multiple extraction methods:
    - Noun phrases (regex-based)
    - TF-IDF top terms
    - Named entities
    - Formulas (if enabled)
    
    Args:
        context_chunks: List of text chunks
        max_phrases: Maximum phrases to return
        include_formulas: Whether to include formula extraction
        
    Returns:
        List of ExtractedPhrase objects, deduplicated and ranked
    """
    combined_text = " ".join(context_chunks)
    
    phrases: List[ExtractedPhrase] = []
    seen_hashes = set()
    
    # 1. Noun phrases
    for np in extract_noun_phrases(combined_text, max_phrases):
        phrase = ExtractedPhrase(text=np, source="noun_phrase", importance_score=1.0)
        if phrase.hash not in seen_hashes:
            seen_hashes.add(phrase.hash)
            phrases.append(phrase)
    
    # 2. TF-IDF terms
    for term, score in extract_tfidf_terms(context_chunks, max_phrases):
        phrase = ExtractedPhrase(text=term, source="tfidf", importance_score=score)
        if phrase.hash not in seen_hashes:
            seen_hashes.add(phrase.hash)
            phrases.append(phrase)
    
    # 3. Named entities
    for entity, etype in extract_named_entities(combined_text):
        phrase = ExtractedPhrase(text=entity, source="entity", importance_score=1.2)
        if phrase.hash not in seen_hashes:
            seen_hashes.add(phrase.hash)
            phrases.append(phrase)
    
    # 4. Formulas
    if include_formulas:
        for formula in extract_formulas(combined_text):
            phrase = ExtractedPhrase(text=formula, source="formula", importance_score=1.1)
            if phrase.hash not in seen_hashes:
                seen_hashes.add(phrase.hash)
                phrases.append(phrase)
    
    # Sort by importance and limit
    phrases.sort(key=lambda p: -p.importance_score)
    
    logger.debug(f"[EXTRACT] Extracted {len(phrases)} phrases from {len(context_chunks)} chunks")
    
    return phrases[:max_phrases]
