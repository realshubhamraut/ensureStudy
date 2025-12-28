"""
Content Normalizer Service

Extracts and normalizes educational content from web pages.
Designed for academic knowledge ingestion.

EXTRACTS:
- Definitions
- Formulas (LaTeX preserved)
- Explanations
- Examples
- Tables

REMOVES:
- Ads, navigation, sidebars
- Scripts, styles
- Comments, metadata
- Non-educational content
"""
import re
import html
import unicodedata
from typing import Optional, List, Dict, Any
from dataclasses import dataclass


@dataclass
class NormalizedContent:
    """Normalized educational content."""
    text: str
    title: str
    summary: str
    word_count: int
    has_formulas: bool
    has_tables: bool
    has_code: bool
    sections: List[str]
    metadata: Dict[str, Any]


# ============================================================================
# Content Cleaning Rules
# ============================================================================

# Patterns to remove (noise)
REMOVE_PATTERNS = [
    r'<script[^>]*>.*?</script>',
    r'<style[^>]*>.*?</style>',
    r'<nav[^>]*>.*?</nav>',
    r'<footer[^>]*>.*?</footer>',
    r'<header[^>]*>.*?</header>',
    r'<aside[^>]*>.*?</aside>',
    r'<!--.*?-->',
    r'\[edit\]',
    r'\[citation needed\]',
    r'\[\d+\]',  # Reference numbers [1], [2], etc.
]

# Patterns to preserve (educational content)
PRESERVE_PATTERNS = {
    'formulas': r'\$.*?\$|\\\(.*?\\\)|\\\[.*?\\\]',
    'code': r'```.*?```|<code>.*?</code>',
    'tables': r'<table>.*?</table>',
}

# Educational keywords to boost relevance
EDUCATIONAL_KEYWORDS = [
    'definition', 'formula', 'equation', 'theorem', 'proof',
    'example', 'solution', 'step', 'method', 'process',
    'concept', 'principle', 'law', 'rule', 'property'
]


# ============================================================================
# Normalization Functions
# ============================================================================

def clean_html_content(html_content: str) -> str:
    """
    Remove HTML noise and extract clean text.
    
    Args:
        html_content: Raw HTML string
        
    Returns:
        Clean text content
    """
    text = html_content
    
    # Remove script, style, nav elements
    for pattern in REMOVE_PATTERNS:
        text = re.sub(pattern, '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove remaining HTML tags but preserve structure
    text = re.sub(r'<br\s*/?>', '\n', text)
    text = re.sub(r'<p[^>]*>', '\n\n', text)
    text = re.sub(r'<h[1-6][^>]*>', '\n\n## ', text)
    text = re.sub(r'</h[1-6]>', '\n', text)
    text = re.sub(r'<li[^>]*>', '\n• ', text)
    text = re.sub(r'<[^>]+>', '', text)
    
    # Decode HTML entities
    text = html.unescape(text)
    
    return text


def normalize_unicode(text: str) -> str:
    """
    Normalize Unicode characters.
    
    - Converts fancy quotes to standard
    - Normalizes dashes
    - Removes zero-width characters
    """
    # Normalize to NFC form
    text = unicodedata.normalize('NFC', text)
    
    # Replace fancy quotes
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace(''', "'").replace(''', "'")
    
    # Normalize dashes
    text = text.replace('–', '-').replace('—', '-')
    
    # Remove zero-width characters
    text = text.replace('\u200b', '').replace('\u200c', '').replace('\u200d', '')
    text = text.replace('\ufeff', '')
    
    return text


def normalize_whitespace(text: str) -> str:
    """
    Clean up whitespace.
    
    - Collapse multiple spaces
    - Normalize line breaks
    - Trim lines
    """
    # Collapse multiple spaces
    text = re.sub(r'[ \t]+', ' ', text)
    
    # Normalize line breaks
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    
    # Trim each line
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)
    
    return text.strip()


def extract_sections(text: str) -> List[str]:
    """
    Extract section headers from text.
    """
    sections = []
    
    # Find headers (lines starting with ##)
    for line in text.split('\n'):
        if line.strip().startswith('##'):
            section = line.strip().lstrip('#').strip()
            if section and len(section) > 2:
                sections.append(section)
    
    return sections


def detect_content_features(text: str) -> Dict[str, bool]:
    """
    Detect special content features.
    """
    return {
        'has_formulas': bool(re.search(r'\$.*?\$|\\[()\[\]]', text)),
        'has_tables': bool(re.search(r'\|.*\|.*\|', text)),
        'has_code': bool(re.search(r'```|`[^`]+`', text)),
        'has_lists': bool(re.search(r'^\s*[•\-\*]\s', text, re.MULTILINE)),
        'has_definitions': bool(re.search(r'definition|defined as|is the', text, re.IGNORECASE))
    }


def calculate_educational_score(text: str) -> float:
    """
    Calculate how educational the content is (0.0 to 1.0).
    """
    text_lower = text.lower()
    
    # Count educational keywords
    keyword_count = sum(1 for kw in EDUCATIONAL_KEYWORDS if kw in text_lower)
    
    # Calculate score
    base_score = min(keyword_count / 10, 0.5)  # Max 0.5 from keywords
    
    # Bonus for structured content
    features = detect_content_features(text)
    if features['has_formulas']:
        base_score += 0.15
    if features['has_definitions']:
        base_score += 0.15
    if features['has_lists']:
        base_score += 0.1
    if features['has_code']:
        base_score += 0.1
    
    return min(base_score, 1.0)


# ============================================================================
# Main Normalization Function
# ============================================================================

def normalize_content(
    raw_content: str,
    title: str = "",
    source_type: str = "html"
) -> NormalizedContent:
    """
    Normalize raw web content into clean educational text.
    
    Args:
        raw_content: Raw HTML or text content
        title: Page title
        source_type: 'html' or 'text'
        
    Returns:
        NormalizedContent object
    """
    # Step 1: Clean HTML if needed
    if source_type == 'html':
        text = clean_html_content(raw_content)
    else:
        text = raw_content
    
    # Step 2: Normalize Unicode
    text = normalize_unicode(text)
    
    # Step 3: Normalize whitespace
    text = normalize_whitespace(text)
    
    # Step 4: Extract features
    sections = extract_sections(text)
    features = detect_content_features(text)
    
    # Step 5: Calculate word count
    word_count = len(text.split())
    
    # Step 6: Generate summary (first 500 chars)
    summary = text[:500].rsplit(' ', 1)[0] + '...' if len(text) > 500 else text
    
    return NormalizedContent(
        text=text,
        title=title,
        summary=summary,
        word_count=word_count,
        has_formulas=features['has_formulas'],
        has_tables=features['has_tables'],
        has_code=features['has_code'],
        sections=sections,
        metadata={
            'educational_score': calculate_educational_score(text),
            'source_type': source_type
        }
    )


def normalize_pdf_content(text: str, title: str = "") -> NormalizedContent:
    """
    Normalize PDF-extracted text.
    """
    # PDF text often has strange line breaks
    text = re.sub(r'(?<![.!?])\n(?=[a-z])', ' ', text)
    
    return normalize_content(text, title, source_type='text')
