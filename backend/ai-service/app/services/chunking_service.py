"""
Chunking Service

Token-based text chunking with overlap for RAG systems.
Designed for educational content with semantic boundary detection.

FEATURES:
- 300-800 token chunks (configurable)
- 20-30% overlap for context preservation
- Semantic boundary detection (paragraphs, sections)
- Metadata preservation per chunk
- LIMITED to 15000 chars max for performance
"""
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# Max chars to process (prevents slow chunking on huge pages)
MAX_CONTENT_CHARS = 15000

# Try to use tiktoken for accurate tokenization
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False


@dataclass
class Chunk:
    """A text chunk with metadata."""
    text: str
    chunk_index: int
    total_chunks: int
    token_count: int
    char_count: int
    start_char: int
    end_char: int
    metadata: Dict[str, Any]


# ============================================================================
# Tokenization
# ============================================================================

def get_tokenizer():
    """Get tiktoken tokenizer (cl100k_base for GPT-4 compatibility)."""
    if TIKTOKEN_AVAILABLE:
        return tiktoken.get_encoding("cl100k_base")
    return None


def count_tokens(text: str) -> int:
    """
    Count tokens in text.
    Falls back to word count * 1.3 if tiktoken unavailable.
    """
    tokenizer = get_tokenizer()
    if tokenizer:
        return len(tokenizer.encode(text))
    # Fallback: approximate based on words
    return int(len(text.split()) * 1.3)


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate text to max tokens."""
    tokenizer = get_tokenizer()
    if tokenizer:
        tokens = tokenizer.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return tokenizer.decode(tokens[:max_tokens])
    
    # Fallback: approximate
    words = text.split()
    max_words = int(max_tokens / 1.3)
    return ' '.join(words[:max_words])


# ============================================================================
# Semantic Boundary Detection
# ============================================================================

def find_semantic_boundaries(text: str) -> List[int]:
    """
    Find semantic boundaries in text (paragraph breaks, section headers).
    
    Returns list of character positions where splits are preferred.
    """
    boundaries = []
    
    # Paragraph breaks (double newlines)
    for match in re.finditer(r'\n\n+', text):
        boundaries.append(match.end())
    
    # Section headers (## or **bold**)
    for match in re.finditer(r'^##.+$', text, re.MULTILINE):
        boundaries.append(match.start())
    
    # Sentence endings followed by newline
    for match in re.finditer(r'[.!?]\n', text):
        boundaries.append(match.end())
    
    return sorted(set(boundaries))


def find_nearest_boundary(text: str, target_pos: int, boundaries: List[int], tolerance: int = 100) -> int:
    """
    Find the nearest semantic boundary to target position.
    
    Args:
        text: Full text
        target_pos: Target character position
        boundaries: List of boundary positions
        tolerance: How far to search for boundary
        
    Returns:
        Best split position
    """
    # Find boundaries within tolerance
    candidates = [b for b in boundaries if abs(b - target_pos) <= tolerance]
    
    if candidates:
        # Return closest boundary
        return min(candidates, key=lambda x: abs(x - target_pos))
    
    # No boundary found, try to find sentence end
    search_start = max(0, target_pos - tolerance)
    search_end = min(len(text), target_pos + tolerance)
    search_text = text[search_start:search_end]
    
    # Look for sentence end
    for match in re.finditer(r'[.!?]\s', search_text):
        return search_start + match.end()
    
    # Fallback to word boundary
    space_pos = text.rfind(' ', search_start, target_pos)
    if space_pos > search_start:
        return space_pos + 1
    
    return target_pos


# ============================================================================
# Main Chunking Function
# ============================================================================

def chunk_text(
    text: str,
    min_tokens: int = 300,
    max_tokens: int = 800,
    overlap_percent: float = 0.25,
    metadata: Optional[Dict[str, Any]] = None
) -> List[Chunk]:
    """
    Chunk text into overlapping segments.
    OPTIMIZED: Fast character-based splitting without heavy regex.
    """
    if not text or not text.strip():
        return []
    
    text = text.strip()
    
    # Limit text size for performance
    if len(text) > MAX_CONTENT_CHARS:
        text = text[:MAX_CONTENT_CHARS]
        last_period = text.rfind('.')
        if last_period > MAX_CONTENT_CHARS * 0.8:
            text = text[:last_period + 1]
    
    # Fast word + token estimation
    word_count = len(text.split())
    total_tokens_est = int(word_count * 1.3)
    
    # If text fits in one chunk, return as-is
    if total_tokens_est <= max_tokens:
        return [Chunk(
            text=text,
            chunk_index=0,
            total_chunks=1,
            token_count=total_tokens_est,
            char_count=len(text),
            start_char=0,
            end_char=len(text),
            metadata=metadata or {}
        )]
    
    # Simple char-based chunk parameters (fast)
    target_tokens = (min_tokens + max_tokens) // 2
    chars_per_token = len(text) / max(total_tokens_est, 1)
    target_chars = int(target_tokens * chars_per_token)
    overlap_chars = int(target_chars * overlap_percent)
    
    chunks = []
    start_char = 0
    
    while start_char < len(text):
        end_char = min(start_char + target_chars, len(text))
        
        # Simple boundary: find nearest space
        if end_char < len(text):
            space_pos = text.rfind(' ', start_char + target_chars // 2, end_char + 100)
            if space_pos > start_char:
                end_char = space_pos
        
        chunk_text_content = text[start_char:end_char].strip()
        
        if chunk_text_content:
            chunk_token_est = int(len(chunk_text_content.split()) * 1.3)
            
            chunks.append(Chunk(
                text=chunk_text_content,
                chunk_index=len(chunks),
                total_chunks=0,
                token_count=chunk_token_est,
                char_count=len(chunk_text_content),
                start_char=start_char,
                end_char=end_char,
                metadata=metadata or {}
            ))
        
        # CRITICAL: Ensure we ALWAYS advance (prevent infinite loop)
        next_start = end_char - overlap_chars
        if next_start <= start_char:
            next_start = start_char + max(target_chars // 2, 100)  # Force advance
        start_char = next_start
        
        # Safety: max 100 chunks
        if len(chunks) >= 100:
            break
    
    # Update total_chunks
    total = len(chunks)
    for chunk in chunks:
        chunk.total_chunks = total
    
    return chunks


def chunk_for_qdrant(
    text: str,
    source_url: str,
    source_type: str,
    source_trust: float,
    min_tokens: int = 300,
    max_tokens: int = 800
) -> List[Dict[str, Any]]:
    """
    Chunk text and prepare for Qdrant storage.
    
    Returns list of dicts ready for embedding and storage.
    """
    from datetime import datetime
    
    chunks = chunk_text(
        text=text,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
        metadata={
            'source_url': source_url,
            'source_type': source_type,
            'source_trust': source_trust,
            'fetched_at': datetime.now().isoformat()
        }
    )
    
    return [
        {
            'text': chunk.text,
            'metadata': {
                **chunk.metadata,
                'chunk_index': chunk.chunk_index,
                'total_chunks': chunk.total_chunks,
                'token_count': chunk.token_count
            }
        }
        for chunk in chunks
    ]


def create_semantic_chunks(text: str, chunk_size: int = 500, overlap: int = 100) -> List[Dict[str, Any]]:
    """
    Wrapper for chunk_text to verify compatibility with MaterialIndexer.
    Returns list of dicts: {'text': ..., 'metadata': ...}
    """
    chunks = chunk_text(
        text=text,
        min_tokens=chunk_size // 2,
        max_tokens=chunk_size,
        overlap_percent=overlap / chunk_size
    )
    
    return [
        {
            "text": c.text,
            "metadata": {
                "chunk_index": c.chunk_index,
                "token_count": c.token_count
            }
        }
        for c in chunks
    ]
