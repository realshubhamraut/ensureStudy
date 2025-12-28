"""
Model Context Protocol (MCP) - Context Assembly

CRITICAL FUNCTION: Controls exactly what the LLM sees.

The LLM NEVER sees:
- Entire documents
- Raw databases
- User history dump

The LLM ONLY sees:
- Selected chunks from retrieval
- The student question
- Clear instructions

MCP Responsibilities:
1. Receive retrieved chunks
2. Filter by relevance score
3. Enforce max token budget (1500-2000 tokens)
4. Order chunks logically: Definition → Explanation → Example
5. Produce a single context_text string
"""
from typing import List, Optional

from ..api.schemas.tutor import RetrievedChunk, AssembledContext, ResponseMode
from ..config import settings


# ============================================================================
# Token Counting (Simple, No External Dependencies)
# ============================================================================

def count_tokens(text: str) -> int:
    """
    Estimate token count.
    Rule of thumb: 1 token ≈ 4 characters for English text.
    """
    return len(text) // 4


# ============================================================================
# Deduplication
# ============================================================================

def _compute_overlap(text1: str, text2: str) -> float:
    """Compute Jaccard similarity between two texts."""
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    if not words1 or not words2:
        return 0.0
    
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    
    return intersection / union if union > 0 else 0.0


def deduplicate_chunks(chunks: List[RetrievedChunk], threshold: float = 0.6) -> List[RetrievedChunk]:
    """
    Remove chunks with high content overlap.
    Keeps the chunk with higher similarity score.
    """
    if not chunks:
        return []
    
    sorted_chunks = sorted(chunks, key=lambda c: c.similarity_score, reverse=True)
    
    kept = []
    for chunk in sorted_chunks:
        is_duplicate = False
        for kept_chunk in kept:
            overlap = _compute_overlap(chunk.text, kept_chunk.text)
            if overlap >= threshold:
                is_duplicate = True
                break
        
        if not is_duplicate:
            kept.append(chunk)
    
    return kept


# ============================================================================
# Content Ordering
# ============================================================================

def _classify_chunk_type(chunk: RetrievedChunk) -> int:
    """
    Classify chunk for ordering: Definition → Explanation → Example
    Returns: 0=definition, 1=explanation, 2=example, 3=other
    """
    text_lower = chunk.text.lower()
    chunk_type = chunk.metadata.get("type", "")
    
    # Check metadata type first
    if chunk_type == "example":
        return 2
    if chunk_type == "definition":
        return 0
    
    # Check text content
    if any(x in text_lower for x in ["example:", "for instance", "consider the case"]):
        return 2
    
    if any(x in text_lower for x in ["=", "formula", "equation", "f = m", "given by"]):
        return 1
    
    if any(x in text_lower for x in ["is defined as", "refers to", "known as", "states that", "is the process"]):
        return 0
    
    return 1  # Default to explanation


def order_chunks(chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
    """
    Order chunks logically: Definition → Explanation → Example
    """
    definitions = []
    explanations = []
    examples = []
    
    for chunk in chunks:
        chunk_type = _classify_chunk_type(chunk)
        if chunk_type == 0:
            definitions.append(chunk)
        elif chunk_type == 2:
            examples.append(chunk)
        else:
            explanations.append(chunk)
    
    return definitions + explanations + examples


# ============================================================================
# Main Context Assembly Function (MCP)
# ============================================================================

def build_context(
    retrieved_chunks: List[RetrievedChunk],
    response_mode: ResponseMode = ResponseMode.SHORT,
    student_metadata: Optional[dict] = None
) -> AssembledContext:
    """
    MODEL CONTEXT PROTOCOL - Controls what the LLM sees.
    
    This function is CRITICAL for:
    - Preventing hallucination
    - Ensuring grounded answers
    - Controlling token usage
    
    Args:
        retrieved_chunks: Chunks from Qdrant search
        response_mode: short (1500 tokens) or detailed (2000 tokens)
        student_metadata: Optional student info (grade level, etc.)
        
    Returns:
        AssembledContext with:
        - context_text: What the LLM will see
        - chunks_used: Which chunks were included
        - total_tokens: Token count
    """
    if not retrieved_chunks:
        return AssembledContext(
            context_text="No relevant study materials found.",
            chunks_used=[],
            total_tokens=10
        )
    
    # Step 1: Determine token budget
    max_tokens = (
        settings.SHORT_MODE_TOKENS 
        if response_mode == ResponseMode.SHORT 
        else settings.MAX_CONTEXT_TOKENS
    )
    
    # Step 2: Filter by relevance score (already applied in retrieval, but double-check)
    relevant_chunks = [c for c in retrieved_chunks if c.similarity_score >= settings.SIMILARITY_THRESHOLD]
    
    if not relevant_chunks:
        relevant_chunks = retrieved_chunks[:3]  # Fallback to top 3
    
    # Step 3: Deduplicate
    unique_chunks = deduplicate_chunks(relevant_chunks)
    
    # Step 4: Order logically (Definition → Explanation → Example)
    ordered_chunks = order_chunks(unique_chunks)
    
    # Step 5: Build context within token budget
    context_parts = []
    used_chunks = []
    current_tokens = 0
    
    for chunk in ordered_chunks:
        # Format chunk with source reference
        chunk_text = f"[{chunk.chunk_id}] {chunk.text}"
        chunk_tokens = count_tokens(chunk_text)
        
        # Check token budget
        if current_tokens + chunk_tokens > max_tokens:
            if not used_chunks:
                # Must include at least one chunk
                chunk_text = chunk_text[:max_tokens * 4]
                context_parts.append(chunk_text)
                used_chunks.append(chunk)
            break
        
        context_parts.append(chunk_text)
        used_chunks.append(chunk)
        current_tokens += chunk_tokens
    
    # Step 6: Assemble final context
    context_text = "\n\n".join(context_parts)
    total_tokens = count_tokens(context_text)
    
    return AssembledContext(
        context_text=context_text,
        chunks_used=used_chunks,
        total_tokens=total_tokens
    )


# Alias for backward compatibility
assemble_context = build_context
