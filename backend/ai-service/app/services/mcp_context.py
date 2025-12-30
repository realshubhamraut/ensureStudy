"""
MCP Context Assembly with Strict Anchor Enforcement

When a topic anchor is active and anchor_hits >= min_anchor_hits,
only anchor chunks are included in LLM context. Web chunks are blocked.

This prevents unrelated web content from contaminating topic-anchored responses.

Flow:
1. Separate chunks by source (anchor, classroom, curated, web)
2. If anchor active + sufficient anchor hits → use anchor chunks ONLY
3. If anchor active + insufficient hits → use classroom/curated, NO web
4. If no anchor → normal flow (classroom, curated, web)
"""
import os
import json
import logging
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

FEATURE_STRICT_ANCHOR = os.getenv("FEATURE_STRICT_ANCHOR", "true").lower() == "true"
ANCHOR_MIN_HITS = int(os.getenv("ANCHOR_MIN_HITS", "1"))
ANCHOR_PRIORITY_MIN_SIM = float(os.getenv("ANCHOR_PRIORITY_MIN_SIM", "0.35"))
TOKEN_BUDGET = int(os.getenv("MCP_TOKEN_BUDGET", "2048"))
ALLOW_WEB_ON_NEW_TOPIC = os.getenv("ALLOW_WEB_ON_NEW_TOPIC", "false").lower() == "true"


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class MCPChunk:
    """A chunk with MCP metadata."""
    id: str
    text: str
    source: str  # 'anchor', 'classroom', 'curated_global', 'web'
    similarity: float
    source_trust: float = 1.0
    topic_anchor_id: Optional[str] = None
    document_id: Optional[str] = None
    
    def score(self) -> float:
        """Compute ranking score."""
        return self.similarity * self.source_trust


@dataclass
class MCPAssemblyResult:
    """Result of MCP context assembly."""
    chunks: List[MCPChunk]
    reason: str
    anchor_hits: int
    web_filtered_count: int
    anchor_sufficient: bool
    allow_broadening_prompt: bool = False
    
    def to_dict(self) -> dict:
        return {
            "chunk_ids": [c.id for c in self.chunks],
            "chunk_sources": [c.source for c in self.chunks],
            "reason": self.reason,
            "anchor_hits": self.anchor_hits,
            "web_filtered_count": self.web_filtered_count,
            "anchor_sufficient": self.anchor_sufficient,
            "allow_broadening_prompt": self.allow_broadening_prompt,
        }


# ============================================================================
# Source Trust Scores
# ============================================================================

SOURCE_TRUST = {
    'anchor': 0.99,
    'classroom': 0.95,
    'curated_global': 0.90,
    'web': 0.60,
}


def get_source_trust(source: str) -> float:
    """Get trust score for a source type."""
    return SOURCE_TRUST.get(source, 0.5)


# ============================================================================
# MCP Context Assembly
# ============================================================================

def classify_chunk_source(chunk: dict, active_anchor_id: Optional[str] = None) -> str:
    """
    Classify a chunk's source type.
    
    Priority:
    1. If chunk has topic_anchor_id matching active anchor → 'anchor'
    2. If chunk has classroom_id → 'classroom'
    3. If chunk has curated=True or trusted domain → 'curated_global'
    4. Otherwise → 'web'
    """
    # Check for anchor match
    if active_anchor_id and chunk.get('topic_anchor_id') == active_anchor_id:
        return 'anchor'
    
    # Check source_type field
    source_type = chunk.get('source_type', '').lower()
    
    if source_type in ('session', 'classroom', 'uploaded', 'document'):
        return 'classroom'
    
    if source_type in ('curated', 'wikipedia', 'academic'):
        return 'curated_global'
    
    # Check for classroom markers
    if chunk.get('classroom_id') or chunk.get('document_id'):
        return 'classroom'
    
    # Check for curated markers
    if chunk.get('is_curated') or chunk.get('trusted_source'):
        return 'curated_global'
    
    # Check URL for known trusted domains
    url = chunk.get('source_url', '') or chunk.get('url', '')
    if url:
        trusted_domains = ['wikipedia.org', 'britannica.com', 'khanacademy.org', '.edu', '.gov']
        if any(domain in url.lower() for domain in trusted_domains):
            return 'curated_global'
    
    return 'web'


def assemble_mcp_context(
    chunks: List[dict],
    active_anchor: Optional[object] = None,
    session_id: str = "",
    request_id: str = "",
    allow_broadening: bool = False,
    config: Dict[str, Any] = None
) -> MCPAssemblyResult:
    """
    Assemble context for LLM with strict anchor enforcement.
    
    CRITICAL RULE:
    - If anchor active + anchor_hits >= min → ONLY anchor chunks
    - If anchor active + insufficient → classroom/curated only, NO web
    - Web chunks allowed only when no anchor OR explicit broadening consent
    
    Args:
        chunks: Raw chunks from retrieval
        active_anchor: Active TopicAnchor (if any)
        session_id: Session ID for logging
        request_id: Request ID for logging
        allow_broadening: User explicitly allowed web results
        config: Configuration overrides
        
    Returns:
        MCPAssemblyResult with selected chunks and metadata
    """
    config = config or {}
    min_anchor_hits = config.get('MIN_ANCHOR_HITS', ANCHOR_MIN_HITS)
    min_sim = config.get('ANCHOR_PRIORITY_MIN_SIM', ANCHOR_PRIORITY_MIN_SIM)
    token_budget = config.get('TOKEN_BUDGET', TOKEN_BUDGET)
    
    # Get anchor ID if active
    anchor_id = None
    anchor_title = None
    if active_anchor:
        anchor_id = getattr(active_anchor, 'id', None)
        anchor_title = getattr(active_anchor, 'canonical_title', None)
    
    # Convert to MCPChunk objects and classify sources
    mcp_chunks = []
    for chunk in chunks:
        source = classify_chunk_source(chunk, anchor_id)
        
        # Override to 'anchor' if content matches anchor title closely
        if active_anchor and anchor_title:
            content = chunk.get('content', '') or chunk.get('text', '')
            if anchor_title.lower() in content.lower():
                # High confidence this is anchor-related
                if source in ('classroom', 'curated_global'):
                    source = 'anchor'
        
        mcp_chunks.append(MCPChunk(
            id=chunk.get('id', f"chunk_{len(mcp_chunks)}"),
            text=chunk.get('content', '') or chunk.get('text', ''),
            source=source,
            similarity=chunk.get('similarity', 0.5),
            source_trust=get_source_trust(source),
            topic_anchor_id=chunk.get('topic_anchor_id'),
            document_id=chunk.get('document_id'),
        ))
    
    # Separate by source
    anchor_chunks = [c for c in mcp_chunks if c.source == 'anchor']
    classroom_chunks = [c for c in mcp_chunks if c.source == 'classroom']
    curated_chunks = [c for c in mcp_chunks if c.source == 'curated_global']
    web_chunks = [c for c in mcp_chunks if c.source == 'web']
    
    # Log retrieval counts
    log_mcp_event(session_id, request_id, 
        action="RETRIEVAL_COUNTS",
        counts={
            'anchor': len(anchor_chunks),
            'classroom': len(classroom_chunks),
            'curated': len(curated_chunks),
            'web': len(web_chunks),
        }
    )
    
    # ========================================
    # ANCHOR ACTIVE PATH
    # ========================================
    if active_anchor and FEATURE_STRICT_ANCHOR:
        # Filter anchor chunks by similarity threshold
        anchor_hits = [c for c in anchor_chunks if c.similarity >= min_sim]
        
        # Also promote high-sim classroom chunks to anchor status
        for c in classroom_chunks:
            if c.similarity >= min_sim + 0.1:  # Higher bar for promotion
                anchor_hits.append(c)
        
        if len(anchor_hits) >= min_anchor_hits:
            # SUFFICIENT ANCHOR HITS → USE ANCHOR ONLY
            selected = sorted(anchor_hits, key=lambda c: c.score(), reverse=True)
            selected = fit_to_token_budget(selected, token_budget)
            
            log_mcp_event(session_id, request_id,
                action="MCP_ASSEMBLE",
                reason="anchor_sufficient",
                anchor_hits=len(selected),
                web_filtered=len(web_chunks),
                selected_ids=[c.id for c in selected]
            )
            
            return MCPAssemblyResult(
                chunks=selected,
                reason="anchor_sufficient",
                anchor_hits=len(anchor_hits),
                web_filtered_count=len(web_chunks),
                anchor_sufficient=True
            )
        
        else:
            # INSUFFICIENT ANCHOR HITS → USE CLASSROOM/CURATED, NO WEB
            allowed = anchor_chunks + classroom_chunks + curated_chunks
            
            if not allowed:
                # No content at all - prompt for broadening
                log_mcp_event(session_id, request_id,
                    action="MCP_ASSEMBLE",
                    reason="anchor_insufficient_no_content",
                    anchor_hits=len(anchor_hits),
                    web_filtered=len(web_chunks)
                )
                
                return MCPAssemblyResult(
                    chunks=[],
                    reason="anchor_insufficient_no_content",
                    anchor_hits=len(anchor_hits),
                    web_filtered_count=len(web_chunks),
                    anchor_sufficient=False,
                    allow_broadening_prompt=True
                )
            
            selected = rank_and_trim(allowed, token_budget)
            
            log_mcp_event(session_id, request_id,
                action="MCP_ASSEMBLE",
                reason="anchor_insufficient_use_classroom_curated",
                anchor_hits=len(anchor_hits),
                classroom_hits=len(classroom_chunks),
                curated_hits=len(curated_chunks),
                web_filtered=len(web_chunks),
                selected_ids=[c.id for c in selected]
            )
            
            return MCPAssemblyResult(
                chunks=selected,
                reason="anchor_insufficient_use_classroom_curated",
                anchor_hits=len(anchor_hits),
                web_filtered_count=len(web_chunks),
                anchor_sufficient=False,
                allow_broadening_prompt=len(selected) == 0
            )
    
    # ========================================
    # NO ANCHOR PATH (or feature disabled)
    # ========================================
    # Normal flow: classroom → curated → web
    allowed = classroom_chunks + curated_chunks
    
    # Only add web if explicitly allowed or no anchor
    if allow_broadening or not active_anchor or ALLOW_WEB_ON_NEW_TOPIC:
        allowed += web_chunks
    
    selected = rank_and_trim(allowed, token_budget)
    
    log_mcp_event(session_id, request_id,
        action="MCP_ASSEMBLE",
        reason="no_anchor_normal_flow" if not active_anchor else "feature_disabled",
        counts={
            'classroom': len(classroom_chunks),
            'curated': len(curated_chunks),
            'web': len(web_chunks),
        },
        selected_ids=[c.id for c in selected]
    )
    
    return MCPAssemblyResult(
        chunks=selected,
        reason="no_anchor_normal_flow",
        anchor_hits=0,
        web_filtered_count=0 if allow_broadening else len(web_chunks),
        anchor_sufficient=True
    )


# ============================================================================
# Helper Functions
# ============================================================================

def rank_and_trim(chunks: List[MCPChunk], token_budget: int) -> List[MCPChunk]:
    """Rank chunks by score and fit to token budget."""
    sorted_chunks = sorted(chunks, key=lambda c: c.score(), reverse=True)
    return fit_to_token_budget(sorted_chunks, token_budget)


def fit_to_token_budget(chunks: List[MCPChunk], token_budget: int) -> List[MCPChunk]:
    """
    Select chunks that fit within token budget.
    
    Uses approximate token count (chars / 4).
    """
    selected = []
    total_tokens = 0
    
    for chunk in chunks:
        # Approximate token count
        chunk_tokens = len(chunk.text) // 4
        
        if total_tokens + chunk_tokens <= token_budget:
            selected.append(chunk)
            total_tokens += chunk_tokens
        else:
            # Budget full
            break
    
    return selected


def log_mcp_event(session_id: str, request_id: str, **payload):
    """Log structured MCP event."""
    event = {
        "ts": datetime.utcnow().isoformat(),
        "session_id": session_id[:8] if session_id else "none",
        "request_id": request_id,
        "component": "MCP",
        **payload
    }
    logger.info(f"[MCP] {json.dumps(event)}")
    print(f"[MCP] {event.get('action', 'event')}: {event.get('reason', '')} "
          f"anchor_hits={event.get('anchor_hits', 'N/A')} "
          f"web_filtered={event.get('web_filtered', 0)}")


# ============================================================================
# Safety Assertion
# ============================================================================

def assert_no_web_in_anchor_context(
    chunks: List[MCPChunk],
    active_anchor: Optional[object],
    anchor_hits: int
) -> bool:
    """
    Safety assertion: When anchor active + sufficient hits, no web chunks allowed.
    
    Returns True if safe, raises AssertionError if violated.
    """
    if not active_anchor:
        return True
    
    if anchor_hits < ANCHOR_MIN_HITS:
        return True
    
    web_in_context = [c for c in chunks if c.source == 'web']
    
    if web_in_context:
        logger.error(
            f"[MCP-SAFETY] WEB CHUNK IN ANCHOR CONTEXT! "
            f"anchor_hits={anchor_hits} web_count={len(web_in_context)}"
        )
        raise AssertionError(
            f"Safety violation: {len(web_in_context)} web chunks in anchor context"
        )
    
    return True


# ============================================================================
# Insufficient Anchor Response
# ============================================================================

INSUFFICIENT_ANCHOR_MESSAGE = (
    "I don't have enough specific information about this topic from your learning materials. "
    "Would you like me to search web resources for more information?"
)

def get_insufficient_anchor_response(anchor_title: str = "") -> str:
    """Get the response message when anchor content is insufficient."""
    if anchor_title:
        return (
            f"I don't have enough specific information about {anchor_title} "
            f"from your learning materials. Would you like me to search web resources?"
        )
    return INSUFFICIENT_ANCHOR_MESSAGE
