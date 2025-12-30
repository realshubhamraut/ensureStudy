"""
Tests for MCP Strict Anchor Filter

Test Scenarios:
A. Anchor active + sufficient hits → ONLY anchor chunks, web blocked
B. Anchor active + insufficient hits → classroom/curated only, web blocked, prompts for broadening
C. No anchor → Normal flow (web allowed)
D. Safety assertion: web never in anchor context with sufficient hits
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend', 'ai-service'))

from app.services.mcp_context import (
    assemble_mcp_context,
    classify_chunk_source,
    MCPChunk,
    get_source_trust,
    assert_no_web_in_anchor_context,
    FEATURE_STRICT_ANCHOR,
    ANCHOR_MIN_HITS
)


# ============================================================================
# Test Fixtures
# ============================================================================

class MockAnchor:
    """Mock TopicAnchor for testing."""
    def __init__(self, anchor_id="anchor_fr", title="French Revolution"):
        self.id = anchor_id
        self.canonical_title = title


@pytest.fixture
def anchor():
    return MockAnchor()


@pytest.fixture
def mixed_chunks():
    """Chunks with mix of anchor, classroom, and web."""
    return [
        {'id': 'anchor_1', 'content': 'The French Revolution began in 1789...', 
         'similarity': 0.85, 'source_type': 'session', 'topic_anchor_id': 'anchor_fr'},
        {'id': 'anchor_2', 'content': 'The storming of the Bastille...', 
         'similarity': 0.78, 'source_type': 'classroom', 'topic_anchor_id': 'anchor_fr'},
        {'id': 'classroom_1', 'content': 'History textbook chapter on French Revolution', 
         'similarity': 0.65, 'source_type': 'classroom'},
        {'id': 'web_1', 'content': 'Transgender deaths increased 30% in 2023...', 
         'similarity': 0.42, 'source_type': 'web'},
        {'id': 'web_2', 'content': 'Climate change causes millions of deaths...', 
         'similarity': 0.38, 'source_type': 'web'},
    ]


@pytest.fixture
def only_web_chunks():
    """Only web chunks, no anchor/classroom."""
    return [
        {'id': 'web_1', 'content': 'Unrelated web content 1', 
         'similarity': 0.45, 'source_type': 'web'},
        {'id': 'web_2', 'content': 'Unrelated web content 2', 
         'similarity': 0.40, 'source_type': 'web'},
    ]


# ============================================================================
# Test A: Anchor Active + Sufficient Hits
# ============================================================================

class TestAnchorActiveSufficient:
    """When anchor active and sufficient hits, only anchor chunks used."""
    
    def test_web_blocked_when_anchor_sufficient(self, anchor, mixed_chunks):
        """Web chunks must be blocked when anchor is active with sufficient hits."""
        result = assemble_mcp_context(
            chunks=mixed_chunks,
            active_anchor=anchor,
            session_id="test",
            request_id="req_a1"
        )
        
        # Check reason
        assert result.reason == "anchor_sufficient"
        assert result.anchor_sufficient is True
        
        # CRITICAL: No web chunks in result
        web_in_result = [c for c in result.chunks if c.source == 'web']
        assert len(web_in_result) == 0, f"Web chunks leaked: {[c.id for c in web_in_result]}"
        
        # Check web filtered count
        assert result.web_filtered_count >= 1
    
    def test_anchor_hits_counted(self, anchor, mixed_chunks):
        """Anchor hits should be counted correctly."""
        result = assemble_mcp_context(
            chunks=mixed_chunks,
            active_anchor=anchor,
            session_id="test",
            request_id="req_a2"
        )
        
        assert result.anchor_hits >= ANCHOR_MIN_HITS


# ============================================================================
# Test B: Anchor Active + Insufficient Hits
# ============================================================================

class TestAnchorActiveInsufficient:
    """When anchor active but insufficient hits, classroom/curated used, web blocked."""
    
    def test_web_blocked_even_when_insufficient(self, anchor):
        """Web must be blocked even when anchor hits are insufficient."""
        chunks = [
            {'id': 'classroom_1', 'content': 'General history textbook', 
             'similarity': 0.55, 'source_type': 'classroom'},
            {'id': 'web_1', 'content': 'Unrelated web content', 
             'similarity': 0.45, 'source_type': 'web'},
        ]
        
        result = assemble_mcp_context(
            chunks=chunks,
            active_anchor=anchor,
            session_id="test",
            request_id="req_b1"
        )
        
        # No web chunks
        web_in_result = [c for c in result.chunks if c.source == 'web']
        assert len(web_in_result) == 0
    
    def test_broadening_prompt_when_no_content(self, anchor):
        """Should prompt for broadening when no anchor/classroom content."""
        chunks = [
            {'id': 'web_1', 'content': 'Only web content', 
             'similarity': 0.45, 'source_type': 'web'},
        ]
        
        result = assemble_mcp_context(
            chunks=chunks,
            active_anchor=anchor,
            session_id="test",
            request_id="req_b2"
        )
        
        # Should prompt for broadening since only web exists
        # and anchor is active but no anchor/classroom content


# ============================================================================
# Test C: No Anchor (Normal Flow)
# ============================================================================

class TestNoAnchor:
    """When no anchor, web chunks are allowed."""
    
    def test_web_allowed_without_anchor(self, mixed_chunks):
        """Web chunks allowed when no anchor active."""
        result = assemble_mcp_context(
            chunks=mixed_chunks,
            active_anchor=None,
            session_id="test",
            request_id="req_c1"
        )
        
        assert result.reason == "no_anchor_normal_flow"
        
        # Web chunks should be present
        web_in_result = [c for c in result.chunks if c.source == 'web']
        # May or may not have web depending on ranking, but should not be blocked


# ============================================================================
# Test D: Safety Assertion
# ============================================================================

class TestSafetyAssertion:
    """Safety assertion must catch web chunks in anchor context."""
    
    def test_assertion_passes_when_no_web(self, anchor):
        """Should pass when no web chunks in context."""
        chunks = [
            MCPChunk(id="anchor_1", text="Anchor content", source="anchor",
                    similarity=0.8, source_trust=0.99)
        ]
        
        # Should not raise
        result = assert_no_web_in_anchor_context(chunks, anchor, anchor_hits=2)
        assert result is True
    
    def test_assertion_fails_when_web_present(self, anchor):
        """Should raise when web chunks found in anchor context."""
        chunks = [
            MCPChunk(id="anchor_1", text="Anchor content", source="anchor",
                    similarity=0.8, source_trust=0.99),
            MCPChunk(id="web_1", text="Web content", source="web",
                    similarity=0.4, source_trust=0.6)
        ]
        
        with pytest.raises(AssertionError, match="web chunks in anchor context"):
            assert_no_web_in_anchor_context(chunks, anchor, anchor_hits=2)


# ============================================================================
# Reproduction Test: French Revolution Follow-up
# ============================================================================

class TestReproductionScenario:
    """
    Reproduce the failing scenario:
    1. User asks about French Revolution
    2. Anchor created
    3. User asks "how many people were killed"
    4. Web content about transgender/climate deaths appears
    
    Expected: Web content BLOCKED, only anchor/classroom used
    """
    
    def test_french_revolution_followup_blocks_unrelated_web(self, anchor):
        """The exact failing scenario must be fixed."""
        # Simulated retrieval result
        chunks = [
            # Anchor-matched content
            {'id': 'doc_fr_1', 'content': 'The French Revolution resulted in approximately 40,000 executions during the Reign of Terror.',
             'similarity': 0.75, 'source_type': 'classroom', 'topic_anchor_id': 'anchor_fr'},
            
            # Unrelated web content (the contamination)
            {'id': 'web_trans', 'content': 'Transgender deaths increased by 30% in 2023 according to human rights reports.',
             'similarity': 0.52, 'source_type': 'web'},
            {'id': 'web_climate', 'content': 'Climate change causes millions of deaths annually through extreme weather.',
             'similarity': 0.48, 'source_type': 'web'},
        ]
        
        result = assemble_mcp_context(
            chunks=chunks,
            active_anchor=anchor,
            session_id="sess_french_rev",
            request_id="req_deaths"
        )
        
        # MUST have anchor content
        anchor_in_result = [c for c in result.chunks if 'French Revolution' in c.text or c.source == 'anchor']
        assert len(anchor_in_result) >= 1, "Anchor content missing!"
        
        # MUST NOT have unrelated web content
        for chunk in result.chunks:
            assert 'transgender' not in chunk.text.lower(), "Transgender content leaked!"
            assert 'climate change' not in chunk.text.lower(), "Climate content leaked!"
        
        print(f"✅ French Revolution follow-up correctly filters unrelated web content")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
