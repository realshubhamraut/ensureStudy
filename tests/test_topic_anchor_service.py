"""
Tests for Topic Anchor Layer (TAL) Service

Test Scenarios:
1. Anchor creation from user query
2. Anchor enforcement (get existing anchor)
3. Anchor clearing and history tracking
4. Entity extraction from context
5. LLM prompt fragment generation
6. Canonical title extraction
"""
import pytest
import sys
import os

# Add path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend', 'ai-service'))

from app.services.topic_anchor_service import (
    TopicAnchorService,
    TopicAnchor,
    get_topic_anchor_service,
    extract_canonical_title,
    FEATURE_TOPIC_ANCHOR
)


# ============================================================================
# Utility Tests
# ============================================================================

class TestExtractCanonicalTitle:
    """Test canonical title extraction from user queries."""
    
    def test_simple_prefix(self):
        """Should remove 'tell me about' prefix."""
        assert extract_canonical_title("tell me about the French Revolution") == "French Revolution"
    
    def test_explain_prefix(self):
        """Should remove 'explain' prefix."""
        assert extract_canonical_title("explain photosynthesis") == "Photosynthesis"
    
    def test_what_is_prefix(self):
        """Should remove 'what is' prefix."""
        assert extract_canonical_title("what is the Pythagorean theorem?") == "Pythagorean Theorem"
    
    def test_complex_prefix(self):
        """Should handle multi-word prefixes."""
        assert extract_canonical_title("can you explain Newton's laws") == "Newton's Laws"
    
    def test_no_prefix(self):
        """Should handle queries without prefix."""
        result = extract_canonical_title("French Revolution causes")
        assert "French" in result
    
    def test_trailing_punctuation(self):
        """Should remove trailing punctuation."""
        assert extract_canonical_title("what is gravity?") == "Gravity"
        assert extract_canonical_title("explain DNA!") == "Dna"


# ============================================================================
# Anchor Service Tests
# ============================================================================

class TestTopicAnchorService:
    """Test the Topic Anchor Service."""
    
    @pytest.fixture
    def service(self):
        """Create fresh service instance."""
        return TopicAnchorService()
    
    def test_create_anchor(self, service):
        """Should create a new anchor."""
        anchor = service.create_anchor(
            session_id="test_session_1",
            request_id="req_001",
            canonical_title="French Revolution",
            source="user_query"
        )
        
        assert anchor is not None
        assert anchor.canonical_title == "French Revolution"
        assert anchor.id.startswith("anchor_")
        assert anchor.source == "user_query"
    
    def test_get_anchor(self, service):
        """Should retrieve existing anchor."""
        # Create
        service.create_anchor(
            session_id="test_session_2",
            request_id="req_002",
            canonical_title="Pythagoras Theorem"
        )
        
        # Get
        anchor = service.get_anchor("test_session_2")
        
        assert anchor is not None
        assert anchor.canonical_title == "Pythagoras Theorem"
    
    def test_get_anchor_nonexistent(self, service):
        """Should return None for nonexistent session."""
        anchor = service.get_anchor("nonexistent_session")
        assert anchor is None
    
    def test_clear_anchor(self, service):
        """Should clear anchor and add to history."""
        # Create
        service.create_anchor(
            session_id="test_session_3",
            request_id="req_003",
            canonical_title="Newton's Laws"
        )
        
        # Clear
        result = service.clear_anchor("test_session_3", reason="user_reset")
        assert result is True
        
        # Verify cleared
        anchor = service.get_anchor("test_session_3")
        assert anchor is None
        
        # Verify history
        history = service.get_topic_history("test_session_3")
        assert len(history) == 1
        assert history[0].canonical_title == "Newton's Laws"
        assert history[0].end_reason == "user_reset"
    
    def test_create_replaces_existing(self, service):
        """Creating new anchor should replace existing."""
        # Create first
        service.create_anchor(
            session_id="test_session_4",
            request_id="req_004",
            canonical_title="First Topic"
        )
        
        # Create second (should replace)
        service.create_anchor(
            session_id="test_session_4",
            request_id="req_005",
            canonical_title="Second Topic"
        )
        
        # Verify replacement
        anchor = service.get_anchor("test_session_4")
        assert anchor.canonical_title == "Second Topic"
        
        # Verify history includes first
        history = service.get_topic_history("test_session_4")
        assert len(history) == 1
        assert history[0].canonical_title == "First Topic"


# ============================================================================
# Anchor Enforcement Tests
# ============================================================================

class TestAnchorEnforcement:
    """Test anchor enforcement for retrieval and LLM."""
    
    @pytest.fixture
    def service(self):
        return TopicAnchorService()
    
    def test_enforce_anchor_returns_filter(self, service):
        """Should return Qdrant filter for anchor."""
        service.create_anchor(
            session_id="enforce_test",
            request_id="req_e01",
            canonical_title="French Revolution"
        )
        
        qdrant_filter, prompt_fragment = service.enforce_anchor(
            session_id="enforce_test",
            request_id="req_e02"
        )
        
        assert qdrant_filter is not None
        assert prompt_fragment is not None
        assert "French Revolution" in prompt_fragment
    
    def test_enforce_no_anchor(self, service):
        """Should return None when no anchor exists."""
        qdrant_filter, prompt_fragment = service.enforce_anchor(
            session_id="no_anchor_session",
            request_id="req_none"
        )
        
        assert qdrant_filter is None
        assert prompt_fragment is None


# ============================================================================
# Entity Extraction Tests
# ============================================================================

class TestEntityExtraction:
    """Test entity extraction from context."""
    
    @pytest.fixture
    def service(self):
        return TopicAnchorService()
    
    def test_extract_entities_from_context(self, service):
        """Should extract named entities from context chunks."""
        anchor = service.create_anchor(
            session_id="entity_test",
            request_id="req_ent",
            canonical_title="French Revolution",
            context_chunks=[
                "The French Revolution began in 1789 when Louis XVI was king.",
                "The storming of the Bastille occurred on July 14.",
                "Napoleon Bonaparte rose to power after the revolution."
            ]
        )
        
        # Should have extracted some entities
        assert len(anchor.locked_entities) > 0
        # Should include year
        assert "1789" in anchor.locked_entities or any("1789" in e for e in anchor.locked_entities)


# ============================================================================
# Scope Generation Tests
# ============================================================================

class TestScopeGeneration:
    """Test subject scope generation."""
    
    @pytest.fixture
    def service(self):
        return TopicAnchorService()
    
    def test_revolution_scope(self, service):
        """Revolution topics should get history-related scope."""
        anchor = service.create_anchor(
            session_id="scope_rev",
            request_id="req_scope",
            canonical_title="French Revolution"
        )
        
        assert "causes" in anchor.subject_scope or "events" in anchor.subject_scope
    
    def test_theorem_scope(self, service):
        """Theorem topics should get math-related scope."""
        anchor = service.create_anchor(
            session_id="scope_thm",
            request_id="req_scope2",
            canonical_title="Pythagorean Theorem"
        )
        
        assert "proof" in anchor.subject_scope or "applications" in anchor.subject_scope


# ============================================================================
# LLM Prompt Fragment Tests
# ============================================================================

class TestPromptFragment:
    """Test LLM system prompt generation."""
    
    def test_prompt_contains_topic(self):
        """Prompt fragment should contain canonical title."""
        anchor = TopicAnchor(
            id="test_anchor",
            canonical_title="French Revolution",
            subject_scope=["causes", "events"],
            locked_entities=["Louis XVI", "1789"]
        )
        
        prompt = anchor.to_prompt_fragment()
        
        assert "French Revolution" in prompt
        assert "causes" in prompt or "events" in prompt
        assert "Louis XVI" in prompt or "1789" in prompt
    
    def test_prompt_has_instruction(self):
        """Prompt should have anchor enforcement instruction."""
        anchor = TopicAnchor(
            id="test_anchor2",
            canonical_title="Photosynthesis"
        )
        
        prompt = anchor.to_prompt_fragment()
        
        assert "anchor scope" in prompt.lower() or "only use" in prompt.lower()


# ============================================================================
# Integration Tests
# ============================================================================

class TestTALIntegration:
    """Integration tests for TAL workflow."""
    
    @pytest.fixture
    def service(self):
        return TopicAnchorService()
    
    def test_full_workflow(self, service):
        """
        Full TAL workflow:
        1. Create anchor from "Tell me about French Revolution"
        2. Enforce anchor for follow-up "What were the consequences?"
        3. Clear anchor when new topic detected
        """
        session_id = "integration_test"
        
        # Step 1: Create anchor
        anchor = service.create_anchor(
            session_id=session_id,
            request_id="req_int_1",
            canonical_title="French Revolution",
            context_chunks=["The French Revolution began in 1789."]
        )
        
        assert anchor is not None
        print(f"Created anchor: {anchor.canonical_title}")
        
        # Step 2: Enforce for follow-up
        qdrant_filter, prompt_fragment = service.enforce_anchor(
            session_id=session_id,
            request_id="req_int_2"
        )
        
        assert prompt_fragment is not None
        assert "French Revolution" in prompt_fragment
        print(f"Prompt fragment includes topic: ✓")
        
        # Step 3: Clear anchor
        service.clear_anchor(session_id, reason="new_topic")
        
        anchor_after = service.get_anchor(session_id)
        assert anchor_after is None
        
        history = service.get_topic_history(session_id)
        assert len(history) == 1
        print(f"History tracked: ✓")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
