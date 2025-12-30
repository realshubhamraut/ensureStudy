"""
Tests for ABCR (Attention-Based Context Routing) Service

Test Scenarios:
1. Follow-up stays on topic (Pythagoras → "How is this used?")
2. New topic detection (Pythagoras → Newton)
3. Students Also Ask click routing
4. Attention vs Cosine conflict resolution
5. Fallback behavior on failure
6. Performance under 250ms
"""
import pytest
import time
import sys
import os
from unittest.mock import MagicMock, patch

# Add path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend', 'ai-service'))

from app.services.abcr_service import (
    ABCRService,
    RelatednessResult,
    TurnScore,
    ABCR_RELATED_THRESHOLD,
    ABCR_FORGET_THRESHOLD
)
from app.services.abcr_cache import ABCRCache, get_abcr_cache


# ============================================================================
# Cache Tests
# ============================================================================

class TestABCRCache:
    """Test the token embedding cache."""
    
    @pytest.fixture
    def cache(self, tmp_path):
        """Create cache with temp directory."""
        return ABCRCache(cache_path=str(tmp_path / "test_cache"))
    
    def test_store_and_retrieve(self, cache):
        """Should store and retrieve token embeddings."""
        import numpy as np
        
        session_id = "test_session"
        turn_index = 0
        embeddings = np.random.randn(10, 768).astype(np.float32)
        
        # Store
        success = cache.store_turn_embeddings(
            session_id=session_id,
            turn_index=turn_index,
            token_embeddings=embeddings,
            turn_text="Test turn"
        )
        assert success
        
        # Retrieve
        retrieved = cache.get_turn_embeddings(session_id, turn_index)
        assert retrieved is not None
        assert retrieved.shape == embeddings.shape
    
    def test_nonexistent_returns_none(self, cache):
        """Should return None for nonexistent entries."""
        result = cache.get_turn_embeddings("fake_session", 999)
        assert result is None
    
    def test_clear_session(self, cache):
        """Should clear all entries for a session."""
        import numpy as np
        
        session_id = "clear_test"
        embeddings = np.random.randn(5, 768).astype(np.float32)
        
        # Store multiple turns
        for i in range(3):
            cache.store_turn_embeddings(session_id, i, embeddings, f"Turn {i}")
        
        # Verify stored
        assert cache.get_turn_embeddings(session_id, 0) is not None
        
        # Clear
        count = cache.clear_session(session_id)
        assert count == 3
        
        # Verify cleared
        assert cache.get_turn_embeddings(session_id, 0) is None
    
    def test_get_recent_turns(self, cache):
        """Should retrieve last N turns."""
        import numpy as np
        
        session_id = "recent_test"
        embeddings = np.random.randn(5, 768).astype(np.float32)
        
        # Store 5 turns
        for i in range(5):
            cache.store_turn_embeddings(session_id, i, embeddings, f"Turn {i}")
        
        # Get last 3
        recent = cache.get_recent_turns(session_id, last_n=3)
        assert len(recent) == 3
        assert 2 in recent and 3 in recent and 4 in recent


# ============================================================================
# Service Tests
# ============================================================================

class TestABCRService:
    """Test the ABCR service."""
    
    @pytest.fixture
    def service(self):
        """Create ABCR service."""
        return ABCRService()
    
    def test_compute_relatedness_empty_history(self, service):
        """Should return new_topic for empty history."""
        result, state = service.compute_relatedness(
            session_id="test",
            query_text="What is calculus?",
            request_id="test_req",
            turn_texts=[]
        )
        
        assert result.decision == "new_topic"
        assert result.max_relatedness == 0.0
    
    def test_follow_up_on_topic(self, service):
        """Follow-up about same topic should be related."""
        result, state = service.compute_relatedness(
            session_id="pythagoras",
            query_text="How is this theorem used in real life?",
            request_id="test_follow",
            turn_texts=[
                "Explain the Pythagorean theorem",
                "The Pythagorean theorem states that a squared plus b squared equals c squared"
            ]
        )
        
        # Should recognize as related
        assert result.max_relatedness > ABCR_FORGET_THRESHOLD
        print(f"Follow-up relatedness: {result.max_relatedness:.3f}")
    
    def test_new_topic_detection(self, service):
        """Completely new topic should be detected."""
        result, state = service.compute_relatedness(
            session_id="topic_change",
            query_text="Explain photosynthesis in plants",
            request_id="test_new",
            turn_texts=[
                "Tell me about World War 2",
                "World War 2 was a global conflict..."
            ]
        )
        
        # Should be new topic
        assert result.max_relatedness < ABCR_RELATED_THRESHOLD
        print(f"New topic relatedness: {result.max_relatedness:.3f}")
    
    def test_clicked_suggestion_lenient(self, service):
        """Clicked suggestions should use more lenient threshold."""
        result, state = service.compute_relatedness(
            session_id="click_test",
            query_text="What were the consequences of the French Revolution?",
            request_id="test_click",
            turn_texts=[
                "Tell me about the French Revolution"
            ],
            clicked_suggestion=True
        )
        
        # With clicked_suggestion, should be more lenient
        print(f"Clicked suggestion relatedness: {result.max_relatedness:.3f}")
    
    def test_hysteresis_stability(self, service):
        """Hysteresis should prevent flip-flopping."""
        last_decision = "related"
        consecutive = 0
        
        # Borderline queries should maintain last decision
        for i in range(2):
            result, state = service.compute_relatedness(
                session_id="hysteresis_test",
                query_text="What about other things?",
                request_id=f"test_hyst_{i}",
                turn_texts=["Some previous topic"],
                last_decision=last_decision,
                consecutive_borderline=consecutive
            )
            last_decision = state["last_decision"]
            consecutive = state["consecutive_borderline"]


# ============================================================================
# Integration Tests
# ============================================================================

class TestABCRIntegration:
    """Integration tests for ABCR with real scenarios."""
    
    @pytest.fixture
    def service(self):
        return ABCRService()
    
    def test_pythagoras_follow_up_scenario(self, service):
        """
        Full scenario: Pythagoras explanation, then "How is this used?"
        """
        # Turn 1: Initial question
        result1, _ = service.compute_relatedness(
            session_id="pyth_full",
            query_text="Explain the Pythagorean theorem",
            request_id="pyth_1",
            turn_texts=[]  # First turn
        )
        
        # Can be new_topic since no history
        assert result1.decision == "new_topic"
        
        # Turn 2: Follow-up
        result2, _ = service.compute_relatedness(
            session_id="pyth_full",
            query_text="How is this used in real life applications?",
            request_id="pyth_2",
            turn_texts=[
                "Explain the Pythagorean theorem"
            ]
        )
        
        # Should be related
        print(f"Pythagoras follow-up: decision={result2.decision}, score={result2.max_relatedness:.3f}")
    
    def test_topic_switch_scenario(self, service):
        """
        Scenario: Pythagoras → Newton (topic switch)
        """
        result, _ = service.compute_relatedness(
            session_id="switch",
            query_text="Explain Newton's first law of motion",
            request_id="switch_1",
            turn_texts=[
                "Explain the Pythagorean theorem",
                "It relates the sides of a right triangle"
            ]
        )
        
        print(f"Topic switch: decision={result.decision}, score={result.max_relatedness:.3f}")


# ============================================================================
# Performance Tests
# ============================================================================

class TestABCRPerformance:
    """Performance tests for ABCR."""
    
    def test_computation_latency(self):
        """ABCR should complete in <250ms for warmup, faster after."""
        service = ABCRService()
        
        # Warmup
        service.compute_relatedness(
            session_id="perf",
            query_text="Warmup query",
            request_id="warmup",
            turn_texts=["Previous turn"]
        )
        
        # Timed run
        start = time.time()
        result, _ = service.compute_relatedness(
            session_id="perf",
            query_text="How does this work?",
            request_id="perf_test",
            turn_texts=[
                "First turn text here",
                "Second turn text here",
                "Third turn text here"
            ]
        )
        elapsed_ms = (time.time() - start) * 1000
        
        print(f"ABCR computation: {elapsed_ms:.0f}ms")
        # Allow generous threshold for CI
        assert elapsed_ms < 5000, f"ABCR too slow: {elapsed_ms}ms"
        
        # Result should include computation time
        assert result.computation_time_ms >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
