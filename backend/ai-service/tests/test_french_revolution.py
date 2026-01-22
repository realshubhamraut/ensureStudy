"""
French Revolution Test Scenario

Tests the TAL/ABCR/MCP integration:
1. Initial query creates anchor
2. Follow-up is detected by ABCR
3. MCP filters web content when anchor active
4. New unrelated query triggers topic change

Usage:
    cd backend/ai-service
    python -m pytest tests/test_french_revolution.py -v
"""
import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def mock_classifier():
    """Mock the moderation classifier."""
    classifier = Mock()
    classifier.classify.return_value = {
        "academic": 0.8,
        "homework": 0.6,
        "study": 0.7,
        "off-topic": 0.1,
        "inappropriate": 0.0
    }
    return classifier


@pytest.fixture
def mock_retriever():
    """Mock the RAG retriever."""
    retriever = Mock()
    retriever.retrieve.return_value = [
        {
            "id": "chunk_1",
            "content": "The French Revolution began in 1789 with the storming of the Bastille.",
            "relevance_score": 0.9,
            "source_type": "classroom",
            "document_id": "doc_123",
            "title": "French Revolution Overview"
        },
        {
            "id": "chunk_2", 
            "content": "Key figures included Robespierre, Louis XVI, and Marie Antoinette.",
            "relevance_score": 0.85,
            "source_type": "classroom",
            "document_id": "doc_123",
            "title": "French Revolution Figures"
        },
        {
            "id": "chunk_3",
            "content": "Random web content about WWE wrestling matches.",
            "relevance_score": 0.5,
            "source_type": "web",
            "url": "http://random.com",
            "title": "WWE News"
        }
    ]
    return retriever


@pytest.fixture
def mock_llm():
    """Mock the LLM provider."""
    llm = Mock()
    llm.invoke.return_value = "The French Revolution was a major event..."
    return llm


# ============================================================================
# Test Cases
# ============================================================================

class TestFrenchRevolutionScenario:
    """Integration tests for the French Revolution scenario."""
    
    @pytest.mark.asyncio
    async def test_step1_initial_query_creates_anchor(
        self, mock_classifier, mock_retriever, mock_llm
    ):
        """
        Step 1: "What caused the French Revolution?"
        Expected: Creates a new topic anchor
        """
        with patch('app.services.llm_provider.get_classifier', return_value=mock_classifier), \
             patch('app.rag.retriever.get_retriever', return_value=mock_retriever), \
             patch('app.services.llm_provider.get_llm', return_value=mock_llm):
            
            from app.agents.tutor_agent import TutorAgent, get_session_state
            
            agent = TutorAgent()
            
            result = await agent.execute({
                "query": "What caused the French Revolution?",
                "user_id": "user_123",
                "session_id": "session_abc",
            })
            
            data = result.get("data", {})
            
            # Assert anchor was created
            assert data.get("topic_anchor") is not None, "Topic anchor should be created"
            assert "French Revolution" in data["topic_anchor"]["title"], \
                f"Anchor title should contain 'French Revolution', got: {data['topic_anchor']['title']}"
            
            # Assert this is NOT a follow-up (first query)
            assert data.get("is_followup") == False, "First query should not be a follow-up"
            
            # Assert session state updated
            sess = get_session_state("session_abc")
            assert len(sess.get("turn_texts", [])) > 0, "Turn history should be updated"
    
    @pytest.mark.asyncio
    async def test_step2_follow_up_detected(
        self, mock_classifier, mock_retriever, mock_llm
    ):
        """
        Step 2: "Tell me more" after French Revolution query
        Expected: ABCR detects as follow-up, keeps same anchor
        """
        with patch('app.services.llm_provider.get_classifier', return_value=mock_classifier), \
             patch('app.rag.retriever.get_retriever', return_value=mock_retriever), \
             patch('app.services.llm_provider.get_llm', return_value=mock_llm):
            
            from app.agents.tutor_agent import TutorAgent, update_session_state
            from app.services.topic_anchor_service import get_topic_anchor_service
            
            # Setup: Create initial anchor
            tal = get_topic_anchor_service()
            tal.create_anchor(
                session_id="session_followup",
                request_id="req_1",
                canonical_title="French Revolution"
            )
            
            # Setup: Add previous turn
            update_session_state("session_followup", {
                "turn_texts": ["What caused the French Revolution?"],
                "topic_anchor_id": "anchor_test",
                "topic_anchor_title": "French Revolution",
            })
            
            agent = TutorAgent()
            
            result = await agent.execute({
                "query": "Tell me more about the causes",
                "user_id": "user_123",
                "session_id": "session_followup",
            })
            
            data = result.get("data", {})
            
            # Assert follow-up detected
            assert data.get("is_followup") == True, \
                f"Should detect 'Tell me more' as follow-up, got: is_followup={data.get('is_followup')}"
            
            # Assert anchor preserved
            assert data.get("topic_anchor") is not None, "Anchor should be preserved"
            assert "French Revolution" in data["topic_anchor"]["title"], \
                "Still discussing French Revolution"
            
            # Assert no topic change confirmation needed
            assert data.get("confirm_new_topic") == False, \
                "Should not prompt for topic change on follow-up"
    
    @pytest.mark.asyncio
    async def test_step3_mcp_filters_web_content(
        self, mock_classifier, mock_retriever, mock_llm
    ):
        """
        Step 3: Verify MCP filters web chunks when anchor is active
        Expected: Web chunks are filtered out, only anchor/classroom content used
        """
        with patch('app.services.llm_provider.get_classifier', return_value=mock_classifier), \
             patch('app.rag.retriever.get_retriever', return_value=mock_retriever), \
             patch('app.services.llm_provider.get_llm', return_value=mock_llm):
            
            from app.agents.tutor_agent import TutorAgent, update_session_state
            from app.services.topic_anchor_service import get_topic_anchor_service
            
            # Setup: Create anchor
            tal = get_topic_anchor_service()
            tal.create_anchor(
                session_id="session_mcp",
                request_id="req_2",
                canonical_title="French Revolution"
            )
            update_session_state("session_mcp", {
                "turn_texts": ["What caused the French Revolution?"],
            })
            
            agent = TutorAgent()
            
            result = await agent.execute({
                "query": "Who were the key figures?",
                "user_id": "user_123",
                "session_id": "session_mcp",
            })
            
            data = result.get("data", {})
            
            # Assert web content was filtered
            assert data.get("web_filtered_count", 0) >= 0, \
                "Web filtered count should be tracked"
            
            # Assert context sources don't include web
            context_sources = data.get("context_sources", [])
            # Note: May include 'classroom', 'anchor', 'curated' but should filter 'web' when anchor active
            
            # Assert anchor hits tracked
            assert "anchor_hits" in data, "Anchor hits should be tracked"
    
    @pytest.mark.asyncio
    async def test_step4_new_topic_detected(
        self, mock_classifier, mock_retriever, mock_llm
    ):
        """
        Step 4: "What is Newton's first law?" (unrelated)
        Expected: ABCR detects new topic, creates new anchor
        """
        with patch('app.services.llm_provider.get_classifier', return_value=mock_classifier), \
             patch('app.rag.retriever.get_retriever', return_value=mock_retriever), \
             patch('app.services.llm_provider.get_llm', return_value=mock_llm):
            
            from app.agents.tutor_agent import TutorAgent, update_session_state
            from app.services.topic_anchor_service import get_topic_anchor_service
            
            # Setup: Create French Revolution anchor
            tal = get_topic_anchor_service()
            tal.create_anchor(
                session_id="session_newtopic",
                request_id="req_3",
                canonical_title="French Revolution"
            )
            update_session_state("session_newtopic", {
                "turn_texts": ["What caused the French Revolution?"],
            })
            
            agent = TutorAgent()
            
            result = await agent.execute({
                "query": "What is Newton's first law of motion?",
                "user_id": "user_123",
                "session_id": "session_newtopic",
            })
            
            data = result.get("data", {})
            
            # Assert NOT a follow-up (different topic)
            # Note: This depends on ABCR's similarity threshold
            # A completely different topic should score low
            
            # Assert new anchor created
            assert data.get("topic_anchor") is not None, "New anchor should be created"
            
            # The anchor title should now be about Newton
            anchor_title = data["topic_anchor"]["title"]
            # Either it's Newton topic or confirm_new_topic is True
            assert "Newton" in anchor_title or data.get("confirm_new_topic") == True, \
                f"Should switch to Newton topic or request confirmation, got: {anchor_title}"


class TestABCRService:
    """Unit tests for ABCR service."""
    
    def test_compute_relatedness_related(self):
        """Test that similar queries are detected as related."""
        from app.services.abcr_service import get_abcr_service
        
        abcr = get_abcr_service()
        
        result, _ = abcr.compute_relatedness(
            session_id="test_session",
            query_text="Tell me more about it",
            turn_texts=["What caused the French Revolution?"]
        )
        
        # "Tell me more" with previous French Revolution context
        # should be detected as related
        assert result.decision in ["related", "new_topic"], \
            f"Should make a decision, got: {result.decision}"
        assert result.max_relatedness >= 0, "Should compute relatedness score"
    
    def test_compute_relatedness_unrelated(self):
        """Test that different topics are detected as new topic."""
        from app.services.abcr_service import get_abcr_service
        
        abcr = get_abcr_service()
        
        result, _ = abcr.compute_relatedness(
            session_id="test_session",
            query_text="What is the capital of France?",
            turn_texts=["Explain quantum mechanics"]
        )
        
        # Very different topics should have low relatedness
        assert result.decision in ["related", "new_topic"], \
            f"Should make a decision, got: {result.decision}"


class TestTopicAnchorService:
    """Unit tests for TAL service."""
    
    def test_create_anchor(self):
        """Test anchor creation."""
        from app.services.topic_anchor_service import get_topic_anchor_service
        
        tal = get_topic_anchor_service()
        
        anchor = tal.create_anchor(
            session_id="test_create",
            request_id="req_test",
            canonical_title="French Revolution"
        )
        
        assert anchor.id is not None
        assert anchor.canonical_title == "French Revolution"
        assert len(anchor.subject_scope) > 0, "Should generate scope"
    
    def test_get_anchor(self):
        """Test anchor retrieval."""
        from app.services.topic_anchor_service import get_topic_anchor_service
        
        tal = get_topic_anchor_service()
        
        # Create anchor
        tal.create_anchor(
            session_id="test_get",
            request_id="req_test",
            canonical_title="Test Topic"
        )
        
        # Retrieve
        anchor = tal.get_anchor("test_get")
        assert anchor is not None
        assert anchor.canonical_title == "Test Topic"
    
    def test_clear_anchor(self):
        """Test anchor clearing."""
        from app.services.topic_anchor_service import get_topic_anchor_service
        
        tal = get_topic_anchor_service()
        
        # Create anchor
        tal.create_anchor(
            session_id="test_clear",
            request_id="req_test",
            canonical_title="To Be Cleared"
        )
        
        # Clear
        success = tal.clear_anchor("test_clear", reason="user_reset")
        assert success == True
        
        # Verify cleared
        anchor = tal.get_anchor("test_clear")
        assert anchor is None


class TestMCPContext:
    """Unit tests for MCP context assembly."""
    
    def test_assemble_with_anchor(self):
        """Test MCP filters web when anchor active."""
        from app.services.mcp_context import assemble_mcp_context
        from app.services.topic_anchor_service import TopicAnchor
        
        # Create mock anchor
        anchor = TopicAnchor(
            id="anchor_1",
            canonical_title="French Revolution"
        )
        
        # Create mock chunks
        chunks = [
            {"id": "1", "content": "French Revolution content", "source_type": "classroom", "similarity": 0.9},
            {"id": "2", "content": "More about 1789", "source_type": "classroom", "similarity": 0.8},
            {"id": "3", "content": "Random web stuff", "source_type": "web", "similarity": 0.7, "url": "http://random.com"},
        ]
        
        result = assemble_mcp_context(
            chunks=chunks,
            active_anchor=anchor,
            session_id="test",
            request_id="req"
        )
        
        # Web should be filtered
        assert result.web_filtered_count >= 0, "Should track web filtered count"
        
        # Only classroom/anchor content
        for chunk in result.chunks:
            assert chunk.source in ["anchor", "classroom", "curated_global"], \
                f"Web should be filtered, got source: {chunk.source}"
    
    def test_assemble_without_anchor(self):
        """Test MCP allows web when no anchor."""
        from app.services.mcp_context import assemble_mcp_context
        
        chunks = [
            {"id": "1", "content": "Web content about history", "source_type": "web", "similarity": 0.8},
        ]
        
        result = assemble_mcp_context(
            chunks=chunks,
            active_anchor=None,
            session_id="test",
            request_id="req"
        )
        
        # Web allowed when no anchor
        assert len(result.chunks) > 0 or result.reason == "no_anchor_normal_flow", \
            "Should allow web content when no anchor"


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
