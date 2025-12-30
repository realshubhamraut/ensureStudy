"""
Tests for Session Service - Resource Chaining and Deduplication

Tests:
1. Session creation and retrieval
2. Turn addition with relatedness detection
3. Resource deduplication by URL, hash, and vector
4. LRU eviction when at capacity
5. Export session JSON
"""
import pytest
import time
from unittest.mock import patch, MagicMock

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend', 'ai-service'))

from app.services.session_service import SessionService, DEFAULT_CONFIG


class TestSessionCreation:
    """Test session lifecycle"""
    
    def test_create_session(self):
        """Test basic session creation"""
        service = SessionService()
        
        session = service.create_session(
            user_id="user_123",
            classroom_id="class_456"
        )
        
        assert session.session_id is not None
        assert session.user_id == "user_123"
        assert session.classroom_id == "class_456"
        assert session.turn_count == 0
        assert session.resource_count == 0
    
    def test_get_session(self):
        """Test session retrieval"""
        service = SessionService()
        created = service.create_session(user_id="user_123")
        
        retrieved = service.get_session(created.session_id)
        
        assert retrieved is not None
        assert retrieved.session_id == created.session_id
        assert retrieved.user_id == created.user_id
    
    def test_get_nonexistent_session(self):
        """Test retrieval of non-existent session returns None"""
        service = SessionService()
        
        result = service.get_session("nonexistent-id")
        
        assert result is None
    
    def test_session_expiration(self):
        """Test that expired sessions return None"""
        service = SessionService()
        
        # Create session with very short TTL
        session = service.create_session(
            user_id="user_123",
            config={"ttl_hours": 0}  # Expire immediately
        )
        
        # Force timestamp to past
        service._sessions[session.session_id]["last_active_at"] = "2020-01-01T00:00:00"
        
        result = service.get_session(session.session_id)
        
        assert result is None


class TestTurnManagement:
    """Test turn tracking and relatedness"""
    
    @patch.object(SessionService, 'embedding_model')
    def test_add_turn(self, mock_model):
        """Test adding a turn to session"""
        mock_model.encode.return_value = [0.1] * 384
        
        service = SessionService()
        service._embedding_model = mock_model
        session = service.create_session(user_id="user_123")
        
        turn = service.add_turn(session.session_id, "What is photosynthesis?")
        
        assert turn is not None
        assert turn.turn_number == 1
        assert turn.question == "What is photosynthesis?"
        assert turn.related is False  # First turn can't be related
    
    @patch.object(SessionService, 'embedding_model')
    def test_relatedness_detection(self, mock_model):
        """Test that follow-up questions are detected as related"""
        # Create embeddings where Q2 is similar to Q1
        q1_embedding = [1.0] + [0.0] * 383
        q2_embedding = [0.95] + [0.0] * 383  # Very similar
        mock_model.encode.side_effect = [q1_embedding, q2_embedding]
        
        service = SessionService()
        service._embedding_model = mock_model
        session = service.create_session(user_id="user_123")
        
        # Add first turn
        service.add_turn(session.session_id, "Explain quadratic equation")
        
        # Add related follow-up
        turn2 = service.add_turn(session.session_id, "How to solve x^2 - 5x + 6 = 0?")
        
        assert turn2.related is True
        assert turn2.relatedness_score > 0.65
    
    @patch.object(SessionService, 'embedding_model')
    def test_unrelated_detection(self, mock_model):
        """Test that unrelated questions are detected"""
        # Create orthogonal embeddings
        q1_embedding = [1.0] + [0.0] * 383
        q2_embedding = [0.0, 1.0] + [0.0] * 382  # Orthogonal
        mock_model.encode.side_effect = [q1_embedding, q2_embedding]
        
        service = SessionService()
        service._embedding_model = mock_model
        session = service.create_session(user_id="user_123")
        
        service.add_turn(session.session_id, "Explain photosynthesis")
        turn2 = service.add_turn(session.session_id, "What is a byte?")
        
        assert turn2.related is False


class TestResourceDeduplication:
    """Test resource append with deduplication"""
    
    def test_append_new_resource(self):
        """Test appending a new resource"""
        service = SessionService()
        session = service.create_session(user_id="user_123")
        
        result = service.append_resource(
            session_id=session.session_id,
            resource_type="article",
            source="wikipedia",
            url="https://en.wikipedia.org/wiki/Pythagorean_theorem",
            title="Pythagorean theorem",
            preview_summary="The Pythagorean theorem..."
        )
        
        assert result.inserted is True
        assert result.reason == "new"
        assert result.resource_id != ""
    
    def test_duplicate_url_rejected(self):
        """Test that duplicate URLs are rejected"""
        service = SessionService()
        session = service.create_session(user_id="user_123")
        
        url = "https://en.wikipedia.org/wiki/Pythagorean_theorem"
        
        # First insert
        result1 = service.append_resource(
            session_id=session.session_id,
            resource_type="article",
            source="wikipedia",
            url=url,
            title="Pythagorean theorem"
        )
        
        # Second insert with same URL
        result2 = service.append_resource(
            session_id=session.session_id,
            resource_type="article",
            source="wikipedia",
            url=url,
            title="Pythagorean theorem - duplicate"
        )
        
        assert result1.inserted is True
        assert result2.inserted is False
        assert result2.reason == "duplicate_url"
    
    def test_duplicate_hash_rejected(self):
        """Test that duplicate content hashes are rejected"""
        service = SessionService()
        session = service.create_session(user_id="user_123")
        
        content_hash = "abc123def456"
        
        # First insert
        result1 = service.append_resource(
            session_id=session.session_id,
            resource_type="text",
            source="web",
            url="https://example1.com",
            title="Example 1",
            content_hash=content_hash
        )
        
        # Second insert with same hash but different URL
        result2 = service.append_resource(
            session_id=session.session_id,
            resource_type="text",
            source="web",
            url="https://example2.com",
            title="Example 2",
            content_hash=content_hash
        )
        
        assert result1.inserted is True
        assert result2.inserted is False
        assert result2.reason == "duplicate_hash"
    
    def test_url_normalization(self):
        """Test that URL normalization catches variations"""
        service = SessionService()
        session = service.create_session(user_id="user_123")
        
        # Insert with http
        result1 = service.append_resource(
            session_id=session.session_id,
            resource_type="article",
            source="web",
            url="http://www.example.com/page/",
            title="Example"
        )
        
        # Insert with https and no trailing slash
        result2 = service.append_resource(
            session_id=session.session_id,
            resource_type="article",
            source="web",
            url="https://example.com/page",
            title="Example"
        )
        
        assert result1.inserted is True
        assert result2.inserted is False
        assert result2.reason == "duplicate_url"


class TestLRUEviction:
    """Test resource list bounding"""
    
    def test_eviction_at_capacity(self):
        """Test that oldest resource is evicted when at capacity"""
        service = SessionService()
        session = service.create_session(
            user_id="user_123",
            config={"max_resources": 3}  # Small limit for testing
        )
        
        # Add 3 resources
        for i in range(3):
            service.append_resource(
                session_id=session.session_id,
                resource_type="text",
                source="web",
                url=f"https://example{i}.com",
                title=f"Resource {i}"
            )
        
        assert len(service._sessions[session.session_id]["resources"]) == 3
        
        # Add 4th resource (should evict oldest)
        service.append_resource(
            session_id=session.session_id,
            resource_type="text",
            source="web",
            url="https://example3.com",
            title="Resource 3"
        )
        
        # Should still have 3 resources
        assert len(service._sessions[session.session_id]["resources"]) == 3
        
        # Oldest (Resource 0) should be gone
        titles = [r["title"] for r in service._sessions[session.session_id]["resources"]]
        assert "Resource 0" not in titles
        assert "Resource 3" in titles


class TestExportSession:
    """Test session export functionality"""
    
    @patch.object(SessionService, 'embedding_model')
    def test_export_session_json(self, mock_model):
        """Test exporting session as JSON"""
        mock_model.encode.return_value = [0.1] * 384
        
        service = SessionService()
        service._embedding_model = mock_model
        session = service.create_session(user_id="user_123", classroom_id="class_456")
        
        # Add a turn
        service.add_turn(session.session_id, "What is gravity?")
        
        # Add a resource
        service.append_resource(
            session_id=session.session_id,
            resource_type="wikipedia",
            source="wikipedia",
            url="https://en.wikipedia.org/wiki/Gravity",
            title="Gravity"
        )
        
        export = service.export_session_json(session.session_id)
        
        assert export is not None
        assert export["session_id"] == session.session_id
        assert export["user_id"] == "user_123"
        assert export["classroom_id"] == "class_456"
        assert len(export["query_chain"]) == 1
        assert export["query_chain"][0]["question"] == "What is gravity?"
        assert len(export["resource_list"]) == 1
        assert export["resource_list"][0]["title"] == "Gravity"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
