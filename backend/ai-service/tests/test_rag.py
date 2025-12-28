"""
Tests for RAG API Endpoints
"""
import pytest
from unittest.mock import patch, MagicMock


class TestRAGQuery:
    """Test RAG query endpoints"""
    
    def test_query_success(self, client, auth_headers, mock_retriever):
        """Test successful RAG query"""
        response = client.post(
            '/api/rag/query',
            headers=auth_headers,
            json={
                'query': 'What is photosynthesis?',
                'top_k': 5
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert 'answer' in data
        assert 'sources' in data
    
    def test_query_without_auth(self, client, mock_retriever):
        """Test RAG query without authentication"""
        response = client.post(
            '/api/rag/query',
            json={'query': 'Test query'}
        )
        
        assert response.status_code in [401, 403]
    
    def test_query_empty_query(self, client, auth_headers):
        """Test RAG query with empty query"""
        response = client.post(
            '/api/rag/query',
            headers=auth_headers,
            json={'query': ''}
        )
        
        assert response.status_code == 422  # Validation error


class TestRAGRetrieve:
    """Test RAG retrieve endpoints"""
    
    def test_retrieve_chunks(self, client, auth_headers, mock_retriever):
        """Test retrieving chunks only"""
        response = client.post(
            '/api/rag/retrieve',
            headers=auth_headers,
            json={
                'query': 'Cell structure',
                'top_k': 3
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert 'chunks' in data
        assert 'count' in data
    
    def test_retrieve_with_filters(self, client, auth_headers, mock_retriever):
        """Test retrieve with subject filter"""
        response = client.post(
            '/api/rag/retrieve',
            headers=auth_headers,
            json={
                'query': 'Mitochondria',
                'subject_filter': 'Biology',
                'difficulty_filter': 'medium'
            }
        )
        
        assert response.status_code == 200


class TestRAGSearch:
    """Test simple search endpoint"""
    
    def test_simple_search(self, client, auth_headers, mock_retriever):
        """Test simple search"""
        response = client.get(
            '/api/rag/search?q=photosynthesis&top_k=5',
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert 'results' in data
    
    def test_search_short_query(self, client, auth_headers):
        """Test search with too short query"""
        response = client.get(
            '/api/rag/search?q=ab',
            headers=auth_headers
        )
        
        assert response.status_code == 422  # Min length validation


class TestCollectionInfo:
    """Test collection info endpoint"""
    
    def test_get_collection_info(self, client, auth_headers):
        """Test getting collection info"""
        with patch('app.rag.qdrant_setup.get_collection_info') as mock:
            mock.return_value = {
                'name': 'test_collection',
                'vectors_count': 100,
                'status': 'ok'
            }
            
            response = client.get(
                '/api/rag/collection-info',
                headers=auth_headers
            )
            
            assert response.status_code == 200
            data = response.json()
            assert 'name' in data
