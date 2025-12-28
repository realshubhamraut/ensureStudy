"""
Pytest Configuration for AI Service Tests
"""
import os
import sys
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope='session')
def mock_qdrant():
    """Mock Qdrant client"""
    with patch('app.rag.qdrant_setup.QdrantClient') as mock:
        mock_client = MagicMock()
        mock_client.get_collections.return_value = MagicMock(collections=[])
        mock_client.search.return_value = []
        mock.return_value = mock_client
        yield mock_client


@pytest.fixture(scope='session')
def mock_openai():
    """Mock OpenAI client"""
    with patch('langchain_openai.ChatOpenAI') as mock_chat:
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="This is a mock response.")
        mock_chat.return_value = mock_llm
        
        with patch('langchain_openai.OpenAIEmbeddings') as mock_embed:
            mock_embeddings = MagicMock()
            mock_embeddings.embed_query.return_value = [0.1] * 1536
            mock_embeddings.embed_documents.return_value = [[0.1] * 1536]
            mock_embed.return_value = mock_embeddings
            
            yield {
                'chat': mock_llm,
                'embeddings': mock_embeddings
            }


@pytest.fixture(scope='session')
def app(mock_qdrant, mock_openai):
    """Create FastAPI app for testing"""
    from app.main import app
    return app


@pytest.fixture(scope='function')
def client(app):
    """FastAPI test client"""
    return TestClient(app)


@pytest.fixture(scope='function')
def auth_token():
    """Create a valid JWT token for testing"""
    import jwt
    from datetime import datetime, timedelta
    
    secret = os.getenv('JWT_SECRET', 'test-jwt-secret-key-32-chars-min')
    
    payload = {
        'user_id': 'test-user-123',
        'role': 'student',
        'type': 'access',
        'iat': datetime.utcnow(),
        'exp': datetime.utcnow() + timedelta(hours=1)
    }
    
    token = jwt.encode(payload, secret, algorithm='HS256')
    return token


@pytest.fixture(scope='function')
def auth_headers(auth_token):
    """Auth headers with valid token"""
    return {'Authorization': f'Bearer {auth_token}'}


@pytest.fixture(scope='function')
def mock_retriever():
    """Mock RAG retriever"""
    with patch('app.rag.retriever.get_retriever') as mock:
        mock_retriever = MagicMock()
        mock_retriever.retrieve_chunks.return_value = [
            {
                'id': 1,
                'text': 'Sample text about the topic.',
                'source': 'textbook.pdf',
                'page': 1,
                'subject': 'Biology',
                'topic': 'Cells',
                'similarity_score': 0.85
            }
        ]
        mock_retriever.answer_with_rag.return_value = {
            'answer': 'This is a test answer based on the sources.',
            'sources': mock_retriever.retrieve_chunks.return_value,
            'query': 'test query',
            'context_used': 1
        }
        mock.return_value = mock_retriever
        yield mock_retriever
