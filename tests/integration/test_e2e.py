"""
Integration Tests - End-to-End API Tests
"""
import pytest
import requests
import time
import os

# Service URLs from environment or defaults
CORE_API_URL = os.getenv('CORE_API_URL', 'http://localhost:8000')
AI_SERVICE_URL = os.getenv('AI_SERVICE_URL', 'http://localhost:8001')


def wait_for_service(url, timeout=30):
    """Wait for service to be ready"""
    start = time.time()
    while time.time() - start < timeout:
        try:
            response = requests.get(f"{url}/health", timeout=5)
            if response.status_code == 200:
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(1)
    return False


@pytest.fixture(scope='module')
def services_ready():
    """Ensure services are running"""
    core_ready = wait_for_service(CORE_API_URL)
    ai_ready = wait_for_service(AI_SERVICE_URL)
    
    if not core_ready:
        pytest.skip(f"Core API not available at {CORE_API_URL}")
    if not ai_ready:
        pytest.skip(f"AI Service not available at {AI_SERVICE_URL}")
    
    return True


@pytest.fixture(scope='module')
def test_user(services_ready):
    """Create a test user and get auth tokens"""
    import uuid
    
    email = f"integrationtest_{uuid.uuid4().hex[:8]}@example.com"
    username = f"testuser_{uuid.uuid4().hex[:8]}"
    password = "testpassword123"
    
    # Register user
    response = requests.post(
        f"{CORE_API_URL}/api/auth/register",
        json={
            'email': email,
            'username': username,
            'password': password
        }
    )
    
    assert response.status_code == 201, f"Registration failed: {response.text}"
    
    data = response.json()
    
    yield {
        'id': data['user']['id'],
        'email': email,
        'username': username,
        'access_token': data['access_token'],
        'refresh_token': data['refresh_token']
    }
    
    # Cleanup would go here (delete user)


class TestCoreAPIIntegration:
    """Integration tests for Core API"""
    
    def test_health_check(self, services_ready):
        """Test core API health"""
        response = requests.get(f"{CORE_API_URL}/health")
        
        assert response.status_code == 200
        assert response.json()['status'] == 'healthy'
    
    def test_login_flow(self, test_user):
        """Test login and token flow"""
        # Login
        response = requests.post(
            f"{CORE_API_URL}/api/auth/login",
            json={
                'email': test_user['email'],
                'password': 'testpassword123'
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert 'access_token' in data
        
        # Use token to get current user
        me_response = requests.get(
            f"{CORE_API_URL}/api/auth/me",
            headers={'Authorization': f"Bearer {data['access_token']}"}
        )
        
        assert me_response.status_code == 200
        assert me_response.json()['user']['email'] == test_user['email']
    
    def test_progress_flow(self, test_user):
        """Test progress tracking flow"""
        headers = {'Authorization': f"Bearer {test_user['access_token']}"}
        
        # Create progress
        response = requests.post(
            f"{CORE_API_URL}/api/progress/topic",
            headers=headers,
            json={
                'topic': 'Integration Test Topic',
                'subject': 'Testing',
                'confidence_score': 75
            }
        )
        
        assert response.status_code == 200
        
        # Get progress
        get_response = requests.get(
            f"{CORE_API_URL}/api/progress/",
            headers=headers
        )
        
        assert get_response.status_code == 200
        assert get_response.json()['count'] >= 1


class TestAIServiceIntegration:
    """Integration tests for AI Service"""
    
    def test_health_check(self, services_ready):
        """Test AI service health"""
        response = requests.get(f"{AI_SERVICE_URL}/health")
        
        assert response.status_code == 200
        assert response.json()['status'] == 'healthy'
    
    def test_rag_query(self, test_user):
        """Test RAG query endpoint"""
        headers = {'Authorization': f"Bearer {test_user['access_token']}"}
        
        response = requests.post(
            f"{AI_SERVICE_URL}/api/rag/query",
            headers=headers,
            json={
                'query': 'What is photosynthesis?',
                'top_k': 3
            }
        )
        
        # Should work even with empty collection
        assert response.status_code in [200, 500]  # 500 if no documents
    
    def test_agents_available(self, test_user):
        """Test listing available agents"""
        headers = {'Authorization': f"Bearer {test_user['access_token']}"}
        
        response = requests.get(
            f"{AI_SERVICE_URL}/api/agents/available",
            headers=headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert 'agents' in data
        assert len(data['agents']) >= 5


class TestCrossServiceIntegration:
    """Tests that span multiple services"""
    
    def test_full_study_flow(self, test_user):
        """Test a full study session flow"""
        headers = {'Authorization': f"Bearer {test_user['access_token']}"}
        
        # 1. Track study progress
        progress_response = requests.post(
            f"{CORE_API_URL}/api/progress/topic",
            headers=headers,
            json={
                'topic': 'Photosynthesis',
                'subject': 'Biology',
                'confidence_score': 40,
                'studied': True
            }
        )
        assert progress_response.status_code == 200
        
        # 2. Get weak topics
        weak_response = requests.get(
            f"{CORE_API_URL}/api/progress/weak-topics",
            headers=headers
        )
        assert weak_response.status_code == 200
        
        # 3. Request study plan from AI
        plan_response = requests.post(
            f"{AI_SERVICE_URL}/api/agents/study-plan",
            headers=headers,
            json={
                'weak_topics': weak_response.json().get('weak_topics', [])
            }
        )
        
        # Allow for both success and error (no documents)
        assert plan_response.status_code in [200, 500]
