"""
Tests for Health and Root Endpoints
"""
import pytest


class TestHealthEndpoints:
    """Test health check endpoints"""
    
    def test_health_check(self, client):
        """Test health endpoint"""
        response = client.get('/health')
        
        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'healthy'
        assert data['service'] == 'ai-service'
    
    def test_root_endpoint(self, client):
        """Test root endpoint"""
        response = client.get('/')
        
        assert response.status_code == 200
        data = response.json()
        assert 'service' in data
        assert 'docs' in data
