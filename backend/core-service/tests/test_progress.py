"""
Tests for Progress Routes
"""
import pytest


class TestProgress:
    """Test progress tracking"""
    
    def test_get_progress_empty(self, client, auth_headers):
        """Test getting progress when empty"""
        response = client.get('/api/progress/', headers=auth_headers)
        
        assert response.status_code == 200
        data = response.get_json()
        assert 'progress' in data
        assert data['count'] == 0
    
    def test_create_progress(self, client, auth_headers):
        """Test creating progress for a topic"""
        response = client.post('/api/progress/topic',
            headers=auth_headers,
            json={
                'topic': 'Photosynthesis',
                'subject': 'Biology',
                'confidence_score': 75.0
            }
        )
        
        assert response.status_code == 200
        data = response.get_json()
        assert data['progress']['topic'] == 'Photosynthesis'
        assert data['progress']['confidence_score'] == 75.0
    
    def test_update_progress(self, client, auth_headers):
        """Test updating existing progress"""
        # Create first
        client.post('/api/progress/topic',
            headers=auth_headers,
            json={
                'topic': 'Cell Division',
                'subject': 'Biology',
                'confidence_score': 50.0
            }
        )
        
        # Update
        response = client.post('/api/progress/topic',
            headers=auth_headers,
            json={
                'topic': 'Cell Division',
                'subject': 'Biology',
                'confidence_score': 80.0,
                'studied': True
            }
        )
        
        assert response.status_code == 200
        data = response.get_json()
        assert data['progress']['confidence_score'] == 80.0
        assert data['progress']['times_studied'] >= 1
    
    def test_weak_topics_detection(self, client, auth_headers):
        """Test automatic weak topic detection"""
        # Create a weak topic (confidence < 50)
        client.post('/api/progress/topic',
            headers=auth_headers,
            json={
                'topic': 'Quantum Physics',
                'subject': 'Physics',
                'confidence_score': 30.0
            }
        )
        
        response = client.get('/api/progress/weak-topics', headers=auth_headers)
        
        assert response.status_code == 200
        data = response.get_json()
        assert data['count'] >= 1
        
        weak_topics = [t['topic'] for t in data['weak_topics']]
        assert 'Quantum Physics' in weak_topics
    
    def test_progress_summary(self, client, auth_headers):
        """Test progress summary endpoint"""
        # Create some progress
        for topic, confidence in [('Topic A', 80), ('Topic B', 60), ('Topic C', 40)]:
            client.post('/api/progress/topic',
                headers=auth_headers,
                json={
                    'topic': topic,
                    'subject': 'Math',
                    'confidence_score': confidence
                }
            )
        
        response = client.get('/api/progress/summary', headers=auth_headers)
        
        assert response.status_code == 200
        data = response.get_json()
        assert 'total_topics' in data
        assert 'average_confidence' in data
        assert 'subjects' in data


class TestProgressFiltering:
    """Test progress filtering by subject"""
    
    def test_filter_by_subject(self, client, auth_headers):
        """Test filtering progress by subject"""
        # Create topics in different subjects
        client.post('/api/progress/topic',
            headers=auth_headers,
            json={'topic': 'Algebra', 'subject': 'Math', 'confidence_score': 70}
        )
        client.post('/api/progress/topic',
            headers=auth_headers,
            json={'topic': 'Cells', 'subject': 'Biology', 'confidence_score': 80}
        )
        
        response = client.get('/api/progress/?subject=Math', headers=auth_headers)
        
        assert response.status_code == 200
        data = response.get_json()
        
        subjects = [p['subject'] for p in data['progress']]
        assert all(s == 'Math' for s in subjects)
