"""
Pytest Configuration for Core Service Tests
"""
import os
import sys
import pytest
from uuid import uuid4

# Add app to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app, db
from app.models.user import User, Leaderboard


@pytest.fixture(scope='session')
def app():
    """Create application for testing"""
    test_config = {
        'TESTING': True,
        'SQLALCHEMY_DATABASE_URI': 'sqlite:///:memory:',
        'SQLALCHEMY_TRACK_MODIFICATIONS': False,
        'SECRET_KEY': 'test-secret-key',
        'JWT_SECRET': 'test-jwt-secret-key-32-chars-min'
    }
    
    app = create_app(test_config)
    
    with app.app_context():
        db.create_all()
        yield app
        db.drop_all()


@pytest.fixture(scope='function')
def client(app):
    """Test client"""
    return app.test_client()


@pytest.fixture(scope='function')
def db_session(app):
    """Database session for testing"""
    with app.app_context():
        yield db.session
        db.session.rollback()


@pytest.fixture(scope='function')
def test_user(app):
    """Create a test user"""
    with app.app_context():
        user = User(
            id=uuid4(),
            email='test@example.com',
            username='testuser',
            first_name='Test',
            last_name='User',
            role='student'
        )
        user.set_password('testpassword123')
        
        leaderboard = Leaderboard(
            id=uuid4(),
            user_id=user.id,
            global_points=0,
            subject_points={},
            badges=[]
        )
        
        db.session.add(user)
        db.session.add(leaderboard)
        db.session.commit()
        
        yield {
            'user': user,
            'id': str(user.id),
            'email': user.email,
            'password': 'testpassword123'
        }
        
        db.session.delete(leaderboard)
        db.session.delete(user)
        db.session.commit()


@pytest.fixture(scope='function')
def auth_headers(client, test_user):
    """Get auth headers with valid token"""
    response = client.post('/api/auth/login', json={
        'email': test_user['email'],
        'password': test_user['password']
    })
    
    data = response.get_json()
    token = data.get('access_token', '')
    
    return {'Authorization': f'Bearer {token}'}
