"""
Seed sample progress data for demo purposes.

This script creates sample progress records for the current user
so the Progress page has data to display.

Run with: python seed_progress_data.py
"""
import os
import sys
from datetime import datetime, timedelta
from uuid import uuid4

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend/core-service'))

from dotenv import load_dotenv
load_dotenv()

from sqlalchemy import create_engine, text

DATABASE_URL = os.environ.get('DATABASE_URL', 'postgresql://user:password@localhost:5432/ensure_study')

# Sample progress data matching the frontend interface
SAMPLE_PROGRESS = [
    {'topic': 'Cell Structure', 'subject': 'Biology', 'confidence': 85, 'is_weak': False, 'times_studied': 8},
    {'topic': 'Photosynthesis', 'subject': 'Biology', 'confidence': 42, 'is_weak': True, 'times_studied': 3},
    {'topic': 'Quadratic Equations', 'subject': 'Math', 'confidence': 38, 'is_weak': True, 'times_studied': 5},
    {'topic': "Newton's Laws", 'subject': 'Physics', 'confidence': 72, 'is_weak': False, 'times_studied': 6},
    {'topic': 'World War II', 'subject': 'History', 'confidence': 35, 'is_weak': True, 'times_studied': 2},
    {'topic': 'Chemical Bonding', 'subject': 'Chemistry', 'confidence': 68, 'is_weak': False, 'times_studied': 4},
    {'topic': 'Thermodynamics', 'subject': 'Physics', 'confidence': 55, 'is_weak': False, 'times_studied': 5},
    {'topic': 'Organic Chemistry', 'subject': 'Chemistry', 'confidence': 45, 'is_weak': True, 'times_studied': 3},
    {'topic': 'Calculus Basics', 'subject': 'Math', 'confidence': 78, 'is_weak': False, 'times_studied': 7},
    {'topic': 'DNA Replication', 'subject': 'Biology', 'confidence': 62, 'is_weak': False, 'times_studied': 4},
]


def seed_progress_data():
    """Seed progress data for all student users"""
    engine = create_engine(DATABASE_URL)
    
    with engine.connect() as conn:
        # Get all student users
        result = conn.execute(text("SELECT id FROM users WHERE role = 'student' LIMIT 10"))
        users = [row[0] for row in result]
        
        if not users:
            print("No student users found. Creating test student...")
            # Create a test student if none exist
            test_id = str(uuid4())
            conn.execute(text("""
                INSERT INTO users (id, email, username, password_hash, role, first_name, last_name, is_active, created_at)
                VALUES (:id, 'test@example.com', 'teststudent', 'hash', 'student', 'Test', 'Student', true, NOW())
                ON CONFLICT (email) DO NOTHING
            """), {'id': test_id})
            conn.commit()
            users = [test_id]
        
        print(f"Seeding progress data for {len(users)} user(s)...")
        
        for user_id in users:
            # Clear existing progress for this user
            conn.execute(text("DELETE FROM progress WHERE user_id = :user_id"), {'user_id': user_id})
            
            for i, p in enumerate(SAMPLE_PROGRESS):
                # Stagger last_studied dates for realistic streak
                days_ago = i % 7  # Within last week
                last_studied = datetime.utcnow() - timedelta(days=days_ago, hours=i*2)
                
                progress_id = str(uuid4())
                conn.execute(text("""
                    INSERT INTO progress (id, user_id, topic, subject, confidence_score, is_weak, times_studied, last_studied, created_at, updated_at)
                    VALUES (:id, :user_id, :topic, :subject, :confidence, :is_weak, :times_studied, :last_studied, NOW(), NOW())
                """), {
                    'id': progress_id,
                    'user_id': user_id,
                    'topic': p['topic'],
                    'subject': p['subject'],
                    'confidence': p['confidence'],
                    'is_weak': p['is_weak'],
                    'times_studied': p['times_studied'],
                    'last_studied': last_studied
                })
            
            print(f"  ✓ Added {len(SAMPLE_PROGRESS)} topics for user {user_id[:8]}...")
        
        conn.commit()
        print("\n✅ Progress data seeded successfully!")


if __name__ == "__main__":
    seed_progress_data()
