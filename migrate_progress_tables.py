"""
Database Migration: Add Progress Tracking Tables

Creates:
- study_streaks: Track daily study activity for streak calculation
- topic_interactions: Track when student interacts with topics

Run with: python migrate_progress_tables.py
"""
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend/core-service'))

from datetime import datetime
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.environ.get('DATABASE_URL', 'postgresql://user:password@localhost:5432/ensure_study')


def run_migration():
    """Run the migration to create progress tracking tables"""
    engine = create_engine(DATABASE_URL)
    
    with engine.connect() as conn:
        # Create study_streaks table
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS study_streaks (
                id VARCHAR(36) PRIMARY KEY,
                user_id VARCHAR(36) NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                study_date DATE NOT NULL,
                session_count INTEGER DEFAULT 0,
                questions_asked INTEGER DEFAULT 0,
                assessments_taken INTEGER DEFAULT 0,
                notes_viewed INTEGER DEFAULT 0,
                total_minutes INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE (user_id, study_date)
            );
        """))
        print("✓ Created study_streaks table")
        
        # Create index for faster lookups
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_study_streaks_user_date 
            ON study_streaks(user_id, study_date DESC);
        """))
        print("✓ Created index on study_streaks")
        
        # Create topic_interactions table
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS topic_interactions (
                id VARCHAR(36) PRIMARY KEY,
                user_id VARCHAR(36) NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                subtopic_id VARCHAR(36) REFERENCES subtopics(id) ON DELETE SET NULL,
                topic_id VARCHAR(36) REFERENCES topics(id) ON DELETE SET NULL,
                interaction_type VARCHAR(30) NOT NULL,
                source_id VARCHAR(36),
                source_type VARCHAR(30),
                detected_topic_name VARCHAR(200),
                confidence_score FLOAT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """))
        print("✓ Created topic_interactions table")
        
        # Create indexes for faster lookups
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_topic_interactions_user 
            ON topic_interactions(user_id);
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_topic_interactions_subtopic 
            ON topic_interactions(subtopic_id);
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_topic_interactions_created 
            ON topic_interactions(created_at DESC);
        """))
        print("✓ Created indexes on topic_interactions")
        
        conn.commit()
        print("\n✅ Migration complete!")


def rollback_migration():
    """Rollback the migration"""
    engine = create_engine(DATABASE_URL)
    
    with engine.connect() as conn:
        conn.execute(text("DROP TABLE IF EXISTS topic_interactions CASCADE;"))
        conn.execute(text("DROP TABLE IF EXISTS study_streaks CASCADE;"))
        conn.commit()
        print("✅ Rollback complete - tables dropped")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--rollback', action='store_true', help='Rollback the migration')
    args = parser.parse_args()
    
    if args.rollback:
        rollback_migration()
    else:
        run_migration()
