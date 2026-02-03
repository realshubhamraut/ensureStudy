"""
Database Migration: Add Learning Agent Tables

Run this script to add the new tables and columns for the Type 5 Learning Agent.

Usage:
    python -c "exec(open('migrations/add_learning_agent_tables.py').read())"
    
Or run the SQL directly in your database client.
"""

# SQL Migration Script
MIGRATION_SQL = """
-- ============================================================================
-- TYPE 5 LEARNING AGENT TABLES MIGRATION
-- ============================================================================

-- 1. Add columns to topic_questions table
ALTER TABLE topic_questions ADD COLUMN IF NOT EXISTS question_hash VARCHAR(64);
ALTER TABLE topic_questions ADD COLUMN IF NOT EXISTS embedding_vector_id VARCHAR(100);
ALTER TABLE topic_questions ADD COLUMN IF NOT EXISTS auto_generated BOOLEAN DEFAULT FALSE;

-- Create index for fast hash lookups (duplicate detection)
CREATE INDEX IF NOT EXISTS idx_topic_question_hash ON topic_questions(question_hash);

-- 2. Create question_effectiveness table
CREATE TABLE IF NOT EXISTS question_effectiveness (
    question_id VARCHAR(36) PRIMARY KEY REFERENCES topic_questions(id) ON DELETE CASCADE,
    discrimination_index FLOAT DEFAULT 0.0,
    difficulty_index FLOAT DEFAULT 0.5,
    distractor_quality JSON,
    effectiveness_score FLOAT DEFAULT 0.5,
    sample_size INTEGER DEFAULT 0,
    total_attempts INTEGER DEFAULT 0,
    correct_attempts INTEGER DEFAULT 0,
    avg_response_time_ms INTEGER,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 3. Create learning_agent_memory table
CREATE TABLE IF NOT EXISTS learning_agent_memory (
    id VARCHAR(36) PRIMARY KEY,
    topic_id VARCHAR(36),
    calibrated_difficulty FLOAT DEFAULT 0.5,
    target_success_rate FLOAT DEFAULT 0.7,
    actual_success_rate FLOAT,
    preferred_question_types JSON,
    avoided_patterns JSON,
    successful_prompts JSON,
    total_questions INTEGER DEFAULT 0,
    effective_questions INTEGER DEFAULT 0,
    needs_more_questions BOOLEAN DEFAULT TRUE,
    learning_iterations INTEGER DEFAULT 0,
    last_learning_at TIMESTAMP,
    improvement_score FLOAT DEFAULT 0.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create index for topic lookups
CREATE INDEX IF NOT EXISTS idx_learning_memory_topic ON learning_agent_memory(topic_id);

-- 4. Backfill question_hash for existing questions (optional, run separately)
-- UPDATE topic_questions 
-- SET question_hash = MD5(LOWER(TRIM(question_text)))
-- WHERE question_hash IS NULL;

-- ============================================================================
-- VERIFICATION QUERIES
-- ============================================================================
-- Check new columns exist:
-- SELECT column_name FROM information_schema.columns WHERE table_name = 'topic_questions' AND column_name IN ('question_hash', 'embedding_vector_id', 'auto_generated');

-- Check new tables exist:
-- SELECT table_name FROM information_schema.tables WHERE table_name IN ('question_effectiveness', 'learning_agent_memory');
"""

def run_migration():
    """Run the migration using SQLAlchemy."""
    import os
    import sys
    
    # Add the app to the path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    try:
        from app import create_app, db
        
        app = create_app()
        
        with app.app_context():
            # Create all tables based on models
            db.create_all()
            print("✓ Created/updated all tables based on models")
            
            # Add indexes if they don't exist
            try:
                db.session.execute("CREATE INDEX IF NOT EXISTS idx_topic_question_hash ON topic_questions(question_hash)")
                db.session.execute("CREATE INDEX IF NOT EXISTS idx_learning_memory_topic ON learning_agent_memory(topic_id)")
                db.session.commit()
                print("✓ Created indexes")
            except Exception as e:
                print(f"Note: Index creation skipped (may already exist): {e}")
            
            print("✓ Learning Agent tables migration complete!")
            
    except Exception as e:
        print(f"Migration error: {e}")
        print("\nYou can also run the following SQL manually:")
        print(MIGRATION_SQL)


if __name__ == "__main__":
    run_migration()
