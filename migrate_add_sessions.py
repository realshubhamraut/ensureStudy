"""
Migration: Add Tutor Sessions tables

Creates:
- tutor_sessions: Session tracking for resource chaining
- session_turns: Query turns with embeddings
- session_resources: Discovered resources with dedup metadata

Run with: python migrate_add_sessions.py
"""
import os
import sys
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sqlalchemy import create_engine, text


def get_db_url():
    """Get database URL from environment"""
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    db = os.getenv("POSTGRES_DB", "ensurestudy")
    user = os.getenv("POSTGRES_USER", "postgres")
    password = os.getenv("POSTGRES_PASSWORD", "postgres")
    return f"postgresql://{user}:{password}@{host}:{port}/{db}"


def run_migration():
    """Run the migration to add session tables"""
    db_url = get_db_url()
    engine = create_engine(db_url)
    
    print(f"🔗 Connecting to database...")
    print(f"   URL: {db_url.replace(os.getenv('POSTGRES_PASSWORD', 'postgres'), '***')}")
    
    with engine.connect() as conn:
        # Create tutor_sessions table
        print("\n📋 Creating tutor_sessions table...")
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS tutor_sessions (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                classroom_id UUID REFERENCES classrooms(id) ON DELETE SET NULL,
                created_at TIMESTAMP NOT NULL DEFAULT NOW(),
                last_active_at TIMESTAMP NOT NULL DEFAULT NOW(),
                expires_at TIMESTAMP,
                config JSONB DEFAULT '{"ttl_hours": 24, "max_resources": 25, "relatedness_threshold": 0.65, "relatedness_lookback": 3}'::jsonb,
                is_active BOOLEAN NOT NULL DEFAULT TRUE
            );
            
            CREATE INDEX IF NOT EXISTS idx_tutor_sessions_user_id ON tutor_sessions(user_id);
            CREATE INDEX IF NOT EXISTS idx_tutor_sessions_classroom_id ON tutor_sessions(classroom_id);
            CREATE INDEX IF NOT EXISTS idx_tutor_sessions_last_active ON tutor_sessions(last_active_at);
        """))
        print("   ✅ tutor_sessions created")
        
        # Create session_turns table
        print("\n📋 Creating session_turns table...")
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS session_turns (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                session_id UUID NOT NULL REFERENCES tutor_sessions(id) ON DELETE CASCADE,
                turn_number INTEGER NOT NULL,
                question TEXT NOT NULL,
                question_hash VARCHAR(64) NOT NULL,
                question_embedding JSONB,
                related_to_previous BOOLEAN DEFAULT FALSE,
                relatedness_score REAL,
                created_at TIMESTAMP NOT NULL DEFAULT NOW(),
                
                UNIQUE(session_id, turn_number)
            );
            
            CREATE INDEX IF NOT EXISTS idx_session_turns_session_id ON session_turns(session_id);
            CREATE INDEX IF NOT EXISTS idx_session_turns_question_hash ON session_turns(question_hash);
        """))
        print("   ✅ session_turns created")
        
        # Create session_resources table
        print("\n📋 Creating session_resources table...")
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS session_resources (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                session_id UUID NOT NULL REFERENCES tutor_sessions(id) ON DELETE CASCADE,
                resource_type VARCHAR(20) NOT NULL,
                source VARCHAR(20) NOT NULL,
                url TEXT,
                canonical_url TEXT,
                content_hash VARCHAR(64),
                title VARCHAR(500),
                preview_summary VARCHAR(300),
                qdrant_collection VARCHAR(100),
                qdrant_point_ids JSONB DEFAULT '[]'::jsonb,
                inline_render BOOLEAN DEFAULT FALSE,
                inline_html TEXT,
                page_number INTEGER,
                bbox JSONB,
                signed_url TEXT,
                inserted_at TIMESTAMP NOT NULL DEFAULT NOW(),
                last_referenced_at TIMESTAMP NOT NULL DEFAULT NOW()
            );
            
            CREATE INDEX IF NOT EXISTS idx_session_resources_session_id ON session_resources(session_id);
            CREATE INDEX IF NOT EXISTS idx_session_resources_canonical_url ON session_resources(canonical_url);
            CREATE INDEX IF NOT EXISTS idx_session_resources_content_hash ON session_resources(content_hash);
            CREATE INDEX IF NOT EXISTS idx_session_resources_last_ref ON session_resources(last_referenced_at);
        """))
        print("   ✅ session_resources created")
        
        conn.commit()
        
    print("\n✅ Migration completed successfully!")
    print("   Tables created:")
    print("   - tutor_sessions (session tracking)")
    print("   - session_turns (query history with embeddings)")
    print("   - session_resources (discovered resources)")


def rollback_migration():
    """Rollback the migration"""
    db_url = get_db_url()
    engine = create_engine(db_url)
    
    print("⚠️  Rolling back session tables...")
    
    with engine.connect() as conn:
        conn.execute(text("DROP TABLE IF EXISTS session_resources CASCADE;"))
        conn.execute(text("DROP TABLE IF EXISTS session_turns CASCADE;"))
        conn.execute(text("DROP TABLE IF EXISTS tutor_sessions CASCADE;"))
        conn.commit()
    
    print("✅ Rollback completed")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run session tables migration")
    parser.add_argument("--rollback", action="store_true", help="Rollback the migration")
    args = parser.parse_args()
    
    if args.rollback:
        rollback_migration()
    else:
        run_migration()
