#!/usr/bin/env python3
"""
Database Migration: Session Intelligence Fields

Adds columns for session intelligence:
- last_topic_vector (JSONB): Centroid of related turn embeddings
- last_decision (VARCHAR): "related" or "new_topic"
- consecutive_borderline (INTEGER): Count for hysteresis

Usage:
    python scripts/migrate_session_intelligence.py
"""
import os
import sys
from sqlalchemy import create_engine, text
from datetime import datetime


def get_database_url():
    """Get database URL from environment or default"""
    return os.getenv(
        "DATABASE_URL",
        "postgresql://ensure_study_user:secure_password_123@localhost:5432/ensure_study"
    )


def run_migration():
    """Run the migration"""
    print(f"[{datetime.now().isoformat()}] Starting Session Intelligence migration...")
    
    database_url = get_database_url()
    print(f"Database: {database_url.split('@')[-1]}")
    
    try:
        engine = create_engine(database_url)
        
        with engine.connect() as conn:
            # Check if columns already exist
            check_query = text("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'tutor_sessions' 
                AND column_name IN ('last_topic_vector', 'last_decision', 'consecutive_borderline')
            """)
            result = conn.execute(check_query)
            existing_columns = [row[0] for row in result]
            
            if len(existing_columns) == 3:
                print("✅ All session intelligence columns already exist. Skipping migration.")
                return 0
            
            # Add missing columns
            migration_sql = []
            
            if 'last_topic_vector' not in existing_columns:
                migration_sql.append("""
                    ALTER TABLE tutor_sessions 
                    ADD COLUMN last_topic_vector JSONB
                """)
                print("  Adding: last_topic_vector (JSONB)")
            
            if 'last_decision' not in existing_columns:
                migration_sql.append("""
                    ALTER TABLE tutor_sessions 
                    ADD COLUMN last_decision VARCHAR(20) DEFAULT 'new_topic'
                """)
                print("  Adding: last_decision (VARCHAR)")
            
            if 'consecutive_borderline' not in existing_columns:
                migration_sql.append("""
                    ALTER TABLE tutor_sessions 
                    ADD COLUMN consecutive_borderline INTEGER DEFAULT 0
                """)
                print("  Adding: consecutive_borderline (INTEGER)")
            
            # Execute migrations
            for sql in migration_sql:
                conn.execute(text(sql))
            
            conn.commit()
            print(f"\n✅ Migration complete! Added {len(migration_sql)} column(s).")
            
            # Verify
            verify_query = text("""
                SELECT column_name, data_type, column_default
                FROM information_schema.columns 
                WHERE table_name = 'tutor_sessions' 
                AND column_name IN ('last_topic_vector', 'last_decision', 'consecutive_borderline')
                ORDER BY column_name
            """)
            result = conn.execute(verify_query)
            
            print("\nVerification:")
            for row in result:
                print(f"  {row[0]}: {row[1]} (default: {row[2]})")
            
            return 0
            
    except Exception as e:
        print(f"\n❌ Migration failed: {e}")
        return 1


def rollback_migration():
    """Rollback the migration (for testing)"""
    print("Rolling back session intelligence columns...")
    
    database_url = get_database_url()
    engine = create_engine(database_url)
    
    with engine.connect() as conn:
        rollback_sql = """
            ALTER TABLE tutor_sessions 
            DROP COLUMN IF EXISTS last_topic_vector,
            DROP COLUMN IF EXISTS last_decision,
            DROP COLUMN IF EXISTS consecutive_borderline
        """
        conn.execute(text(rollback_sql))
        conn.commit()
        print("✅ Rollback complete")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--rollback":
        sys.exit(rollback_migration())
    else:
        sys.exit(run_migration())
