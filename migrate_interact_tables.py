"""
Database Migration: Create Interact Module Tables

Creates:
- conversations: Chat threads
- conversation_participants: Users in conversations
- messages: Chat messages with moderation
- interaction_analytics: Usage tracking

Run with: python migrate_interact_tables.py
"""
import os
import sys
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

DATABASE_URL = os.environ.get('DATABASE_URL', 'postgresql://user:password@localhost:5432/ensure_study')


def run_migration():
    """Run the migration to create interact tables"""
    engine = create_engine(DATABASE_URL)
    
    with engine.connect() as conn:
        # Create conversations table
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS conversations (
                id VARCHAR(36) PRIMARY KEY,
                type VARCHAR(20) NOT NULL DEFAULT 'direct',
                title VARCHAR(200),
                created_by VARCHAR(36) NOT NULL REFERENCES users(id),
                classroom_id VARCHAR(36) REFERENCES classrooms(id),
                is_moderated BOOLEAN DEFAULT true,
                is_active BOOLEAN DEFAULT true,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_message_at TIMESTAMP
            );
        """))
        print("✓ Created conversations table")
        
        # Create conversation_participants table
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS conversation_participants (
                id VARCHAR(36) PRIMARY KEY,
                conversation_id VARCHAR(36) NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                user_id VARCHAR(36) NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                role VARCHAR(20),
                joined_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_read_at TIMESTAMP,
                is_muted BOOLEAN DEFAULT false,
                is_admin BOOLEAN DEFAULT false,
                UNIQUE(conversation_id, user_id)
            );
        """))
        print("✓ Created conversation_participants table")
        
        # Create messages table
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS messages (
                id VARCHAR(36) PRIMARY KEY,
                conversation_id VARCHAR(36) NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                sender_id VARCHAR(36) NOT NULL REFERENCES users(id),
                content TEXT NOT NULL,
                message_type VARCHAR(20) DEFAULT 'text',
                reply_to_id VARCHAR(36) REFERENCES messages(id),
                is_flagged BOOLEAN DEFAULT false,
                flag_reason VARCHAR(200),
                moderated_by VARCHAR(36) REFERENCES users(id),
                moderated_at TIMESTAMP,
                moderation_action VARCHAR(20),
                is_deleted BOOLEAN DEFAULT false,
                is_edited BOOLEAN DEFAULT false,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                edited_at TIMESTAMP
            );
        """))
        print("✓ Created messages table")
        
        # Create interaction_analytics table
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS interaction_analytics (
                id VARCHAR(36) PRIMARY KEY,
                user_id VARCHAR(36) NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                date DATE NOT NULL,
                messages_sent INTEGER DEFAULT 0,
                messages_received INTEGER DEFAULT 0,
                conversations_started INTEGER DEFAULT 0,
                avg_response_time_seconds INTEGER,
                UNIQUE(user_id, date)
            );
        """))
        print("✓ Created interaction_analytics table")
        
        # Create indexes
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_conv_participants_user ON conversation_participants(user_id);
            CREATE INDEX IF NOT EXISTS idx_conv_participants_conv ON conversation_participants(conversation_id);
            CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id);
            CREATE INDEX IF NOT EXISTS idx_messages_sender ON messages(sender_id);
            CREATE INDEX IF NOT EXISTS idx_messages_created ON messages(created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_messages_flagged ON messages(is_flagged) WHERE is_flagged = true;
            CREATE INDEX IF NOT EXISTS idx_analytics_user_date ON interaction_analytics(user_id, date);
        """))
        print("✓ Created indexes")
        
        conn.commit()
        print("\n✅ Interact migration complete!")


def rollback_migration():
    """Rollback the migration"""
    engine = create_engine(DATABASE_URL)
    
    with engine.connect() as conn:
        conn.execute(text("DROP TABLE IF EXISTS interaction_analytics CASCADE;"))
        conn.execute(text("DROP TABLE IF EXISTS messages CASCADE;"))
        conn.execute(text("DROP TABLE IF EXISTS conversation_participants CASCADE;"))
        conn.execute(text("DROP TABLE IF EXISTS conversations CASCADE;"))
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
