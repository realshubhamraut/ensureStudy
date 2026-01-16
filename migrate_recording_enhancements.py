"""
Migration: Add enhanced recording fields for transcription and analytics
"""
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sqlalchemy import create_engine, text

DATABASE_URL = os.getenv(
    'DATABASE_URL',
    'postgresql://ensure_study_user:secure_password_123@localhost:5432/ensure_study'
)

def migrate():
    """Add enhanced fields to meeting_recordings table"""
    engine = create_engine(DATABASE_URL)
    
    with engine.connect() as conn:
        # Add new columns to meeting_recordings
        migrations = [
            # Speaker and language info
            "ALTER TABLE meeting_recordings ADD COLUMN IF NOT EXISTS speaker_count INTEGER DEFAULT 0",
            "ALTER TABLE meeting_recordings ADD COLUMN IF NOT EXISTS language VARCHAR(10) DEFAULT 'en'",
            "ALTER TABLE meeting_recordings ADD COLUMN IF NOT EXISTS word_count INTEGER DEFAULT 0",
            
            # Processing progress (0-100)
            "ALTER TABLE meeting_recordings ADD COLUMN IF NOT EXISTS processing_progress INTEGER DEFAULT 0",
            
            # Thumbnail for VOD display
            "ALTER TABLE meeting_recordings ADD COLUMN IF NOT EXISTS thumbnail_url VARCHAR(500)",
            
            # Key topics extracted from transcript (JSON array as text)
            "ALTER TABLE meeting_recordings ADD COLUMN IF NOT EXISTS key_topics TEXT",
            
            # MongoDB document ID reference for transcript
            "ALTER TABLE meeting_recordings ADD COLUMN IF NOT EXISTS mongo_transcript_id VARCHAR(36)",
            
            # Qdrant index status
            "ALTER TABLE meeting_recordings ADD COLUMN IF NOT EXISTS is_indexed BOOLEAN DEFAULT FALSE",
            "ALTER TABLE meeting_recordings ADD COLUMN IF NOT EXISTS indexed_at TIMESTAMP",
            
            # Update status enum to include more states
            # Status: uploading, processing, transcribing, embedding, ready, failed
        ]
        
        for migration in migrations:
            try:
                conn.execute(text(migration))
                print(f"✅ {migration[:60]}...")
            except Exception as e:
                if 'already exists' in str(e).lower() or 'duplicate' in str(e).lower():
                    print(f"⏭️  Column already exists, skipping...")
                else:
                    print(f"❌ Error: {e}")
        
        conn.commit()
        print("\n✅ Migration complete!")


def rollback():
    """Remove added columns (for development only)"""
    engine = create_engine(DATABASE_URL)
    
    with engine.connect() as conn:
        columns = [
            'speaker_count', 'language', 'word_count', 'processing_progress',
            'thumbnail_url', 'key_topics', 'mongo_transcript_id', 
            'is_indexed', 'indexed_at'
        ]
        
        for col in columns:
            try:
                conn.execute(text(f"ALTER TABLE meeting_recordings DROP COLUMN IF EXISTS {col}"))
                print(f"Dropped {col}")
            except Exception as e:
                print(f"Error dropping {col}: {e}")
        
        conn.commit()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == 'rollback':
        rollback()
    else:
        migrate()
