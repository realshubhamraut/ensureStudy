"""
Migration: Add transcript_text column to meeting_recordings table

Run: python migrate_add_transcript_text.py
"""
import sys
sys.path.insert(0, 'backend/core-service')

from app import create_app, db
from sqlalchemy import text

def migrate():
    app = create_app()
    with app.app_context():
        # Check if column already exists
        result = db.session.execute(text("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'meeting_recordings' AND column_name = 'transcript_text'
        """))
        exists = result.fetchone()
        
        if exists:
            print("transcript_text column already exists")
            return
        
        # Add the column
        print("Adding transcript_text column to meeting_recordings table...")
        db.session.execute(text("""
            ALTER TABLE meeting_recordings 
            ADD COLUMN transcript_text TEXT
        """))
        db.session.commit()
        print("Migration complete!")

if __name__ == "__main__":
    migrate()
