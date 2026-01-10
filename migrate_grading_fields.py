"""
Migration: Add AI Grading Fields to Submissions
Run with: python migrate_grading_fields.py
"""
import os
import sys

# Add parent dir to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from backend.core_service.app import create_app, db
from sqlalchemy import text

def migrate():
    """Add AI grading fields to submissions table"""
    app = create_app()
    
    with app.app_context():
        conn = db.engine.connect()
        
        # Check if columns already exist
        result = conn.execute(text("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'submissions' 
            AND column_name IN ('ai_graded', 'ai_confidence', 'graded_at', 'detailed_feedback')
        """))
        existing_columns = [row[0] for row in result]
        
        migrations = []
        
        if 'ai_graded' not in existing_columns:
            migrations.append("ALTER TABLE submissions ADD COLUMN ai_graded BOOLEAN DEFAULT FALSE")
            
        if 'ai_confidence' not in existing_columns:
            migrations.append("ALTER TABLE submissions ADD COLUMN ai_confidence FLOAT")
            
        if 'graded_at' not in existing_columns:
            migrations.append("ALTER TABLE submissions ADD COLUMN graded_at TIMESTAMP")
            
        if 'detailed_feedback' not in existing_columns:
            migrations.append("ALTER TABLE submissions ADD COLUMN detailed_feedback JSONB")
        
        if migrations:
            print(f"Running {len(migrations)} migrations...")
            for sql in migrations:
                print(f"  Executing: {sql[:60]}...")
                conn.execute(text(sql))
            conn.commit()
            print("✓ Migrations complete!")
        else:
            print("✓ All columns already exist, no migration needed.")
        
        conn.close()

if __name__ == '__main__':
    migrate()
