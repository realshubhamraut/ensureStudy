"""
Migration script to add meetings tables.
Run: python migrate_meetings.py
"""
import sys
sys.path.insert(0, 'backend/core-service')

from app import create_app, db
from app.models.meeting import Meeting, MeetingParticipant, MeetingRecording

if __name__ == '__main__':
    app = create_app()
    
    with app.app_context():
        print("Creating meetings tables...")
        
        # Create tables
        db.create_all()
        
        print("✅ Meetings tables created successfully!")
        print("   - meetings")
        print("   - meeting_participants")
        print("   - meeting_recordings")
