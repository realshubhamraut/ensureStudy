"""
Meeting Models for Live Video Conferencing System
Supports scheduling, live sessions, participants, and recordings
"""
from datetime import datetime
from app import db
import uuid


def generate_uuid():
    return str(uuid.uuid4())


class Meeting(db.Model):
    """Live meeting/video call in a classroom"""
    __tablename__ = "meetings"
    
    id = db.Column(db.String(36), primary_key=True, default=generate_uuid)
    classroom_id = db.Column(db.String(36), db.ForeignKey("classrooms.id"), nullable=False, index=True)
    
    # Meeting info
    title = db.Column(db.String(255), nullable=False)
    description = db.Column(db.Text)
    
    # Host (teacher)
    host_id = db.Column(db.String(36), db.ForeignKey("users.id"), nullable=False)
    
    # Status: scheduled, live, ended, cancelled
    status = db.Column(db.String(20), default='scheduled', index=True)
    
    # Scheduling
    scheduled_at = db.Column(db.DateTime, index=True)
    started_at = db.Column(db.DateTime)
    ended_at = db.Column(db.DateTime)
    duration_minutes = db.Column(db.Integer)
    
    # Settings
    max_participants = db.Column(db.Integer, default=50)
    is_recording_enabled = db.Column(db.Boolean, default=True)
    
    # WebRTC room info
    room_id = db.Column(db.String(100), unique=True)
    meeting_link = db.Column(db.String(500))
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    classroom = db.relationship("Classroom", backref=db.backref("meetings", lazy="dynamic"))
    host = db.relationship("User", foreign_keys=[host_id], backref="hosted_meetings")
    
    def to_dict(self, include_participants=False, include_recordings=False):
        data = {
            "id": self.id,
            "classroom_id": self.classroom_id,
            "title": self.title,
            "description": self.description,
            "host_id": self.host_id,
            "status": self.status,
            "scheduled_at": self.scheduled_at.isoformat() if self.scheduled_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "duration_minutes": self.duration_minutes,
            "max_participants": self.max_participants,
            "is_recording_enabled": self.is_recording_enabled,
            "room_id": self.room_id,
            "meeting_link": self.meeting_link,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "participant_count": len(list(self.participants)) if hasattr(self, 'participants') else 0
        }
        
        if include_participants:
            data["participants"] = [p.to_dict() for p in self.participants]
        
        if include_recordings:
            data["recordings"] = [r.to_dict() for r in self.recordings]
        
        if self.host:
            data["host"] = {
                "id": self.host.id,
                "name": f"{self.host.first_name or ''} {self.host.last_name or ''}".strip() or self.host.username,
                "email": self.host.email
            }
        
        return data


class MeetingParticipant(db.Model):
    """Track participants in a meeting"""
    __tablename__ = "meeting_participants"
    
    id = db.Column(db.String(36), primary_key=True, default=generate_uuid)
    meeting_id = db.Column(db.String(36), db.ForeignKey("meetings.id"), nullable=False, index=True)
    user_id = db.Column(db.String(36), db.ForeignKey("users.id"), nullable=False, index=True)
    
    # Role: host, co-host, attendee
    role = db.Column(db.String(20), default='attendee')
    
    # Attendance tracking
    joined_at = db.Column(db.DateTime)
    left_at = db.Column(db.DateTime)
    duration_seconds = db.Column(db.Integer)
    
    # Connection info
    connection_quality = db.Column(db.String(20))  # good, fair, poor
    device_type = db.Column(db.String(50))  # desktop, mobile, tablet
    
    # Unique constraint: one user per meeting
    __table_args__ = (
        db.UniqueConstraint('meeting_id', 'user_id', name='unique_meeting_participant'),
    )
    
    # Relationships
    meeting = db.relationship("Meeting", backref=db.backref("participants", lazy="dynamic"))
    user = db.relationship("User", backref="meeting_participations")
    
    def to_dict(self):
        data = {
            "id": self.id,
            "meeting_id": self.meeting_id,
            "user_id": self.user_id,
            "role": self.role,
            "joined_at": self.joined_at.isoformat() if self.joined_at else None,
            "left_at": self.left_at.isoformat() if self.left_at else None,
            "duration_seconds": self.duration_seconds,
            "connection_quality": self.connection_quality,
            "device_type": self.device_type
        }
        
        if self.user:
            data["user"] = {
                "id": self.user.id,
                "name": f"{self.user.first_name or ''} {self.user.last_name or ''}".strip() or self.user.username,
                "email": self.user.email
            }
        
        return data


class MeetingRecording(db.Model):
    """Recording of a meeting session"""
    __tablename__ = "meeting_recordings"
    
    id = db.Column(db.String(36), primary_key=True, default=generate_uuid)
    meeting_id = db.Column(db.String(36), db.ForeignKey("meetings.id"), nullable=False, index=True)
    
    # Storage info
    storage_url = db.Column(db.String(500), nullable=False)
    storage_provider = db.Column(db.String(50), default='local')  # local, s3, gcs
    
    # File info
    file_size = db.Column(db.BigInteger)  # bytes
    duration_seconds = db.Column(db.Integer)
    format = db.Column(db.String(20), default='webm')  # webm, mp4
    
    # Processing status: uploading, processing, ready, failed
    status = db.Column(db.String(20), default='uploading', index=True)
    processed_at = db.Column(db.DateTime)
    error_message = db.Column(db.Text)
    
    # Transcript reference (stored in MongoDB)
    transcript_id = db.Column(db.String(36))
    has_transcript = db.Column(db.Boolean, default=False)
    transcript_text = db.Column(db.Text)  # Full transcript text
    
    # Summary (brief version stored here, full in MongoDB)
    summary_brief = db.Column(db.Text)
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    meeting = db.relationship("Meeting", backref=db.backref("recordings", lazy="dynamic"))
    
    def to_dict(self):
        return {
            "id": self.id,
            "meeting_id": self.meeting_id,
            "storage_url": self.storage_url,
            "storage_provider": self.storage_provider,
            "file_size": self.file_size,
            "duration_seconds": self.duration_seconds,
            "format": self.format,
            "status": self.status,
            "processed_at": self.processed_at.isoformat() if self.processed_at else None,
            "error_message": self.error_message,
            "transcript_id": self.transcript_id,
            "has_transcript": self.has_transcript,
            "transcript_text": self.transcript_text,
            "summary_brief": self.summary_brief,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


# Helper functions
def create_meeting(classroom_id: str, host_id: str, title: str, **kwargs) -> Meeting:
    """Create a new meeting"""
    import os
    meeting_id = generate_uuid()
    room_id = f"{classroom_id[:8]}-{meeting_id[:8]}"
    # Use environment variable for frontend URL, default to localhost for development
    frontend_url = os.getenv('FRONTEND_URL', 'http://localhost:3000')
    meeting = Meeting(
        id=meeting_id,
        classroom_id=classroom_id,
        host_id=host_id,
        title=title,
        room_id=room_id,
        meeting_link=f"{frontend_url}/meet/{meeting_id}",  # Use meeting id, not room_id
        **kwargs
    )
    db.session.add(meeting)
    db.session.commit()
    return meeting


def start_meeting(meeting_id: str) -> Meeting:
    """Start a scheduled meeting"""
    meeting = Meeting.query.get(meeting_id)
    if meeting:
        meeting.status = 'live'
        meeting.started_at = datetime.utcnow()
        db.session.commit()
    return meeting


def end_meeting(meeting_id: str) -> Meeting:
    """End a live meeting"""
    meeting = Meeting.query.get(meeting_id)
    if meeting and meeting.started_at:
        meeting.status = 'ended'
        meeting.ended_at = datetime.utcnow()
        delta = meeting.ended_at - meeting.started_at
        meeting.duration_minutes = int(delta.total_seconds() / 60)
        db.session.commit()
    return meeting
