"""
Notification model for system notifications
"""
from datetime import datetime
from app import db
import uuid

class Notification(db.Model):
    __tablename__ = "notifications"
    
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = db.Column(db.String(36), db.ForeignKey("users.id"), nullable=False, index=True)
    
    # Notification type: stream, material, assignment, assessment, message, meet, result
    type = db.Column(db.String(50), nullable=False)
    title = db.Column(db.String(255), nullable=False)
    message = db.Column(db.Text)
    
    # Link to source (classroom_id, assignment_id, conversation_id etc)
    source_id = db.Column(db.String(36))
    source_type = db.Column(db.String(50))  # classroom, assignment, conversation, etc
    
    # Link for navigation
    action_url = db.Column(db.String(500))
    
    # Status
    is_read = db.Column(db.Boolean, default=False)
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    read_at = db.Column(db.DateTime)
    
    # Relationship
    user = db.relationship("User", backref=db.backref("notifications", lazy="dynamic"))
    
    def to_dict(self):
        return {
            "id": self.id,
            "user_id": self.user_id,
            "type": self.type,
            "title": self.title,
            "message": self.message,
            "source_id": self.source_id,
            "source_type": self.source_type,
            "action_url": self.action_url,
            "is_read": self.is_read,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "read_at": self.read_at.isoformat() if self.read_at else None
        }


def create_notification(user_id: str, type: str, title: str, message: str = None,
                        source_id: str = None, source_type: str = None, action_url: str = None):
    """Helper to create a notification"""
    notification = Notification(
        user_id=user_id,
        type=type,
        title=title,
        message=message,
        source_id=source_id,
        source_type=source_type,
        action_url=action_url
    )
    db.session.add(notification)
    db.session.commit()
    return notification


def notify_classroom_members(classroom_id: str, notification_type: str, title: str, message: str = None,
                             exclude_user_id: str = None, action_url: str = None, source_id: str = None):
    """Send notification to all classroom members"""
    from app.models.classroom import StudentClassroom, Classroom
    
    # Get classroom teacher
    classroom = Classroom.query.get(classroom_id)
    
    # Get all student members
    members = StudentClassroom.query.filter_by(classroom_id=classroom_id).all()
    
    notifications_created = 0
    
    for member in members:
        if exclude_user_id and member.student_id == exclude_user_id:
            continue
        
        try:
            create_notification(
                user_id=member.student_id,
                type=notification_type,
                title=title,
                message=message,
                source_id=source_id or classroom_id,
                source_type="classroom",
                action_url=action_url
            )
            notifications_created += 1
        except Exception as e:
            print(f"Failed to create notification for {member.student_id}: {e}")
    
    return notifications_created
