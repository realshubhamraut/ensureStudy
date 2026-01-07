"""
Announcement Model - Classroom announcements/stream posts
"""
from datetime import datetime
import uuid
from app import db


class Announcement(db.Model):
    __tablename__ = 'announcements'

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    classroom_id = db.Column(db.String(36), db.ForeignKey('classrooms.id'), nullable=False)
    teacher_id = db.Column(db.String(36), db.ForeignKey('users.id'), nullable=False)
    message = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    classroom = db.relationship('Classroom', backref=db.backref('announcements', lazy='dynamic'))
    teacher = db.relationship('User', backref=db.backref('announcements', lazy='dynamic'))

    def to_dict(self):
        return {
            'id': self.id,
            'classroom_id': self.classroom_id,
            'teacher_id': self.teacher_id,
            'message': self.message,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'teacher_name': f"{self.teacher.first_name} {self.teacher.last_name}" if self.teacher else None
        }
