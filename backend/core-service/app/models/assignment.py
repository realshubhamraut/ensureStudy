from app import db
from datetime import datetime
import uuid


class Assignment(db.Model):
    __tablename__ = 'assignments'
    
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    classroom_id = db.Column(db.String(36), db.ForeignKey('classrooms.id'), nullable=False)
    title = db.Column(db.String(255), nullable=False)
    description = db.Column(db.Text)
    due_date = db.Column(db.DateTime)
    points = db.Column(db.Integer)
    status = db.Column(db.String(20), default='published')  # draft, published
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    classroom = db.relationship('Classroom', backref='assignments')
    attachments = db.relationship('AssignmentAttachment', backref='assignment', cascade='all, delete-orphan')
    submissions = db.relationship('Submission', backref='assignment', cascade='all, delete-orphan')
    
    def to_dict(self, include_attachments=True, include_submissions=False, student_id=None):
        data = {
            'id': self.id,
            'classroom_id': self.classroom_id,
            'title': self.title,
            'description': self.description,
            'due_date': self.due_date.isoformat() if self.due_date else None,
            'points': self.points,
            'status': self.status,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
        }
        
        if include_attachments:
            data['attachments'] = [att.to_dict() for att in self.attachments]
        
        # For student view: include their submission
        if student_id:
            student_submission = next((s for s in self.submissions if s.student_id == student_id), None)
            data['my_submission'] = student_submission.to_dict(include_files=True) if student_submission else None
        
        # For teacher view: include submission count
        if include_submissions:
            data['submission_count'] = len(self.submissions)
            data['submissions'] = [s.to_dict(include_student=True, include_files=True) for s in self.submissions]
        
        return data


class AssignmentAttachment(db.Model):
    __tablename__ = 'assignment_attachments'
    
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    assignment_id = db.Column(db.String(36), db.ForeignKey('assignments.id'), nullable=False)
    type = db.Column(db.String(20), nullable=False)  # 'file' or 'link'
    url = db.Column(db.String(500))
    filename = db.Column(db.String(255))
    file_size = db.Column(db.Integer)
    
    def to_dict(self):
        return {
            'id': self.id,
            'assignment_id': self.assignment_id,
            'type': self.type,
            'url': self.url,
            'filename': self.filename,
            'file_size': self.file_size
        }


class Submission(db.Model):
    __tablename__ = 'submissions'
    
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    assignment_id = db.Column(db.String(36), db.ForeignKey('assignments.id'), nullable=False)
    student_id = db.Column(db.String(36), db.ForeignKey('users.id'), nullable=False)
    submitted_at = db.Column(db.DateTime, default=datetime.utcnow)
    status = db.Column(db.String(20), default='submitted')  # submitted, graded, returned
    grade = db.Column(db.Integer)
    feedback = db.Column(db.Text)
    
    # Relationships
    student = db.relationship('User', backref='submissions')
    files = db.relationship('SubmissionFile', backref='submission', cascade='all, delete-orphan')
    
    def to_dict(self, include_student=False, include_files=True):
        data = {
            'id': self.id,
            'assignment_id': self.assignment_id,
            'student_id': self.student_id,
            'submitted_at': self.submitted_at.isoformat() if self.submitted_at else None,
            'status': self.status,
            'grade': self.grade,
            'feedback': self.feedback
        }
        
        if include_student and self.student:
            data['student'] = {
                'id': self.student.id,
                'name': f"{self.student.first_name or ''} {self.student.last_name or ''}".strip() or self.student.username,
                'email': self.student.email
            }
        
        if include_files:
            data['files'] = [f.to_dict() for f in self.files]
        
        return data


class SubmissionFile(db.Model):
    __tablename__ = 'submission_files'
    
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    submission_id = db.Column(db.String(36), db.ForeignKey('submissions.id'), nullable=False)
    url = db.Column(db.String(500))
    filename = db.Column(db.String(255))
    file_size = db.Column(db.Integer)
    type = db.Column(db.String(20), default='file')  # 'file' or 'link'
    
    def to_dict(self):
        return {
            'id': self.id,
            'submission_id': self.submission_id,
            'url': self.url,
            'filename': self.filename,
            'file_size': self.file_size,
            'type': self.type
        }
