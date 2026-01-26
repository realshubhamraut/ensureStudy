"""
Classroom Model for Google Classroom-style join codes
"""
from datetime import datetime
from app import db
import uuid
import secrets


def generate_uuid():
    return str(uuid.uuid4())


def generate_join_code():
    """Generate a 6-character alphanumeric join code"""
    return secrets.token_urlsafe(4)[:6].upper()


class Classroom(db.Model):
    """Classroom with join code for students"""
    __tablename__ = "classrooms"
    
    id = db.Column(db.String(36), primary_key=True, default=generate_uuid)
    name = db.Column(db.String(100), nullable=False)
    grade = db.Column(db.String(20))  # "10", "11", "12"
    section = db.Column(db.String(20))  # "A", "B", "Science"
    subject = db.Column(db.String(100))  # Optional: specific subject
    
    # Join code (like Google Classroom)
    join_code = db.Column(db.String(6), unique=True, default=generate_join_code)
    
    # Creator (teacher)
    teacher_id = db.Column(db.String(36), db.ForeignKey("users.id"), nullable=False)
    
    # Organization (from teacher)
    organization_id = db.Column(db.String(36), db.ForeignKey("organizations.id"), nullable=False)
    
    # Status
    is_active = db.Column(db.Boolean, default=True)  # Can students join?
    
    # Syllabus
    syllabus_url = db.Column(db.String(500))  # Uploaded file URL (PDF, image)
    syllabus_content = db.Column(db.Text)      # Text/markdown content
    syllabus_filename = db.Column(db.String(255))  # Original filename
    syllabus_uploaded_at = db.Column(db.DateTime)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    teacher = db.relationship("User", foreign_keys=[teacher_id], backref="created_classrooms")
    organization = db.relationship("Organization", backref="classrooms")
    
    def regenerate_code(self):
        """Generate a new join code"""
        self.join_code = generate_join_code()
        return self.join_code
    
    def to_dict(self, include_students=False, include_teacher=False):
        data = {
            "id": self.id,
            "name": self.name,
            "grade": self.grade,
            "section": self.section,
            "subject": self.subject,
            "join_code": self.join_code,
            "teacher_id": self.teacher_id,
            "organization_id": self.organization_id,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "student_count": len(self.student_enrollments) if hasattr(self, 'student_enrollments') else 0,
            # Syllabus fields
            "syllabus_url": self.syllabus_url,
            "syllabus_content": self.syllabus_content,
            "syllabus_filename": self.syllabus_filename,
            "syllabus_uploaded_at": self.syllabus_uploaded_at.isoformat() if self.syllabus_uploaded_at else None,
            "has_syllabus": bool(self.syllabus_url or self.syllabus_content)
        }
        
        if include_teacher and self.teacher:
            data["teacher"] = {
                "id": self.teacher.id,
                "name": f"{self.teacher.first_name or ''} {self.teacher.last_name or ''}".strip() or self.teacher.username,
                "email": self.teacher.email
            }
        
        if include_students and hasattr(self, 'student_enrollments'):
            data["students"] = [e.to_dict() for e in self.student_enrollments]
        
        return data


class StudentClassroom(db.Model):
    """Join table: links students to classrooms"""
    __tablename__ = "student_classrooms"
    
    id = db.Column(db.String(36), primary_key=True, default=generate_uuid)
    student_id = db.Column(db.String(36), db.ForeignKey("users.id"), nullable=False)
    classroom_id = db.Column(db.String(36), db.ForeignKey("classrooms.id"), nullable=False)
    
    # When they joined
    joined_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Status
    is_active = db.Column(db.Boolean, default=True)
    
    # Unique constraint: one student per classroom
    __table_args__ = (
        db.UniqueConstraint('student_id', 'classroom_id', name='unique_student_classroom'),
    )
    
    # Relationships
    student = db.relationship("User", backref="classroom_enrollments")
    classroom = db.relationship("Classroom", backref="student_enrollments")
    
    def to_dict(self):
        return {
            "id": self.id,
            "student_id": self.student_id,
            "classroom_id": self.classroom_id,
            "joined_at": self.joined_at.isoformat() if self.joined_at else None,
            "is_active": self.is_active
        }


class ClassroomMaterial(db.Model):
    """Study materials uploaded to a classroom"""
    __tablename__ = "classroom_materials"
    
    id = db.Column(db.String(36), primary_key=True, default=generate_uuid)
    classroom_id = db.Column(db.String(36), db.ForeignKey("classrooms.id"), nullable=False)
    
    # File info
    name = db.Column(db.String(255), nullable=False)  # Original filename
    file_url = db.Column(db.String(500), nullable=False)  # Stored file URL
    file_type = db.Column(db.String(100))  # MIME type
    file_size = db.Column(db.Integer)  # Size in bytes
    
    # Organization
    subject = db.Column(db.String(100))  # Subject category
    description = db.Column(db.Text)  # Optional description
    
    # Uploader
    uploaded_by = db.Column(db.String(36), db.ForeignKey("users.id"), nullable=False)
    
    # Source tracking (for web-downloaded PDFs)
    source = db.Column(db.String(20), default='upload')  # 'upload' or 'web'
    source_url = db.Column(db.String(500))  # Original web URL before download
    
    # Timestamps
    uploaded_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Status
    is_active = db.Column(db.Boolean, default=True)
    
    # Indexing status for RAG
    indexing_status = db.Column(db.String(20), default='pending')  # pending, processing, completed, failed
    indexed_at = db.Column(db.DateTime)
    chunk_count = db.Column(db.Integer, default=0)
    indexing_error = db.Column(db.Text)
    
    # Relationships
    classroom = db.relationship("Classroom", backref="materials")
    uploader = db.relationship("User", backref="uploaded_materials")
    
    def to_dict(self):
        return {
            "id": self.id,
            "classroom_id": self.classroom_id,
            "name": self.name,
            "url": self.file_url,
            "type": self.file_type,
            "size": self.file_size,
            "subject": self.subject,
            "description": self.description,
            "uploaded_by": self.uploaded_by,
            "uploaded_at": self.uploaded_at.isoformat() if self.uploaded_at else None,
            "is_active": self.is_active,
            # Source tracking
            "source": self.source or "upload",  # 'upload' or 'web'
            "source_url": self.source_url,  # Original web URL
            # Indexing status for RAG
            "indexing_status": self.indexing_status,
            "indexed_at": self.indexed_at.isoformat() if self.indexed_at else None,
            "chunk_count": self.chunk_count
        }
