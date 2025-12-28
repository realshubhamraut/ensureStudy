"""
Exam Evaluation Models for Teacher Answer Evaluation System
"""
from datetime import datetime
from app import db
import uuid


def generate_uuid():
    return str(uuid.uuid4())


class ExamSession(db.Model):
    """Exam session created by teacher for evaluation"""
    __tablename__ = "exam_sessions"
    
    id = db.Column(db.String(36), primary_key=True, default=generate_uuid)
    name = db.Column(db.String(255), nullable=False)  # e.g., "Physics Mid Term - December 2024"
    exam_type = db.Column(db.String(50), nullable=False)  # mid_term, unit_test, surprise_test, etc.
    subject = db.Column(db.String(100), nullable=False)
    class_name = db.Column(db.String(50), nullable=False)  # Class 10, Class 12, etc.
    date = db.Column(db.Date, nullable=False)
    
    teacher_id = db.Column(db.String(36), db.ForeignKey("users.id"), nullable=False)
    organization_id = db.Column(db.String(36), db.ForeignKey("organizations.id"), nullable=False)
    
    # Exam details
    total_marks = db.Column(db.Integer, default=0)
    question_paper = db.Column(db.JSON, default=dict)  # Stores parsed question paper data
    
    # Status: in_progress, evaluating, completed, results_declared
    status = db.Column(db.String(30), default="in_progress")
    results_declared_at = db.Column(db.DateTime, nullable=True)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    evaluations = db.relationship("StudentEvaluation", backref="exam_session", cascade="all, delete-orphan", lazy="dynamic")
    
    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "exam_type": self.exam_type,
            "subject": self.subject,
            "class_name": self.class_name,
            "date": self.date.isoformat() if self.date else None,
            "teacher_id": self.teacher_id,
            "organization_id": self.organization_id,
            "total_marks": self.total_marks,
            "question_paper": self.question_paper,
            "status": self.status,
            "results_declared_at": self.results_declared_at.isoformat() if self.results_declared_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "evaluation_count": self.evaluations.count() if self.evaluations else 0
        }


class StudentEvaluation(db.Model):
    """Individual student's evaluation for an exam session"""
    __tablename__ = "student_evaluations"
    
    id = db.Column(db.String(36), primary_key=True, default=generate_uuid)
    exam_session_id = db.Column(db.String(36), db.ForeignKey("exam_sessions.id"), nullable=False)
    student_id = db.Column(db.String(36), db.ForeignKey("users.id"), nullable=False)
    
    # Per-question evaluations stored as JSON array
    # Each item: {question_number, extracted_text, score, max_marks, feedback, image_url}
    question_evaluations = db.Column(db.JSON, default=list)
    
    # Aggregate scores
    total_score = db.Column(db.Float, default=0)
    max_score = db.Column(db.Float, default=0)
    percentage = db.Column(db.Float, default=0)
    grade = db.Column(db.String(10), nullable=True)  # A+, A, B, C, D, F
    
    # Status: pending, in_progress, evaluated
    status = db.Column(db.String(30), default="pending")
    
    # Teacher feedback
    teacher_comments = db.Column(db.Text, nullable=True)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    __table_args__ = (
        db.UniqueConstraint('exam_session_id', 'student_id', name='unique_exam_student'),
    )
    
    def calculate_grade(self):
        """Calculate grade based on percentage"""
        if self.percentage >= 90:
            self.grade = "A+"
        elif self.percentage >= 80:
            self.grade = "A"
        elif self.percentage >= 70:
            self.grade = "B"
        elif self.percentage >= 60:
            self.grade = "C"
        elif self.percentage >= 50:
            self.grade = "D"
        else:
            self.grade = "F"
        return self.grade
    
    def to_dict(self, include_student=False):
        result = {
            "id": self.id,
            "exam_session_id": self.exam_session_id,
            "student_id": self.student_id,
            "question_evaluations": self.question_evaluations,
            "total_score": self.total_score,
            "max_score": self.max_score,
            "percentage": self.percentage,
            "grade": self.grade,
            "status": self.status,
            "teacher_comments": self.teacher_comments,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }
        
        if include_student:
            from app.models.user import User
            student = User.query.get(self.student_id)
            if student:
                result["student"] = {
                    "id": student.id,
                    "name": f"{student.first_name or ''} {student.last_name or ''}".strip(),
                    "email": student.email,
                    "username": student.username
                }
        
        return result
