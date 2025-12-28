"""
Student Profile Model for Personalization
"""
from datetime import datetime
from app import db
import uuid


def generate_uuid():
    return str(uuid.uuid4())


class StudentProfile(db.Model):
    """Extended profile for student personalization"""
    __tablename__ = "student_profiles"
    
    id = db.Column(db.String(36), primary_key=True, default=generate_uuid)
    user_id = db.Column(db.String(36), db.ForeignKey("users.id"), nullable=False, unique=True)
    
    # Academic info
    grade_level = db.Column(db.String(20))  # "10", "11", "12", "UG", etc.
    board = db.Column(db.String(50))  # CBSE, ICSE, State Board
    stream = db.Column(db.String(50))  # Science, Commerce, Arts
    
    # Target exams (JSON array)
    target_exams = db.Column(db.JSON, default=list)  # ["JEE", "NEET", "BOARDS"]
    
    # Subjects of focus (JSON array)
    subjects = db.Column(db.JSON, default=list)  # ["Physics", "Chemistry", "Math"]
    
    # Study goals (JSON)
    study_goals = db.Column(db.JSON, default=dict)  # {"target_rank": 1000, "hours_per_day": 4}
    
    # Learning preferences
    learning_style = db.Column(db.String(50))  # visual, auditory, kinesthetic
    preferred_study_time = db.Column(db.String(20))  # morning, afternoon, night
    
    # Onboarding
    onboarding_complete = db.Column(db.Boolean, default=False)
    onboarding_step = db.Column(db.Integer, default=0)
    
    # Student's link code (for parents to link)
    link_code = db.Column(db.String(8), unique=True)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def generate_link_code(self):
        """Generate a unique link code for parent linking"""
        import secrets
        self.link_code = secrets.token_urlsafe(6)[:8].upper()
        return self.link_code
    
    def to_dict(self):
        return {
            "id": self.id,
            "user_id": self.user_id,
            "grade_level": self.grade_level,
            "board": self.board,
            "stream": self.stream,
            "target_exams": self.target_exams,
            "subjects": self.subjects,
            "study_goals": self.study_goals,
            "learning_style": self.learning_style,
            "preferred_study_time": self.preferred_study_time,
            "onboarding_complete": self.onboarding_complete,
            "link_code": self.link_code
        }


class ParentStudentLink(db.Model):
    """Links parents to their children (students)"""
    __tablename__ = "parent_student_links"
    
    id = db.Column(db.String(36), primary_key=True, default=generate_uuid)
    parent_id = db.Column(db.String(36), db.ForeignKey("users.id"), nullable=False)
    student_id = db.Column(db.String(36), db.ForeignKey("users.id"), nullable=False)
    
    relationship_type = db.Column(db.String(20), default="parent")  # mother, father, guardian
    
    # Access permissions
    can_view_progress = db.Column(db.Boolean, default=True)
    can_view_assessments = db.Column(db.Boolean, default=True)
    can_receive_reports = db.Column(db.Boolean, default=True)
    
    is_verified = db.Column(db.Boolean, default=False)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Unique constraint: one parent-student pair
    __table_args__ = (
        db.UniqueConstraint('parent_id', 'student_id', name='unique_parent_student'),
    )
    
    def to_dict(self):
        return {
            "id": self.id,
            "parent_id": self.parent_id,
            "student_id": self.student_id,
            "relationship_type": self.relationship_type,
            "is_verified": self.is_verified,
            "can_view_progress": self.can_view_progress,
            "can_view_assessments": self.can_view_assessments
        }


class TeacherClassAssignment(db.Model):
    """Assigns teachers to classes/subjects"""
    __tablename__ = "teacher_class_assignments"
    
    id = db.Column(db.String(36), primary_key=True, default=generate_uuid)
    teacher_id = db.Column(db.String(36), db.ForeignKey("users.id"), nullable=False)
    
    # What the teacher teaches
    class_name = db.Column(db.String(50))  # "12A", "11B"
    subject = db.Column(db.String(100))
    
    organization_id = db.Column(db.String(36), db.ForeignKey("organizations.id"), nullable=False)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            "id": self.id,
            "teacher_id": self.teacher_id,
            "class_name": self.class_name,
            "subject": self.subject,
            "organization_id": self.organization_id
        }
