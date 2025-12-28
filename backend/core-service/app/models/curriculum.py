"""
Curriculum Models - Subject, Topic, Subtopic Hierarchy
"""
from datetime import datetime
from app import db
import uuid


def generate_uuid():
    return str(uuid.uuid4())


class Subject(db.Model):
    """Subject (e.g., Physics, Chemistry, Math)"""
    __tablename__ = "subjects"
    
    id = db.Column(db.String(36), primary_key=True, default=generate_uuid)
    name = db.Column(db.String(100), nullable=False)
    code = db.Column(db.String(20), unique=True)  # PHY, CHE, MAT
    description = db.Column(db.Text)
    icon = db.Column(db.String(50))  # Icon name or URL
    color = db.Column(db.String(7))  # Hex color code
    
    # Which grades/boards this subject applies to
    grade_levels = db.Column(db.JSON, default=list)  # ["11", "12"]
    boards = db.Column(db.JSON, default=list)  # ["CBSE", "ICSE"]
    
    order = db.Column(db.Integer, default=0)
    is_active = db.Column(db.Boolean, default=True)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    topics = db.relationship("Topic", backref="subject", cascade="all, delete-orphan", lazy="dynamic")
    
    def to_dict(self, include_topics=False):
        data = {
            "id": self.id,
            "name": self.name,
            "code": self.code,
            "description": self.description,
            "icon": self.icon,
            "color": self.color,
            "grade_levels": self.grade_levels,
            "boards": self.boards,
            "order": self.order,
            "topic_count": self.topics.count()
        }
        if include_topics:
            data["topics"] = [t.to_dict() for t in self.topics.order_by(Topic.order)]
        return data


class Topic(db.Model):
    """Topic within a subject (e.g., Mechanics, Thermodynamics)"""
    __tablename__ = "topics"
    
    id = db.Column(db.String(36), primary_key=True, default=generate_uuid)
    subject_id = db.Column(db.String(36), db.ForeignKey("subjects.id"), nullable=False)
    
    name = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text)
    
    # Learning metadata
    estimated_hours = db.Column(db.Float, default=2.0)
    difficulty = db.Column(db.String(20), default="medium")  # easy, medium, hard
    
    order = db.Column(db.Integer, default=0)
    is_active = db.Column(db.Boolean, default=True)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    subtopics = db.relationship("Subtopic", backref="topic", cascade="all, delete-orphan", lazy="dynamic")
    
    def to_dict(self, include_subtopics=False):
        data = {
            "id": self.id,
            "subject_id": self.subject_id,
            "name": self.name,
            "description": self.description,
            "estimated_hours": self.estimated_hours,
            "difficulty": self.difficulty,
            "order": self.order,
            "subtopic_count": self.subtopics.count()
        }
        if include_subtopics:
            data["subtopics"] = [s.to_dict() for s in self.subtopics.order_by(Subtopic.order)]
        return data


class Subtopic(db.Model):
    """Subtopic within a topic (e.g., Newton's Laws, Work-Energy Theorem)"""
    __tablename__ = "subtopics"
    
    id = db.Column(db.String(36), primary_key=True, default=generate_uuid)
    topic_id = db.Column(db.String(36), db.ForeignKey("topics.id"), nullable=False)
    
    name = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text)
    
    # Learning content
    key_concepts = db.Column(db.JSON, default=list)  # ["F=ma", "Action-Reaction"]
    learning_objectives = db.Column(db.JSON, default=list)
    
    # Assessment info
    has_assessment = db.Column(db.Boolean, default=True)
    question_count = db.Column(db.Integer, default=10)
    
    estimated_minutes = db.Column(db.Integer, default=30)
    difficulty = db.Column(db.String(20), default="medium")
    
    order = db.Column(db.Integer, default=0)
    is_active = db.Column(db.Boolean, default=True)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            "id": self.id,
            "topic_id": self.topic_id,
            "name": self.name,
            "description": self.description,
            "key_concepts": self.key_concepts,
            "learning_objectives": self.learning_objectives,
            "has_assessment": self.has_assessment,
            "question_count": self.question_count,
            "estimated_minutes": self.estimated_minutes,
            "difficulty": self.difficulty,
            "order": self.order
        }


class SubtopicAssessment(db.Model):
    """MCQ Assessment for a subtopic"""
    __tablename__ = "subtopic_assessments"
    
    id = db.Column(db.String(36), primary_key=True, default=generate_uuid)
    subtopic_id = db.Column(db.String(36), db.ForeignKey("subtopics.id"), nullable=False)
    
    title = db.Column(db.String(300))
    description = db.Column(db.Text)
    
    # Questions stored as JSON array
    questions = db.Column(db.JSON, nullable=False, default=list)
    # [{
    #   "id": "q1",
    #   "question": "What is the unit of force?",
    #   "options": ["Newton", "Joule", "Watt", "Pascal"],
    #   "correct_answer": 0,
    #   "explanation": "Force is measured in Newtons (N)",
    #   "difficulty": "easy"
    # }]
    
    time_limit_minutes = db.Column(db.Integer, default=15)
    passing_score = db.Column(db.Float, default=60.0)  # Percentage
    
    is_adaptive = db.Column(db.Boolean, default=False)
    is_active = db.Column(db.Boolean, default=True)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationship
    subtopic = db.relationship("Subtopic", backref="assessments")
    
    def to_dict(self, include_questions=False):
        data = {
            "id": self.id,
            "subtopic_id": self.subtopic_id,
            "title": self.title,
            "description": self.description,
            "question_count": len(self.questions) if self.questions else 0,
            "time_limit_minutes": self.time_limit_minutes,
            "passing_score": self.passing_score,
            "is_adaptive": self.is_adaptive
        }
        if include_questions:
            data["questions"] = self.questions
        return data


class StudentSubtopicProgress(db.Model):
    """Track student progress per subtopic"""
    __tablename__ = "student_subtopic_progress"
    
    id = db.Column(db.String(36), primary_key=True, default=generate_uuid)
    user_id = db.Column(db.String(36), db.ForeignKey("users.id"), nullable=False)
    subtopic_id = db.Column(db.String(36), db.ForeignKey("subtopics.id"), nullable=False)
    
    # Progress status
    status = db.Column(db.String(20), default="not_started")  # not_started, in_progress, completed, mastered
    
    # Assessment results
    attempts = db.Column(db.Integer, default=0)
    best_score = db.Column(db.Float, default=0.0)
    last_score = db.Column(db.Float)
    average_score = db.Column(db.Float, default=0.0)
    
    # Time spent
    total_time_minutes = db.Column(db.Integer, default=0)
    
    # Mastery
    mastery_level = db.Column(db.Float, default=0.0)  # 0-100
    
    first_attempt_at = db.Column(db.DateTime)
    last_attempt_at = db.Column(db.DateTime)
    completed_at = db.Column(db.DateTime)
    
    __table_args__ = (
        db.UniqueConstraint('user_id', 'subtopic_id', name='unique_user_subtopic'),
    )
    
    def to_dict(self):
        return {
            "id": self.id,
            "user_id": self.user_id,
            "subtopic_id": self.subtopic_id,
            "status": self.status,
            "attempts": self.attempts,
            "best_score": self.best_score,
            "last_score": self.last_score,
            "average_score": self.average_score,
            "mastery_level": self.mastery_level,
            "total_time_minutes": self.total_time_minutes,
            "last_attempt_at": self.last_attempt_at.isoformat() if self.last_attempt_at else None
        }
