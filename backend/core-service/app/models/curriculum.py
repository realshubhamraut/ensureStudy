"""
Curriculum Models - Subject, Topic, Subtopic Hierarchy
"""
from datetime import datetime
from app import db
from sqlalchemy.types import JSON
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


class Syllabus(db.Model):
    """Syllabus document linked to classroom for topic extraction"""
    __tablename__ = "syllabi"
    
    id = db.Column(db.String(36), primary_key=True, default=generate_uuid)
    classroom_id = db.Column(db.String(36), db.ForeignKey("classrooms.id"), nullable=False, index=True)
    document_id = db.Column(db.String(36), nullable=True)  # PDF/document reference (no FK - stored externally)
    
    title = db.Column(db.String(300), nullable=False)
    subject_id = db.Column(db.String(36), db.ForeignKey("subjects.id"), nullable=True)  # Link to Subject
    academic_year = db.Column(db.String(20))  # "2025-26"
    description = db.Column(db.Text)
    
    # Extraction status
    extraction_status = db.Column(db.String(20), default="pending")  # pending, processing, completed, failed
    extraction_error = db.Column(db.Text)
    extracted_topics_count = db.Column(db.Integer, default=0)
    
    created_by = db.Column(db.String(36), db.ForeignKey("users.id"))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    subject = db.relationship("Subject", backref="syllabi")
    
    def to_dict(self):
        return {
            "id": str(self.id),
            "classroom_id": str(self.classroom_id),
            "document_id": str(self.document_id) if self.document_id else None,
            "title": self.title,
            "subject_id": str(self.subject_id) if self.subject_id else None,
            "academic_year": self.academic_year,
            "description": self.description,
            "extraction_status": self.extraction_status,
            "extracted_topics_count": self.extracted_topics_count,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


class QuestionBank(db.Model):
    """Question bank for a classroom/subject - collection of questions"""
    __tablename__ = "question_banks"
    
    id = db.Column(db.String(36), primary_key=True, default=generate_uuid)
    classroom_id = db.Column(db.String(36), db.ForeignKey("classrooms.id"), nullable=False, index=True)
    subject_id = db.Column(db.String(36), db.ForeignKey("subjects.id"), nullable=True)
    
    name = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text)
    total_questions = db.Column(db.Integer, default=0)
    
    # Source tracking
    source_type = db.Column(db.String(50), default="generated")  # generated, imported, manual
    source_document_id = db.Column(db.String(36))  # Reference to source PDF if applicable
    
    created_by = db.Column(db.String(36), db.ForeignKey("users.id"))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    questions = db.relationship("Question", backref="question_bank", cascade="all, delete-orphan")
    
    def to_dict(self):
        return {
            "id": str(self.id),
            "classroom_id": str(self.classroom_id),
            "subject_id": str(self.subject_id) if self.subject_id else None,
            "name": self.name,
            "description": self.description,
            "total_questions": self.total_questions,
            "source_type": self.source_type,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


class Question(db.Model):
    """Individual question with topic linkage and analytics"""
    __tablename__ = "questions"
    
    id = db.Column(db.String(36), primary_key=True, default=generate_uuid)
    question_bank_id = db.Column(db.String(36), db.ForeignKey("question_banks.id"), nullable=True, index=True)
    
    # Topic hierarchy linkage
    topic_id = db.Column(db.String(36), db.ForeignKey("topics.id"), nullable=True, index=True)
    subtopic_id = db.Column(db.String(36), db.ForeignKey("subtopics.id"), nullable=True, index=True)
    
    # Question content
    question_type = db.Column(db.String(20), nullable=False)  # mcq, descriptive, short_answer
    question_text = db.Column(db.Text, nullable=False)
    
    # For MCQ: store options as JSON array
    options = db.Column(JSON, default=list)  # [{"id": "A", "text": "Option A"}, ...]
    correct_answer = db.Column(db.String(500))  # For MCQ: "A", "B", etc. For descriptive: key points
    explanation = db.Column(db.Text)  # Explanation shown after answer
    
    # Key points for descriptive answers (used for evaluation)
    key_points = db.Column(JSON, default=list)  # ["point1", "point2", ...]
    
    # Difficulty and metadata
    difficulty = db.Column(db.String(20), default="medium")  # easy, medium, hard
    marks = db.Column(db.Integer, default=1)
    time_estimate_seconds = db.Column(db.Integer, default=60)
    
    # Source tracking
    source_chunk_id = db.Column(db.String(100))  # Qdrant point ID if generated from chunk
    source_content_preview = db.Column(db.Text)  # First 500 chars of source content
    
    # Analytics
    times_used = db.Column(db.Integer, default=0)
    times_correct = db.Column(db.Integer, default=0)
    times_incorrect = db.Column(db.Integer, default=0)
    average_time_taken = db.Column(db.Float)  # Average time students take
    difficulty_rating = db.Column(db.Float)  # Calculated from success rate
    
    # Status
    is_active = db.Column(db.Boolean, default=True)
    review_status = db.Column(db.String(20), default="pending")  # pending, approved, rejected
    
    created_by = db.Column(db.String(36), db.ForeignKey("users.id"))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    topic = db.relationship("Topic", backref="questions")
    subtopic = db.relationship("Subtopic", backref="questions")
    
    __table_args__ = (
        db.Index('idx_question_topic', 'topic_id'),
        db.Index('idx_question_type_difficulty', 'question_type', 'difficulty'),
    )
    
    def to_dict(self, include_answer: bool = False):
        """Convert to dictionary. Set include_answer=False to hide correct answer for students."""
        data = {
            "id": str(self.id),
            "question_bank_id": str(self.question_bank_id) if self.question_bank_id else None,
            "topic_id": str(self.topic_id) if self.topic_id else None,
            "subtopic_id": str(self.subtopic_id) if self.subtopic_id else None,
            "question_type": self.question_type,
            "question_text": self.question_text,
            "options": self.options,
            "difficulty": self.difficulty,
            "marks": self.marks,
            "time_estimate_seconds": self.time_estimate_seconds,
            "times_used": self.times_used,
            "is_active": self.is_active,
            "review_status": self.review_status,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }
        
        if include_answer:
            data["correct_answer"] = self.correct_answer
            data["explanation"] = self.explanation
            data["key_points"] = self.key_points
        
        return data
    
    def update_analytics(self, was_correct: bool, time_taken_seconds: int):
        """Update question analytics after a student answers"""
        self.times_used += 1
        if was_correct:
            self.times_correct += 1
        else:
            self.times_incorrect += 1
        
        # Update average time
        if self.average_time_taken is None:
            self.average_time_taken = float(time_taken_seconds)
        else:
            # Running average
            self.average_time_taken = (
                (self.average_time_taken * (self.times_used - 1) + time_taken_seconds) / 
                self.times_used
            )
        
        # Update difficulty rating based on success rate
        if self.times_used >= 5:  # Only calculate after 5 attempts
            success_rate = self.times_correct / self.times_used
            # Invert: lower success rate = higher difficulty
            self.difficulty_rating = round(1 - success_rate, 2)
