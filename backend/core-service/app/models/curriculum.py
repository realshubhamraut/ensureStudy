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


# ============================================================================
# CLASSROOM TOPIC HIERARCHY MODELS
# Classroom → Chapter → ClassroomTopic → TopicQuestion
# ============================================================================

# Color palette for chapters
CHAPTER_COLORS = [
    '#3B82F6',  # Blue
    '#10B981',  # Emerald
    '#F59E0B',  # Amber
    '#EF4444',  # Red
    '#8B5CF6',  # Violet
    '#EC4899',  # Pink
    '#06B6D4',  # Cyan
    '#84CC16',  # Lime
    '#F97316',  # Orange
    '#14B8A6',  # Teal
]


class Chapter(db.Model):
    """
    Chapter/Lesson within a classroom syllabus.
    Groups related topics together.
    Hierarchy: Classroom → Chapters → ClassroomTopics
    """
    __tablename__ = "chapters"
    
    id = db.Column(db.String(36), primary_key=True, default=generate_uuid)
    classroom_id = db.Column(db.String(36), db.ForeignKey("classrooms.id"), nullable=False, index=True)
    syllabus_id = db.Column(db.String(36), db.ForeignKey("syllabi.id"), nullable=True, index=True)
    
    name = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text)
    
    # UI styling
    color = db.Column(db.String(7), default='#3B82F6')  # Hex color for chapter grouping
    icon = db.Column(db.String(50))  # Optional icon name
    
    # Metadata
    estimated_hours = db.Column(db.Float, default=2.0)
    order = db.Column(db.Integer, default=0)
    is_active = db.Column(db.Boolean, default=True)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    topics = db.relationship("ClassroomTopic", backref="chapter", cascade="all, delete-orphan", lazy="dynamic")
    
    __table_args__ = (
        db.Index('idx_chapter_classroom', 'classroom_id'),
    )
    
    def to_dict(self, include_topics=False):
        data = {
            "id": self.id,
            "classroom_id": self.classroom_id,
            "syllabus_id": self.syllabus_id,
            "name": self.name,
            "description": self.description,
            "color": self.color,
            "icon": self.icon,
            "estimated_hours": self.estimated_hours,
            "order": self.order,
            "is_active": self.is_active,
            "topic_count": self.topics.count() if self.topics else 0,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }
        if include_topics:
            data["topics"] = [t.to_dict() for t in self.topics.filter_by(is_active=True).order_by(ClassroomTopic.order)]
        return data


class ClassroomTopic(db.Model):
    """
    Topic extracted from classroom syllabus.
    Different from personal curriculum Topics - these are shared with all enrolled students.
    """
    __tablename__ = "classroom_topics"
    
    id = db.Column(db.String(36), primary_key=True, default=generate_uuid)
    chapter_id = db.Column(db.String(36), db.ForeignKey("chapters.id"), nullable=False, index=True)
    classroom_id = db.Column(db.String(36), db.ForeignKey("classrooms.id"), nullable=False, index=True)
    
    name = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text)
    
    # Learning metadata
    difficulty = db.Column(db.String(20), default="medium")  # easy, medium, hard
    estimated_hours = db.Column(db.Float, default=1.0)
    key_concepts = db.Column(JSON, default=list)  # ["concept1", "concept2"]
    
    # Ordering
    order = db.Column(db.Integer, default=0)
    is_active = db.Column(db.Boolean, default=True)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    questions = db.relationship("TopicQuestion", backref="topic", cascade="all, delete-orphan", lazy="dynamic")
    student_scores = db.relationship("StudentTopicScore", backref="topic", cascade="all, delete-orphan", lazy="dynamic")
    descriptive_questions = db.relationship("DescriptiveQuestion", backref="topic", cascade="all, delete-orphan", lazy="dynamic")
    
    __table_args__ = (
        db.Index('idx_classroom_topic_chapter', 'chapter_id'),
        db.Index('idx_classroom_topic_classroom', 'classroom_id'),
    )
    
    def to_dict(self, include_questions=False):
        data = {
            "id": self.id,
            "chapter_id": self.chapter_id,
            "classroom_id": self.classroom_id,
            "name": self.name,
            "description": self.description,
            "difficulty": self.difficulty,
            "estimated_hours": self.estimated_hours,
            "key_concepts": self.key_concepts or [],
            "order": self.order,
            "is_active": self.is_active,
            "question_count": self.questions.count() if self.questions else 0,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }
        if include_questions:
            data["questions"] = [q.to_dict() for q in self.questions.filter_by(is_active=True)]
        return data


class TopicQuestion(db.Model):
    """
    Question linked to a classroom topic.
    Supports both MCQ and Descriptive question types.
    """
    __tablename__ = "topic_questions"
    
    id = db.Column(db.String(36), primary_key=True, default=generate_uuid)
    classroom_topic_id = db.Column(db.String(36), db.ForeignKey("classroom_topics.id"), nullable=False, index=True)
    
    # Question content
    question_type = db.Column(db.String(20), nullable=False)  # "mcq" or "descriptive"
    question_text = db.Column(db.Text, nullable=False)
    
    # For MCQ: options array
    # [{"id": "A", "text": "Option A", "is_correct": false}, ...]
    options = db.Column(JSON, default=list)
    
    # For MCQ: correct option id ("A", "B", "C", "D")
    correct_answer = db.Column(db.String(10))
    
    # For Descriptive: model/expected answer
    expected_answer = db.Column(db.Text)
    
    # Scoring rubric for descriptive answers
    # ["key point 1", "key point 2", ...]
    key_points = db.Column(JSON, default=list)
    
    # Explanation shown after answer
    explanation = db.Column(db.Text)
    
    # Scoring
    marks = db.Column(db.Integer, default=1)
    difficulty = db.Column(db.String(20), default="medium")  # easy, medium, hard
    time_estimate_seconds = db.Column(db.Integer, default=60)
    
    # Analytics
    times_used = db.Column(db.Integer, default=0)
    times_correct = db.Column(db.Integer, default=0)
    times_incorrect = db.Column(db.Integer, default=0)
    average_score = db.Column(db.Float)  # For descriptive questions
    
    # Status
    is_active = db.Column(db.Boolean, default=True)
    source = db.Column(db.String(30), default="generated")  # "generated", "manual", "imported"
    
    # Learning Agent fields for duplicate detection
    question_hash = db.Column(db.String(64), index=True)  # SHA256 hash of normalized question text
    embedding_vector_id = db.Column(db.String(100))  # Qdrant point ID for semantic search
    auto_generated = db.Column(db.Boolean, default=False)  # True if generated by Learning Agent
    
    created_by = db.Column(db.String(36), db.ForeignKey("users.id"))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    responses = db.relationship("StudentQuestionResponse", backref="question", cascade="all, delete-orphan", lazy="dynamic")
    
    __table_args__ = (
        db.Index('idx_topic_question_topic', 'classroom_topic_id'),
        db.Index('idx_topic_question_type', 'question_type'),
    )
    
    def to_dict(self, include_answer=False):
        """Convert to dictionary. Set include_answer=False to hide correct answer for students."""
        data = {
            "id": self.id,
            "classroom_topic_id": self.classroom_topic_id,
            "question_type": self.question_type,
            "question_text": self.question_text,
            "options": self.options if self.question_type == "mcq" else None,
            "marks": self.marks,
            "difficulty": self.difficulty,
            "time_estimate_seconds": self.time_estimate_seconds,
            "is_active": self.is_active,
            "source": self.source
        }
        
        if include_answer:
            data["correct_answer"] = self.correct_answer
            data["expected_answer"] = self.expected_answer
            data["key_points"] = self.key_points
            data["explanation"] = self.explanation
        
        return data
    
    def update_analytics(self, score_awarded: float, max_score: float):
        """Update question analytics after a student answers"""
        self.times_used += 1
        
        if self.question_type == "mcq":
            if score_awarded == max_score:
                self.times_correct += 1
            else:
                self.times_incorrect += 1
        else:
            # For descriptive, update average score
            if self.average_score is None:
                self.average_score = score_awarded / max_score * 100
            else:
                self.average_score = (
                    (self.average_score * (self.times_used - 1) + (score_awarded / max_score * 100)) /
                    self.times_used
                )


class StudentTopicScore(db.Model):
    """
    Track cumulative scores and mastery per student per classroom topic.
    Updated after MCQ tests and mock interviews.
    """
    __tablename__ = "student_topic_scores"
    
    id = db.Column(db.String(36), primary_key=True, default=generate_uuid)
    user_id = db.Column(db.String(36), db.ForeignKey("users.id"), nullable=False, index=True)
    classroom_topic_id = db.Column(db.String(36), db.ForeignKey("classroom_topics.id"), nullable=False, index=True)
    
    # Cumulative scores
    total_score = db.Column(db.Float, default=0.0)
    max_possible_score = db.Column(db.Float, default=0.0)
    mastery_percentage = db.Column(db.Float, default=0.0)  # 0-100
    
    # MCQ performance
    mcq_attempts = db.Column(db.Integer, default=0)
    mcq_correct = db.Column(db.Integer, default=0)
    mcq_total_score = db.Column(db.Float, default=0.0)
    mcq_max_score = db.Column(db.Float, default=0.0)
    
    # Descriptive performance (mock interview, etc.)
    descriptive_attempts = db.Column(db.Integer, default=0)
    descriptive_total_score = db.Column(db.Float, default=0.0)
    descriptive_max_score = db.Column(db.Float, default=0.0)
    descriptive_avg_score = db.Column(db.Float, default=0.0)
    
    # Status tracking
    status = db.Column(db.String(20), default="not_started")  # not_started, learning, practicing, mastered
    
    first_activity_at = db.Column(db.DateTime)
    last_activity_at = db.Column(db.DateTime)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    __table_args__ = (
        db.UniqueConstraint('user_id', 'classroom_topic_id', name='unique_user_classroom_topic'),
        db.Index('idx_student_topic_score_user', 'user_id'),
        db.Index('idx_student_topic_score_topic', 'classroom_topic_id'),
    )
    
    def to_dict(self):
        return {
            "id": self.id,
            "user_id": self.user_id,
            "classroom_topic_id": self.classroom_topic_id,
            "total_score": self.total_score,
            "max_possible_score": self.max_possible_score,
            "mastery_percentage": round(self.mastery_percentage, 1),
            "mcq_attempts": self.mcq_attempts,
            "mcq_correct": self.mcq_correct,
            "mcq_accuracy": round((self.mcq_correct / self.mcq_attempts * 100) if self.mcq_attempts > 0 else 0, 1),
            "descriptive_attempts": self.descriptive_attempts,
            "descriptive_avg_score": round(self.descriptive_avg_score, 1),
            "status": self.status,
            "last_activity_at": self.last_activity_at.isoformat() if self.last_activity_at else None
        }
    
    def update_mcq_score(self, correct: bool, marks: int):
        """Update after MCQ answer"""
        self.mcq_attempts += 1
        self.mcq_max_score += marks
        self.max_possible_score += marks
        
        if correct:
            self.mcq_correct += 1
            self.mcq_total_score += marks
            self.total_score += marks
        
        self._recalculate_mastery()
        self.last_activity_at = datetime.utcnow()
        if not self.first_activity_at:
            self.first_activity_at = datetime.utcnow()
    
    def update_descriptive_score(self, score_awarded: float, max_score: float):
        """Update after descriptive answer"""
        self.descriptive_attempts += 1
        self.descriptive_total_score += score_awarded
        self.descriptive_max_score += max_score
        self.total_score += score_awarded
        self.max_possible_score += max_score
        
        # Update running average
        self.descriptive_avg_score = (
            self.descriptive_total_score / self.descriptive_max_score * 100
        ) if self.descriptive_max_score > 0 else 0
        
        self._recalculate_mastery()
        self.last_activity_at = datetime.utcnow()
        if not self.first_activity_at:
            self.first_activity_at = datetime.utcnow()
    
    def _recalculate_mastery(self):
        """Recalculate mastery percentage based on all scores"""
        if self.max_possible_score > 0:
            self.mastery_percentage = (self.total_score / self.max_possible_score) * 100
        else:
            self.mastery_percentage = 0.0
        
        # Update status based on mastery
        if self.mastery_percentage >= 80:
            self.status = "mastered"
        elif self.mastery_percentage >= 50:
            self.status = "practicing"
        elif self.mastery_percentage > 0:
            self.status = "learning"
        else:
            self.status = "not_started"


class StudentQuestionResponse(db.Model):
    """
    Individual question response record.
    Stores both MCQ and descriptive answers with scoring details.
    """
    __tablename__ = "student_question_responses"
    
    id = db.Column(db.String(36), primary_key=True, default=generate_uuid)
    user_id = db.Column(db.String(36), db.ForeignKey("users.id"), nullable=False, index=True)
    question_id = db.Column(db.String(36), db.ForeignKey("topic_questions.id"), nullable=False, index=True)
    
    # Response content
    response_type = db.Column(db.String(20), nullable=False)  # "mcq" or "descriptive"
    
    # For MCQ
    selected_option = db.Column(db.String(10))  # "A", "B", "C", "D"
    is_correct = db.Column(db.Boolean)
    
    # For Descriptive
    descriptive_response = db.Column(db.Text)
    matched_key_points = db.Column(JSON, default=list)  # Which key points were covered
    
    # Scoring
    score_awarded = db.Column(db.Float, default=0.0)
    max_score = db.Column(db.Float, default=1.0)
    score_percentage = db.Column(db.Float, default=0.0)  # 0-100
    
    # AI evaluation (for descriptive)
    ai_feedback = db.Column(db.Text)
    ai_confidence = db.Column(db.Float)  # 0-1
    
    # Timing
    response_time_ms = db.Column(db.Integer)  # Time taken to answer
    
    # Source tracking
    source = db.Column(db.String(30), nullable=False)  # "assessment", "mock_interview", "practice"
    source_session_id = db.Column(db.String(36))  # Reference to exam/interview session
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        db.Index('idx_response_user', 'user_id'),
        db.Index('idx_response_question', 'question_id'),
        db.Index('idx_response_source', 'source', 'source_session_id'),
    )
    
    def to_dict(self, include_feedback=True):
        data = {
            "id": self.id,
            "user_id": self.user_id,
            "question_id": self.question_id,
            "response_type": self.response_type,
            "selected_option": self.selected_option if self.response_type == "mcq" else None,
            "is_correct": self.is_correct if self.response_type == "mcq" else None,
            "descriptive_response": self.descriptive_response if self.response_type == "descriptive" else None,
            "score_awarded": self.score_awarded,
            "max_score": self.max_score,
            "score_percentage": round(self.score_percentage, 1),
            "response_time_ms": self.response_time_ms,
            "source": self.source,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }
        
        if include_feedback and self.response_type == "descriptive":
            data["ai_feedback"] = self.ai_feedback
            data["matched_key_points"] = self.matched_key_points
        
        return data


class StudyScheduleEntry(db.Model):
    """
    Student's study schedule entry - maps a topic to a planned study date.
    Used by students to plan their learning schedule via drag-and-drop calendar.
    """
    __tablename__ = "study_schedule_entries"
    
    id = db.Column(db.String(36), primary_key=True, default=generate_uuid)
    user_id = db.Column(db.String(36), db.ForeignKey("users.id"), nullable=False, index=True)
    
    # Can reference either a ClassroomTopic (from enrolled classroom) or a generic Topic
    classroom_topic_id = db.Column(db.String(36), db.ForeignKey("classroom_topics.id"), nullable=True, index=True)
    topic_id = db.Column(db.String(36), db.ForeignKey("topics.id"), nullable=True)
    
    # Denormalized for easy display (avoid joins when fetching schedule)
    topic_name = db.Column(db.String(200), nullable=False)
    topic_description = db.Column(db.String(500))  # Topic description/subtopics
    subject_name = db.Column(db.String(100))  # Subject/Classroom name
    chapter_name = db.Column(db.String(200))  # Chapter name if applicable
    
    # Scheduling
    scheduled_date = db.Column(db.Date, nullable=False, index=True)
    estimated_hours = db.Column(db.Float, default=1.0)
    
    # Status tracking
    status = db.Column(db.String(20), default="scheduled")  # scheduled, in_progress, completed, skipped
    completed_at = db.Column(db.DateTime)
    actual_hours = db.Column(db.Float)  # How long student actually studied
    
    # Notes
    notes = db.Column(db.Text)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    classroom_topic = db.relationship("ClassroomTopic", backref="schedule_entries")
    
    __table_args__ = (
        db.Index('idx_schedule_user_date', 'user_id', 'scheduled_date'),
        db.Index('idx_schedule_user_topic', 'user_id', 'classroom_topic_id'),
    )
    
    def to_dict(self):
        return {
            "id": self.id,
            "user_id": self.user_id,
            "classroom_topic_id": self.classroom_topic_id,
            "topic_id": self.topic_id,
            "topic_name": self.topic_name,
            "topic_description": self.topic_description,
            "subject_name": self.subject_name,
            "chapter_name": self.chapter_name,
            "scheduled_date": self.scheduled_date.isoformat() if self.scheduled_date else None,
            "estimated_hours": self.estimated_hours,
            "status": self.status,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "actual_hours": self.actual_hours,
            "notes": self.notes,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


class QuestionEffectiveness(db.Model):
    """
    Tracks question quality metrics for the Type 5 Learning Agent.
    Used to identify which questions best assess student knowledge.
    """
    __tablename__ = "question_effectiveness"
    
    question_id = db.Column(db.String(36), db.ForeignKey("topic_questions.id"), primary_key=True)
    
    # Psychometric metrics
    discrimination_index = db.Column(db.Float, default=0.0)  # -1 to 1, how well it separates strong/weak students
    difficulty_index = db.Column(db.Float, default=0.5)  # 0 to 1, % of students who answered correctly
    
    # Distractor quality for MCQs
    distractor_quality = db.Column(JSON, default=dict)  # {"A": 0.8, "B": 0.2, ...} selection frequency
    
    # Combined effectiveness score (weighted average)
    effectiveness_score = db.Column(db.Float, default=0.5)  # 0 to 1
    
    # Sample size for statistical confidence
    sample_size = db.Column(db.Integer, default=0)
    
    # Performance tracking
    total_attempts = db.Column(db.Integer, default=0)
    correct_attempts = db.Column(db.Integer, default=0)
    avg_response_time_ms = db.Column(db.Integer)  # Average time to answer
    
    last_updated = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship
    question = db.relationship("TopicQuestion", backref=db.backref("effectiveness", uselist=False))
    
    def to_dict(self):
        return {
            "question_id": self.question_id,
            "discrimination_index": self.discrimination_index,
            "difficulty_index": self.difficulty_index,
            "distractor_quality": self.distractor_quality,
            "effectiveness_score": self.effectiveness_score,
            "sample_size": self.sample_size,
            "total_attempts": self.total_attempts,
            "correct_attempts": self.correct_attempts,
            "avg_response_time_ms": self.avg_response_time_ms,
            "last_updated": self.last_updated.isoformat() if self.last_updated else None
        }


class LearningAgentMemory(db.Model):
    """
    Persistent memory for the Type 5 Learning Agent.
    Stores learned parameters that improve question generation over time.
    """
    __tablename__ = "learning_agent_memory"
    
    id = db.Column(db.String(36), primary_key=True, default=generate_uuid)
    topic_id = db.Column(db.String(36), index=True)  # Can be classroom_topic_id
    
    # Learned difficulty calibration
    calibrated_difficulty = db.Column(db.Float, default=0.5)  # Optimal difficulty for this topic
    target_success_rate = db.Column(db.Float, default=0.7)  # Target % correct answers
    actual_success_rate = db.Column(db.Float)  # Current observed success rate
    
    # Generation strategy learned over time
    preferred_question_types = db.Column(JSON, default=list)  # ["mcq", "descriptive"]
    avoided_patterns = db.Column(JSON, default=list)  # Question patterns that confused students
    successful_prompts = db.Column(JSON, default=list)  # Prompt templates that generated good questions
    
    # Question bank status
    total_questions = db.Column(db.Integer, default=0)
    effective_questions = db.Column(db.Integer, default=0)  # Questions with high effectiveness score
    needs_more_questions = db.Column(db.Boolean, default=True)
    
    # Learning metrics
    learning_iterations = db.Column(db.Integer, default=0)  # How many times agent has learned
    last_learning_at = db.Column(db.DateTime)
    improvement_score = db.Column(db.Float, default=0.0)  # How much agent has improved over time
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    __table_args__ = (
        db.Index('idx_learning_memory_topic', 'topic_id'),
    )
    
    def to_dict(self):
        return {
            "id": self.id,
            "topic_id": self.topic_id,
            "calibrated_difficulty": self.calibrated_difficulty,
            "target_success_rate": self.target_success_rate,
            "actual_success_rate": self.actual_success_rate,
            "preferred_question_types": self.preferred_question_types,
            "avoided_patterns": self.avoided_patterns,
            "successful_prompts": self.successful_prompts,
            "total_questions": self.total_questions,
            "effective_questions": self.effective_questions,
            "needs_more_questions": self.needs_more_questions,
            "learning_iterations": self.learning_iterations,
            "improvement_score": self.improvement_score,
            "last_learning_at": self.last_learning_at.isoformat() if self.last_learning_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }
