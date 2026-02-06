"""
Database Models for ensureStudy Core Service
"""
from datetime import datetime
import uuid
import os
from sqlalchemy import JSON, String
from werkzeug.security import generate_password_hash, check_password_hash
from app import db

# Use String for UUID to support both SQLite and PostgreSQL
def generate_uuid():
    return str(uuid.uuid4())



class User(db.Model):
    """User model for students, teachers, parents, and admins"""
    __tablename__ = "users"
    
    id = db.Column(db.String(36), primary_key=True, default=generate_uuid)
    email = db.Column(db.String(255), unique=True, nullable=False, index=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(20), nullable=False, default="student")  # admin, teacher, student, parent
    first_name = db.Column(db.String(100))
    last_name = db.Column(db.String(100))
    
    # Organization (school) relationship
    organization_id = db.Column(db.String(36), db.ForeignKey("organizations.id"), nullable=True, index=True)
    invited_by = db.Column(db.String(36), db.ForeignKey("users.id"), nullable=True)  # Who invited this user
    
    # Legacy fields (kept for compatibility)
    school_id = db.Column(db.String(100), index=True)
    class_id = db.Column(db.String(100), index=True)
    
    avatar_url = db.Column(db.String(500))
    bio = db.Column(db.Text)
    phone = db.Column(db.String(20))
    
    # Consent
    audio_consent = db.Column(db.Boolean, default=False)
    data_sharing_consent = db.Column(db.Boolean, default=False)
    
    is_active = db.Column(db.Boolean, default=True)
    email_verified = db.Column(db.Boolean, default=False)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    progress_records = db.relationship("Progress", backref="user", cascade="all, delete-orphan", lazy="dynamic")
    assessment_results = db.relationship("AssessmentResult", backref="user", cascade="all, delete-orphan", lazy="dynamic")
    chat_sessions = db.relationship("ChatSession", backref="user", cascade="all, delete-orphan", lazy="dynamic")
    leaderboard_entry = db.relationship("Leaderboard", backref="user", uselist=False, cascade="all, delete-orphan")
    
    # New relationships
    student_profile = db.relationship("StudentProfile", backref="user", uselist=False, cascade="all, delete-orphan")
    invited_users = db.relationship("User", backref=db.backref("inviter", remote_side=[id]), lazy="dynamic")
    
    def set_password(self, password: str):
        """Hash and set the password"""
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password: str) -> bool:
        """Verify password against hash"""
        return check_password_hash(self.password_hash, password)
    
    def to_dict(self):
        """Serialize user to dictionary"""
        return {
            "id": str(self.id),
            "email": self.email,
            "username": self.username,
            "role": self.role,
            "first_name": self.first_name,
            "last_name": self.last_name,
            "school_id": self.school_id,
            "class_id": self.class_id,
            "avatar_url": self.avatar_url,
            "bio": self.bio,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


class Progress(db.Model):
    """Track student progress per topic"""
    __tablename__ = "progress"
    
    id = db.Column(db.String(36), primary_key=True, default=generate_uuid)
    user_id = db.Column(db.String(36), db.ForeignKey("users.id"), nullable=False, index=True)
    topic = db.Column(db.String(200), nullable=False, index=True)
    subject = db.Column(db.String(100), nullable=False, index=True)
    confidence_score = db.Column(db.Float, default=0.0)
    assessment_scores = db.Column(JSON, default=list)
    last_studied = db.Column(db.DateTime)
    times_studied = db.Column(db.Integer, default=0)
    is_weak = db.Column(db.Boolean, default=False, index=True)
    notes = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    __table_args__ = (
        db.Index('idx_progress_user_subject', 'user_id', 'subject'),
    )
    
    def to_dict(self):
        return {
            "id": str(self.id),
            "user_id": str(self.user_id),
            "topic": self.topic,
            "subject": self.subject,
            "confidence_score": self.confidence_score,
            "assessment_scores": self.assessment_scores,
            "last_studied": self.last_studied.isoformat() if self.last_studied else None,
            "times_studied": self.times_studied,
            "is_weak": self.is_weak
        }


class Assessment(db.Model):
    """Assessment/Quiz definitions"""
    __tablename__ = "assessments"
    
    id = db.Column(db.String(36), primary_key=True, default=generate_uuid)
    topic = db.Column(db.String(200), nullable=False, index=True)
    subject = db.Column(db.String(100), nullable=False, index=True)
    title = db.Column(db.String(300))
    description = db.Column(db.Text)
    questions = db.Column(JSON, nullable=False)  # Array of question objects
    difficulty = db.Column(db.String(20), default="medium")  # easy, medium, hard, mixed
    time_limit_minutes = db.Column(db.Integer, default=30)
    is_adaptive = db.Column(db.Boolean, default=False)
    scheduled_date = db.Column(db.DateTime)
    created_by = db.Column(db.String(36), db.ForeignKey("users.id"))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # NEW: Assessment type and creation context
    assessment_type = db.Column(db.String(30), default="teacher_created")  # teacher_created, self_practice, student_challenge
    classroom_id = db.Column(db.String(36), db.ForeignKey("classrooms.id"), nullable=True, index=True)
    
    # NEW: AI generation options
    use_ai_questions = db.Column(db.Boolean, default=False)
    
    # NEW: Source filters used to create assessment (topic/chapter IDs)
    source_topics = db.Column(JSON, default=list)  # List of ClassroomTopic IDs
    source_chapters = db.Column(JSON, default=list)  # List of Chapter IDs
    include_weak_topics = db.Column(db.Boolean, default=False)
    
    # NEW: Challenge support
    is_challenge = db.Column(db.Boolean, default=False)
    original_assessment_id = db.Column(db.String(36), nullable=True)  # If cloned from challenge
    
    # NEW: Revision assessment tracking
    is_revision_assessment = db.Column(db.Boolean, default=False)  # Auto-generated from revision schedule
    revision_date = db.Column(db.Date, nullable=True, index=True)  # Date this revision assessment is for
    
    # Relationships
    results = db.relationship("AssessmentResult", backref="assessment", cascade="all, delete-orphan")
    
    def to_dict(self, include_questions=True):
        data = {
            "id": str(self.id),
            "topic": self.topic,
            "subject": self.subject,
            "title": self.title,
            "description": self.description,
            "difficulty": self.difficulty,
            "time_limit_minutes": self.time_limit_minutes,
            "is_adaptive": self.is_adaptive,
            "scheduled_date": self.scheduled_date.isoformat() if self.scheduled_date else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "created_by": self.created_by,
            "assessment_type": self.assessment_type,
            "classroom_id": self.classroom_id,
            "use_ai_questions": self.use_ai_questions,
            "source_topics": self.source_topics or [],
            "source_chapters": self.source_chapters or [],
            "include_weak_topics": self.include_weak_topics,
            "is_challenge": self.is_challenge,
            "is_revision_assessment": self.is_revision_assessment,
            "revision_date": self.revision_date.isoformat() if self.revision_date else None,
            "num_questions": len(self.questions) if self.questions else 0
        }
        if include_questions:
            data["questions"] = self.questions
        return data


class AssessmentResult(db.Model):
    """Student assessment submission results"""
    __tablename__ = "assessment_results"
    
    id = db.Column(db.String(36), primary_key=True, default=generate_uuid)
    user_id = db.Column(db.String(36), db.ForeignKey("users.id"), nullable=False, index=True)
    assessment_id = db.Column(db.String(36), db.ForeignKey("assessments.id"), nullable=False, index=True)
    answers = db.Column(JSON, nullable=False)  # Student's answers
    score = db.Column(db.Float, nullable=False)
    max_score = db.Column(db.Float, default=100.0)
    time_taken_seconds = db.Column(db.Integer)
    confidence_score = db.Column(db.Float)  # Self-reported confidence
    feedback = db.Column(JSON)  # AI-generated feedback per question
    completed_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        db.Index('idx_result_user_assessment', 'user_id', 'assessment_id'),
    )
    
    def to_dict(self):
        return {
            "id": str(self.id),
            "user_id": str(self.user_id),
            "assessment_id": str(self.assessment_id),
            "answers": self.answers,
            "score": self.score,
            "max_score": self.max_score,
            "time_taken_seconds": self.time_taken_seconds,
            "confidence_score": self.confidence_score,
            "feedback": self.feedback,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None
        }


class ChatSession(db.Model):
    """Chat session with AI tutor"""
    __tablename__ = "chat_sessions"
    
    id = db.Column(db.String(36), primary_key=True, default=generate_uuid)
    user_id = db.Column(db.String(36), db.ForeignKey("users.id"), nullable=False, index=True)
    title = db.Column(db.String(200))
    messages = db.Column(JSON, default=list)  # Array of message objects
    context = db.Column(JSON, default=dict)  # Session context/state
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def to_dict(self):
        return {
            "id": str(self.id),
            "user_id": str(self.user_id),
            "title": self.title,
            "messages": self.messages,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }


class ModerationLog(db.Model):
    """Log of content moderation decisions"""
    __tablename__ = "moderation_logs"
    
    id = db.Column(db.String(36), primary_key=True, default=generate_uuid)
    user_id = db.Column(db.String(36), db.ForeignKey("users.id"), nullable=False, index=True)
    message_text = db.Column(db.Text, nullable=False)
    classification = db.Column(db.String(50), nullable=False)  # academic, off_topic, inappropriate
    confidence = db.Column(db.Float)
    was_blocked = db.Column(db.Boolean, default=False)
    reason = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            "id": str(self.id),
            "user_id": str(self.user_id),
            "classification": self.classification,
            "confidence": self.confidence,
            "was_blocked": self.was_blocked,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


class Leaderboard(db.Model):
    """Gamification leaderboard"""
    __tablename__ = "leaderboard"
    
    id = db.Column(db.String(36), primary_key=True, default=generate_uuid)
    user_id = db.Column(db.String(36), db.ForeignKey("users.id"), nullable=False, unique=True)
    global_points = db.Column(db.Integer, default=0, index=True)
    subject_points = db.Column(JSON, default=dict)  # {"Math": 100, "Biology": 50}
    class_points = db.Column(db.Integer, default=0)
    badges = db.Column(JSON, default=list)  # ["First Quiz", "100 Days Streak"]
    study_streak = db.Column(db.Integer, default=0)
    longest_streak = db.Column(db.Integer, default=0)
    last_study_date = db.Column(db.Date)
    level = db.Column(db.Integer, default=1)
    xp = db.Column(db.Integer, default=0)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def to_dict(self):
        return {
            "id": str(self.id),
            "user_id": str(self.user_id),
            "global_points": self.global_points,
            "subject_points": self.subject_points,
            "class_points": self.class_points,
            "badges": self.badges,
            "study_streak": self.study_streak,
            "longest_streak": self.longest_streak,
            "level": self.level,
            "xp": self.xp
        }


class StudyNote(db.Model):
    """AI-generated or user study notes"""
    __tablename__ = "study_notes"
    
    id = db.Column(db.String(36), primary_key=True, default=generate_uuid)
    user_id = db.Column(db.String(36), db.ForeignKey("users.id"), nullable=False, index=True)
    topic = db.Column(db.String(200), nullable=False, index=True)
    subject = db.Column(db.String(100), nullable=False)
    title = db.Column(db.String(300), nullable=False)
    content = db.Column(db.Text, nullable=False)
    key_terms = db.Column(JSON, default=list)  # [{term, definition, source}]
    sources = db.Column(JSON, default=list)  # Source citations
    is_ai_generated = db.Column(db.Boolean, default=True)
    is_public = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def to_dict(self):
        return {
            "id": str(self.id),
            "user_id": str(self.user_id),
            "topic": self.topic,
            "subject": self.subject,
            "title": self.title,
            "content": self.content,
            "key_terms": self.key_terms,
            "sources": self.sources,
            "is_ai_generated": self.is_ai_generated,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


class AssessmentChallenge(db.Model):
    """Track assessment challenges between students"""
    __tablename__ = "assessment_challenges"
    
    id = db.Column(db.String(36), primary_key=True, default=generate_uuid)
    
    # The assessment being shared
    assessment_id = db.Column(db.String(36), db.ForeignKey("assessments.id"), nullable=False, index=True)
    
    # Sender & recipient
    sender_id = db.Column(db.String(36), db.ForeignKey("users.id"), nullable=False, index=True)
    recipient_id = db.Column(db.String(36), db.ForeignKey("users.id"), nullable=False, index=True)
    
    # Challenge status: pending, accepted, declined, completed
    status = db.Column(db.String(20), default="pending", index=True)
    
    # Results comparison
    sender_score = db.Column(db.Float)  # Sender's score on the assessment
    recipient_score = db.Column(db.Float)  # Recipient's score on the assessment
    recipient_assessment_id = db.Column(db.String(36))  # Cloned assessment for recipient tracking
    
    # Optional message
    challenge_message = db.Column(db.Text)
    
    # Timestamps
    sent_at = db.Column(db.DateTime, default=datetime.utcnow)
    responded_at = db.Column(db.DateTime)
    completed_at = db.Column(db.DateTime)
    
    # Relationships
    assessment = db.relationship("Assessment", backref="challenges")
    sender = db.relationship("User", foreign_keys=[sender_id], backref="sent_challenges")
    recipient = db.relationship("User", foreign_keys=[recipient_id], backref="received_challenges")
    
    __table_args__ = (
        db.Index('idx_challenge_sender', 'sender_id'),
        db.Index('idx_challenge_recipient', 'recipient_id'),
        db.Index('idx_challenge_status', 'status'),
    )
    
    def to_dict(self, include_assessment=False):
        data = {
            "id": str(self.id),
            "assessment_id": str(self.assessment_id),
            "sender_id": str(self.sender_id),
            "recipient_id": str(self.recipient_id),
            "status": self.status,
            "sender_score": self.sender_score,
            "recipient_score": self.recipient_score,
            "challenge_message": self.challenge_message,
            "sent_at": self.sent_at.isoformat() if self.sent_at else None,
            "responded_at": self.responded_at.isoformat() if self.responded_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            # Include sender/recipient info
            "sender_name": f"{self.sender.first_name or ''} {self.sender.last_name or ''}".strip() or self.sender.username if self.sender else None,
            "recipient_name": f"{self.recipient.first_name or ''} {self.recipient.last_name or ''}".strip() or self.recipient.username if self.recipient else None
        }
        if include_assessment and self.assessment:
            data["assessment"] = self.assessment.to_dict(include_questions=False)
        return data
