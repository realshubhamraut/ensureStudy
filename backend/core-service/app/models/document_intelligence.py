"""
Database Schema Extensions for Document Intelligence

NEW TABLES (additive only):
- formulas: Mathematical formula extraction
- web_fetches: Web content tracking  
- model_training_logs: ML model versioning

EXTENSIONS to existing tables:
- Enhanced note_search_history with answer/confidence fields
"""
from datetime import datetime
from app import db
import uuid


def generate_uuid():
    return str(uuid.uuid4())


class Formula(db.Model):
    """Extracted mathematical formulas from documents"""
    __tablename__ = "formulas"
    
    id = db.Column(db.String(36), primary_key=True, default=generate_uuid)
    
    # Parent document/page
    job_id = db.Column(db.String(36), db.ForeignKey("note_processing_jobs.id", ondelete="CASCADE"), nullable=False)
    page_id = db.Column(db.String(36), db.ForeignKey("digitized_note_pages.id", ondelete="CASCADE"))
    
    # Formula data
    formula_index = db.Column(db.Integer, nullable=False)  # Position in document
    latex = db.Column(db.Text, nullable=False)  # LaTeX representation
    plain_text = db.Column(db.Text)  # Plain text fallback
    
    # Image storage
    image_s3_path = db.Column(db.String(512))
    image_s3_bucket = db.Column(db.String(128))
    
    # Extraction metadata
    bounding_box = db.Column(db.JSON)  # {x, y, width, height}
    confidence = db.Column(db.Float)  # 0-1 confidence score
    formula_type = db.Column(db.String(50))  # equation, expression, symbol
    page_number = db.Column(db.Integer)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    job = db.relationship("NoteProcessingJob", backref="formulas")
    page = db.relationship("DigitizedNotePage", backref="formulas")
    
    __table_args__ = (
        db.Index('idx_formulas_job', 'job_id'),
        db.Index('idx_formulas_page', 'page_id'),
        db.CheckConstraint('formula_index >= 0', name='check_formula_index'),
        db.CheckConstraint('confidence >= 0 AND confidence <= 1', name='check_formula_confidence'),
    )
    
    def to_dict(self):
        return {
            "id": self.id,
            "job_id": self.job_id,
            "page_id": self.page_id,
            "formula_index": self.formula_index,
            "latex": self.latex,
            "plain_text": self.plain_text,
            "image_s3_path": self.image_s3_path,
            "confidence": self.confidence,
            "formula_type": self.formula_type,
            "page_number": self.page_number,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


class WebFetch(db.Model):
    """Tracking for web-fetched educational content"""
    __tablename__ = "web_fetches"
    
    id = db.Column(db.String(36), primary_key=True, default=generate_uuid)
    
    # Source information
    source_type = db.Column(db.String(50), nullable=False)  # wikipedia, youtube, khan_academy, etc
    source_url = db.Column(db.Text, nullable=False, unique=True)
    topic = db.Column(db.String(255))
    
    # Content metadata
    title = db.Column(db.String(500))
    author = db.Column(db.String(255))
    publish_date = db.Column(db.Date)
    content_type = db.Column(db.String(100))  # article, video_transcript, pdf
    content_hash = db.Column(db.String(64))  # SHA256 for deduplication
    
    # Storage
    content_s3_path = db.Column(db.String(512))
    content_s3_bucket = db.Column(db.String(128))
    
    # Processing status
    fetch_status = db.Column(db.String(20), nullable=False, default='pending')  # pending, success, failed
    retry_count = db.Column(db.Integer, default=0)
    error_message = db.Column(db.Text)
    
    # Indexing status
    indexed = db.Column(db.Boolean, default=False)
    indexed_at = db.Column(db.DateTime)
    
    # Initiated by
    fetch_initiated_by = db.Column(db.String(36), db.ForeignKey("users.id", ondelete="SET NULL"))
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    fetched_at = db.Column(db.DateTime)
    
    # Relationships
    initiator = db.relationship("User", backref="web_fetch_requests")
    
    __table_args__ = (
        db.Index('idx_web_fetch_source_type', 'source_type'),
        db.Index('idx_web_fetch_status', 'fetch_status'),
        db.Index('idx_web_fetch_topic', 'topic'),
        db.Index('idx_web_fetch_indexed', 'indexed'),
        db.CheckConstraint("fetch_status IN ('pending', 'success', 'failed')", name='check_fetch_status'),
        db.CheckConstraint("source_type IN ('wikipedia', 'youtube', 'khan_academy', 'mit_ocw', 'ncert', 'pdf', 'article')", name='check_source_type'),
        db.CheckConstraint('retry_count >= 0', name='check_retry_count'),
    )
    
    def to_dict(self):
        return {
            "id": self.id,
            "source_type": self.source_type,
            "source_url": self.source_url,
            "topic": self.topic,
            "title": self.title,
            "author": self.author,
            "content_type": self.content_type,
            "fetch_status": self.fetch_status,
            "indexed": self.indexed,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "fetched_at": self.fetched_at.isoformat() if self.fetched_at else None
        }


class ModelTrainingLog(db.Model):
    """ML model training history and versioning"""
    __tablename__ = "model_training_logs"
    
    id = db.Column(db.String(36), primary_key=True, default=generate_uuid)
    
    # Model identification
    model_name = db.Column(db.String(100), nullable=False)  # htr_model, grading_siamese, etc
    model_version = db.Column(db.String(20), nullable=False)
    
    # Training info
    training_dataset = db.Column(db.String(255))
    training_notebook_path = db.Column(db.String(512))
    
    # Model storage
    model_s3_path = db.Column(db.String(512))
    model_s3_bucket = db.Column(db.String(128))
    
    # Hyperparameters and metrics
    hyperparameters = db.Column(db.JSON)  # {learning_rate, batch_size, etc}
    metrics = db.Column(db.JSON)  # {accuracy, loss, f1, etc}
    
    # Training metadata
    training_duration_seconds = db.Column(db.Integer)
    trained_by = db.Column(db.String(36), db.ForeignKey("users.id", ondelete="SET NULL"))
    
    # Deployment status
    deployed = db.Column(db.Boolean, default=False)
    deployed_at = db.Column(db.DateTime)
    
    # Timestamps
    training_started_at = db.Column(db.DateTime, nullable=False)
    training_completed_at = db.Column(db.DateTime)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    trainer = db.relationship("User", backref="trained_models")
    
    __table_args__ = (
        db.UniqueConstraint('model_name', 'model_version', name='unique_model_version'),
        db.Index('idx_model_name', 'model_name'),
        db.Index('idx_model_deployed', 'deployed'),
        db.Index('idx_model_training_started', 'training_started_at'),
        db.CheckConstraint("model_name IN ('htr_model', 'grading_siamese', 'speech_fluency', 'gesture_analysis', 'moderation', 'difficulty')", name='check_model_name'),
    )
    
    def to_dict(self):
        return {
            "id": self.id,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "training_dataset": self.training_dataset,
            "hyperparameters": self.hyperparameters,
            "metrics": self.metrics,
            "training_duration_seconds": self.training_duration_seconds,
            "deployed": self.deployed,
            "deployed_at": self.deployed_at.isoformat() if self.deployed_at else None,
            "training_started_at": self.training_started_at.isoformat() if self.training_started_at else None,
            "training_completed_at": self.training_completed_at.isoformat() if self.training_completed_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


class RAGQuery(db.Model):
    """Enhanced AI tutor query logs with answers and sources"""
    __tablename__ = "rag_queries"
    
    id = db.Column(db.String(36), primary_key=True, default=generate_uuid)
    
    # Context
    student_id = db.Column(db.String(36), db.ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    classroom_id = db.Column(db.String(36), db.ForeignKey("classrooms.id", ondelete="CASCADE"), nullable=False)
    
    # Question
    question_text = db.Column(db.Text, nullable=False)
    question_hash = db.Column(db.String(64), nullable=False)  # For cache lookup
    
    # Answer
    answer_text = db.Column(db.Text)
    sources_used = db.Column(db.JSON)  # [{chunk_id, confidence, text_preview}]
    confidence_score = db.Column(db.Float)  # 0-1
    
    # Performance
    response_time_ms = db.Column(db.Integer)
    was_cached = db.Column(db.Boolean, default=False)
    used_web_sources = db.Column(db.Boolean, default=False)
    
    # LLM info
    llm_model = db.Column(db.String(100))
    llm_tokens_used = db.Column(db.Integer)
    
    # Student feedback
    student_rating = db.Column(db.Integer)  # 1-5
    student_feedback = db.Column(db.Text)
    
    # Error tracking
    error_occurred = db.Column(db.Boolean, default=False)
    error_message = db.Column(db.Text)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    student = db.relationship("User", backref="rag_queries")
    classroom = db.relationship("Classroom", backref="rag_queries")
    
    __table_args__ = (
        db.Index('idx_rag_student', 'student_id'),
        db.Index('idx_rag_classroom', 'classroom_id'),
        db.Index('idx_rag_question_hash', 'question_hash'),
        db.Index('idx_rag_created', 'created_at'),
        db.Index('idx_rag_confidence', 'confidence_score'),
        db.CheckConstraint('confidence_score >= 0 AND confidence_score <= 1', name='check_rag_confidence'),
        db.CheckConstraint('student_rating >= 1 AND student_rating <= 5', name='check_rag_rating'),
    )
    
    def to_dict(self, include_answer=True):
        data = {
            "id": self.id,
            "student_id": self.student_id,
            "classroom_id": self.classroom_id,
            "question_text": self.question_text,
            "confidence_score": self.confidence_score,
            "response_time_ms": self.response_time_ms,
            "was_cached": self.was_cached,
            "student_rating": self.student_rating,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }
        
        if include_answer:
            data["answer_text"] = self.answer_text
            data["sources_used"] = self.sources_used
        
        return data
