"""
Digitized Notes Models for Video-to-Notes Processing
Supports: Upload tracking, page extraction, OCR results, vector embeddings
"""
from datetime import datetime
from app import db
import uuid


def generate_uuid():
    return str(uuid.uuid4())


class NoteProcessingJob(db.Model):
    """Tracks video/image upload processing jobs"""
    __tablename__ = "note_processing_jobs"
    
    id = db.Column(db.String(36), primary_key=True, default=generate_uuid)
    
    # Owner
    student_id = db.Column(db.String(36), db.ForeignKey("users.id"), nullable=False)
    classroom_id = db.Column(db.String(36), db.ForeignKey("classrooms.id"), nullable=False)
    
    # Job metadata
    title = db.Column(db.String(255), nullable=False)  # User-provided title
    description = db.Column(db.Text)  # Optional description
    
    # INTEGRATION NOTE: Extended to support document processing
    source_type = db.Column(db.String(20), nullable=False)  # 'video', 'images', or 'document'
    source_url = db.Column(db.Text, nullable=False)  # S3/storage URL
    source_filename = db.Column(db.String(255))
    source_size_bytes = db.Column(db.BigInteger)
    
    # Processing status
    status = db.Column(db.String(20), default='pending')  # pending, processing, completed, failed
    progress_percent = db.Column(db.Integer, default=0)  # 0-100
    current_step = db.Column(db.String(100))  # Current processing step description
    
    # Results
    total_pages = db.Column(db.Integer, default=0)
    processed_pages = db.Column(db.Integer, default=0)
    avg_confidence = db.Column(db.Float)  # Average OCR confidence
    pdf_path = db.Column(db.Text)  # Path to combined PDF document
    
    # Error handling
    error_message = db.Column(db.Text)
    retry_count = db.Column(db.Integer, default=0)
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    started_at = db.Column(db.DateTime)
    completed_at = db.Column(db.DateTime)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    student = db.relationship("User", backref="note_jobs")
    classroom = db.relationship("Classroom", backref="note_jobs")
    pages = db.relationship("DigitizedNotePage", backref="job", cascade="all, delete-orphan")
    
    def to_dict(self, include_pages=False):
        data = {
            "id": self.id,
            "student_id": self.student_id,
            "classroom_id": self.classroom_id,
            "title": self.title,
            "description": self.description,
            "source_type": self.source_type,
            "source_url": self.source_url,
            "source_filename": self.source_filename,
            "source_size_bytes": self.source_size_bytes,
            "status": self.status,
            "progress_percent": self.progress_percent,
            "current_step": self.current_step,
            "total_pages": self.total_pages,
            "processed_pages": self.processed_pages,
            "avg_confidence": self.avg_confidence,
            "pdf_path": self.pdf_path,
            "error_message": self.error_message,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }
        
        if include_pages:
            data["pages"] = [p.to_dict() for p in self.pages]
        
        return data


class DigitizedNotePage(db.Model):
    """Individual extracted page from notes"""
    __tablename__ = "digitized_note_pages"
    
    id = db.Column(db.String(36), primary_key=True, default=generate_uuid)
    job_id = db.Column(db.String(36), db.ForeignKey("note_processing_jobs.id"), nullable=False)
    
    # Page metadata
    page_number = db.Column(db.Integer, nullable=False)
    frame_timestamp = db.Column(db.Float)  # Video timestamp if from video
    
    # Image URLs
    original_image_url = db.Column(db.Text)
    enhanced_image_url = db.Column(db.Text)
    thumbnail_url = db.Column(db.Text)
    
    # OCR results
    extracted_text = db.Column(db.Text)
    confidence_score = db.Column(db.Float)  # 0-1 OCR confidence
    
    # Image quality metrics
    brightness = db.Column(db.Float)
    contrast = db.Column(db.Float)
    sharpness = db.Column(db.Float)
    
    # Processing status
    status = db.Column(db.String(20), default='pending')  # pending, enhanced, ocr_done, embedded
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    embeddings = db.relationship("NoteEmbedding", backref="page", cascade="all, delete-orphan")
    
    def to_dict(self, include_text=True):
        data = {
            "id": self.id,
            "job_id": self.job_id,
            "page_number": self.page_number,
            "frame_timestamp": self.frame_timestamp,
            "original_image_url": self.original_image_url,
            "enhanced_image_url": self.enhanced_image_url,
            "thumbnail_url": self.thumbnail_url,
            "confidence_score": self.confidence_score,
            "status": self.status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
        
        if include_text:
            data["extracted_text"] = self.extracted_text
        
        return data


class NoteEmbedding(db.Model):
    """Vector embeddings for semantic search"""
    __tablename__ = "note_embeddings"
    
    id = db.Column(db.String(36), primary_key=True, default=generate_uuid)
    page_id = db.Column(db.String(36), db.ForeignKey("digitized_note_pages.id"), nullable=False)
    
    # Chunk info
    chunk_index = db.Column(db.Integer, nullable=False)
    chunk_text = db.Column(db.Text, nullable=False)
    chunk_start_char = db.Column(db.Integer)
    chunk_end_char = db.Column(db.Integer)
    
    # Vector DB reference
    qdrant_collection = db.Column(db.String(100), default='notes_embeddings')
    qdrant_point_id = db.Column(db.String(36))  # UUID in Qdrant
    
    # Metadata
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            "id": self.id,
            "page_id": self.page_id,
            "chunk_index": self.chunk_index,
            "chunk_text": self.chunk_text[:200] + "..." if len(self.chunk_text) > 200 else self.chunk_text,
            "qdrant_point_id": self.qdrant_point_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class NoteSearchHistory(db.Model):
    """Track search queries for analytics"""
    __tablename__ = "note_search_history"
    
    id = db.Column(db.String(36), primary_key=True, default=generate_uuid)
    student_id = db.Column(db.String(36), db.ForeignKey("users.id"), nullable=False)
    classroom_id = db.Column(db.String(36), db.ForeignKey("classrooms.id"))
    
    query = db.Column(db.Text, nullable=False)
    result_count = db.Column(db.Integer, default=0)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    student = db.relationship("User", backref="note_searches")
