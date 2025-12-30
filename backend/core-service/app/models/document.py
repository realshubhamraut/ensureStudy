"""
Document Models for Teacher Materials Ingestion
SQLAlchemy models for documents, pages, chunks, and quality reports.
"""
from datetime import datetime
import uuid
from app import db


def generate_uuid():
    return str(uuid.uuid4())


class Document(db.Model):
    """
    Main record for uploaded teacher documents (PDFs/images).
    Tracks ingestion status and file metadata.
    """
    __tablename__ = "documents"
    
    id = db.Column(db.String(36), primary_key=True, default=generate_uuid)
    class_id = db.Column(db.String(36), nullable=False, index=True)
    title = db.Column(db.String(255), nullable=False)
    filename = db.Column(db.String(255), nullable=False)
    s3_path = db.Column(db.String(500), nullable=False)
    file_hash = db.Column(db.String(64), nullable=False, index=True)
    file_size = db.Column(db.Integer)
    mime_type = db.Column(db.String(100))
    uploaded_by = db.Column(db.String(36), nullable=False, index=True)
    uploaded_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Ingestion status: uploaded, processing, indexed, error
    status = db.Column(db.String(20), default='uploaded', index=True)
    requires_manual_review = db.Column(db.Boolean, default=False)
    error_message = db.Column(db.Text)
    version = db.Column(db.Integer, default=1)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    pages = db.relationship("DocumentPage", backref="document", cascade="all, delete-orphan", lazy="dynamic")
    chunks = db.relationship("DocumentChunk", backref="document", cascade="all, delete-orphan", lazy="dynamic")
    quality_report = db.relationship("DocumentQualityReport", backref="document", uselist=False, cascade="all, delete-orphan")
    
    def to_dict(self):
        return {
            "id": self.id,
            "class_id": self.class_id,
            "title": self.title,
            "filename": self.filename,
            "file_size": self.file_size,
            "mime_type": self.mime_type,
            "status": self.status,
            "requires_manual_review": self.requires_manual_review,
            "error_message": self.error_message,
            "version": self.version,
            "uploaded_at": self.uploaded_at.isoformat() if self.uploaded_at else None,
            "page_count": self.pages.count() if self.pages else 0,
            "chunk_count": self.chunks.count() if self.chunks else 0
        }
    
    def to_status_dict(self):
        """Minimal status response for polling."""
        return {
            "doc_id": self.id,
            "status": self.status,
            "requires_manual_review": self.requires_manual_review,
            "error_message": self.error_message,
            "page_count": self.pages.count() if self.pages else 0,
            "chunk_count": self.chunks.count() if self.chunks else 0
        }


class DocumentPage(db.Model):
    """
    Per-page OCR results and metadata.
    Links to the processed JSON stored in S3.
    """
    __tablename__ = "document_pages"
    
    id = db.Column(db.String(36), primary_key=True, default=generate_uuid)
    document_id = db.Column(db.String(36), db.ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True)
    page_number = db.Column(db.Integer, nullable=False)
    s3_page_json_path = db.Column(db.String(500))
    s3_page_image_path = db.Column(db.String(500))
    ocr_confidence = db.Column(db.Float)
    text_length = db.Column(db.Integer, default=0)
    block_count = db.Column(db.Integer, default=0)
    ocr_method = db.Column(db.String(20))  # nanonets, tesseract, paddleocr
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        db.UniqueConstraint('document_id', 'page_number', name='uq_doc_page'),
    )
    
    def to_dict(self):
        return {
            "id": self.id,
            "page_number": self.page_number,
            "ocr_confidence": self.ocr_confidence,
            "text_length": self.text_length,
            "block_count": self.block_count,
            "ocr_method": self.ocr_method
        }


class DocumentChunk(db.Model):
    """
    Chunked text for RAG retrieval.
    Each chunk links to a Qdrant vector ID.
    """
    __tablename__ = "document_chunks"
    
    id = db.Column(db.String(36), primary_key=True, default=generate_uuid)
    document_id = db.Column(db.String(36), db.ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True)
    page_number = db.Column(db.Integer, nullable=False)
    block_id = db.Column(db.String(36))
    chunk_index = db.Column(db.Integer, nullable=False)
    preview_text = db.Column(db.String(200))
    full_text = db.Column(db.Text)
    bbox_json = db.Column(db.Text)  # JSON: [x1, y1, x2, y2]
    qdrant_id = db.Column(db.String(36), index=True)
    token_count = db.Column(db.Integer)
    content_type = db.Column(db.String(20), default='text')  # text, formula_image, table
    embedding_hash = db.Column(db.String(64))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            "id": self.id,
            "chunk_id": self.id,
            "page_number": self.page_number,
            "preview_text": self.preview_text,
            "bbox": self.bbox_json,
            "token_count": self.token_count,
            "content_type": self.content_type
        }


class DocumentQualityReport(db.Model):
    """
    Quality metrics for processed documents.
    Used to flag documents needing manual review.
    """
    __tablename__ = "document_quality_reports"
    
    id = db.Column(db.String(36), primary_key=True, default=generate_uuid)
    document_id = db.Column(db.String(36), db.ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, unique=True)
    avg_ocr_confidence = db.Column(db.Float)
    min_ocr_confidence = db.Column(db.Float)
    pages_processed = db.Column(db.Integer, default=0)
    pages_failed = db.Column(db.Integer, default=0)
    flagged_pages = db.Column(db.ARRAY(db.Integer))  # PostgreSQL array
    total_chunks = db.Column(db.Integer, default=0)
    total_tokens = db.Column(db.Integer, default=0)
    processing_time_ms = db.Column(db.Integer)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            "document_id": self.document_id,
            "avg_ocr_confidence": self.avg_ocr_confidence,
            "min_ocr_confidence": self.min_ocr_confidence,
            "pages_processed": self.pages_processed,
            "pages_failed": self.pages_failed,
            "flagged_pages": self.flagged_pages or [],
            "total_chunks": self.total_chunks,
            "total_tokens": self.total_tokens,
            "processing_time_ms": self.processing_time_ms
        }
