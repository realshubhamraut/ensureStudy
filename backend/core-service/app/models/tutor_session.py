"""
Tutor Session Model - Tracks user sessions for resource chaining

This model stores session state including query chain and resource list
for implementing follow-up question detection and resource deduplication.
"""
import uuid
from datetime import datetime, timedelta
from typing import Optional, List
from sqlalchemy import Column, String, DateTime, ForeignKey, JSON, Boolean, Integer, Text, Float, Enum as SQLEnum
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from enum import Enum

from . import db


class ResourceType(str, Enum):
    """Type of resource in session"""
    TEXT = "text"
    PDF = "pdf"
    IMAGE = "image"
    VIDEO = "video"
    FLOWCHART = "flowchart"
    WIKIPEDIA = "wikipedia"
    ARTICLE = "article"


class ResourceSource(str, Enum):
    """Source of the resource"""
    CLASSROOM = "classroom"
    WIKIPEDIA = "wikipedia"
    WEB = "web"
    YOUTUBE = "youtube"


class TutorSession(db.Model):
    """
    Tracks a user's tutoring session for resource chaining.
    
    Sessions maintain:
    - Query chain with embeddings for relatedness detection
    - Resource list with deduplication
    - Configuration for TTL and limits
    """
    __tablename__ = 'tutor_sessions'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=False, index=True)
    classroom_id = Column(UUID(as_uuid=True), ForeignKey('classrooms.id'), nullable=True, index=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_active_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    expires_at = Column(DateTime, nullable=True)
    
    # Configuration
    config = Column(JSON, default=lambda: {
        "ttl_hours": 24,
        "max_resources": 25,
        "relatedness_threshold": 0.65,
        "relatedness_lookback": 3
    })
    
    # Status
    is_active = Column(Boolean, default=True, nullable=False)
    
    # Relationships
    user = relationship("User", backref="tutor_sessions")
    turns = relationship("SessionTurn", back_populates="session", cascade="all, delete-orphan", order_by="SessionTurn.turn_number")
    resources = relationship("SessionResource", back_populates="session", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<TutorSession {self.id} user={self.user_id}>"
    
    def is_expired(self) -> bool:
        """Check if session has expired"""
        if self.expires_at:
            return datetime.utcnow() > self.expires_at
        ttl_hours = self.config.get("ttl_hours", 24)
        return datetime.utcnow() > self.last_active_at + timedelta(hours=ttl_hours)
    
    def to_dict(self) -> dict:
        return {
            "session_id": str(self.id),
            "user_id": str(self.user_id),
            "classroom_id": str(self.classroom_id) if self.classroom_id else None,
            "created_at": self.created_at.isoformat(),
            "last_active_at": self.last_active_at.isoformat(),
            "is_active": self.is_active,
            "turn_count": len(self.turns) if self.turns else 0,
            "resource_count": len(self.resources) if self.resources else 0,
            "config": self.config
        }


class SessionTurn(db.Model):
    """
    A single query turn within a session.
    
    Stores the question, its embedding (for relatedness detection),
    and whether it was related to previous turns.
    """
    __tablename__ = 'session_turns'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey('tutor_sessions.id'), nullable=False, index=True)
    
    # Turn info
    turn_number = Column(Integer, nullable=False)
    question = Column(Text, nullable=False)
    question_hash = Column(String(64), nullable=False, index=True)  # SHA256 of question
    
    # Embedding stored as JSON array (384 floats for MiniLM)
    question_embedding = Column(JSON, nullable=True)
    
    # Relatedness
    related_to_previous = Column(Boolean, default=False)
    relatedness_score = Column(Float, nullable=True)  # Highest similarity to prev turns
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationship
    session = relationship("TutorSession", back_populates="turns")
    
    def __repr__(self):
        return f"<SessionTurn {self.turn_number} session={self.session_id}>"
    
    def to_dict(self) -> dict:
        return {
            "turn": self.turn_number,
            "question": self.question,
            "related": self.related_to_previous,
            "relatedness_score": self.relatedness_score,
            "timestamp": self.created_at.isoformat()
        }


class SessionResource(db.Model):
    """
    A resource discovered during a session.
    
    Stores metadata for deduplication (hash, URL) and display
    (preview, inline render flag).
    """
    __tablename__ = 'session_resources'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey('tutor_sessions.id'), nullable=False, index=True)
    
    # Resource identification
    resource_type = Column(String(20), nullable=False)  # text, pdf, image, video, flowchart, wikipedia
    source = Column(String(20), nullable=False)  # classroom, wikipedia, web, youtube
    
    # URLs and dedup
    url = Column(Text, nullable=True)
    canonical_url = Column(Text, nullable=True, index=True)  # Normalized URL for dedup
    content_hash = Column(String(64), nullable=True, index=True)  # SHA256 of content
    
    # Metadata
    title = Column(String(500), nullable=True)
    preview_summary = Column(String(300), nullable=True)  # 180-300 char preview
    qdrant_collection = Column(String(100), nullable=True)
    qdrant_point_ids = Column(JSON, default=list)  # List of Qdrant point IDs
    
    # Rendering
    inline_render = Column(Boolean, default=False)  # True for Wikipedia
    inline_html = Column(Text, nullable=True)  # Sanitized HTML for inline display
    
    # For documents
    page_number = Column(Integer, nullable=True)
    bbox = Column(JSON, nullable=True)  # Bounding box for highlight
    signed_url = Column(Text, nullable=True)  # Signed URL for PDFs
    
    # Timestamps
    inserted_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_referenced_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationship
    session = relationship("TutorSession", back_populates="resources")
    
    def __repr__(self):
        return f"<SessionResource {self.title} type={self.resource_type}>"
    
    def to_dict(self) -> dict:
        return {
            "resource_id": str(self.id),
            "type": self.resource_type,
            "source": self.source,
            "url": self.url,
            "title": self.title,
            "preview_summary": self.preview_summary,
            "inline_render": self.inline_render,
            "inserted_at": self.inserted_at.isoformat(),
            "last_referenced_at": self.last_referenced_at.isoformat(),
            "hash": self.content_hash
        }
    
    def update_reference(self):
        """Update last_referenced_at timestamp"""
        self.last_referenced_at = datetime.utcnow()
