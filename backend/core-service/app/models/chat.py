"""
Chat History Database Models
Stores AI tutor conversations and messages server-side
"""
from datetime import datetime
from app import db
import uuid


def generate_uuid():
    return str(uuid.uuid4())


class ChatConversation(db.Model):
    """A conversation thread with the AI tutor"""
    __tablename__ = 'chat_conversations'
    
    id = db.Column(db.String(36), primary_key=True, default=generate_uuid)
    user_id = db.Column(db.String(36), db.ForeignKey('users.id'), nullable=False, index=True)
    title = db.Column(db.String(200), nullable=False, default='New Conversation')
    pinned = db.Column(db.Boolean, default=False)
    subject = db.Column(db.String(50))  # math, science, general, etc.
    classroom_id = db.Column(db.String(36), db.ForeignKey('classrooms.id'), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    messages = db.relationship('ChatMessage', backref='conversation', lazy='dynamic', 
                              order_by='ChatMessage.timestamp', cascade='all, delete-orphan')
    sources = db.relationship('ChatSource', backref='conversation', lazy='dynamic',
                             cascade='all, delete-orphan')
    
    def to_dict(self, include_messages=False):
        result = {
            'id': self.id,
            'title': self.title,
            'pinned': self.pinned,
            'subject': self.subject,
            'classroom_id': self.classroom_id,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'message_count': self.messages.count() if self.messages else 0
        }
        if include_messages:
            result['messages'] = [m.to_dict() for m in self.messages.all()]
            result['sources'] = [s.to_dict() for s in self.sources.all()]
        return result


class ChatMessage(db.Model):
    """A single message in a conversation"""
    __tablename__ = 'chat_messages'
    
    id = db.Column(db.String(36), primary_key=True, default=generate_uuid)
    conversation_id = db.Column(db.String(36), db.ForeignKey('chat_conversations.id'), nullable=False, index=True)
    type = db.Column(db.String(20), nullable=False)  # 'user' or 'assistant'
    content = db.Column(db.Text, nullable=False)
    
    # Store full AI response as JSON for detailed view, sources, etc.
    response_json = db.Column(db.JSON, nullable=True)
    
    # Optional fields
    subject = db.Column(db.String(50))
    image_url = db.Column(db.String(500))  # If user sent an image
    
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    
    def to_dict(self):
        return {
            'id': self.id,
            'type': self.type,
            'content': self.content,
            'response': self.response_json,
            'subject': self.subject,
            'image_url': self.image_url,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None
        }


class ChatSource(db.Model):
    """Sources/resources associated with a conversation"""
    __tablename__ = 'chat_sources'
    
    id = db.Column(db.String(36), primary_key=True, default=generate_uuid)
    conversation_id = db.Column(db.String(36), db.ForeignKey('chat_conversations.id'), nullable=False, index=True)
    
    source_type = db.Column(db.String(20), nullable=False)  # pdf, video, article, image, flowchart
    title = db.Column(db.String(300), nullable=False)
    url = db.Column(db.String(1000))
    thumbnail_url = db.Column(db.String(1000))
    relevance = db.Column(db.Integer)  # 0-100
    snippet = db.Column(db.Text)
    source_name = db.Column(db.String(100))  # e.g., "YouTube", "Wikipedia"
    
    # Additional metadata as JSON (renamed from 'metadata' which is reserved in SQLAlchemy)
    source_metadata = db.Column(db.JSON, nullable=True)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        result = {
            'id': self.id,
            'type': self.source_type,
            'title': self.title,
            'url': self.url,
            'thumbnailUrl': self.thumbnail_url,
            'relevance': self.relevance,
            'snippet': self.snippet,
            'source': self.source_name
        }
        # Merge additional metadata
        if self.source_metadata:
            result.update(self.source_metadata)
        return result

