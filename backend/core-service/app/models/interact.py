"""
Interact Module Models - Conversations and Messages

Tables:
- conversations: Chat threads (direct, group, parent-teacher-student)
- conversation_participants: Users in each conversation
- messages: Chat messages with moderation flags
- interaction_analytics: Daily interaction tracking
"""
from datetime import datetime, date
from app import db
import uuid


def generate_uuid():
    return str(uuid.uuid4())


class Conversation(db.Model):
    """A conversation thread (direct, group, or three-way)"""
    __tablename__ = "conversations"
    
    id = db.Column(db.String(36), primary_key=True, default=generate_uuid)
    
    # Type of conversation
    type = db.Column(db.String(20), nullable=False, default='direct')  # 'direct', 'group', 'parent_teacher_student'
    title = db.Column(db.String(200))  # For group chats
    
    # Context
    created_by = db.Column(db.String(36), db.ForeignKey("users.id"), nullable=False)
    classroom_id = db.Column(db.String(36), db.ForeignKey("classrooms.id"), nullable=True)
    
    # Settings
    is_moderated = db.Column(db.Boolean, default=True)
    is_active = db.Column(db.Boolean, default=True)
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_message_at = db.Column(db.DateTime)
    
    # Relationships
    participants = db.relationship("ConversationParticipant", backref="conversation", cascade="all, delete-orphan")
    messages = db.relationship("Message", backref="conversation", cascade="all, delete-orphan", lazy="dynamic")
    creator = db.relationship("User", foreign_keys=[created_by])
    
    def to_dict(self, include_participants=False, include_last_message=False):
        data = {
            "id": self.id,
            "type": self.type,
            "title": self.title,
            "created_by": self.created_by,
            "classroom_id": self.classroom_id,
            "is_moderated": self.is_moderated,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "last_message_at": self.last_message_at.isoformat() if self.last_message_at else None
        }
        if include_participants:
            data["participants"] = [p.to_dict() for p in self.participants]
        if include_last_message:
            last_msg = self.messages.order_by(Message.created_at.desc()).first()
            data["last_message"] = last_msg.to_dict() if last_msg else None
        return data


class ConversationParticipant(db.Model):
    """A user participating in a conversation"""
    __tablename__ = "conversation_participants"
    
    id = db.Column(db.String(36), primary_key=True, default=generate_uuid)
    conversation_id = db.Column(db.String(36), db.ForeignKey("conversations.id"), nullable=False, index=True)
    user_id = db.Column(db.String(36), db.ForeignKey("users.id"), nullable=False, index=True)
    
    # User's role in this conversation
    role = db.Column(db.String(20))  # 'student', 'teacher', 'parent'
    
    # Read status
    joined_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_read_at = db.Column(db.DateTime)
    
    # Settings
    is_muted = db.Column(db.Boolean, default=False)
    is_admin = db.Column(db.Boolean, default=False)  # Can add/remove participants
    
    # Relationship to get user details
    user = db.relationship("User", foreign_keys=[user_id])
    
    __table_args__ = (
        db.UniqueConstraint('conversation_id', 'user_id', name='unique_conversation_participant'),
    )
    
    def to_dict(self):
        return {
            "id": self.id,
            "conversation_id": self.conversation_id,
            "user_id": self.user_id,
            "role": self.role,
            "joined_at": self.joined_at.isoformat() if self.joined_at else None,
            "last_read_at": self.last_read_at.isoformat() if self.last_read_at else None,
            "is_muted": self.is_muted,
            "is_admin": self.is_admin,
            "user": {
                "id": self.user.id,
                "username": self.user.username,
                "first_name": self.user.first_name,
                "last_name": self.user.last_name,
                "avatar_url": self.user.avatar_url,
                "role": self.user.role
            } if self.user else None
        }


class Message(db.Model):
    """A chat message in a conversation"""
    __tablename__ = "messages"
    
    id = db.Column(db.String(36), primary_key=True, default=generate_uuid)
    conversation_id = db.Column(db.String(36), db.ForeignKey("conversations.id"), nullable=False, index=True)
    sender_id = db.Column(db.String(36), db.ForeignKey("users.id"), nullable=False, index=True)
    
    # Content
    content = db.Column(db.Text, nullable=False)
    message_type = db.Column(db.String(20), default='text')  # 'text', 'image', 'file', 'system'
    
    # For threaded replies
    reply_to_id = db.Column(db.String(36), db.ForeignKey("messages.id"), nullable=True)
    
    # Moderation
    is_flagged = db.Column(db.Boolean, default=False, index=True)
    flag_reason = db.Column(db.String(200))
    moderated_by = db.Column(db.String(36), db.ForeignKey("users.id"), nullable=True)
    moderated_at = db.Column(db.DateTime)
    moderation_action = db.Column(db.String(20))  # 'approved', 'hidden', 'deleted'
    
    # Status
    is_deleted = db.Column(db.Boolean, default=False)
    is_edited = db.Column(db.Boolean, default=False)
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    edited_at = db.Column(db.DateTime)
    
    # Relationships
    sender = db.relationship("User", foreign_keys=[sender_id])
    moderator = db.relationship("User", foreign_keys=[moderated_by])
    reply_to = db.relationship("Message", remote_side=[id], backref="replies")
    
    def to_dict(self, include_sender=True):
        data = {
            "id": self.id,
            "conversation_id": self.conversation_id,
            "sender_id": self.sender_id,
            "content": self.content if not self.is_deleted else "[Message deleted]",
            "message_type": self.message_type,
            "reply_to_id": self.reply_to_id,
            "is_flagged": self.is_flagged,
            "is_deleted": self.is_deleted,
            "is_edited": self.is_edited,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "edited_at": self.edited_at.isoformat() if self.edited_at else None
        }
        if include_sender and self.sender:
            data["sender"] = {
                "id": self.sender.id,
                "username": self.sender.username,
                "first_name": self.sender.first_name,
                "last_name": self.sender.last_name,
                "avatar_url": self.sender.avatar_url,
                "role": self.sender.role
            }
        return data


class InteractionAnalytics(db.Model):
    """Daily interaction tracking per user"""
    __tablename__ = "interaction_analytics"
    
    id = db.Column(db.String(36), primary_key=True, default=generate_uuid)
    user_id = db.Column(db.String(36), db.ForeignKey("users.id"), nullable=False, index=True)
    date = db.Column(db.Date, nullable=False)
    
    # Counts
    messages_sent = db.Column(db.Integer, default=0)
    messages_received = db.Column(db.Integer, default=0)
    conversations_started = db.Column(db.Integer, default=0)
    
    # Response metrics
    avg_response_time_seconds = db.Column(db.Integer)
    
    __table_args__ = (
        db.UniqueConstraint('user_id', 'date', name='unique_user_analytics_date'),
    )
    
    def to_dict(self):
        return {
            "id": self.id,
            "user_id": self.user_id,
            "date": self.date.isoformat(),
            "messages_sent": self.messages_sent,
            "messages_received": self.messages_received,
            "conversations_started": self.conversations_started,
            "avg_response_time_seconds": self.avg_response_time_seconds
        }


# Helper functions
def get_or_create_direct_conversation(user1_id: str, user2_id: str) -> Conversation:
    """Get existing direct conversation or create new one"""
    # Find existing direct conversation between these two users
    existing = db.session.query(Conversation).join(
        ConversationParticipant
    ).filter(
        Conversation.type == 'direct',
        ConversationParticipant.user_id.in_([user1_id, user2_id])
    ).group_by(
        Conversation.id
    ).having(
        db.func.count(ConversationParticipant.id) == 2
    ).first()
    
    if existing:
        return existing
    
    # Create new conversation
    conv = Conversation(
        type='direct',
        created_by=user1_id
    )
    db.session.add(conv)
    db.session.flush()
    
    # Add participants
    for uid in [user1_id, user2_id]:
        participant = ConversationParticipant(
            conversation_id=conv.id,
            user_id=uid
        )
        db.session.add(participant)
    
    db.session.commit()
    return conv


def record_message_analytics(sender_id: str, recipient_ids: list):
    """Update interaction analytics when message is sent"""
    today = date.today()
    
    # Update sender analytics
    sender_analytics = db.session.query(InteractionAnalytics).filter(
        InteractionAnalytics.user_id == sender_id,
        InteractionAnalytics.date == today
    ).first()
    
    if not sender_analytics:
        sender_analytics = InteractionAnalytics(
            user_id=sender_id, 
            date=today,
            messages_sent=0,
            messages_received=0,
            conversations_started=0
        )
        db.session.add(sender_analytics)
        db.session.flush()  # Ensure defaults are set
    
    # Handle None case
    if sender_analytics.messages_sent is None:
        sender_analytics.messages_sent = 0
    sender_analytics.messages_sent += 1
    
    # Update recipient analytics
    for rid in recipient_ids:
        recipient_analytics = db.session.query(InteractionAnalytics).filter(
            InteractionAnalytics.user_id == rid,
            InteractionAnalytics.date == today
        ).first()
        
        if not recipient_analytics:
            recipient_analytics = InteractionAnalytics(
                user_id=rid, 
                date=today,
                messages_sent=0,
                messages_received=0,
                conversations_started=0
            )
            db.session.add(recipient_analytics)
            db.session.flush()
        
        # Handle None case
        if recipient_analytics.messages_received is None:
            recipient_analytics.messages_received = 0
        recipient_analytics.messages_received += 1
    
    db.session.commit()
