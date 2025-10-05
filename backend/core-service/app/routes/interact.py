"""
Interact Module API Routes

Provides endpoints for:
- Conversations management
- Messages (send, edit, delete)
- Contact discovery
- Moderation
"""
from flask import Blueprint, request, jsonify
from datetime import datetime, date
from uuid import uuid4
from sqlalchemy import or_, and_, func, desc
from app import db
from app.models.user import User
from app.models.interact import (
    Conversation, ConversationParticipant, Message, InteractionAnalytics,
    get_or_create_direct_conversation, record_message_analytics
)
from app.routes.users import require_auth

interact_bp = Blueprint("interact", __name__, url_prefix="/api/interact")


# ============ CONVERSATIONS ============

@interact_bp.route("/conversations", methods=["GET"])
@require_auth
def get_conversations():
    """Get all conversations for the current user"""
    user_id = request.user_id
    
    # Get conversations where user is a participant
    conversations = db.session.query(Conversation).join(
        ConversationParticipant
    ).filter(
        ConversationParticipant.user_id == user_id,
        Conversation.is_active == True
    ).order_by(
        desc(Conversation.last_message_at),
        desc(Conversation.created_at)
    ).all()
    
    result = []
    for conv in conversations:
        conv_data = conv.to_dict(include_participants=True, include_last_message=True)
        
        # Get unread count
        participant = next((p for p in conv.participants if p.user_id == user_id), None)
        if participant:
            unread_count = conv.messages.filter(
                Message.created_at > (participant.last_read_at or datetime.min),
                Message.sender_id != user_id,
                Message.is_deleted == False
            ).count()
            conv_data["unread_count"] = unread_count
        
        result.append(conv_data)
    
    return jsonify(result), 200


@interact_bp.route("/conversations", methods=["POST"])
@require_auth
def create_conversation():
    """Create a new conversation"""
    user_id = request.user_id
    data = request.get_json()
    
    conv_type = data.get("type", "direct")
    participant_ids = data.get("participant_ids", [])
    title = data.get("title")
    
    if not participant_ids:
        return jsonify({"error": "At least one participant required"}), 400
    
    # For direct conversations, check if one already exists
    if conv_type == "direct" and len(participant_ids) == 1:
        existing = get_or_create_direct_conversation(user_id, participant_ids[0])
        return jsonify(existing.to_dict(include_participants=True)), 200
    
    # Create new conversation
    conv = Conversation(
        type=conv_type,
        title=title,
        created_by=user_id
    )
    db.session.add(conv)
    db.session.flush()
    
    # Add creator as participant
    creator_participant = ConversationParticipant(
        conversation_id=conv.id,
        user_id=user_id,
        is_admin=True
    )
    db.session.add(creator_participant)
    
    # Add other participants
    for pid in participant_ids:
        if pid != user_id:
            participant = ConversationParticipant(
                conversation_id=conv.id,
                user_id=pid
            )
            db.session.add(participant)
    
    # Update analytics
    today = date.today()
    analytics = db.session.query(InteractionAnalytics).filter(
        InteractionAnalytics.user_id == user_id,
        InteractionAnalytics.date == today
    ).first()
    if not analytics:
        analytics = InteractionAnalytics(user_id=user_id, date=today)
        db.session.add(analytics)
    analytics.conversations_started += 1
    
    db.session.commit()
    
    return jsonify(conv.to_dict(include_participants=True)), 201


@interact_bp.route("/conversations/<conversation_id>", methods=["GET"])
@require_auth
def get_conversation(conversation_id):
    """Get a specific conversation"""
    user_id = request.user_id
    
    conv = Conversation.query.get(conversation_id)
    if not conv:
        return jsonify({"error": "Conversation not found"}), 404
    
    # Check if user is a participant
    is_participant = any(p.user_id == user_id for p in conv.participants)
    if not is_participant:
        return jsonify({"error": "Access denied"}), 403
    
    return jsonify(conv.to_dict(include_participants=True)), 200


@interact_bp.route("/conversations/<conversation_id>/messages", methods=["GET"])
@require_auth
def get_messages(conversation_id):
    """Get messages for a conversation"""
    user_id = request.user_id
    limit = request.args.get("limit", 50, type=int)
    before = request.args.get("before")  # Cursor for pagination
    
    conv = Conversation.query.get(conversation_id)
    if not conv:
        return jsonify({"error": "Conversation not found"}), 404
    
    # Check if user is a participant
    is_participant = any(p.user_id == user_id for p in conv.participants)
    if not is_participant:
        return jsonify({"error": "Access denied"}), 403
    
    # Get messages
    query = conv.messages.filter(Message.is_deleted == False)
    
    if before:
        before_msg = Message.query.get(before)
        if before_msg:
            query = query.filter(Message.created_at < before_msg.created_at)
    
    messages = query.order_by(desc(Message.created_at)).limit(limit).all()
    
    # Get other participants' last_read_at for read receipts
    other_participants = [p for p in conv.participants if p.user_id != user_id]
    
    # Update last_read_at for this user
    participant = next((p for p in conv.participants if p.user_id == user_id), None)
    if participant:
        participant.last_read_at = datetime.utcnow()
        db.session.commit()
    
    # Return in chronological order
    messages.reverse()
    
    # Build response with read status
    result = []
    for m in messages:
        msg_dict = m.to_dict()
        # A message is "read" if any other participant has read it (last_read_at >= message created_at)
        if m.sender_id == user_id:
            # For sender's messages, check if recipients have read it
            msg_dict["is_read"] = any(
                p.last_read_at and p.last_read_at >= m.created_at 
                for p in other_participants
            )
        else:
            msg_dict["is_read"] = True  # Received messages are always "read" by default
        result.append(msg_dict)
    
    return jsonify(result), 200


# ============ MESSAGES ============

@interact_bp.route("/messages", methods=["POST"])
@require_auth
def send_message():
    """Send a message in a conversation"""
    user_id = request.user_id
    data = request.get_json()
    
    conversation_id = data.get("conversation_id")
    content = data.get("content", "").strip()
    message_type = data.get("message_type", "text")
    reply_to_id = data.get("reply_to_id")
    
    if not conversation_id or not content:
        return jsonify({"error": "Conversation ID and content required"}), 400
    
    conv = Conversation.query.get(conversation_id)
    if not conv:
        return jsonify({"error": "Conversation not found"}), 404
    
    # Check if user is a participant
    is_participant = any(p.user_id == user_id for p in conv.participants)
    if not is_participant:
        return jsonify({"error": "Access denied"}), 403
    
    # Create message (no moderation - free messaging)
    message = Message(
        conversation_id=conversation_id,
        sender_id=user_id,
        content=content,
        message_type=message_type,
        reply_to_id=reply_to_id
    )
    db.session.add(message)
    
    # Update conversation last_message_at
    conv.last_message_at = datetime.utcnow()
    
    # Update analytics
    recipient_ids = [p.user_id for p in conv.participants if p.user_id != user_id]
    record_message_analytics(user_id, recipient_ids)
    
    db.session.commit()
    
    return jsonify(message.to_dict()), 201


@interact_bp.route("/messages/<message_id>", methods=["PUT"])
@require_auth
def edit_message(message_id):
    """Edit a message"""
    user_id = request.user_id
    data = request.get_json()
    
    message = Message.query.get(message_id)
    if not message:
        return jsonify({"error": "Message not found"}), 404
    
    if message.sender_id != user_id:
        return jsonify({"error": "Can only edit own messages"}), 403
    
    new_content = data.get("content", "").strip()
    if not new_content:
        return jsonify({"error": "Content required"}), 400
    
    message.content = new_content
    message.is_edited = True
    message.edited_at = datetime.utcnow()
    message.is_flagged = check_profanity(new_content)
    
    db.session.commit()
    
    return jsonify(message.to_dict()), 200


@interact_bp.route("/messages/<message_id>", methods=["DELETE"])
@require_auth
def delete_message(message_id):
    """Delete a message (soft delete)"""
    user_id = request.user_id
    
    message = Message.query.get(message_id)
    if not message:
        return jsonify({"error": "Message not found"}), 404
    
    # Users can delete their own messages, admins can delete any
    if message.sender_id != user_id:
        # Check if user is admin of this conversation
        conv = message.conversation
        is_admin = any(p.user_id == user_id and p.is_admin for p in conv.participants)
        if not is_admin:
            return jsonify({"error": "Can only delete own messages"}), 403
    
    message.is_deleted = True
    db.session.commit()
    
    return jsonify({"success": True}), 200


# ============ CONTACTS ============

@interact_bp.route("/contacts", methods=["GET"])
@require_auth
def get_contacts():
    """Get available contacts for the current user"""
    user_id = request.user_id
    role_filter = request.args.get("role")  # 'student', 'teacher', 'parent'
    search = request.args.get("search", "")
    
    # Get current user
    current_user = User.query.get(user_id)
    if not current_user:
        return jsonify({"error": "User not found"}), 404
    
    # Build contact query based on user's role
    query = User.query.filter(
        User.id != user_id,
        User.is_active == True
    )
    
    # Filter by role
    if role_filter:
        query = query.filter(User.role == role_filter)
    else:
        # Default: show appropriate contacts based on user's role
        if current_user.role == 'student':
            # Students can see other students and teachers
            query = query.filter(User.role.in_(['student', 'teacher']))
        elif current_user.role == 'teacher':
            # Teachers can see students, other teachers, and parents
            query = query.filter(User.role.in_(['student', 'teacher', 'parent']))
        elif current_user.role == 'parent':
            # Parents can see teachers
            query = query.filter(User.role == 'teacher')
    
    # Search filter
    if search:
        search_term = f"%{search}%"
        query = query.filter(or_(
            User.username.ilike(search_term),
            User.first_name.ilike(search_term),
            User.last_name.ilike(search_term),
            User.email.ilike(search_term)
        ))
    
    # Limit results
    contacts = query.order_by(User.first_name, User.last_name).limit(50).all()
    
    return jsonify([
        {
            "id": u.id,
            "username": u.username,
            "first_name": u.first_name,
            "last_name": u.last_name,
            "email": u.email,
            "role": u.role,
            "avatar_url": u.avatar_url
        }
        for u in contacts
    ]), 200


@interact_bp.route("/contacts/<contact_id>/profile", methods=["GET"])
@require_auth
def get_contact_profile(contact_id):
    """Get detailed profile for a contact"""
    user = User.query.get(contact_id)
    if not user:
        return jsonify({"error": "Contact not found"}), 404
    
    return jsonify({
        "id": user.id,
        "username": user.username,
        "first_name": user.first_name,
        "last_name": user.last_name,
        "email": user.email,
        "role": user.role,
        "avatar_url": user.avatar_url,
        "bio": user.bio,
        "created_at": user.created_at.isoformat() if user.created_at else None
    }), 200


# ============ MODERATION ============

@interact_bp.route("/messages/<message_id>/flag", methods=["POST"])
@require_auth
def flag_message(message_id):
    """Flag a message for moderation"""
    user_id = request.user_id
    data = request.get_json()
    
    message = Message.query.get(message_id)
    if not message:
        return jsonify({"error": "Message not found"}), 404
    
    message.is_flagged = True
    message.flag_reason = data.get("reason", "Reported by user")
    db.session.commit()
    
    return jsonify({"success": True}), 200


@interact_bp.route("/messages/<message_id>/moderate", methods=["POST"])
@require_auth
def moderate_message(message_id):
    """Teacher/admin moderation action on a message"""
    user_id = request.user_id
    data = request.get_json()
    
    # Check if user is teacher or admin
    user = User.query.get(user_id)
    if user.role not in ['teacher', 'admin']:
        return jsonify({"error": "Only teachers and admins can moderate"}), 403
    
    message = Message.query.get(message_id)
    if not message:
        return jsonify({"error": "Message not found"}), 404
    
    action = data.get("action")  # 'approved', 'hidden', 'deleted'
    if action not in ['approved', 'hidden', 'deleted']:
        return jsonify({"error": "Invalid action"}), 400
    
    message.moderated_by = user_id
    message.moderated_at = datetime.utcnow()
    message.moderation_action = action
    
    if action == 'approved':
        message.is_flagged = False
    elif action == 'deleted':
        message.is_deleted = True
    
    db.session.commit()
    
    return jsonify({"success": True}), 200


@interact_bp.route("/moderation/queue", methods=["GET"])
@require_auth
def get_moderation_queue():
    """Get flagged messages for moderation (teachers/admins only)"""
    user_id = request.user_id
    
    user = User.query.get(user_id)
    if user.role not in ['teacher', 'admin']:
        return jsonify({"error": "Access denied"}), 403
    
    flagged = Message.query.filter(
        Message.is_flagged == True,
        Message.moderated_at == None,
        Message.is_deleted == False
    ).order_by(desc(Message.created_at)).limit(50).all()
    
    return jsonify([m.to_dict() for m in flagged]), 200


# ============ ANALYTICS ============

@interact_bp.route("/analytics", methods=["GET"])
@require_auth
def get_my_analytics():
    """Get interaction analytics for current user"""
    user_id = request.user_id
    days = request.args.get("days", 7, type=int)
    
    from datetime import timedelta
    start_date = date.today() - timedelta(days=days)
    
    analytics = InteractionAnalytics.query.filter(
        InteractionAnalytics.user_id == user_id,
        InteractionAnalytics.date >= start_date
    ).order_by(InteractionAnalytics.date).all()
    
    total_sent = sum(a.messages_sent for a in analytics)
    total_received = sum(a.messages_received for a in analytics)
    
    return jsonify({
        "total_messages_sent": total_sent,
        "total_messages_received": total_received,
        "days": [a.to_dict() for a in analytics]
    }), 200


# ============ HELPERS ============

def check_profanity(content: str) -> bool:
    """Simple profanity check - replace with proper filter in production"""
    # Basic list - in production use a proper library like better-profanity
    bad_words = ['badword1', 'badword2']  # Add real words
    content_lower = content.lower()
    return any(word in content_lower for word in bad_words)
