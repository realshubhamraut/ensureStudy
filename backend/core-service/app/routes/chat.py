"""
Chat History API Routes
Server-side persistence for AI tutor conversations
"""
from flask import Blueprint, request, jsonify
from datetime import datetime
from app import db
from app.models.chat import ChatConversation, ChatMessage, ChatSource
from app.routes.assignment import token_required

chat_bp = Blueprint('chat', __name__, url_prefix='/api/chat')


@chat_bp.route('/conversations', methods=['GET'])
@token_required
def list_conversations():
    """Get all conversations for the current user"""
    user_id = request.current_user.id
    
    # Optional filters
    pinned_only = request.args.get('pinned', 'false').lower() == 'true'
    limit = min(int(request.args.get('limit', 50)), 100)
    
    query = ChatConversation.query.filter_by(user_id=user_id)
    
    if pinned_only:
        query = query.filter_by(pinned=True)
    
    # Order by pinned first, then by updated_at desc
    conversations = query.order_by(
        ChatConversation.pinned.desc(),
        ChatConversation.updated_at.desc()
    ).limit(limit).all()
    
    return jsonify({
        'success': True,
        'conversations': [c.to_dict() for c in conversations]
    })


@chat_bp.route('/conversations', methods=['POST'])
@token_required
def create_conversation():
    """Create a new conversation"""
    user_id = request.current_user.id
    data = request.get_json() or {}
    
    conversation = ChatConversation(
        user_id=user_id,
        title=data.get('title', 'New Conversation'),
        subject=data.get('subject'),
        classroom_id=data.get('classroom_id'),
        pinned=data.get('pinned', False)
    )
    
    db.session.add(conversation)
    db.session.commit()
    
    return jsonify({
        'success': True,
        'conversation': conversation.to_dict()
    }), 201


@chat_bp.route('/conversations/<conversation_id>', methods=['GET'])
@token_required
def get_conversation(conversation_id):
    """Get a single conversation with all messages"""
    user_id = request.current_user.id
    
    conversation = ChatConversation.query.filter_by(
        id=conversation_id,
        user_id=user_id
    ).first()
    
    if not conversation:
        return jsonify({'success': False, 'error': 'Conversation not found'}), 404
    
    return jsonify({
        'success': True,
        'conversation': conversation.to_dict(include_messages=True)
    })


@chat_bp.route('/conversations/<conversation_id>', methods=['PUT'])
@token_required
def update_conversation(conversation_id):
    """Update conversation (title, pinned status)"""
    user_id = request.current_user.id
    data = request.get_json() or {}
    
    conversation = ChatConversation.query.filter_by(
        id=conversation_id,
        user_id=user_id
    ).first()
    
    if not conversation:
        return jsonify({'success': False, 'error': 'Conversation not found'}), 404
    
    # Update allowed fields
    if 'title' in data:
        conversation.title = data['title'][:200]
    if 'pinned' in data:
        conversation.pinned = bool(data['pinned'])
    if 'subject' in data:
        conversation.subject = data['subject']
    
    conversation.updated_at = datetime.utcnow()
    db.session.commit()
    
    return jsonify({
        'success': True,
        'conversation': conversation.to_dict()
    })


@chat_bp.route('/conversations/<conversation_id>', methods=['DELETE'])
@token_required
def delete_conversation(conversation_id):
    """Delete a conversation and all its messages"""
    user_id = request.current_user.id
    
    conversation = ChatConversation.query.filter_by(
        id=conversation_id,
        user_id=user_id
    ).first()
    
    if not conversation:
        return jsonify({'success': False, 'error': 'Conversation not found'}), 404
    
    db.session.delete(conversation)
    db.session.commit()
    
    return jsonify({'success': True, 'message': 'Conversation deleted'})


@chat_bp.route('/conversations/<conversation_id>/messages', methods=['POST'])
@token_required
def add_message(conversation_id):
    """Add a message to a conversation (auto-creates conversation if needed)"""
    user_id = request.current_user.id
    data = request.get_json()
    
    if not data or 'type' not in data or 'content' not in data:
        return jsonify({'success': False, 'error': 'type and content are required'}), 400
    
    conversation = ChatConversation.query.filter_by(
        id=conversation_id,
        user_id=user_id
    ).first()
    
    # Auto-create conversation if it doesn't exist
    if not conversation:
        conversation = ChatConversation(
            id=conversation_id,
            user_id=user_id,
            title=data.get('content', 'New Conversation')[:50] + ('...' if len(data.get('content', '')) > 50 else ''),
            subject=data.get('subject'),
            classroom_id=data.get('classroom_id')
        )
        db.session.add(conversation)
        db.session.flush()  # Get the ID before committing
    
    # Create message
    message = ChatMessage(
        conversation_id=conversation_id,
        type=data['type'],
        content=data['content'],
        response_json=data.get('response'),
        subject=data.get('subject'),
        image_url=data.get('image_url')
    )
    
    # Update conversation title if it's the first user message
    if conversation.messages.count() == 0 and data['type'] == 'user':
        conversation.title = data['content'][:50] + ('...' if len(data['content']) > 50 else '')
    
    conversation.updated_at = datetime.utcnow()
    
    db.session.add(message)
    db.session.commit()
    
    return jsonify({
        'success': True,
        'message': message.to_dict(),
        'conversation_created': conversation is not None
    }), 201


@chat_bp.route('/conversations/<conversation_id>/sources', methods=['POST'])
@token_required
def add_sources(conversation_id):
    """Add sources to a conversation (auto-creates conversation if needed)"""
    user_id = request.current_user.id
    data = request.get_json()
    
    if not data or 'sources' not in data:
        return jsonify({'success': False, 'error': 'sources array is required'}), 400
    
    conversation = ChatConversation.query.filter_by(
        id=conversation_id,
        user_id=user_id
    ).first()
    
    # Auto-create conversation if it doesn't exist
    if not conversation:
        conversation = ChatConversation(
            id=conversation_id,
            user_id=user_id,
            title='New Conversation'
        )
        db.session.add(conversation)
        db.session.flush()
    
    # Clear existing sources and add new ones
    ChatSource.query.filter_by(conversation_id=conversation_id).delete()
    
    added_sources = []
    for src in data['sources']:
        source = ChatSource(
            conversation_id=conversation_id,
            source_type=src.get('type', 'article'),
            title=src.get('title', 'Untitled'),
            url=src.get('url'),
            thumbnail_url=src.get('thumbnailUrl'),
            relevance=src.get('relevance'),
            snippet=src.get('snippet'),
            source_name=src.get('source'),
            source_metadata={
                'embedUrl': src.get('embedUrl'),
                'duration': src.get('duration'),
                'cachedContent': src.get('cachedContent'),
                'mermaidCode': src.get('mermaidCode')
            }
        )
        db.session.add(source)
        added_sources.append(source)
    
    db.session.commit()
    
    return jsonify({
        'success': True,
        'sources': [s.to_dict() for s in added_sources]
    }), 201
