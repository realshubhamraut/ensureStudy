"""
Feedback API Routes

Endpoints for collecting and managing agent interaction feedback
for the Type 5 Learning Agent system.
"""
from flask import Blueprint, request, jsonify, g
from sqlalchemy import func, desc
from datetime import datetime, timedelta
import uuid

from app import db
from app.models.feedback import (
    AgentInteraction, InteractionFeedback, LearningExample,
    AgentPerformanceMetrics, FeedbackType
)
from app.routes.assignment import token_required

feedback_bp = Blueprint('feedback', __name__)


# ============================================================================
# Interaction Logging
# ============================================================================

@feedback_bp.route('/api/feedback/interactions', methods=['POST'])
@token_required
def log_interaction():
    """
    Log an agent interaction for learning.
    Called by the AI service after generating a response.
    """
    data = request.get_json()
    
    if not data or not data.get('agent_type') or not data.get('query') or not data.get('response'):
        return jsonify({'error': 'Missing required fields: agent_type, query, response'}), 400
    
    try:
        interaction = AgentInteraction(
            agent_type=data['agent_type'],
            session_id=data.get('session_id') or str(uuid.uuid4()),
            user_id=g.current_user.id if hasattr(g, 'current_user') else None,
            query=data['query'],
            response=data['response'],
            response_json=data.get('metadata', {}),
            topic=data.get('topic'),
            classroom_id=data.get('classroom_id'),
            response_time_ms=data.get('response_time_ms'),
            token_count=data.get('token_count')
        )
        
        db.session.add(interaction)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'interaction_id': str(interaction.id)
        }), 201
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500


# ============================================================================
# Feedback Collection
# ============================================================================

@feedback_bp.route('/api/feedback/submit', methods=['POST'])
@token_required
def submit_feedback():
    """
    Submit feedback for an agent interaction.
    
    Body:
        interaction_id: UUID of the interaction
        feedback_type: 'thumbs' | 'rating' | 'correction' | 'report'
        feedback_value: int (1/-1 for thumbs, 1-5 for rating)
        feedback_text: optional text for corrections/reports
    """
    data = request.get_json()
    
    if not data or not data.get('interaction_id') or not data.get('feedback_type'):
        return jsonify({'error': 'Missing required fields'}), 400
    
    try:
        # Validate interaction exists
        interaction = AgentInteraction.query.get(data['interaction_id'])
        if not interaction:
            return jsonify({'error': 'Interaction not found'}), 404
        
        # Parse feedback type
        try:
            feedback_type = FeedbackType(data['feedback_type'])
        except ValueError:
            return jsonify({'error': f"Invalid feedback_type. Must be one of: {[t.value for t in FeedbackType]}"}), 400
        
        # Validate feedback value
        feedback_value = data.get('feedback_value', 0)
        if feedback_type == FeedbackType.THUMBS and feedback_value not in [1, -1]:
            return jsonify({'error': 'Thumbs feedback_value must be 1 or -1'}), 400
        if feedback_type == FeedbackType.RATING and not (1 <= feedback_value <= 5):
            return jsonify({'error': 'Rating feedback_value must be 1-5'}), 400
        
        feedback = InteractionFeedback(
            interaction_id=interaction.id,
            user_id=g.current_user.id if hasattr(g, 'current_user') else None,
            feedback_type=feedback_type,
            feedback_value=feedback_value,
            feedback_text=data.get('feedback_text')
        )
        
        db.session.add(feedback)
        db.session.commit()
        
        # If positive feedback, consider adding as learning example
        if feedback_type == FeedbackType.THUMBS and feedback_value == 1:
            _maybe_create_learning_example(interaction)
        
        return jsonify({
            'success': True,
            'feedback_id': str(feedback.id)
        }), 201
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500


def _maybe_create_learning_example(interaction: AgentInteraction):
    """
    Automatically create a learning example from highly-rated interactions.
    Only creates if this interaction has multiple positive feedbacks.
    """
    positive_count = InteractionFeedback.query.filter(
        InteractionFeedback.interaction_id == interaction.id,
        InteractionFeedback.feedback_value > 0
    ).count()
    
    # Require at least 2 positive feedbacks to become an example
    if positive_count >= 2:
        existing = LearningExample.query.filter(
            LearningExample.query == interaction.query,
            LearningExample.agent_type == interaction.agent_type
        ).first()
        
        if not existing:
            example = LearningExample(
                agent_type=interaction.agent_type,
                topic=interaction.topic or 'general',
                query=interaction.query,
                good_response=interaction.response,
                source='user_feedback',
                feedback_score=positive_count
            )
            db.session.add(example)
            db.session.commit()


# ============================================================================
# Learning Examples
# ============================================================================

@feedback_bp.route('/api/feedback/examples', methods=['GET'])
def get_learning_examples():
    """
    Get learning examples for few-shot prompting.
    
    Query params:
        agent_type: required
        topic: optional filter
        limit: max examples (default 5)
    """
    agent_type = request.args.get('agent_type')
    if not agent_type:
        return jsonify({'error': 'agent_type is required'}), 400
    
    topic = request.args.get('topic')
    limit = int(request.args.get('limit', 5))
    
    query = LearningExample.query.filter(
        LearningExample.agent_type == agent_type,
        LearningExample.is_active == 'active'
    )
    
    if topic:
        # Fuzzy match on topic
        query = query.filter(LearningExample.topic.ilike(f'%{topic}%'))
    
    # Order by weight and feedback score
    examples = query.order_by(
        desc(LearningExample.weight),
        desc(LearningExample.feedback_score)
    ).limit(limit).all()
    
    return jsonify({
        'examples': [{
            'id': str(ex.id),
            'topic': ex.topic,
            'query': ex.query,
            'good_response': ex.good_response,
            'weight': ex.weight
        } for ex in examples]
    })


@feedback_bp.route('/api/feedback/examples', methods=['POST'])
@token_required
def create_learning_example():
    """
    Manually create a learning example (admin/expert).
    """
    data = request.get_json()
    
    required = ['agent_type', 'topic', 'query', 'good_response']
    if not all(data.get(f) for f in required):
        return jsonify({'error': f'Missing required fields: {required}'}), 400
    
    try:
        example = LearningExample(
            agent_type=data['agent_type'],
            topic=data['topic'],
            query=data['query'],
            good_response=data['good_response'],
            bad_response=data.get('bad_response'),
            source=data.get('source', 'expert'),
            weight=float(data.get('weight', 1.0))
        )
        
        db.session.add(example)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'example_id': str(example.id)
        }), 201
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500


# ============================================================================
# Performance Metrics
# ============================================================================

@feedback_bp.route('/api/feedback/stats/<agent_type>', methods=['GET'])
def get_agent_stats(agent_type: str):
    """
    Get feedback statistics for an agent.
    
    Query params:
        days: number of days to look back (default 7)
    """
    days = int(request.args.get('days', 7))
    start_date = datetime.utcnow() - timedelta(days=days)
    
    # Total interactions
    total_interactions = AgentInteraction.query.filter(
        AgentInteraction.agent_type == agent_type,
        AgentInteraction.created_at >= start_date
    ).count()
    
    # Feedback counts
    feedback_query = db.session.query(
        InteractionFeedback.feedback_value,
        func.count(InteractionFeedback.id)
    ).join(AgentInteraction).filter(
        AgentInteraction.agent_type == agent_type,
        InteractionFeedback.created_at >= start_date,
        InteractionFeedback.feedback_type == FeedbackType.THUMBS
    ).group_by(InteractionFeedback.feedback_value).all()
    
    positive = sum(count for value, count in feedback_query if value > 0)
    negative = sum(count for value, count in feedback_query if value < 0)
    
    # Topic breakdown
    topic_stats = db.session.query(
        AgentInteraction.topic,
        func.count(AgentInteraction.id),
        func.avg(InteractionFeedback.feedback_value)
    ).outerjoin(InteractionFeedback).filter(
        AgentInteraction.agent_type == agent_type,
        AgentInteraction.created_at >= start_date,
        AgentInteraction.topic.isnot(None)
    ).group_by(AgentInteraction.topic).limit(10).all()
    
    return jsonify({
        'agent_type': agent_type,
        'period_days': days,
        'total_interactions': total_interactions,
        'feedback': {
            'positive': positive,
            'negative': negative,
            'satisfaction_rate': positive / (positive + negative) if (positive + negative) > 0 else None
        },
        'top_topics': [{
            'topic': topic,
            'count': count,
            'avg_feedback': float(avg) if avg else None
        } for topic, count, avg in topic_stats if topic]
    })


# ============================================================================
# Learning Examples Count (for use by AI service)
# ============================================================================

@feedback_bp.route('/api/feedback/examples/count', methods=['GET'])
def count_learning_examples():
    """Count active learning examples by agent type."""
    counts = db.session.query(
        LearningExample.agent_type,
        func.count(LearningExample.id)
    ).filter(
        LearningExample.is_active == 'active'
    ).group_by(LearningExample.agent_type).all()
    
    return jsonify({
        'counts': {agent: count for agent, count in counts}
    })
