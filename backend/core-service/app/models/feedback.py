"""
Feedback Models for Learning Agent System

Stores interaction feedback for continuous agent improvement.
"""
from datetime import datetime
import uuid
import enum

from app import db


def generate_uuid():
    return str(uuid.uuid4())


class FeedbackType(enum.Enum):
    """Types of feedback"""
    THUMBS = "thumbs"        # 👍/👎
    RATING = "rating"        # 1-5 stars
    CORRECTION = "correction"  # Text correction
    REPORT = "report"        # Issue report


class AgentInteraction(db.Model):
    """Stores all agent interactions for learning"""
    __tablename__ = "agent_interactions"

    id = db.Column(db.String(36), primary_key=True, default=generate_uuid)
    agent_type = db.Column(db.String(50), nullable=False, index=True)  # 'tutor', 'curriculum', etc.
    session_id = db.Column(db.String(36), nullable=False, index=True)
    user_id = db.Column(db.String(36), db.ForeignKey("users.id"), nullable=True)
    
    # Interaction data
    query = db.Column(db.Text, nullable=False)
    response = db.Column(db.Text, nullable=False)
    response_json = db.Column(db.JSON, default={})  # sources, confidence, topic, etc.
    
    # Context
    topic = db.Column(db.String(255), nullable=True, index=True)
    classroom_id = db.Column(db.String(36), nullable=True)
    
    # Metrics
    response_time_ms = db.Column(db.Integer, nullable=True)
    token_count = db.Column(db.Integer, nullable=True)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    
    # Relationships
    feedback = db.relationship("InteractionFeedback", back_populates="interaction", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<AgentInteraction {self.agent_type}:{self.id}>"


class InteractionFeedback(db.Model):
    """User feedback on agent interactions"""
    __tablename__ = "interaction_feedback"

    id = db.Column(db.String(36), primary_key=True, default=generate_uuid)
    interaction_id = db.Column(db.String(36), db.ForeignKey("agent_interactions.id"), nullable=False)
    user_id = db.Column(db.String(36), db.ForeignKey("users.id"), nullable=True)
    
    # Feedback data
    feedback_type = db.Column(db.Enum(FeedbackType), nullable=False)
    feedback_value = db.Column(db.Integer, nullable=False)  # 1=positive, -1=negative, 1-5 for ratings
    feedback_text = db.Column(db.Text, nullable=True)  # Optional text correction/comment
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    interaction = db.relationship("AgentInteraction", back_populates="feedback")

    def __repr__(self):
        return f"<InteractionFeedback {self.feedback_type.value}:{self.feedback_value}>"


class LearningExample(db.Model):
    """Curated examples for few-shot learning"""
    __tablename__ = "learning_examples"

    id = db.Column(db.String(36), primary_key=True, default=generate_uuid)
    agent_type = db.Column(db.String(50), nullable=False, index=True)
    
    # Example content
    topic = db.Column(db.String(255), nullable=False, index=True)
    query = db.Column(db.Text, nullable=False)
    good_response = db.Column(db.Text, nullable=False)  # Positive example
    bad_response = db.Column(db.Text, nullable=True)    # Optional negative example
    
    # Example metadata
    source = db.Column(db.String(50), nullable=False)  # 'user_feedback', 'expert', 'a_b_test'
    weight = db.Column(db.Float, default=1.0)  # Importance weight for sampling
    feedback_score = db.Column(db.Float, nullable=True)  # Aggregate feedback score
    use_count = db.Column(db.Integer, default=0)  # How often used in prompts
    
    # Status
    is_active = db.Column(db.String(10), default="active")  # 'active', 'retired'
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<LearningExample {self.topic}:{self.id}>"


class AgentPerformanceMetrics(db.Model):
    """Aggregated performance metrics for monitoring"""
    __tablename__ = "agent_performance_metrics"

    id = db.Column(db.String(36), primary_key=True, default=generate_uuid)
    agent_type = db.Column(db.String(50), nullable=False, index=True)
    
    # Time period
    period_start = db.Column(db.DateTime, nullable=False, index=True)
    period_end = db.Column(db.DateTime, nullable=False)
    
    # Metrics
    total_interactions = db.Column(db.Integer, default=0)
    positive_feedback_count = db.Column(db.Integer, default=0)
    negative_feedback_count = db.Column(db.Integer, default=0)
    avg_response_time_ms = db.Column(db.Float, nullable=True)
    satisfaction_rate = db.Column(db.Float, nullable=True)  # positive / total rated
    
    # Topic breakdown
    topic_metrics_json = db.Column(db.JSON, default={})
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<AgentPerformanceMetrics {self.agent_type}:{self.period_start}>"
