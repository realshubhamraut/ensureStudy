"""
Progress Tracking Models - Study Streaks and Topic Interactions

New tables required for the Progress page to display real data.
"""
from datetime import datetime, date
from app import db
import uuid


def generate_uuid():
    return str(uuid.uuid4())


class StudyStreak(db.Model):
    """Track daily study activity for streak calculation"""
    __tablename__ = "study_streaks"
    
    id = db.Column(db.String(36), primary_key=True, default=generate_uuid)
    user_id = db.Column(db.String(36), db.ForeignKey("users.id"), nullable=False, index=True)
    
    # The study day
    study_date = db.Column(db.Date, nullable=False)
    
    # Activity counts
    session_count = db.Column(db.Integer, default=0)  # AI chat sessions
    questions_asked = db.Column(db.Integer, default=0)  # Total questions
    assessments_taken = db.Column(db.Integer, default=0)  # Assessments completed
    notes_viewed = db.Column(db.Integer, default=0)  # Notes opened
    
    # Time tracking
    total_minutes = db.Column(db.Integer, default=0)  # Total study time
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    __table_args__ = (
        db.UniqueConstraint('user_id', 'study_date', name='unique_user_study_date'),
    )
    
    def to_dict(self):
        return {
            "id": self.id,
            "user_id": self.user_id,
            "study_date": self.study_date.isoformat(),
            "session_count": self.session_count,
            "questions_asked": self.questions_asked,
            "assessments_taken": self.assessments_taken,
            "notes_viewed": self.notes_viewed,
            "total_minutes": self.total_minutes
        }


class TopicInteraction(db.Model):
    """Track when student interacts with a topic"""
    __tablename__ = "topic_interactions"
    
    id = db.Column(db.String(36), primary_key=True, default=generate_uuid)
    user_id = db.Column(db.String(36), db.ForeignKey("users.id"), nullable=False, index=True)
    
    # Can link to either subtopic (preferred) or topic
    subtopic_id = db.Column(db.String(36), db.ForeignKey("subtopics.id"), nullable=True, index=True)
    topic_id = db.Column(db.String(36), db.ForeignKey("topics.id"), nullable=True, index=True)
    
    # Type of interaction
    interaction_type = db.Column(db.String(30), nullable=False)  # 'chat', 'assessment', 'notes_view', 'video_watch'
    
    # Optional: source reference
    source_id = db.Column(db.String(36), nullable=True)  # session_turn_id, assessment_result_id, etc.
    source_type = db.Column(db.String(30), nullable=True)  # 'session_turn', 'assessment_result'
    
    # Detected topic info (if from AI classification)
    detected_topic_name = db.Column(db.String(200), nullable=True)  # Raw detected name
    confidence_score = db.Column(db.Float, nullable=True)  # ML confidence 0-1
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    
    # Relationships
    subtopic = db.relationship("Subtopic", backref="interactions")
    topic = db.relationship("Topic", backref="interactions")
    
    def to_dict(self):
        return {
            "id": self.id,
            "user_id": self.user_id,
            "subtopic_id": self.subtopic_id,
            "topic_id": self.topic_id,
            "interaction_type": self.interaction_type,
            "detected_topic_name": self.detected_topic_name,
            "confidence_score": self.confidence_score,
            "created_at": self.created_at.isoformat()
        }


# Helper functions for progress calculation
def calculate_study_streak(user_id: str) -> int:
    """Calculate consecutive study days streak for a user"""
    from sqlalchemy import desc
    
    today = date.today()
    streak = 0
    current_date = today
    
    while True:
        has_activity = db.session.query(StudyStreak).filter(
            StudyStreak.user_id == user_id,
            StudyStreak.study_date == current_date,
            db.or_(
                StudyStreak.questions_asked > 0,
                StudyStreak.assessments_taken > 0,
                StudyStreak.notes_viewed > 0
            )
        ).first()
        
        if has_activity:
            streak += 1
            current_date = current_date.replace(day=current_date.day - 1) if current_date.day > 1 else None
            if current_date is None:
                break
        else:
            # Allow one day gap for today (might not have studied today yet)
            if current_date == today and streak == 0:
                current_date = current_date.replace(day=current_date.day - 1) if current_date.day > 1 else None
                continue
            break
    
    return streak


def record_study_activity(user_id: str, activity_type: str, minutes: int = 0):
    """Record study activity for the day"""
    today = date.today()
    
    streak = db.session.query(StudyStreak).filter(
        StudyStreak.user_id == user_id,
        StudyStreak.study_date == today
    ).first()
    
    if not streak:
        streak = StudyStreak(user_id=user_id, study_date=today)
        db.session.add(streak)
    
    if activity_type == 'chat':
        streak.questions_asked += 1
    elif activity_type == 'session':
        streak.session_count += 1
    elif activity_type == 'assessment':
        streak.assessments_taken += 1
    elif activity_type == 'notes':
        streak.notes_viewed += 1
    
    if minutes > 0:
        streak.total_minutes += minutes
    
    db.session.commit()
    return streak
