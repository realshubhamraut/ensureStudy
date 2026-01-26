"""
Spaced Repetition Service - Optimized review scheduling based on forgetting curves

Implements:
- SM-2 algorithm for optimal review intervals
- Topic review scheduling based on mastery decay
- Learning style adaptation (visual, auditory, kinesthetic, reading/writing)
- Integration with Research Agent for resource suggestions

References:
- SuperMemo SM-2: https://www.supermemo.com/en/archives1990-2015/english/ol/sm2
"""
import os
import math
import logging
from typing import Optional, Dict, List, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict, field
from enum import Enum

logger = logging.getLogger(__name__)


# ============================================================================
# Enums and Types
# ============================================================================

class LearningStyle(Enum):
    """VARK learning style model"""
    VISUAL = "visual"           # Diagrams, charts, videos
    AUDITORY = "auditory"       # Lectures, podcasts, discussions
    READING = "reading"         # Text, notes, articles
    KINESTHETIC = "kinesthetic" # Practice, experiments, hands-on


class ReviewQuality(Enum):
    """SM-2 quality of response (0-5)"""
    BLACKOUT = 0      # Complete failure
    INCORRECT = 1     # Incorrect but remembered something
    HARD = 2          # Correct but very difficult
    MEDIUM = 3        # Correct with some hesitation
    EASY = 4          # Correct with some thought
    PERFECT = 5       # Perfect recall


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class ReviewItem:
    """A single item in the spaced repetition system"""
    topic_id: str
    topic_name: str
    easiness_factor: float = 2.5   # EF from SM-2 (min 1.3)
    interval: int = 1              # Days until next review
    repetitions: int = 0           # Successful repetitions in a row
    next_review: str = ""          # ISO date string
    last_review: str = ""          # ISO date string
    mastery: float = 0.0           # 0.0 to 1.0
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class LearningProfile:
    """Student's learning profile for personalization"""
    user_id: str
    primary_style: LearningStyle = LearningStyle.VISUAL
    secondary_style: Optional[LearningStyle] = None
    preferred_session_minutes: int = 30
    best_study_time: str = "morning"  # morning, afternoon, evening, night
    retention_strength: float = 1.0   # Multiplier for interval (>1 = good memory)
    topics_per_session: int = 3
    review_items: Dict[str, ReviewItem] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            **{k: v for k, v in asdict(self).items() if k != "review_items"},
            "primary_style": self.primary_style.value,
            "secondary_style": self.secondary_style.value if self.secondary_style else None,
            "review_items": {k: v.to_dict() for k, v in self.review_items.items()}
        }


@dataclass 
class ResourceSuggestion:
    """Learning resource suggestion"""
    topic: str
    resource_type: str  # video, article, pdf, exercise, quiz
    title: str
    url: str
    description: str
    duration_min: int
    difficulty: str
    learning_styles: List[str]
    relevance_score: float


# ============================================================================
# Spaced Repetition Service
# ============================================================================

class SpacedRepetitionService:
    """
    Implements SM-2 spaced repetition algorithm with learning style adaptation.
    
    The SM-2 algorithm calculates:
    1. Easiness Factor (EF): How easy the topic is for the student
    2. Interval: Days until next review
    3. Repetitions: Count of successful reviews
    
    Formula:
    - EF' = EF + (0.1 - (5-q) * (0.08 + (5-q) * 0.02))
    - Where q is quality of response (0-5)
    - EF minimum is 1.3
    
    Interval calculation:
    - First review: 1 day
    - Second review: 6 days  
    - Subsequent: interval * EF
    """
    
    def __init__(self):
        self._profiles: Dict[str, LearningProfile] = {}
        
        # Resource type preferences by learning style
        self.style_resources = {
            LearningStyle.VISUAL: ["video", "diagram", "infographic", "animation"],
            LearningStyle.AUDITORY: ["podcast", "lecture", "audiobook", "discussion"],
            LearningStyle.READING: ["article", "pdf", "textbook", "notes"],
            LearningStyle.KINESTHETIC: ["exercise", "lab", "project", "simulation"]
        }
    
    # ========================================================================
    # Profile Management
    # ========================================================================
    
    def get_or_create_profile(self, user_id: str) -> LearningProfile:
        """Get or create a learning profile for user"""
        if user_id not in self._profiles:
            self._profiles[user_id] = LearningProfile(user_id=user_id)
        return self._profiles[user_id]
    
    def update_learning_style(
        self, 
        user_id: str, 
        primary: str,
        secondary: Optional[str] = None
    ) -> LearningProfile:
        """Update user's learning style preference"""
        profile = self.get_or_create_profile(user_id)
        profile.primary_style = LearningStyle(primary)
        if secondary:
            profile.secondary_style = LearningStyle(secondary)
        return profile
    
    def update_study_preferences(
        self,
        user_id: str,
        session_minutes: int = None,
        best_time: str = None,
        topics_per_session: int = None
    ) -> LearningProfile:
        """Update study preferences"""
        profile = self.get_or_create_profile(user_id)
        
        if session_minutes:
            profile.preferred_session_minutes = session_minutes
        if best_time:
            profile.best_study_time = best_time
        if topics_per_session:
            profile.topics_per_session = topics_per_session
            
        return profile
    
    # ========================================================================
    # SM-2 Algorithm
    # ========================================================================
    
    def calculate_next_review(
        self,
        item: ReviewItem,
        quality: ReviewQuality
    ) -> ReviewItem:
        """
        Calculate next review date using SM-2 algorithm.
        
        Args:
            item: Current review item state
            quality: Quality of response (0-5)
            
        Returns:
            Updated ReviewItem with new interval and next_review date
        """
        q = quality.value
        
        # Update easiness factor
        ef_change = 0.1 - (5 - q) * (0.08 + (5 - q) * 0.02)
        new_ef = max(1.3, item.easiness_factor + ef_change)
        
        # Calculate interval
        if q < 3:
            # Failed - reset to beginning
            new_interval = 1
            new_reps = 0
        else:
            # Success - increase interval
            if item.repetitions == 0:
                new_interval = 1
            elif item.repetitions == 1:
                new_interval = 6
            else:
                new_interval = round(item.interval * item.easiness_factor)
            new_reps = item.repetitions + 1
        
        # Calculate next review date
        next_date = datetime.now() + timedelta(days=new_interval)
        
        # Update mastery based on quality and repetitions
        if q >= 3:
            # Increase mastery for successful reviews
            mastery_gain = 0.1 * (q - 2)  # 0.1 for q=3, 0.2 for q=4, 0.3 for q=5
            new_mastery = min(1.0, item.mastery + mastery_gain)
        else:
            # Decrease mastery for failed reviews
            new_mastery = max(0.0, item.mastery - 0.2)
        
        return ReviewItem(
            topic_id=item.topic_id,
            topic_name=item.topic_name,
            easiness_factor=new_ef,
            interval=new_interval,
            repetitions=new_reps,
            next_review=next_date.strftime("%Y-%m-%d"),
            last_review=datetime.now().strftime("%Y-%m-%d"),
            mastery=new_mastery
        )
    
    def record_review(
        self,
        user_id: str,
        topic_id: str,
        topic_name: str,
        quality: int
    ) -> ReviewItem:
        """
        Record a review and calculate next review date.
        
        Args:
            user_id: Student ID
            topic_id: Topic being reviewed
            topic_name: Human-readable topic name
            quality: Quality of response (0-5)
            
        Returns:
            Updated ReviewItem
        """
        profile = self.get_or_create_profile(user_id)
        
        # Get or create review item
        if topic_id in profile.review_items:
            item = profile.review_items[topic_id]
        else:
            item = ReviewItem(topic_id=topic_id, topic_name=topic_name)
        
        # Calculate next review
        quality_enum = ReviewQuality(min(5, max(0, quality)))
        updated_item = self.calculate_next_review(item, quality_enum)
        
        # Apply retention strength modifier
        if profile.retention_strength != 1.0:
            updated_item.interval = round(
                updated_item.interval * profile.retention_strength
            )
        
        # Store updated item
        profile.review_items[topic_id] = updated_item
        
        logger.info(
            f"[SPACED-REP] User {user_id[:8]}... Topic '{topic_name}' "
            f"Q={quality} -> Interval={updated_item.interval}d, "
            f"Mastery={updated_item.mastery:.0%}"
        )
        
        return updated_item
    
    # ========================================================================
    # Review Scheduling
    # ========================================================================
    
    def get_due_reviews(
        self, 
        user_id: str,
        limit: int = 10
    ) -> List[ReviewItem]:
        """
        Get topics due for review today or overdue.
        
        Args:
            user_id: Student ID
            limit: Maximum number of items to return
            
        Returns:
            List of ReviewItems sorted by urgency
        """
        profile = self.get_or_create_profile(user_id)
        today = datetime.now().strftime("%Y-%m-%d")
        
        due_items = []
        for item in profile.review_items.values():
            if item.next_review <= today:
                # Calculate urgency (days overdue)
                if item.next_review:
                    try:
                        next_date = datetime.strptime(item.next_review, "%Y-%m-%d")
                        days_overdue = (datetime.now() - next_date).days
                    except ValueError:
                        days_overdue = 0
                else:
                    days_overdue = 0
                
                due_items.append((days_overdue, item))
        
        # Sort by urgency (most overdue first)
        due_items.sort(key=lambda x: x[0], reverse=True)
        
        return [item for _, item in due_items[:limit]]
    
    def get_optimal_study_session(
        self,
        user_id: str,
        available_minutes: int = None
    ) -> Dict[str, Any]:
        """
        Generate an optimal study session based on spaced repetition.
        
        Includes:
        - Topics due for review
        - New topics to learn
        - Resource suggestions based on learning style
        
        Args:
            user_id: Student ID
            available_minutes: Optional time available (uses profile default)
            
        Returns:
            Study session plan
        """
        profile = self.get_or_create_profile(user_id)
        minutes = available_minutes or profile.preferred_session_minutes
        
        # Get due reviews
        due_reviews = self.get_due_reviews(user_id, limit=profile.topics_per_session)
        
        # Build session
        session = {
            "user_id": user_id,
            "learning_style": profile.primary_style.value,
            "total_minutes": minutes,
            "review_topics": [],
            "estimated_review_time": 0,
            "suggestions": []
        }
        
        # Allocate time for reviews (about 5-10 min each)
        review_time_each = min(10, minutes // max(1, len(due_reviews)))
        for item in due_reviews:
            session["review_topics"].append({
                "topic_id": item.topic_id,
                "topic_name": item.topic_name,
                "current_mastery": round(item.mastery * 100),
                "days_since_review": self._days_since_last_review(item),
                "allocated_minutes": review_time_each
            })
            session["estimated_review_time"] += review_time_each
        
        # Get resource suggestions for remaining time
        remaining_time = minutes - session["estimated_review_time"]
        if remaining_time > 0 and due_reviews:
            session["suggestions"] = self._get_style_based_resources(
                topics=[item.topic_name for item in due_reviews],
                style=profile.primary_style,
                max_duration=remaining_time
            )
        
        return session
    
    def _days_since_last_review(self, item: ReviewItem) -> int:
        """Calculate days since last review"""
        if not item.last_review:
            return -1
        try:
            last = datetime.strptime(item.last_review, "%Y-%m-%d")
            return (datetime.now() - last).days
        except ValueError:
            return -1
    
    def _get_style_based_resources(
        self,
        topics: List[str],
        style: LearningStyle,
        max_duration: int
    ) -> List[Dict]:
        """
        Generate resource suggestions based on learning style.
        
        In production, this would call the Research Agent.
        For now, returns template suggestions.
        """
        resource_types = self.style_resources.get(style, ["article"])
        
        suggestions = []
        for topic in topics[:3]:  # Limit to 3 topics
            primary_type = resource_types[0]
            
            suggestions.append({
                "topic": topic,
                "resource_type": primary_type,
                "title": f"{primary_type.title()}: {topic}",
                "description": f"Recommended {primary_type} for reviewing {topic}",
                "estimated_minutes": min(15, max_duration // len(topics)),
                "learning_style": style.value,
                "search_query": f"{topic} {primary_type} tutorial educational"
            })
        
        return suggestions
    
    # ========================================================================
    # Learning Style Quiz
    # ========================================================================
    
    def get_learning_style_quiz(self) -> Dict[str, Any]:
        """
        Returns a VARK-style quiz to determine learning preference.
        
        The quiz asks about preferences in different scenarios.
        """
        return {
            "title": "Discover Your Learning Style",
            "description": "Answer these questions to find your preferred learning style",
            "questions": [
                {
                    "id": "q1",
                    "text": "When learning a new concept, I prefer to:",
                    "options": [
                        {"id": "a", "text": "Watch a video or look at diagrams", "style": "visual"},
                        {"id": "b", "text": "Listen to a lecture or explanation", "style": "auditory"},
                        {"id": "c", "text": "Read about it in a textbook or article", "style": "reading"},
                        {"id": "d", "text": "Try it out myself with practice problems", "style": "kinesthetic"}
                    ]
                },
                {
                    "id": "q2",
                    "text": "When I need to remember something, I usually:",
                    "options": [
                        {"id": "a", "text": "Visualize it in my mind", "style": "visual"},
                        {"id": "b", "text": "Repeat it out loud or discuss it", "style": "auditory"},
                        {"id": "c", "text": "Write it down or read it again", "style": "reading"},
                        {"id": "d", "text": "Associate it with an action or movement", "style": "kinesthetic"}
                    ]
                },
                {
                    "id": "q3",
                    "text": "In class, I learn best when the teacher:",
                    "options": [
                        {"id": "a", "text": "Uses slides, charts, or writes on the board", "style": "visual"},
                        {"id": "b", "text": "Explains concepts verbally", "style": "auditory"},
                        {"id": "c", "text": "Gives handouts or reading materials", "style": "reading"},
                        {"id": "d", "text": "Uses demonstrations or hands-on activities", "style": "kinesthetic"}
                    ]
                },
                {
                    "id": "q4",
                    "text": "When studying for a test, I prefer to:",
                    "options": [
                        {"id": "a", "text": "Look at my notes and highlight key points", "style": "visual"},
                        {"id": "b", "text": "Explain concepts to others or myself", "style": "auditory"},
                        {"id": "c", "text": "Review written materials and summaries", "style": "reading"},
                        {"id": "d", "text": "Practice problems and examples", "style": "kinesthetic"}
                    ]
                },
                {
                    "id": "q5",
                    "text": "I find it easiest to follow:",
                    "options": [
                        {"id": "a", "text": "Maps, diagrams, or flowcharts", "style": "visual"},
                        {"id": "b", "text": "Verbal directions", "style": "auditory"},
                        {"id": "c", "text": "Written instructions", "style": "reading"},
                        {"id": "d", "text": "Physical demonstrations", "style": "kinesthetic"}
                    ]
                }
            ]
        }
    
    def analyze_learning_style_quiz(
        self,
        responses: Dict[str, str]
    ) -> Tuple[LearningStyle, Optional[LearningStyle]]:
        """
        Analyze quiz responses to determine learning style.
        
        Args:
            responses: Map of question_id -> selected option style
            
        Returns:
            Tuple of (primary_style, secondary_style or None)
        """
        # Count style preferences
        counts = {
            "visual": 0,
            "auditory": 0,
            "reading": 0,
            "kinesthetic": 0
        }
        
        for style in responses.values():
            if style in counts:
                counts[style] += 1
        
        # Sort by count
        sorted_styles = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        
        primary = LearningStyle(sorted_styles[0][0])
        
        # Secondary style if there's a close second
        secondary = None
        if len(sorted_styles) > 1 and sorted_styles[1][1] >= sorted_styles[0][1] - 1:
            secondary = LearningStyle(sorted_styles[1][0])
        
        return primary, secondary


# ============================================================================
# Singleton
# ============================================================================

_spaced_rep_service: Optional[SpacedRepetitionService] = None


def get_spaced_repetition_service() -> SpacedRepetitionService:
    """Get singleton spaced repetition service"""
    global _spaced_rep_service
    if _spaced_rep_service is None:
        _spaced_rep_service = SpacedRepetitionService()
    return _spaced_rep_service
