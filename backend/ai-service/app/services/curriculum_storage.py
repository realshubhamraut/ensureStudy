"""
Curriculum Storage Service - Persistence for Curricula and Progress

Features:
- Redis caching for fast curriculum lookups
- In-memory fallback when Redis unavailable
- Progress tracking per user/topic
- Quiz storage for evaluation
- Adaptive scheduling when student falls behind

Storage Patterns:
- curriculum:{curriculum_id} -> Full curriculum JSON
- curriculum:user:{user_id} -> Set of curriculum IDs
- curriculum:quiz:{quiz_id} -> Quiz JSON
- progress:{curriculum_id}:{topic_id} -> Progress JSON
"""
import os
import json
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

# Try to import redis
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("[CURRICULUM-STORE] redis not installed, using memory only")


# ============================================================================
# Configuration
# ============================================================================

CURRICULUM_TTL = int(os.getenv("CURRICULUM_CACHE_TTL", "604800"))  # 7 days
QUIZ_TTL = int(os.getenv("QUIZ_CACHE_TTL", "86400"))  # 24 hours
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Prefixes
PREFIX_CURRICULUM = "curriculum:"
PREFIX_USER_CURRICULA = "curriculum:user:"
PREFIX_QUIZ = "curriculum:quiz:"
PREFIX_PROGRESS = "progress:"


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class TopicProgress:
    """Progress on a single topic"""
    topic_id: str
    topic_name: str
    status: str  # not_started, in_progress, completed
    mastery: float  # 0.0 to 1.0
    time_spent_minutes: int
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    quiz_attempts: int = 0
    last_quiz_score: float = 0.0
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class CurriculumProgress:
    """Overall curriculum progress"""
    curriculum_id: str
    user_id: str
    total_topics: int
    completed_topics: int
    in_progress_topics: int
    overall_mastery: float
    current_day: int
    days_behind: int  # Positive = behind schedule
    total_time_spent_minutes: int
    last_active: str
    topic_progress: Dict[str, TopicProgress]
    
    def to_dict(self) -> Dict:
        return {
            **{k: v for k, v in asdict(self).items() if k != "topic_progress"},
            "topic_progress": {k: v.to_dict() for k, v in self.topic_progress.items()}
        }


# ============================================================================
# Storage Service
# ============================================================================

class CurriculumStorageService:
    """
    Stores and retrieves curricula, quizzes, and progress.
    
    Uses Redis when available with in-memory fallback.
    """
    
    def __init__(self, redis_url: str = None):
        self.redis_url = redis_url or REDIS_URL
        self._client = None
        self._available = REDIS_AVAILABLE
        
        # In-memory fallback storage
        self._memory_curricula: Dict[str, Dict] = {}
        self._memory_quizzes: Dict[str, Dict] = {}
        self._memory_progress: Dict[str, Dict] = {}
        self._memory_user_curricula: Dict[str, List[str]] = {}
    
    @property
    def client(self) -> Optional[Any]:
        """Lazy load Redis client"""
        if not self._available:
            return None
            
        if self._client is None:
            try:
                self._client = redis.from_url(self.redis_url, decode_responses=True)
                self._client.ping()
                logger.info("[CURRICULUM-STORE] Connected to Redis")
            except Exception as e:
                logger.warning(f"[CURRICULUM-STORE] Redis unavailable: {e}, using memory")
                self._available = False
                self._client = None
                
        return self._client
    
    @property
    def is_redis_available(self) -> bool:
        """Check if Redis is available"""
        return self.client is not None
    
    # ========================================================================
    # Curriculum Storage
    # ========================================================================
    
    def save_curriculum(self, curriculum: Dict) -> bool:
        """
        Save curriculum to storage.
        
        Args:
            curriculum: Curriculum dict with 'id' and 'user_id'
            
        Returns:
            True if saved successfully
        """
        curriculum_id = curriculum.get("id")
        user_id = curriculum.get("user_id")
        
        if not curriculum_id or not user_id:
            logger.error("[CURRICULUM-STORE] Curriculum must have id and user_id")
            return False
        
        # Add metadata
        curriculum["stored_at"] = datetime.utcnow().isoformat()
        
        if self.is_redis_available:
            try:
                # Save curriculum
                key = f"{PREFIX_CURRICULUM}{curriculum_id}"
                self.client.setex(key, CURRICULUM_TTL, json.dumps(curriculum, default=str))
                
                # Add to user's curriculum set
                user_key = f"{PREFIX_USER_CURRICULA}{user_id}"
                self.client.sadd(user_key, curriculum_id)
                self.client.expire(user_key, CURRICULUM_TTL)
                
                logger.info(f"[CURRICULUM-STORE] Saved curriculum {curriculum_id[:8]}...")
                return True
                
            except Exception as e:
                logger.error(f"[CURRICULUM-STORE] Redis save failed: {e}")
        
        # Memory fallback
        self._memory_curricula[curriculum_id] = curriculum
        if user_id not in self._memory_user_curricula:
            self._memory_user_curricula[user_id] = []
        if curriculum_id not in self._memory_user_curricula[user_id]:
            self._memory_user_curricula[user_id].append(curriculum_id)
        
        logger.info(f"[CURRICULUM-STORE] Saved curriculum to memory {curriculum_id[:8]}...")
        return True
    
    def get_curriculum(self, curriculum_id: str) -> Optional[Dict]:
        """Get curriculum by ID"""
        if self.is_redis_available:
            try:
                key = f"{PREFIX_CURRICULUM}{curriculum_id}"
                data = self.client.get(key)
                if data:
                    return json.loads(data)
            except Exception as e:
                logger.error(f"[CURRICULUM-STORE] Redis get failed: {e}")
        
        # Memory fallback
        return self._memory_curricula.get(curriculum_id)
    
    def get_user_curricula(self, user_id: str) -> List[Dict]:
        """Get all curricula for a user"""
        curriculum_ids = []
        
        if self.is_redis_available:
            try:
                user_key = f"{PREFIX_USER_CURRICULA}{user_id}"
                curriculum_ids = list(self.client.smembers(user_key))
            except Exception as e:
                logger.error(f"[CURRICULUM-STORE] Redis get user curricula failed: {e}")
        else:
            curriculum_ids = self._memory_user_curricula.get(user_id, [])
        
        # Fetch each curriculum
        curricula = []
        for cid in curriculum_ids:
            curriculum = self.get_curriculum(cid)
            if curriculum:
                curricula.append(curriculum)
        
        return curricula
    
    def delete_curriculum(self, curriculum_id: str, user_id: str) -> bool:
        """Delete a curriculum and all associated progress data"""
        # First, get the curriculum to find all topics
        curriculum = self.get_curriculum(curriculum_id)
        topic_ids = []
        if curriculum:
            topics = curriculum.get("topics", [])
            topic_ids = [t.get("id", t.get("name", "")) for t in topics]
        
        if self.is_redis_available:
            try:
                # Delete curriculum
                self.client.delete(f"{PREFIX_CURRICULUM}{curriculum_id}")
                # Remove from user's curriculum set
                self.client.srem(f"{PREFIX_USER_CURRICULA}{user_id}", curriculum_id)
                
                # Delete all topic progress for this curriculum
                for topic_id in topic_ids:
                    progress_key = f"{PREFIX_PROGRESS}{curriculum_id}:{topic_id}"
                    self.client.delete(progress_key)
                
                logger.info(f"[CURRICULUM-STORE] Deleted curriculum {curriculum_id[:8]}... and {len(topic_ids)} topic progress entries")
                return True
            except Exception as e:
                logger.error(f"[CURRICULUM-STORE] Redis delete failed: {e}")
        
        # Memory fallback
        if curriculum_id in self._memory_curricula:
            del self._memory_curricula[curriculum_id]
        if user_id in self._memory_user_curricula:
            self._memory_user_curricula[user_id] = [
                c for c in self._memory_user_curricula[user_id] if c != curriculum_id
            ]
        # Delete topic progress from memory
        for topic_id in topic_ids:
            progress_key = f"{curriculum_id}:{topic_id}"
            if progress_key in self._memory_progress:
                del self._memory_progress[progress_key]
        
        logger.info(f"[CURRICULUM-STORE] Deleted curriculum from memory {curriculum_id[:8]}... and {len(topic_ids)} topic progress entries")
        return True
    
    # ========================================================================
    # Quiz Storage
    # ========================================================================
    
    def save_quiz(self, quiz: Dict) -> bool:
        """Save diagnostic quiz for later evaluation"""
        quiz_id = quiz.get("id")
        if not quiz_id:
            return False
        
        quiz["stored_at"] = datetime.utcnow().isoformat()
        
        if self.is_redis_available:
            try:
                key = f"{PREFIX_QUIZ}{quiz_id}"
                self.client.setex(key, QUIZ_TTL, json.dumps(quiz, default=str))
                logger.info(f"[CURRICULUM-STORE] Saved quiz {quiz_id}")
                return True
            except Exception as e:
                logger.error(f"[CURRICULUM-STORE] Quiz save failed: {e}")
        
        # Memory fallback
        self._memory_quizzes[quiz_id] = quiz
        return True
    
    def get_quiz(self, quiz_id: str) -> Optional[Dict]:
        """Get quiz by ID"""
        if self.is_redis_available:
            try:
                key = f"{PREFIX_QUIZ}{quiz_id}"
                data = self.client.get(key)
                if data:
                    return json.loads(data)
            except Exception as e:
                logger.error(f"[CURRICULUM-STORE] Quiz get failed: {e}")
        
        return self._memory_quizzes.get(quiz_id)
    
    # ========================================================================
    # Progress Tracking
    # ========================================================================
    
    def update_topic_progress(
        self,
        curriculum_id: str,
        topic_id: str,
        status: str,
        mastery: float = None,
        time_spent_minutes: int = None,
        quiz_score: float = None
    ) -> TopicProgress:
        """
        Update progress on a topic.
        
        Args:
            curriculum_id: Curriculum ID
            topic_id: Topic ID
            status: "not_started", "in_progress", "completed"
            mastery: Optional mastery score update
            time_spent_minutes: Time to add
            quiz_score: Optional quiz score to record
        """
        key = f"{curriculum_id}:{topic_id}"
        
        # Get existing progress
        existing = self._get_topic_progress_raw(key)
        
        now = datetime.utcnow().isoformat()
        
        if existing:
            progress = TopicProgress(**existing)
        else:
            progress = TopicProgress(
                topic_id=topic_id,
                topic_name=topic_id,  # Will be overwritten
                status="not_started",
                mastery=0.0,
                time_spent_minutes=0
            )
        
        # Update fields
        progress.status = status
        
        if mastery is not None:
            progress.mastery = mastery
        
        if time_spent_minutes:
            progress.time_spent_minutes += time_spent_minutes
        
        if quiz_score is not None:
            progress.quiz_attempts += 1
            progress.last_quiz_score = quiz_score
            # Update mastery based on quiz
            progress.mastery = max(progress.mastery, quiz_score)
        
        if status == "in_progress" and not progress.started_at:
            progress.started_at = now
        
        if status == "completed":
            progress.completed_at = now
        
        # Save
        self._save_topic_progress_raw(key, progress.to_dict())
        
        logger.info(f"[CURRICULUM-STORE] Updated progress {topic_id}: {status}")
        return progress
    
    def _get_topic_progress_raw(self, key: str) -> Optional[Dict]:
        """Get raw topic progress"""
        if self.is_redis_available:
            try:
                data = self.client.get(f"{PREFIX_PROGRESS}{key}")
                if data:
                    return json.loads(data)
            except Exception as e:
                logger.error(f"[CURRICULUM-STORE] Progress get failed: {e}")
        
        return self._memory_progress.get(key)
    
    def _save_topic_progress_raw(self, key: str, data: Dict) -> bool:
        """Save raw topic progress"""
        if self.is_redis_available:
            try:
                self.client.setex(
                    f"{PREFIX_PROGRESS}{key}",
                    CURRICULUM_TTL,
                    json.dumps(data, default=str)
                )
                return True
            except Exception as e:
                logger.error(f"[CURRICULUM-STORE] Progress save failed: {e}")
        
        self._memory_progress[key] = data
        return True
    
    def get_curriculum_progress(self, curriculum_id: str) -> Optional[CurriculumProgress]:
        """
        Get overall progress for a curriculum.
        
        Aggregates all topic progress into summary stats.
        """
        curriculum = self.get_curriculum(curriculum_id)
        if not curriculum:
            return None
        
        topics = curriculum.get("topics", [])
        user_id = curriculum.get("user_id", "")
        
        topic_progress = {}
        completed = 0
        in_progress = 0
        total_mastery = 0.0
        total_time = 0
        
        for topic in topics:
            topic_id = topic.get("id", topic.get("name", ""))
            key = f"{curriculum_id}:{topic_id}"
            raw = self._get_topic_progress_raw(key)
            
            if raw:
                tp = TopicProgress(**raw)
            else:
                tp = TopicProgress(
                    topic_id=topic_id,
                    topic_name=topic.get("name", topic_id),
                    status="not_started",
                    mastery=0.0,
                    time_spent_minutes=0
                )
            
            topic_progress[topic_id] = tp
            total_mastery += tp.mastery
            total_time += tp.time_spent_minutes
            
            if tp.status == "completed":
                completed += 1
            elif tp.status == "in_progress":
                in_progress += 1
        
        # Calculate days behind
        start_date = datetime.strptime(
            curriculum.get("start_date", datetime.now().strftime("%Y-%m-%d")),
            "%Y-%m-%d"
        )
        current_day = (datetime.now() - start_date).days + 1
        
        # Expected progress based on schedule
        daily_goals = curriculum.get("daily_goals", [])
        expected_topics = sum(
            len(g.get("topics", []))
            for g in daily_goals
            if g.get("day", 0) <= current_day
        )
        
        days_behind = max(0, expected_topics - completed)
        
        return CurriculumProgress(
            curriculum_id=curriculum_id,
            user_id=user_id,
            total_topics=len(topics),
            completed_topics=completed,
            in_progress_topics=in_progress,
            overall_mastery=total_mastery / len(topics) if topics else 0.0,
            current_day=current_day,
            days_behind=days_behind,
            total_time_spent_minutes=total_time,
            last_active=datetime.utcnow().isoformat(),
            topic_progress=topic_progress
        )
    
    # ========================================================================
    # Adaptive Scheduling
    # ========================================================================
    
    def reschedule_curriculum(
        self,
        curriculum_id: str,
        new_hours_per_day: float = None,
        extend_days: int = None
    ) -> Optional[Dict]:
        """
        Reschedule a curriculum when student falls behind.
        
        Args:
            curriculum_id: Curriculum ID
            new_hours_per_day: New daily study hours
            extend_days: Days to extend deadline
            
        Returns:
            Updated curriculum or None
        """
        curriculum = self.get_curriculum(curriculum_id)
        if not curriculum:
            return None
        
        progress = self.get_curriculum_progress(curriculum_id)
        if not progress:
            return curriculum
        
        # Get remaining topics
        remaining_topics = []
        for topic in curriculum.get("topics", []):
            topic_id = topic.get("id", topic.get("name", ""))
            tp = progress.topic_progress.get(topic_id)
            if not tp or tp.status != "completed":
                remaining_topics.append(topic)
        
        if not remaining_topics:
            # All done!
            return curriculum
        
        # Calculate new schedule
        hours_per_day = new_hours_per_day or curriculum.get("hours_per_day", 2.0)
        total_remaining_hours = sum(t.get("estimated_hours", 2.0) for t in remaining_topics)
        days_needed = int(total_remaining_hours / hours_per_day) + 1
        
        if extend_days:
            days_needed = max(days_needed, extend_days)
        
        # Update end date
        new_end = datetime.now() + timedelta(days=days_needed)
        curriculum["end_date"] = new_end.strftime("%Y-%m-%d")
        curriculum["rescheduled_at"] = datetime.utcnow().isoformat()
        curriculum["days_behind"] = progress.days_behind
        
        # Regenerate daily goals for remaining topics
        new_goals = []
        current_day = 1
        current_topics = []
        current_hours = 0
        
        for topic in remaining_topics:
            topic_hours = topic.get("estimated_hours", 2.0)
            
            if current_hours + topic_hours > hours_per_day and current_topics:
                new_goals.append({
                    "day": current_day,
                    "date": (datetime.now() + timedelta(days=current_day - 1)).strftime("%Y-%m-%d"),
                    "topics": [t.get("name", "") for t in current_topics],
                    "total_hours": current_hours,
                    "milestone": None
                })
                current_day += 1
                current_topics = []
                current_hours = 0
            
            current_topics.append(topic)
            current_hours += topic_hours
        
        if current_topics:
            new_goals.append({
                "day": current_day,
                "date": (datetime.now() + timedelta(days=current_day - 1)).strftime("%Y-%m-%d"),
                "topics": [t.get("name", "") for t in current_topics],
                "total_hours": current_hours,
                "milestone": "🎯 Catch-up Complete"
            })
        
        curriculum["daily_goals"] = new_goals
        curriculum["total_days"] = len(new_goals)
        
        # Save updated curriculum
        self.save_curriculum(curriculum)
        
        logger.info(
            f"[CURRICULUM-STORE] Rescheduled {curriculum_id[:8]}... "
            f"({len(remaining_topics)} topics, {len(new_goals)} days)"
        )
        
        return curriculum
    
    # ========================================================================
    # Analytics
    # ========================================================================
    
    def get_storage_stats(self) -> Dict:
        """Get storage statistics"""
        stats = {
            "redis_available": self.is_redis_available,
            "memory_curricula": len(self._memory_curricula),
            "memory_quizzes": len(self._memory_quizzes),
            "memory_progress": len(self._memory_progress),
        }
        
        if self.is_redis_available:
            try:
                # Count Redis keys
                curriculum_keys = list(self.client.scan_iter(match=f"{PREFIX_CURRICULUM}*"))
                quiz_keys = list(self.client.scan_iter(match=f"{PREFIX_QUIZ}*"))
                progress_keys = list(self.client.scan_iter(match=f"{PREFIX_PROGRESS}*"))
                
                stats["redis_curricula"] = len(curriculum_keys)
                stats["redis_quizzes"] = len(quiz_keys)
                stats["redis_progress"] = len(progress_keys)
            except Exception as e:
                stats["redis_error"] = str(e)
        
        return stats


# ============================================================================
# Singleton
# ============================================================================

_storage_service: Optional[CurriculumStorageService] = None


def get_curriculum_storage() -> CurriculumStorageService:
    """Get singleton storage service"""
    global _storage_service
    if _storage_service is None:
        _storage_service = CurriculumStorageService()
    return _storage_service
