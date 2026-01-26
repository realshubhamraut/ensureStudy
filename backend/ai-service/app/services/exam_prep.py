"""
Exam Prep Service - Intensified study scheduling before exams

Features:
- Exam-focused curriculum intensification
- Low-mastery topic prioritization
- Timed practice quiz generation
- Resource recommendations via Research Agent
- Daily prep schedules with review sessions
"""
import os
import logging
from typing import Optional, Dict, List, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class ExamInfo:
    """Exam metadata"""
    exam_id: str
    name: str
    subject: str
    date: str  # YYYY-MM-DD
    curriculum_id: str
    topics: List[str]
    total_marks: int = 100
    duration_minutes: int = 120


@dataclass
class PrepDay:
    """A single day in the prep schedule"""
    day: int
    date: str
    focus_topics: List[str]
    activities: List[Dict]
    total_hours: float
    is_review_day: bool = False
    is_exam_day: bool = False


@dataclass
class ExamPrepPlan:
    """Complete exam preparation plan"""
    exam_id: str
    exam_name: str
    exam_date: str
    days_until_exam: int
    curriculum_id: str
    total_prep_days: int
    hours_per_day: float
    
    # Topic prioritization
    weak_topics: List[Dict]  # Low mastery topics
    strong_topics: List[Dict]  # High mastery topics
    
    # Schedule
    prep_days: List[PrepDay]
    review_days: List[int]  # Day numbers that are review days
    
    # Resources
    recommended_resources: List[Dict]
    practice_tests: List[Dict]
    
    def to_dict(self) -> Dict:
        return {
            **{k: v for k, v in asdict(self).items() if k != "prep_days"},
            "prep_days": [asdict(d) for d in self.prep_days]
        }


# ============================================================================
# Exam Prep Service
# ============================================================================

class ExamPrepService:
    """
    Creates intensive exam preparation plans.
    
    Strategy:
    1. Calculate days until exam
    2. Identify weak topics from progress data
    3. Allocate more time to weak topics
    4. Insert review days using spaced repetition
    5. Generate practice tests for final days
    """
    
    def __init__(self):
        self._exams: Dict[str, ExamInfo] = {}
        self._plans: Dict[str, ExamPrepPlan] = {}
    
    async def create_exam_prep_plan(
        self,
        exam_name: str,
        exam_date: str,
        curriculum_id: str,
        user_id: str,
        hours_per_day: float = 3.0,
        include_resources: bool = True
    ) -> ExamPrepPlan:
        """
        Create an intensive exam prep plan.
        
        Args:
            exam_name: Name of the exam
            exam_date: Exam date (YYYY-MM-DD)
            curriculum_id: Associated curriculum ID
            user_id: Student ID
            hours_per_day: Daily study hours
            include_resources: Whether to fetch resource recommendations
            
        Returns:
            Complete ExamPrepPlan
        """
        exam_id = f"exam_{curriculum_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Get curriculum and progress
        from app.services.curriculum_storage import get_curriculum_storage
        storage = get_curriculum_storage()
        
        curriculum = storage.get_curriculum(curriculum_id)
        if not curriculum:
            raise ValueError(f"Curriculum {curriculum_id} not found")
        
        progress = storage.get_curriculum_progress(curriculum_id)
        
        # Calculate days until exam
        exam_dt = datetime.strptime(exam_date, "%Y-%m-%d")
        days_until = (exam_dt - datetime.now()).days
        
        if days_until < 1:
            raise ValueError("Exam date must be in the future")
        
        # Get topics with their mastery
        topics = curriculum.get("topics", [])
        topic_mastery = []
        
        for topic in topics:
            topic_id = topic.get("id", topic.get("name", ""))
            mastery = 0.0
            
            if progress:
                for tp in progress.topic_progress.values():
                    if tp.topic_id == topic_id:
                        mastery = tp.mastery
                        break
            
            topic_mastery.append({
                "id": topic_id,
                "name": topic.get("name", ""),
                "mastery": mastery,
                "estimated_hours": topic.get("estimated_hours", 2.0)
            })
        
        # Sort by mastery - weak topics first
        topic_mastery.sort(key=lambda x: x["mastery"])
        
        weak_topics = [t for t in topic_mastery if t["mastery"] < 0.6]
        strong_topics = [t for t in topic_mastery if t["mastery"] >= 0.6]
        
        # Create prep schedule
        prep_days = self._generate_prep_schedule(
            topics=topic_mastery,
            weak_topics=weak_topics,
            days_until=days_until,
            hours_per_day=hours_per_day,
            exam_date=exam_date
        )
        
        # Get resources if requested
        resources = []
        practice_tests = []
        
        if include_resources:
            resources = await self._get_topic_resources(
                topics=[t["name"] for t in weak_topics[:5]],
                subject=curriculum.get("subject_name", "")
            )
            
            practice_tests = self._generate_practice_test_schedule(
                topics=[t["name"] for t in topic_mastery],
                days_until=days_until
            )
        
        # Build plan
        review_days = [d.day for d in prep_days if d.is_review_day]
        
        plan = ExamPrepPlan(
            exam_id=exam_id,
            exam_name=exam_name,
            exam_date=exam_date,
            days_until_exam=days_until,
            curriculum_id=curriculum_id,
            total_prep_days=len(prep_days),
            hours_per_day=hours_per_day,
            weak_topics=weak_topics,
            strong_topics=strong_topics,
            prep_days=prep_days,
            review_days=review_days,
            recommended_resources=resources,
            practice_tests=practice_tests
        )
        
        # Store plan
        self._plans[exam_id] = plan
        
        # Store exam info
        self._exams[exam_id] = ExamInfo(
            exam_id=exam_id,
            name=exam_name,
            subject=curriculum.get("subject_name", ""),
            date=exam_date,
            curriculum_id=curriculum_id,
            topics=[t["name"] for t in topic_mastery]
        )
        
        logger.info(
            f"[EXAM-PREP] Created plan for '{exam_name}' on {exam_date} "
            f"({days_until} days, {len(weak_topics)} weak topics)"
        )
        
        return plan
    
    def _generate_prep_schedule(
        self,
        topics: List[Dict],
        weak_topics: List[Dict],
        days_until: int,
        hours_per_day: float,
        exam_date: str
    ) -> List[PrepDay]:
        """Generate daily prep schedule"""
        prep_days = []
        
        # Strategy:
        # - First 60% of time: Focus on weak topics
        # - Next 30%: Review all topics
        # - Last 10%: Practice tests and light review
        
        weak_end = int(days_until * 0.6)
        review_end = int(days_until * 0.9)
        
        # Build topic queue with weak topics weighted higher
        topic_queue = []
        for t in weak_topics:
            # Add weak topics 2x
            topic_queue.append(t)
            topic_queue.append(t)
        for t in topics:
            if t not in weak_topics:
                topic_queue.append(t)
        
        topic_idx = 0
        review_intervals = [3, 7, 14]  # Review every 3, 7, 14 days
        
        for day in range(1, days_until + 1):
            date = (datetime.now() + timedelta(days=day)).strftime("%Y-%m-%d")
            is_review = day in review_intervals or day > review_end
            is_exam = day == days_until
            
            activities = []
            focus_topics = []
            
            if is_exam:
                # Exam day - light review only
                activities.append({
                    "type": "light_review",
                    "description": "Quick review of key concepts",
                    "duration_min": 30
                })
                activities.append({
                    "type": "exam",
                    "description": f"Take exam: {exam_date}",
                    "duration_min": 120
                })
            elif is_review:
                # Review day - practice tests and spaced review
                activities.append({
                    "type": "practice_test",
                    "description": "Timed practice test",
                    "duration_min": 60
                })
                activities.append({
                    "type": "review",
                    "description": "Review incorrect answers",
                    "duration_min": int(hours_per_day * 60 - 60)
                })
                # Add topics from weak list
                focus_topics = [t["name"] for t in weak_topics[:3]]
            else:
                # Regular study day
                time_per_topic = (hours_per_day * 60) / min(3, len(topics))
                
                for _ in range(min(3, len(topics))):
                    if topic_idx < len(topic_queue):
                        topic = topic_queue[topic_idx]
                        topic_idx = (topic_idx + 1) % len(topic_queue)
                        
                        focus_topics.append(topic["name"])
                        activities.append({
                            "type": "study",
                            "topic": topic["name"],
                            "description": f"Study: {topic['name']}",
                            "duration_min": int(time_per_topic * 0.7)
                        })
                        activities.append({
                            "type": "practice",
                            "topic": topic["name"],
                            "description": f"Practice problems: {topic['name']}",
                            "duration_min": int(time_per_topic * 0.3)
                        })
            
            prep_days.append(PrepDay(
                day=day,
                date=date,
                focus_topics=focus_topics,
                activities=activities,
                total_hours=hours_per_day,
                is_review_day=is_review,
                is_exam_day=is_exam
            ))
        
        return prep_days
    
    async def _get_topic_resources(
        self,
        topics: List[str],
        subject: str
    ) -> List[Dict]:
        """Get resource recommendations using Research Agent"""
        resources = []
        
        try:
            from app.agents import get_research_agent
            agent = get_research_agent()
            
            for topic in topics[:3]:  # Limit to 3 topics
                result = await agent.execute({
                    "query": f"{topic} {subject} study guide tutorial",
                    "user_id": "system",
                    "search_web": True,
                    "search_youtube": True,
                    "search_pdfs": False,
                    "max_results": 3
                })
                
                if result.get("success"):
                    for web_result in result.get("web_results", [])[:2]:
                        resources.append({
                            "topic": topic,
                            "type": "article",
                            "title": web_result.get("title", ""),
                            "url": web_result.get("url", ""),
                            "description": web_result.get("snippet", "")
                        })
                    
                    for yt_result in result.get("youtube_results", [])[:1]:
                        resources.append({
                            "topic": topic,
                            "type": "video",
                            "title": yt_result.get("title", ""),
                            "url": yt_result.get("url", ""),
                            "description": yt_result.get("description", "")
                        })
        
        except Exception as e:
            logger.warning(f"[EXAM-PREP] Resource fetch failed: {e}")
        
        return resources
    
    def _generate_practice_test_schedule(
        self,
        topics: List[str],
        days_until: int
    ) -> List[Dict]:
        """Generate practice test schedule"""
        tests = []
        
        # Full mock tests in final week
        if days_until >= 7:
            tests.append({
                "day": days_until - 5,
                "type": "full_mock",
                "description": "Full mock exam simulation",
                "topics": topics,
                "duration_min": 90
            })
        
        if days_until >= 3:
            tests.append({
                "day": days_until - 2,
                "type": "topic_test",
                "description": "Weak topics focused test",
                "topics": topics[:5],
                "duration_min": 45
            })
        
        if days_until >= 1:
            tests.append({
                "day": days_until - 1,
                "type": "quick_review",
                "description": "Quick review quiz",
                "topics": topics[:3],
                "duration_min": 20
            })
        
        return tests
    
    def get_exam(self, exam_id: str) -> Optional[ExamInfo]:
        """Get exam info by ID"""
        return self._exams.get(exam_id)
    
    def get_prep_plan(self, exam_id: str) -> Optional[ExamPrepPlan]:
        """Get prep plan by exam ID"""
        return self._plans.get(exam_id)
    
    def get_user_exams(self, curriculum_id: str) -> List[ExamInfo]:
        """Get all exams for a curriculum"""
        return [e for e in self._exams.values() if e.curriculum_id == curriculum_id]


# ============================================================================
# Singleton
# ============================================================================

_exam_prep_service: Optional[ExamPrepService] = None


def get_exam_prep_service() -> ExamPrepService:
    """Get singleton exam prep service"""
    global _exam_prep_service
    if _exam_prep_service is None:
        _exam_prep_service = ExamPrepService()
    return _exam_prep_service
