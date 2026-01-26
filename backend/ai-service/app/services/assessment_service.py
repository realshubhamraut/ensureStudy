"""
Knowledge Assessment Service

Provides diagnostic assessments for curriculum topics to:
1. Generate diagnostic quizzes for each topic
2. Evaluate student responses and calculate mastery
3. Track knowledge gaps and adjust curriculum
"""
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import json

logger = logging.getLogger(__name__)


# ============================================================================
# Data Models
# ============================================================================

@dataclass
class DiagnosticQuestion:
    """A single diagnostic question for a topic"""
    id: str
    topic_id: str
    topic_name: str
    question_text: str
    question_type: str  # mcq, short_answer
    options: List[Dict[str, str]]  # For MCQ
    correct_answer: str
    difficulty: str
    points: int = 1
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class AssessmentResult:
    """Result of evaluating a student's answers"""
    topic_id: str
    topic_name: str
    total_questions: int
    correct_answers: int
    mastery_score: float  # 0.0 to 1.0
    knowledge_level: str  # beginner, intermediate, advanced
    weak_areas: List[str]
    strong_areas: List[str]
    time_taken_seconds: int
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass  
class DiagnosticQuiz:
    """Complete diagnostic quiz for a curriculum"""
    id: str
    curriculum_id: str
    user_id: str
    topics: List[str]
    questions: List[DiagnosticQuestion]
    total_points: int
    estimated_time_minutes: int
    created_at: str
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "curriculum_id": self.curriculum_id,
            "user_id": self.user_id,
            "topics": self.topics,
            "questions": [q.to_dict() for q in self.questions],
            "total_points": self.total_points,
            "estimated_time_minutes": self.estimated_time_minutes,
            "created_at": self.created_at
        }


# ============================================================================
# Assessment Service
# ============================================================================

class KnowledgeAssessmentService:
    """
    Service for generating and evaluating diagnostic assessments.
    
    This helps:
    1. Establish baseline knowledge before starting curriculum
    2. Identify topics student already knows
    3. Prioritize topics that need more attention
    4. Skip unnecessary content
    """
    
    def __init__(self):
        self._question_generator = None
        self._llm_provider = None
    
    @property
    def question_generator(self):
        """Lazy load question generator"""
        if self._question_generator is None:
            from app.services.question_generator import get_question_generator
            self._question_generator = get_question_generator()
        return self._question_generator
    
    @property
    def llm_provider(self):
        """Lazy load LLM provider"""
        if self._llm_provider is None:
            from app.services.llm_provider import get_llm_provider
            self._llm_provider = get_llm_provider()
        return self._llm_provider
    
    async def generate_diagnostic_quiz(
        self,
        curriculum_id: str,
        user_id: str,
        topics: List[Dict[str, Any]],
        questions_per_topic: int = 3,
        difficulty: str = "mixed"
    ) -> DiagnosticQuiz:
        """
        Generate a diagnostic quiz covering all curriculum topics.
        
        Args:
            curriculum_id: ID of the curriculum
            user_id: Student ID
            topics: List of topic dicts with 'id', 'name', 'description'
            questions_per_topic: Number of questions per topic (default 3)
            difficulty: "easy", "medium", "hard", or "mixed"
            
        Returns:
            DiagnosticQuiz with questions for all topics
        """
        import uuid
        
        quiz_id = f"quiz_{uuid.uuid4().hex[:12]}"
        all_questions = []
        topic_names = []
        
        logger.info(f"[ASSESSMENT] Generating diagnostic quiz for {len(topics)} topics")
        
        for topic in topics:
            topic_id = topic.get("id", topic.get("name", "unknown"))
            topic_name = topic.get("name", "Unknown Topic")
            topic_desc = topic.get("description", topic_name)
            topic_names.append(topic_name)
            
            try:
                # Generate questions for this topic
                questions = await self._generate_topic_questions(
                    topic_id=topic_id,
                    topic_name=topic_name,
                    topic_description=topic_desc,
                    num_questions=questions_per_topic,
                    difficulty=difficulty
                )
                all_questions.extend(questions)
                
            except Exception as e:
                logger.warning(f"[ASSESSMENT] Failed to generate questions for {topic_name}: {e}")
                # Add a fallback question
                all_questions.append(DiagnosticQuestion(
                    id=f"q_{uuid.uuid4().hex[:8]}",
                    topic_id=topic_id,
                    topic_name=topic_name,
                    question_text=f"How familiar are you with {topic_name}?",
                    question_type="mcq",
                    options=[
                        {"id": "A", "text": "Not familiar at all"},
                        {"id": "B", "text": "Somewhat familiar"},
                        {"id": "C", "text": "Fairly familiar"},
                        {"id": "D", "text": "Very familiar"}
                    ],
                    correct_answer="D",  # Not really a correct answer
                    difficulty="easy",
                    points=1
                ))
        
        # Calculate totals
        total_points = sum(q.points for q in all_questions)
        estimated_time = len(all_questions) * 1  # 1 minute per question avg
        
        quiz = DiagnosticQuiz(
            id=quiz_id,
            curriculum_id=curriculum_id,
            user_id=user_id,
            topics=topic_names,
            questions=all_questions,
            total_points=total_points,
            estimated_time_minutes=estimated_time,
            created_at=datetime.now().isoformat()
        )
        
        logger.info(f"[ASSESSMENT] Created quiz with {len(all_questions)} questions")
        return quiz
    
    async def _generate_topic_questions(
        self,
        topic_id: str,
        topic_name: str,
        topic_description: str,
        num_questions: int,
        difficulty: str
    ) -> List[DiagnosticQuestion]:
        """Generate diagnostic MCQ questions for a single topic"""
        import uuid
        
        # Create content for question generation
        content = f"""
Topic: {topic_name}

Description: {topic_description}

This is a diagnostic assessment to determine the student's existing knowledge level on this topic.
Generate questions that test understanding at different levels:
- Basic recall/definition questions
- Application/understanding questions  
- Analysis/synthesis questions (if advanced)
"""
        
        # Use existing question generator
        try:
            generated = await self.question_generator.generate_questions(
                content=content,
                question_type="mcq",
                num_questions=num_questions,
                difficulty=difficulty if difficulty != "mixed" else "medium"
            )
            
            # Convert to DiagnosticQuestion format
            questions = []
            for i, gq in enumerate(generated):
                questions.append(DiagnosticQuestion(
                    id=f"q_{uuid.uuid4().hex[:8]}",
                    topic_id=topic_id,
                    topic_name=topic_name,
                    question_text=gq.question_text,
                    question_type="mcq",
                    options=gq.options,
                    correct_answer=gq.correct_answer,
                    difficulty=gq.difficulty,
                    points=self._get_points_for_difficulty(gq.difficulty)
                ))
            
            return questions
            
        except Exception as e:
            logger.error(f"[ASSESSMENT] Question generation failed: {e}")
            return []
    
    def _get_points_for_difficulty(self, difficulty: str) -> int:
        """Get point value based on difficulty"""
        return {
            "easy": 1,
            "medium": 2, 
            "hard": 3
        }.get(difficulty.lower(), 1)
    
    async def evaluate_responses(
        self,
        quiz: DiagnosticQuiz,
        responses: Dict[str, str]
    ) -> List[AssessmentResult]:
        """
        Evaluate student responses and calculate mastery per topic.
        
        Args:
            quiz: The DiagnosticQuiz that was taken
            responses: Dict mapping question_id -> student_answer
            
        Returns:
            List of AssessmentResult, one per topic
        """
        # Group questions by topic
        topic_questions: Dict[str, List[DiagnosticQuestion]] = {}
        for q in quiz.questions:
            if q.topic_id not in topic_questions:
                topic_questions[q.topic_id] = []
            topic_questions[q.topic_id].append(q)
        
        results = []
        
        for topic_id, questions in topic_questions.items():
            topic_name = questions[0].topic_name if questions else "Unknown"
            
            correct = 0
            total = len(questions)
            earned_points = 0
            max_points = sum(q.points for q in questions)
            weak_areas = []
            strong_areas = []
            
            for q in questions:
                student_answer = responses.get(q.id, "").strip().upper()
                correct_normalized = q.correct_answer.strip().upper()
                
                if student_answer == correct_normalized:
                    correct += 1
                    earned_points += q.points
                    strong_areas.append(q.question_text[:50] + "...")
                else:
                    weak_areas.append(q.question_text[:50] + "...")
            
            # Calculate mastery score (weighted by points)
            mastery = earned_points / max_points if max_points > 0 else 0.0
            
            # Determine knowledge level
            if mastery >= 0.8:
                level = "advanced"
            elif mastery >= 0.5:
                level = "intermediate"
            else:
                level = "beginner"
            
            results.append(AssessmentResult(
                topic_id=topic_id,
                topic_name=topic_name,
                total_questions=total,
                correct_answers=correct,
                mastery_score=round(mastery, 2),
                knowledge_level=level,
                weak_areas=weak_areas[:3],  # Top 3 weak areas
                strong_areas=strong_areas[:3],
                time_taken_seconds=0  # Would be tracked on frontend
            ))
        
        logger.info(f"[ASSESSMENT] Evaluated {len(results)} topic results")
        return results
    
    async def quick_assessment(
        self,
        topics: List[Dict[str, Any]],
        assumed_level: str = "beginner"
    ) -> Dict[str, float]:
        """
        Quick assessment without quiz - assume knowledge level based on context.
        
        Used when:
        - Student skips diagnostic quiz
        - No prior data available
        - Fast curriculum generation needed
        
        Args:
            topics: List of topics
            assumed_level: "beginner", "intermediate", "advanced"
            
        Returns:
            Dict mapping topic_id -> assumed mastery score
        """
        default_mastery = {
            "beginner": 0.0,
            "intermediate": 0.4,
            "advanced": 0.7
        }.get(assumed_level, 0.0)
        
        return {
            topic.get("id", topic.get("name")): default_mastery
            for topic in topics
        }
    
    def calculate_adjusted_hours(
        self,
        base_hours: float,
        mastery_score: float
    ) -> float:
        """
        Adjust estimated study hours based on existing mastery.
        
        If student already knows 80% of the topic, they need less time.
        If student is a beginner, they might need more time.
        """
        # Scale factor: 1.0 for beginner (0% mastery), 0.2 for advanced (100%)
        # This means advanced students study at 20% of base time
        scale = 1.0 - (mastery_score * 0.8)
        
        adjusted = base_hours * scale
        
        # Minimum 30 minutes for any topic (review time)
        return max(adjusted, 0.5)
    
    def prioritize_topics(
        self,
        topics: List[Dict[str, Any]],
        assessment_results: List[AssessmentResult]
    ) -> List[Dict[str, Any]]:
        """
        Reorder topics to prioritize weak areas.
        
        Lower mastery topics are prioritized higher (more focus needed).
        """
        # Create mastery lookup
        mastery_lookup = {
            r.topic_id: r.mastery_score
            for r in assessment_results
        }
        
        # Add mastery score to topics
        for topic in topics:
            topic_id = topic.get("id", topic.get("name"))
            topic["mastery"] = mastery_lookup.get(topic_id, 0.0)
        
        # Sort by mastery (ascending - weakest first)
        # But respect dependencies - this is handled by the curriculum agent
        return sorted(topics, key=lambda t: t.get("mastery", 0.0))


# ============================================================================
# Singleton
# ============================================================================

_assessment_service: Optional[KnowledgeAssessmentService] = None


def get_assessment_service() -> KnowledgeAssessmentService:
    """Get singleton assessment service"""
    global _assessment_service
    if _assessment_service is None:
        _assessment_service = KnowledgeAssessmentService()
    return _assessment_service
