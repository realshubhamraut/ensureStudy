"""
Interview Answer Evaluator Service

Evaluates user answers against interview questions using LLM.
Provides scoring, feedback, and identification of key points covered.
"""

import logging
import re
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Result from answer evaluation."""
    score: float  # 0-100
    feedback: str  # Overall feedback
    key_points_covered: List[str]  # What the user got right
    key_points_missed: List[str]  # What the user missed
    clarity_score: float  # How clear/articulate the answer was
    relevance_score: float  # How relevant to the question
    completeness_score: float  # How complete the answer is
    suggestions: List[str]  # Improvement suggestions
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": round(self.score, 1),
            "feedback": self.feedback,
            "key_points_covered": self.key_points_covered,
            "key_points_missed": self.key_points_missed,
            "clarity_score": round(self.clarity_score, 1),
            "relevance_score": round(self.relevance_score, 1),
            "completeness_score": round(self.completeness_score, 1),
            "suggestions": self.suggestions,
            "breakdown": {
                "clarity": round(self.clarity_score, 1),
                "relevance": round(self.relevance_score, 1),
                "completeness": round(self.completeness_score, 1)
            },
            "timestamp": self.timestamp
        }


class InterviewEvaluator:
    """
    Evaluates interview answers using LLM.
    
    Provides:
    - Score calculation (0-100)
    - Key points identification
    - Feedback generation
    - Improvement suggestions
    """
    
    def __init__(self):
        self._groq_client = None
        logger.info("[InterviewEvaluator] Initialized")
    
    @property
    def groq_client(self):
        """Lazy load Groq client."""
        if self._groq_client is None:
            try:
                from groq import Groq
                import os
                self._groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
                logger.info("[InterviewEvaluator] Groq client initialized")
            except Exception as e:
                logger.error(f"[InterviewEvaluator] Failed to init Groq: {e}")
                self._groq_client = None
        return self._groq_client
    
    async def evaluate_answer(
        self,
        question: str,
        user_answer: str,
        expected_answer: Optional[str] = None,
        subject: Optional[str] = None,
        difficulty: str = "medium"
    ) -> EvaluationResult:
        """
        Evaluate a user's answer to an interview question.
        
        Args:
            question: The interview question asked
            user_answer: The user's transcribed answer
            expected_answer: Optional expected/ideal answer
            subject: Optional subject context (e.g., "physics", "general")
            difficulty: Question difficulty level
            
        Returns:
            EvaluationResult with scores and feedback
        """
        logger.info(f"[InterviewEvaluator] Evaluating answer for question: {question[:50]}...")
        logger.info(f"[InterviewEvaluator] User answer length: {len(user_answer)} chars")
        
        # Handle empty or very short answers
        if not user_answer or len(user_answer.strip()) < 10:
            logger.warning("[InterviewEvaluator] Answer too short")
            return EvaluationResult(
                score=20.0,
                feedback="Your answer was too brief. Try to provide more detail and explanation.",
                key_points_covered=[],
                key_points_missed=["Detailed explanation", "Examples", "Clear reasoning"],
                clarity_score=30.0,
                relevance_score=20.0,
                completeness_score=10.0,
                suggestions=[
                    "Provide a more detailed response",
                    "Include specific examples",
                    "Explain your reasoning step by step"
                ]
            )
        
        # Try LLM evaluation
        if self.groq_client:
            try:
                result = await self._evaluate_with_llm(
                    question, user_answer, expected_answer, subject, difficulty
                )
                logger.info(f"[InterviewEvaluator] LLM evaluation score: {result.score}")
                return result
            except Exception as e:
                logger.error(f"[InterviewEvaluator] LLM evaluation failed: {e}")
        
        # Fallback to heuristic evaluation
        logger.info("[InterviewEvaluator] Using heuristic evaluation")
        return self._evaluate_heuristic(question, user_answer)
    
    async def _evaluate_with_llm(
        self,
        question: str,
        user_answer: str,
        expected_answer: Optional[str],
        subject: Optional[str],
        difficulty: str
    ) -> EvaluationResult:
        """Evaluate using Groq LLM."""
        
        context = f"Subject: {subject}" if subject else "General interview"
        expected_context = f"\n\nExpected answer for reference:\n{expected_answer}" if expected_answer else ""
        
        prompt = f"""You are an expert interview evaluator. Evaluate the following interview answer.

Context: {context}
Difficulty: {difficulty}

Question: {question}

User's Answer: {user_answer}
{expected_context}

Evaluate the answer on these criteria (score each 0-100):
1. CLARITY: How clear and well-articulated is the answer?
2. RELEVANCE: How relevant is the answer to the question asked?
3. COMPLETENESS: How complete and thorough is the answer?

Also identify:
- KEY_POINTS_COVERED: What important points did the user cover? (list 2-4 items)
- KEY_POINTS_MISSED: What important points did the user miss? (list 2-4 items)
- SUGGESTIONS: How can the user improve? (list 2-3 specific suggestions)

Respond in this exact format:
CLARITY_SCORE: [number]
RELEVANCE_SCORE: [number]
COMPLETENESS_SCORE: [number]
OVERALL_FEEDBACK: [one paragraph feedback]
KEY_POINTS_COVERED: [point1] | [point2] | [point3]
KEY_POINTS_MISSED: [point1] | [point2]
SUGGESTIONS: [suggestion1] | [suggestion2] | [suggestion3]"""

        response = self.groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=800
        )
        
        text = response.choices[0].message.content
        logger.debug(f"[InterviewEvaluator] LLM response: {text[:200]}...")
        
        # Parse response
        return self._parse_llm_response(text)
    
    def _parse_llm_response(self, text: str) -> EvaluationResult:
        """Parse LLM response into EvaluationResult."""
        
        def extract_value(pattern: str, default: str = "") -> str:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            return match.group(1).strip() if match else default
        
        def extract_score(pattern: str, default: float = 70.0) -> float:
            value = extract_value(pattern, str(default))
            try:
                return min(100, max(0, float(value)))
            except:
                return default
        
        clarity = extract_score(r"CLARITY_SCORE:\s*(\d+(?:\.\d+)?)")
        relevance = extract_score(r"RELEVANCE_SCORE:\s*(\d+(?:\.\d+)?)")
        completeness = extract_score(r"COMPLETENESS_SCORE:\s*(\d+(?:\.\d+)?)")
        
        # Calculate overall score (weighted average)
        overall = (clarity * 0.3 + relevance * 0.4 + completeness * 0.3)
        
        feedback = extract_value(r"OVERALL_FEEDBACK:\s*(.+?)(?=KEY_POINTS|$)", "Good attempt. Keep practicing!")
        
        covered_str = extract_value(r"KEY_POINTS_COVERED:\s*(.+?)(?=KEY_POINTS_MISSED|$)")
        covered = [p.strip() for p in covered_str.split("|") if p.strip()][:4]
        
        missed_str = extract_value(r"KEY_POINTS_MISSED:\s*(.+?)(?=SUGGESTIONS|$)")
        missed = [p.strip() for p in missed_str.split("|") if p.strip()][:4]
        
        suggestions_str = extract_value(r"SUGGESTIONS:\s*(.+?)$")
        suggestions = [s.strip() for s in suggestions_str.split("|") if s.strip()][:3]
        
        return EvaluationResult(
            score=overall,
            feedback=feedback,
            key_points_covered=covered if covered else ["Addressed the question"],
            key_points_missed=missed if missed else [],
            clarity_score=clarity,
            relevance_score=relevance,
            completeness_score=completeness,
            suggestions=suggestions if suggestions else ["Continue practicing for more confidence"]
        )
    
    def _evaluate_heuristic(self, question: str, user_answer: str) -> EvaluationResult:
        """Fallback heuristic evaluation when LLM is unavailable."""
        
        # Basic metrics
        word_count = len(user_answer.split())
        sentence_count = len(re.findall(r'[.!?]+', user_answer)) or 1
        avg_sentence_length = word_count / sentence_count
        
        # Clarity score based on structure
        clarity = 70.0
        if word_count > 30:
            clarity += 10
        if avg_sentence_length > 5 and avg_sentence_length < 25:
            clarity += 10
        if any(word in user_answer.lower() for word in ["because", "therefore", "for example", "such as"]):
            clarity += 10
        
        # Relevance - check if keywords from question appear in answer
        question_words = set(question.lower().split()) - {"the", "a", "an", "is", "are", "what", "how", "why", "can", "you"}
        answer_words = set(user_answer.lower().split())
        overlap = len(question_words & answer_words) / max(len(question_words), 1)
        relevance = min(100, 50 + overlap * 50)
        
        # Completeness based on length
        completeness = min(100, 40 + (word_count / 50) * 30)
        
        # Overall score
        overall = clarity * 0.3 + relevance * 0.4 + completeness * 0.3
        
        # Generate feedback
        if overall >= 80:
            feedback = "Excellent answer! You covered the topic well with clear explanations."
        elif overall >= 60:
            feedback = "Good answer. Consider adding more specific examples or details."
        elif overall >= 40:
            feedback = "Your answer addresses the question but could be more complete. Try to elaborate more."
        else:
            feedback = "Try to provide more detail and stay focused on the question asked."
        
        return EvaluationResult(
            score=overall,
            feedback=feedback,
            key_points_covered=["Addressed the main topic"] if relevance > 50 else [],
            key_points_missed=["More specific examples", "Deeper explanation"],
            clarity_score=clarity,
            relevance_score=relevance,
            completeness_score=completeness,
            suggestions=[
                "Include specific examples to support your points",
                "Explain concepts in more detail",
                "Structure your answer with a clear beginning, middle, and end"
            ]
        )


# Singleton instance
_evaluator = None


def get_interview_evaluator() -> InterviewEvaluator:
    """Get singleton InterviewEvaluator instance."""
    global _evaluator
    if _evaluator is None:
        _evaluator = InterviewEvaluator()
    return _evaluator
