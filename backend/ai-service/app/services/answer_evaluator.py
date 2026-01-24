"""
Answer Evaluator Service
========================

Evaluates student answers using LLM for semantic understanding.
Supports MCQ validation, descriptive answer scoring, and constructive feedback.

Technologies:
- HuggingFace LLM for semantic evaluation
- Keyword matching for preliminary scoring
- Rubric-based evaluation for consistency
"""

import os
import re
import logging
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Structured evaluation result"""
    question_id: str
    question_type: str
    is_correct: bool  # For MCQ
    score: float  # Normalized 0-10
    max_score: float
    correctness_score: float
    relevance_score: float
    completeness_score: float
    feedback: str
    improvements: str
    topics_to_study: List[str]
    time_taken_seconds: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AnswerEvaluatorService:
    """
    Service for evaluating student answers.
    
    Features:
    - MCQ validation (instant, no LLM needed)
    - Descriptive answer evaluation using LLM
    - Keyword matching for preliminary checks
    - Rubric-based scoring
    - Constructive feedback generation
    """
    
    # Scoring weights
    CORRECTNESS_WEIGHT = 0.5
    RELEVANCE_WEIGHT = 0.3
    COMPLETENESS_WEIGHT = 0.2
    
    # Stop words for keyword matching
    STOP_WORDS = {
        'this', 'that', 'with', 'from', 'have', 'been', 'were', 'they', 'their',
        'what', 'when', 'where', 'which', 'there', 'these', 'those', 'about',
        'would', 'could', 'should', 'into', 'more', 'some', 'such', 'than',
        'then', 'them', 'will', 'also', 'only', 'other', 'being', 'because',
        'answer', 'question', 'following', 'given', 'above', 'below'
    }
    
    def __init__(self):
        """Initialize the answer evaluator"""
        self.api_key = os.getenv("HUGGINGFACE_API_KEY", "")
        self.model_name = os.getenv("LLM_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")
        
        self._llm_client = None
        
        logger.info(f"[AnswerEvaluator] Initialized with model: {self.model_name}")
    
    @property
    def llm_client(self):
        """Lazy load HuggingFace inference client"""
        if self._llm_client is None:
            try:
                from huggingface_hub import InferenceClient
                self._llm_client = InferenceClient(token=self.api_key)
                logger.info("[AnswerEvaluator] LLM client initialized")
            except Exception as e:
                logger.error(f"[AnswerEvaluator] Failed to init LLM client: {e}")
                raise
        return self._llm_client
    
    def evaluate_mcq(
        self,
        question_id: str,
        student_answer: str,
        correct_answer: str,
        time_taken_seconds: Optional[int] = None
    ) -> EvaluationResult:
        """
        Evaluate an MCQ answer (simple comparison).
        
        Args:
            question_id: Question identifier
            student_answer: Student's selected option (A, B, C, D)
            correct_answer: Correct option
            time_taken_seconds: Time taken to answer
            
        Returns:
            EvaluationResult with is_correct flag
        """
        # Normalize answers
        student_answer = student_answer.strip().upper() if student_answer else ""
        correct_answer = correct_answer.strip().upper() if correct_answer else ""
        
        is_correct = student_answer == correct_answer
        score = 10.0 if is_correct else 0.0
        
        return EvaluationResult(
            question_id=question_id,
            question_type="mcq",
            is_correct=is_correct,
            score=score,
            max_score=10.0,
            correctness_score=score,
            relevance_score=score,
            completeness_score=score,
            feedback="Correct!" if is_correct else f"Incorrect. The correct answer is {correct_answer}.",
            improvements="" if is_correct else "Review this topic to understand why this answer is correct.",
            topics_to_study=[],
            time_taken_seconds=time_taken_seconds
        )
    
    def _extract_keywords(self, text: str) -> set:
        """Extract meaningful keywords from text"""
        words = set(
            word.lower() 
            for word in re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        )
        return words - self.STOP_WORDS
    
    def _calculate_keyword_match(
        self,
        student_answer: str,
        reference_content: str
    ) -> tuple:
        """
        Calculate keyword overlap between answer and reference.
        
        Returns:
            (match_percentage, matching_keywords, missing_keywords)
        """
        answer_keywords = self._extract_keywords(student_answer)
        reference_keywords = self._extract_keywords(reference_content)
        
        matching = answer_keywords.intersection(reference_keywords)
        missing = reference_keywords - answer_keywords
        
        if not answer_keywords:
            return 0.0, set(), reference_keywords
        
        match_pct = len(matching) / len(answer_keywords) * 100
        return match_pct, matching, missing
    
    def _create_evaluation_prompt(
        self,
        question: str,
        student_answer: str,
        reference_content: str,
        key_points: Optional[List[str]] = None
    ) -> str:
        """Create prompt for LLM evaluation"""
        key_points_text = ""
        if key_points:
            key_points_text = f"\nKEY POINTS TO COVER:\n" + "\n".join(f"- {kp}" for kp in key_points)
        
        return f"""You are a STRICT educational evaluator. Evaluate the student's answer and provide specific feedback.

QUESTION: {question}

REFERENCE CONTENT:
{reference_content[:2500]}
{key_points_text}

STUDENT'S ANSWER:
{student_answer}

SCORING RUBRIC (0-10 each):
- CORRECTNESS: 0=completely wrong, 5=partially correct, 10=fully correct
- RELEVANCE: 0=off-topic, 5=partially relevant, 10=fully relevant
- COMPLETENESS: 0=missing all points, 5=some points covered, 10=all points covered

IMPORTANT: Be STRICT but fair. Partial credit for partial answers.

Provide your evaluation in this EXACT format:
CORRECTNESS_SCORE: [0-10]
RELEVANCE_SCORE: [0-10]
COMPLETENESS_SCORE: [0-10]
OVERALL_SCORE: [0-10]
FEEDBACK: [What was correct and what was wrong, be specific]
IMPROVEMENTS: [Specific suggestions for improvement, reference the question]
TOPICS_TO_STUDY: [2-3 comma-separated topics to review]

Evaluate now:"""
    
    def _parse_llm_evaluation(self, response: str) -> Dict[str, Any]:
        """Parse LLM evaluation response"""
        result = {
            "correctness_score": 5.0,
            "relevance_score": 5.0,
            "completeness_score": 5.0,
            "overall_score": 5.0,
            "feedback": "",
            "improvements": "",
            "topics_to_study": []
        }
        
        # Extract numeric scores
        score_patterns = {
            "correctness_score": r'CORRECTNESS_SCORE:\s*(\d+(?:\.\d+)?)',
            "relevance_score": r'RELEVANCE_SCORE:\s*(\d+(?:\.\d+)?)',
            "completeness_score": r'COMPLETENESS_SCORE:\s*(\d+(?:\.\d+)?)',
            "overall_score": r'OVERALL_SCORE:\s*(\d+(?:\.\d+)?)'
        }
        
        for key, pattern in score_patterns.items():
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                score = float(match.group(1))
                result[key] = min(max(score, 0), 10)  # Clamp to 0-10
        
        # Extract text fields
        text_patterns = {
            "feedback": r'FEEDBACK:\s*(.+?)(?=IMPROVEMENTS:|$)',
            "improvements": r'IMPROVEMENTS:\s*(.+?)(?=TOPICS_TO_STUDY:|$)',
            "topics_to_study": r'TOPICS_TO_STUDY:\s*(.+?)$'
        }
        
        for key, pattern in text_patterns.items():
            match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
            if match:
                text = match.group(1).strip()
                if key == "topics_to_study":
                    result[key] = [t.strip() for t in text.split(',') if t.strip()]
                else:
                    result[key] = text
        
        return result
    
    async def evaluate_descriptive(
        self,
        question_id: str,
        question_text: str,
        student_answer: str,
        reference_content: str,
        key_points: Optional[List[str]] = None,
        time_taken_seconds: Optional[int] = None
    ) -> EvaluationResult:
        """
        Evaluate a descriptive answer using LLM.
        
        Args:
            question_id: Question identifier
            question_text: The question asked
            student_answer: Student's answer text
            reference_content: Reference content for verification
            key_points: Expected key points in the answer
            time_taken_seconds: Time taken to answer
            
        Returns:
            EvaluationResult with detailed feedback
        """
        # Handle empty answer
        if not student_answer or not student_answer.strip():
            return EvaluationResult(
                question_id=question_id,
                question_type="descriptive",
                is_correct=False,
                score=0.0,
                max_score=10.0,
                correctness_score=0.0,
                relevance_score=0.0,
                completeness_score=0.0,
                feedback="No answer provided. Please attempt to answer the question.",
                improvements="Review the topic and provide a complete answer covering the key points.",
                topics_to_study=["Review the entire topic"],
                time_taken_seconds=time_taken_seconds
            )
        
        # Pre-check: Keyword matching
        match_pct, matching_kw, missing_kw = self._calculate_keyword_match(
            student_answer, reference_content
        )
        
        logger.info(f"[AnswerEvaluator] Keyword match: {match_pct:.1f}%")
        
        # If NO keywords match at all, return 0 immediately
        if len(matching_kw) == 0 and len(self._extract_keywords(student_answer)) > 2:
            logger.warning("[AnswerEvaluator] Zero keyword match - returning 0 score")
            
            key_terms = list(self._extract_keywords(reference_content))[:5]
            
            return EvaluationResult(
                question_id=question_id,
                question_type="descriptive",
                is_correct=False,
                score=0.0,
                max_score=10.0,
                correctness_score=0.0,
                relevance_score=0.0,
                completeness_score=0.0,
                feedback="Your answer does not contain any relevant information from the topic. The answer appears to be completely unrelated to the question.",
                improvements=f"To answer this question, your answer should include key concepts such as: {', '.join(key_terms)}. Please re-read the topic and focus on the specific question being asked.",
                topics_to_study=key_terms[:3],
                time_taken_seconds=time_taken_seconds
            )
        
        try:
            # Create evaluation prompt
            prompt = self._create_evaluation_prompt(
                question_text, student_answer, reference_content, key_points
            )
            
            # Call LLM
            logger.info("[AnswerEvaluator] Calling LLM for evaluation...")
            
            messages = [
                {"role": "system", "content": "You are a strict but fair educational evaluator. Provide detailed, constructive feedback."},
                {"role": "user", "content": prompt}
            ]
            
            response = self.llm_client.chat_completion(
                messages=messages,
                model=self.model_name,
                max_tokens=800,
                temperature=0.3  # Lower temperature for consistency
            )
            
            response_text = response.choices[0].message.content or ""
            logger.info(f"[AnswerEvaluator] Received LLM response ({len(response_text)} chars)")
            
            # Parse evaluation
            parsed = self._parse_llm_evaluation(response_text)
            
            # Calculate weighted overall score if not provided well
            if parsed["overall_score"] == 5.0:
                parsed["overall_score"] = round(
                    parsed["correctness_score"] * self.CORRECTNESS_WEIGHT +
                    parsed["relevance_score"] * self.RELEVANCE_WEIGHT +
                    parsed["completeness_score"] * self.COMPLETENESS_WEIGHT,
                    1
                )
            
            return EvaluationResult(
                question_id=question_id,
                question_type="descriptive",
                is_correct=parsed["overall_score"] >= 7.0,
                score=parsed["overall_score"],
                max_score=10.0,
                correctness_score=parsed["correctness_score"],
                relevance_score=parsed["relevance_score"],
                completeness_score=parsed["completeness_score"],
                feedback=parsed["feedback"],
                improvements=parsed["improvements"],
                topics_to_study=parsed["topics_to_study"],
                time_taken_seconds=time_taken_seconds
            )
            
        except Exception as e:
            logger.error(f"[AnswerEvaluator] Error in LLM evaluation: {e}")
            import traceback
            traceback.print_exc()
            
            # Fallback to keyword-based scoring
            score = min(match_pct / 10, 10)  # Convert to 0-10 scale
            
            return EvaluationResult(
                question_id=question_id,
                question_type="descriptive",
                is_correct=score >= 7.0,
                score=score,
                max_score=10.0,
                correctness_score=score,
                relevance_score=score,
                completeness_score=score,
                feedback=f"Keyword-based evaluation: {match_pct:.0f}% relevant terms found.",
                improvements="Could not perform detailed evaluation. Please review your answer.",
                topics_to_study=["Review the topic material"],
                time_taken_seconds=time_taken_seconds
            )
    
    async def evaluate_batch(
        self,
        answers: List[Dict[str, Any]],
        reference_content: str
    ) -> List[EvaluationResult]:
        """
        Evaluate multiple answers.
        
        Args:
            answers: List of dicts with 'question_id', 'question_type', 'question_text',
                    'student_answer', 'correct_answer' (for MCQ), 'key_points'
            reference_content: Common reference content
            
        Returns:
            List of EvaluationResult objects
        """
        results = []
        
        for i, ans in enumerate(answers):
            logger.info(f"[AnswerEvaluator] Evaluating answer {i+1}/{len(answers)}...")
            
            if ans.get("question_type") == "mcq":
                result = self.evaluate_mcq(
                    question_id=ans.get("question_id", str(i+1)),
                    student_answer=ans.get("student_answer", ""),
                    correct_answer=ans.get("correct_answer", ""),
                    time_taken_seconds=ans.get("time_taken_seconds")
                )
            else:
                result = await self.evaluate_descriptive(
                    question_id=ans.get("question_id", str(i+1)),
                    question_text=ans.get("question_text", ""),
                    student_answer=ans.get("student_answer", ""),
                    reference_content=reference_content,
                    key_points=ans.get("key_points"),
                    time_taken_seconds=ans.get("time_taken_seconds")
                )
            
            results.append(result)
        
        return results
    
    def calculate_total_score(
        self,
        evaluations: List[EvaluationResult]
    ) -> Dict[str, Any]:
        """
        Calculate total and average scores from evaluations.
        
        Returns:
            Summary statistics dictionary
        """
        if not evaluations:
            return {
                "total_score": 0.0,
                "max_possible": 0.0,
                "percentage": 0.0,
                "average_score": 0.0,
                "num_questions": 0,
                "num_correct": 0,
                "num_incorrect": 0
            }
        
        total = sum(e.score for e in evaluations)
        max_possible = sum(e.max_score for e in evaluations)
        num_correct = sum(1 for e in evaluations if e.is_correct)
        
        return {
            "total_score": round(total, 1),
            "max_possible": max_possible,
            "percentage": round((total / max_possible) * 100, 1) if max_possible > 0 else 0,
            "average_score": round(total / len(evaluations), 1),
            "num_questions": len(evaluations),
            "num_correct": num_correct,
            "num_incorrect": len(evaluations) - num_correct
        }


# Singleton instance
_evaluator_instance = None


def get_answer_evaluator() -> AnswerEvaluatorService:
    """Get singleton instance of AnswerEvaluatorService"""
    global _evaluator_instance
    if _evaluator_instance is None:
        _evaluator_instance = AnswerEvaluatorService()
    return _evaluator_instance
