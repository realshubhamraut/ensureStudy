"""
Assignment Grading Service
AI-powered automatic grading for student submissions

Pipeline:
1. Parse teacher's assignment PDF → Extract questions
2. Parse student's submission PDF → Extract answers (OCR if handwritten)
3. Match answers to questions semantically
4. Score each answer based on relevance and correctness
5. Generate feedback and overall grade

Uses FREE, LOCAL models:
- sentence-transformers for semantic matching
- FLAN-T5 or rule-based scoring
"""
import os
import asyncio
import aiohttp
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime

from .pdf_extractor import pdf_extractor, Question, ExtractedContent

# Embeddings for semantic matching
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
except ImportError:
    SentenceTransformer = None
    np = None
    print("[Grading] sentence-transformers not installed")


@dataclass
class AnswerGrade:
    """Grade for a single answer"""
    question_number: int
    question_text: str
    student_answer: str
    points_earned: float
    max_points: float
    percentage: float
    feedback: str
    confidence: float


@dataclass
class GradingResult:
    """Complete grading result for a submission"""
    submission_id: str
    assignment_id: str
    total_grade: float
    max_points: float
    percentage: float
    overall_feedback: str
    question_grades: List[AnswerGrade] = field(default_factory=list)
    confidence: float = 0.0
    graded_at: str = ""
    ocr_used: bool = False
    error: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            'submission_id': self.submission_id,
            'assignment_id': self.assignment_id,
            'total_grade': round(self.total_grade, 1),
            'max_points': self.max_points,
            'percentage': round(self.percentage, 1),
            'overall_feedback': self.overall_feedback,
            'question_grades': [
                {
                    'question_number': g.question_number,
                    'question_text': g.question_text[:200],
                    'student_answer': g.student_answer[:500],
                    'points_earned': round(g.points_earned, 1),
                    'max_points': g.max_points,
                    'percentage': round(g.percentage, 1),
                    'feedback': g.feedback,
                    'confidence': round(g.confidence, 2)
                }
                for g in self.question_grades
            ],
            'confidence': round(self.confidence, 2),
            'graded_at': self.graded_at,
            'ocr_used': self.ocr_used,
            'error': self.error
        }


class AssignmentGradingService:
    """
    AI-powered assignment grading service.
    
    Uses semantic similarity to match answers to questions,
    and rule-based + LLM scoring for grading.
    """
    
    def __init__(self):
        self._model: Optional[SentenceTransformer] = None
        self.callback_url = os.getenv('CORE_SERVICE_URL', 'http://localhost:9000') + '/api/grading/callback'
    
    @property
    def embedding_model(self) -> Optional[SentenceTransformer]:
        """Lazy-load embedding model"""
        if self._model is None and SentenceTransformer is not None:
            print("[Grading] Loading embedding model...")
            self._model = SentenceTransformer('all-mpnet-base-v2')
        return self._model
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between two texts"""
        if self.embedding_model is None or np is None:
            return 0.5  # Default if model not available
        
        embeddings = self.embedding_model.encode([text1, text2], normalize_embeddings=True)
        similarity = float(np.dot(embeddings[0], embeddings[1]))
        return max(0.0, min(1.0, similarity))
    
    def extract_answers_by_question(self, text: str, questions: List[Question]) -> Dict[int, str]:
        """
        Extract student answers matching question numbers.
        
        Looks for patterns like:
        - 1. Answer text
        - Q1: Answer text
        - Answer 1: text
        """
        import re
        
        answers = {}
        
        # Split text into potential answer blocks
        # Try to find numbered sections
        blocks = re.split(r'\n(?=\s*(?:Q|Question|Answer|Ans)?\s*\d+\s*[.)::])', text, flags=re.IGNORECASE)
        
        for block in blocks:
            # Extract question number from block start
            match = re.match(r'\s*(?:Q|Question|Answer|Ans)?\s*(\d+)\s*[.)::]\s*(.+)', block, re.IGNORECASE | re.DOTALL)
            if match:
                q_num = int(match.group(1))
                answer_text = match.group(2).strip()
                if q_num <= len(questions) and len(answer_text) > 5:
                    answers[q_num] = answer_text[:2000]  # Limit answer length
        
        # If no structured answers found, try semantic matching
        if not answers and questions:
            # Split by paragraphs and match semantically
            paragraphs = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 20]
            
            for i, question in enumerate(questions):
                best_match = ""
                best_score = 0.3  # Minimum threshold
                
                for para in paragraphs:
                    # Simple keyword matching as fallback
                    q_words = set(question.text.lower().split())
                    a_words = set(para.lower().split())
                    overlap = len(q_words & a_words) / max(len(q_words), 1)
                    
                    if overlap > best_score:
                        best_score = overlap
                        best_match = para
                
                if best_match:
                    answers[question.number] = best_match[:2000]
        
        return answers
    
    def grade_answer(
        self,
        question: Question,
        answer: str,
        max_points: float
    ) -> AnswerGrade:
        """
        Grade a single answer against a question.
        
        Scoring rubric:
        - 100%: Complete, relevant answer with key concepts
        - 75%: Mostly correct, minor gaps
        - 50%: Partial answer, some relevant content
        - 25%: Minimal relevance, attempted
        - 0%: No answer or completely irrelevant
        """
        if not answer or len(answer.strip()) < 5:
            return AnswerGrade(
                question_number=question.number,
                question_text=question.text,
                student_answer="[No answer provided]",
                points_earned=0.0,
                max_points=max_points,
                percentage=0.0,
                feedback="No answer was provided for this question.",
                confidence=1.0
            )
        
        # Compute semantic similarity
        similarity = self.compute_similarity(question.text, answer)
        
        # Length-based bonus (longer answers often more complete)
        length_factor = min(1.0, len(answer) / 200)  # Bonus up to 200 chars
        
        # Combined score
        raw_score = (similarity * 0.7) + (length_factor * 0.3)
        
        # Map to rubric
        if raw_score >= 0.7:
            percentage = 100.0
            feedback = "Excellent answer! Comprehensive and relevant response."
        elif raw_score >= 0.55:
            percentage = 75.0
            feedback = "Good answer with minor areas for improvement."
        elif raw_score >= 0.4:
            percentage = 50.0
            feedback = "Partial answer. Some key concepts are present but incomplete."
        elif raw_score >= 0.25:
            percentage = 25.0
            feedback = "Limited response. The answer needs more detail and relevance."
        else:
            percentage = 0.0
            feedback = "The answer does not adequately address the question."
        
        points_earned = (percentage / 100.0) * max_points
        
        # Confidence based on how clear the matching was
        confidence = min(1.0, similarity + 0.3) if similarity > 0.3 else 0.5
        
        return AnswerGrade(
            question_number=question.number,
            question_text=question.text,
            student_answer=answer[:500],
            points_earned=points_earned,
            max_points=max_points,
            percentage=percentage,
            feedback=feedback,
            confidence=confidence
        )
    
    async def grade_submission(
        self,
        assignment_id: str,
        submission_id: str,
        teacher_pdf_url: str,
        student_pdf_urls: List[str],
        max_points: int = 100,
        classroom_id: Optional[str] = None,
        student_id: Optional[str] = None
    ) -> GradingResult:
        """
        Main grading pipeline.
        
        1. Extract questions from teacher PDF
        2. Extract answers from student submission(s)
        3. Match and grade each answer
        4. Calculate total grade
        5. Send callback to core service
        """
        print(f"[Grading] Starting grading for submission {submission_id}")
        graded_at = datetime.utcnow().isoformat()
        
        try:
            # Step 1: Extract questions from teacher's assignment
            print(f"[Grading] Extracting questions from teacher PDF...")
            teacher_content = await pdf_extractor.extract_from_url(teacher_pdf_url, use_ocr=False)
            questions = teacher_content.questions
            
            if not questions:
                # If no structured questions found, treat entire document as one question
                questions = [Question(
                    number=1,
                    text=teacher_content.full_text[:500] or "Complete the assignment",
                    points=max_points
                )]
            
            print(f"[Grading] Found {len(questions)} questions")
            
            # Step 2: Extract answers from student submission(s)
            print(f"[Grading] Extracting answers from {len(student_pdf_urls)} student file(s)...")
            student_text = ""
            ocr_used = False
            
            for pdf_url in student_pdf_urls:
                content = await pdf_extractor.extract_from_url(pdf_url, use_ocr=True)
                student_text += "\n\n" + content.full_text
                if content.ocr_used:
                    ocr_used = True
            
            student_text = student_text.strip()
            print(f"[Grading] Extracted {len(student_text)} chars from student submission, OCR={ocr_used}")
            
            if not student_text:
                return GradingResult(
                    submission_id=submission_id,
                    assignment_id=assignment_id,
                    total_grade=0.0,
                    max_points=float(max_points),
                    percentage=0.0,
                    overall_feedback="Could not extract any text from the submission. Please ensure the PDF is readable.",
                    confidence=0.5,
                    graded_at=graded_at,
                    ocr_used=ocr_used,
                    error="No text extracted from submission"
                )
            
            # Step 3: Match answers to questions
            answers = self.extract_answers_by_question(student_text, questions)
            print(f"[Grading] Matched {len(answers)} answers to questions")
            
            # Step 4: Grade each answer
            points_per_question = max_points / len(questions)
            question_grades = []
            total_earned = 0.0
            total_confidence = 0.0
            
            for question in questions:
                q_points = question.points or points_per_question
                answer = answers.get(question.number, "")
                
                grade = self.grade_answer(question, answer, q_points)
                question_grades.append(grade)
                total_earned += grade.points_earned
                total_confidence += grade.confidence
            
            # Step 5: Calculate overall grade
            overall_percentage = (total_earned / max_points * 100) if max_points > 0 else 0
            avg_confidence = total_confidence / len(questions) if questions else 0.5
            
            # Generate overall feedback
            if overall_percentage >= 90:
                overall_feedback = "Excellent work! Your submission demonstrates strong understanding of the material."
            elif overall_percentage >= 75:
                overall_feedback = "Good job! Your answers show solid comprehension with room for minor improvements."
            elif overall_percentage >= 60:
                overall_feedback = "Satisfactory work. Review the feedback on individual questions to improve."
            elif overall_percentage >= 40:
                overall_feedback = "Your submission needs improvement. Please review the material and consider the feedback provided."
            else:
                overall_feedback = "Significant improvement needed. Please review all topics and consider seeking help."
            
            result = GradingResult(
                submission_id=submission_id,
                assignment_id=assignment_id,
                total_grade=total_earned,
                max_points=float(max_points),
                percentage=overall_percentage,
                overall_feedback=overall_feedback,
                question_grades=question_grades,
                confidence=avg_confidence,
                graded_at=graded_at,
                ocr_used=ocr_used
            )
            
            print(f"[Grading] Complete: {total_earned}/{max_points} ({overall_percentage:.1f}%), confidence={avg_confidence:.2f}")
            
            # Step 6: Send callback to core service
            await self.send_grading_callback(result, student_id)
            
            return result
            
        except Exception as e:
            print(f"[Grading] Error: {e}")
            import traceback
            traceback.print_exc()
            
            result = GradingResult(
                submission_id=submission_id,
                assignment_id=assignment_id,
                total_grade=0.0,
                max_points=float(max_points),
                percentage=0.0,
                overall_feedback="Automatic grading failed. The teacher will review your submission manually.",
                confidence=0.0,
                graded_at=graded_at,
                error=str(e)
            )
            
            # Still try to send callback (to update status)
            await self.send_grading_callback(result, student_id, failed=True)
            
            return result
    
    async def send_grading_callback(
        self,
        result: GradingResult,
        student_id: Optional[str] = None,
        failed: bool = False
    ):
        """Send grading results back to core service"""
        try:
            payload = {
                'submission_id': result.submission_id,
                'assignment_id': result.assignment_id,
                'grade': int(round(result.total_grade)),
                'max_points': int(result.max_points),
                'percentage': result.percentage,
                'feedback': result.overall_feedback,
                'detailed_feedback': result.to_dict(),
                'confidence': result.confidence,
                'ai_graded': True,
                'status': 'failed_grading' if failed else 'graded',
                'student_id': student_id
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.callback_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        print(f"[Grading] Callback sent successfully")
                    else:
                        print(f"[Grading] Callback failed: HTTP {response.status}")
                        
        except Exception as e:
            print(f"[Grading] Callback error: {e}")


# Singleton instance
grading_service = AssignmentGradingService()
