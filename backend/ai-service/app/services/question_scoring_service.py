"""
Question Generation and Scoring Service

Handles:
- Generating MCQ questions from topics
- Generating descriptive questions
- Scoring MCQ answers (correct/incorrect)
- Scoring descriptive answers with LLM (key points matching)
- Updating student topic scores

Design:
- MCQ: 4 options, exactly 1 correct
- Descriptive: Expected answer + key points for partial credit
- Scoring: MCQ = full marks if correct, Descriptive = 0-100% based on key points
"""

import os
import json
import logging
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class MCQOption:
    id: str  # A, B, C, D
    text: str
    is_correct: bool


@dataclass
class GeneratedMCQ:
    question_text: str
    options: List[MCQOption]
    correct_answer: str
    explanation: str
    difficulty: str = "medium"


@dataclass
class GeneratedDescriptive:
    question_text: str
    expected_answer: str
    key_points: List[str]
    explanation: str
    difficulty: str = "medium"


@dataclass
class ScoringResult:
    score_awarded: float
    max_score: float
    score_percentage: float
    is_correct: Optional[bool]  # For MCQ
    matched_key_points: List[str]  # For descriptive
    feedback: str
    confidence: float = 1.0


# MCQ Generation Prompt
MCQ_GENERATION_PROMPT = """You are an expert educator creating high-quality MCQ questions.

**TOPIC**: {topic_name}
**DESCRIPTION**: {topic_description}
**DIFFICULTY**: {difficulty}
**KEY CONCEPTS**: {key_concepts}

Generate {count} multiple-choice questions. Each question must:
1. Test understanding of the topic (not just memorization)
2. Have exactly 4 options (A, B, C, D)
3. Have exactly ONE correct answer
4. Include plausible distractors for wrong answers
5. Include an explanation for the correct answer

**RESPONSE FORMAT** (JSON only):
{{
  "questions": [
    {{
      "question": "What is...?",
      "options": [
        {{"id": "A", "text": "Option A", "is_correct": false}},
        {{"id": "B", "text": "Option B", "is_correct": true}},
        {{"id": "C", "text": "Option C", "is_correct": false}},
        {{"id": "D", "text": "Option D", "is_correct": false}}
      ],
      "correct_answer": "B",
      "explanation": "B is correct because...",
      "difficulty": "{difficulty}"
    }}
  ]
}}

Return ONLY valid JSON.
"""


# Descriptive Generation Prompt
DESCRIPTIVE_GENERATION_PROMPT = """You are an expert educator creating descriptive/essay questions.

**TOPIC**: {topic_name}
**DESCRIPTION**: {topic_description}
**DIFFICULTY**: {difficulty}
**KEY CONCEPTS**: {key_concepts}

Generate {count} descriptive questions. Each question must:
1. Require a detailed written response
2. Have a model answer
3. Have 3-5 key points for scoring
4. Be appropriate for the difficulty level

**RESPONSE FORMAT** (JSON only):
{{
  "questions": [
    {{
      "question": "Explain the concept of...?",
      "expected_answer": "A complete model answer that covers all key points...",
      "key_points": [
        "First key concept that must be mentioned",
        "Second important point",
        "Third critical aspect"
      ],
      "explanation": "This question tests understanding of...",
      "difficulty": "{difficulty}"
    }}
  ]
}}

Return ONLY valid JSON.
"""


# Descriptive Answer Scoring Prompt
DESCRIPTIVE_SCORING_PROMPT = """You are an expert grader evaluating a student's answer.

**QUESTION**: {question}

**EXPECTED ANSWER KEY POINTS**:
{key_points}

**STUDENT'S ANSWER**:
{student_answer}

Evaluate the student's answer:
1. Check which key points are covered (even partially)
2. Consider clarity and accuracy of explanation
3. Provide constructive feedback

**RESPONSE FORMAT** (JSON only):
{{
  "matched_points": ["point 1", "point 2"],
  "score_percentage": 75,
  "feedback": "Good explanation of X, but missing details about Y...",
  "confidence": 0.9
}}

Return ONLY valid JSON.
"""


class QuestionGenerator:
    """Generates MCQ and descriptive questions for topics."""
    
    def generate_mcq(
        self,
        topic_name: str,
        topic_description: str = "",
        key_concepts: List[str] = None,
        difficulty: str = "medium",
        count: int = 5
    ) -> List[GeneratedMCQ]:
        """Generate MCQ questions for a topic."""
        
        prompt = MCQ_GENERATION_PROMPT.format(
            topic_name=topic_name,
            topic_description=topic_description or "No description provided",
            key_concepts=", ".join(key_concepts or []),
            difficulty=difficulty,
            count=count
        )
        
        # Try LLM providers
        response = self._call_llm(prompt)
        if not response:
            logger.warning("MCQ generation failed, returning empty list")
            return []
        
        return self._parse_mcq_response(response)
    
    def generate_descriptive(
        self,
        topic_name: str,
        topic_description: str = "",
        key_concepts: List[str] = None,
        difficulty: str = "medium",
        count: int = 3
    ) -> List[GeneratedDescriptive]:
        """Generate descriptive questions for a topic."""
        
        prompt = DESCRIPTIVE_GENERATION_PROMPT.format(
            topic_name=topic_name,
            topic_description=topic_description or "No description provided",
            key_concepts=", ".join(key_concepts or []),
            difficulty=difficulty,
            count=count
        )
        
        response = self._call_llm(prompt)
        if not response:
            logger.warning("Descriptive generation failed, returning empty list")
            return []
        
        return self._parse_descriptive_response(response)
    
    def _call_llm(self, prompt: str) -> Optional[str]:
        """Call LLM (Mistral > Groq) for generation."""
        
        # Try Mistral
        result = self._try_mistral(prompt)
        if result:
            return result
        
        # Try Groq
        result = self._try_groq(prompt)
        if result:
            return result
        
        return None
    
    def _try_mistral(self, prompt: str) -> Optional[str]:
        """Try Mistral API."""
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            return None
        
        try:
            from mistralai import Mistral
            
            client = Mistral(api_key=api_key)
            response = client.chat.complete(
                model="mistral-large-latest",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=4000
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.warning(f"Mistral question generation failed: {e}")
            return None
    
    def _try_groq(self, prompt: str) -> Optional[str]:
        """Try Groq API."""
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            return None
        
        try:
            from groq import Groq
            
            client = Groq(api_key=api_key)
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=4000
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.warning(f"Groq question generation failed: {e}")
            return None
    
    def _parse_mcq_response(self, content: str) -> List[GeneratedMCQ]:
        """Parse LLM response into MCQ objects."""
        try:
            # Clean markdown
            content = content.strip()
            if content.startswith("```"):
                lines = content.split("\n")
                content = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
            
            data = json.loads(content)
            questions = []
            
            for q in data.get("questions", []):
                options = [
                    MCQOption(
                        id=opt["id"],
                        text=opt["text"],
                        is_correct=opt.get("is_correct", False)
                    )
                    for opt in q.get("options", [])
                ]
                
                questions.append(GeneratedMCQ(
                    question_text=q["question"],
                    options=options,
                    correct_answer=q.get("correct_answer", "A"),
                    explanation=q.get("explanation", ""),
                    difficulty=q.get("difficulty", "medium")
                ))
            
            return questions
        except Exception as e:
            logger.error(f"Failed to parse MCQ response: {e}")
            return []
    
    def _parse_descriptive_response(self, content: str) -> List[GeneratedDescriptive]:
        """Parse LLM response into descriptive question objects."""
        try:
            content = content.strip()
            if content.startswith("```"):
                lines = content.split("\n")
                content = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
            
            data = json.loads(content)
            questions = []
            
            for q in data.get("questions", []):
                questions.append(GeneratedDescriptive(
                    question_text=q["question"],
                    expected_answer=q.get("expected_answer", ""),
                    key_points=q.get("key_points", []),
                    explanation=q.get("explanation", ""),
                    difficulty=q.get("difficulty", "medium")
                ))
            
            return questions
        except Exception as e:
            logger.error(f"Failed to parse descriptive response: {e}")
            return []


class AnswerScorer:
    """Scores MCQ and descriptive answers."""
    
    def score_mcq(
        self,
        selected_option: str,
        correct_answer: str,
        marks: int = 1
    ) -> ScoringResult:
        """Score an MCQ answer."""
        is_correct = selected_option.upper() == correct_answer.upper()
        
        return ScoringResult(
            score_awarded=float(marks) if is_correct else 0.0,
            max_score=float(marks),
            score_percentage=100.0 if is_correct else 0.0,
            is_correct=is_correct,
            matched_key_points=[],
            feedback="Correct!" if is_correct else f"Incorrect. The correct answer is {correct_answer}.",
            confidence=1.0
        )
    
    def score_descriptive(
        self,
        student_answer: str,
        question_text: str,
        key_points: List[str],
        marks: int = 10
    ) -> ScoringResult:
        """Score a descriptive answer using LLM."""
        
        if not student_answer or not student_answer.strip():
            return ScoringResult(
                score_awarded=0.0,
                max_score=float(marks),
                score_percentage=0.0,
                is_correct=None,
                matched_key_points=[],
                feedback="No answer provided.",
                confidence=1.0
            )
        
        # Format key points for prompt
        key_points_str = "\n".join(f"- {kp}" for kp in key_points)
        
        prompt = DESCRIPTIVE_SCORING_PROMPT.format(
            question=question_text,
            key_points=key_points_str,
            student_answer=student_answer
        )
        
        # Call LLM
        response = self._call_llm(prompt)
        if not response:
            # Fallback: simple keyword matching
            return self._fallback_scoring(student_answer, key_points, marks)
        
        return self._parse_scoring_response(response, marks)
    
    def _call_llm(self, prompt: str) -> Optional[str]:
        """Call LLM for scoring."""
        # Try Mistral
        result = self._try_mistral(prompt)
        if result:
            return result
        
        # Try Groq  
        result = self._try_groq(prompt)
        if result:
            return result
        
        return None
    
    def _try_mistral(self, prompt: str) -> Optional[str]:
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            return None
        
        try:
            from mistralai import Mistral
            
            client = Mistral(api_key=api_key)
            response = client.chat.complete(
                model="mistral-large-latest",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1000
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.warning(f"Mistral scoring failed: {e}")
            return None
    
    def _try_groq(self, prompt: str) -> Optional[str]:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            return None
        
        try:
            from groq import Groq
            
            client = Groq(api_key=api_key)
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1000
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.warning(f"Groq scoring failed: {e}")
            return None
    
    def _parse_scoring_response(self, content: str, marks: int) -> ScoringResult:
        """Parse LLM scoring response."""
        try:
            content = content.strip()
            if content.startswith("```"):
                lines = content.split("\n")
                content = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
            
            data = json.loads(content)
            
            score_pct = float(data.get("score_percentage", 0))
            score_awarded = (score_pct / 100) * marks
            
            return ScoringResult(
                score_awarded=score_awarded,
                max_score=float(marks),
                score_percentage=score_pct,
                is_correct=None,
                matched_key_points=data.get("matched_points", []),
                feedback=data.get("feedback", ""),
                confidence=float(data.get("confidence", 0.8))
            )
        except Exception as e:
            logger.error(f"Failed to parse scoring response: {e}")
            return self._fallback_scoring("", [], marks)
    
    def _fallback_scoring(
        self,
        student_answer: str,
        key_points: List[str],
        marks: int
    ) -> ScoringResult:
        """Fallback keyword-based scoring."""
        answer_lower = student_answer.lower()
        matched = []
        
        for kp in key_points:
            # Simple keyword matching
            keywords = [w.lower() for w in kp.split() if len(w) > 3]
            if any(kw in answer_lower for kw in keywords[:3]):
                matched.append(kp)
        
        score_pct = (len(matched) / len(key_points) * 100) if key_points else 0
        score_awarded = (score_pct / 100) * marks
        
        return ScoringResult(
            score_awarded=score_awarded,
            max_score=float(marks),
            score_percentage=score_pct,
            is_correct=None,
            matched_key_points=matched,
            feedback=f"Matched {len(matched)} of {len(key_points)} key points.",
            confidence=0.5  # Low confidence for fallback
        )


# Singleton instances
_question_generator: Optional[QuestionGenerator] = None
_answer_scorer: Optional[AnswerScorer] = None


def get_question_generator() -> QuestionGenerator:
    global _question_generator
    if _question_generator is None:
        _question_generator = QuestionGenerator()
    return _question_generator


def get_answer_scorer() -> AnswerScorer:
    global _answer_scorer
    if _answer_scorer is None:
        _answer_scorer = AnswerScorer()
    return _answer_scorer
