"""
LLM-Based Fluency Evaluator Service

Evaluates speech fluency using Groq LLM for more intelligent analysis.
Analyzes:
- Filler word usage and patterns
- Sentence structure and flow
- Speaking pace estimation
- Coherence and clarity
"""

import os
import re
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class FluencyEvaluationResult:
    """Result from LLM fluency evaluation."""
    score: float  # 0-100
    wpm_score: float  # Estimated based on word count/structure
    filler_score: float  # Filler word penalty
    coherence_score: float  # Flow and coherence
    feedback: str  # Overall feedback
    filler_words_found: List[str]  # Detected fillers
    suggestions: List[str]  # Improvement tips
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": round(self.score, 1),
            "wpm_score": round(self.wpm_score, 1),
            "filler_score": round(self.filler_score, 1),
            "coherence_score": round(self.coherence_score, 1),
            "feedback": self.feedback,
            "filler_words_found": self.filler_words_found,
            "suggestions": self.suggestions,
            "timestamp": self.timestamp
        }


# Common filler patterns for detection
FILLER_PATTERNS = [
    r"\bum+\b", r"\buh+\b", r"\blike\b", r"\byou know\b",
    r"\bbasically\b", r"\bactually\b", r"\bliterally\b",
    r"\bi mean\b", r"\bkind of\b", r"\bsort of\b", r"\bright\b",
    r"\bso\b", r"\bwell\b", r"\banyways?\b"
]


class FluencyEvaluator:
    """
    Evaluates speech fluency using LLM.
    
    Provides:
    - Fluency score (0-100)
    - Filler word detection
    - Coherence analysis
    - Improvement suggestions
    """
    
    def __init__(self):
        self._groq_client = None
        logger.info("[FluencyEvaluator] Initialized")
    
    @property
    def groq_client(self):
        """Lazy load Groq client."""
        if self._groq_client is None:
            try:
                from groq import Groq
                self._groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
                logger.info("[FluencyEvaluator] Groq client initialized")
            except Exception as e:
                logger.error(f"[FluencyEvaluator] Failed to init Groq: {e}")
                self._groq_client = None
        return self._groq_client
    
    def _detect_fillers(self, text: str) -> tuple:
        """Detect filler words in text."""
        text_lower = text.lower()
        detected = []
        total_count = 0
        
        for pattern in FILLER_PATTERNS:
            matches = re.findall(pattern, text_lower)
            if matches:
                detected.extend(matches)
                total_count += len(matches)
        
        return total_count, list(set(detected))
    
    async def evaluate_fluency(
        self,
        transcript: str,
        duration_seconds: float = 0.0,
        context: str = "communication practice"
    ) -> FluencyEvaluationResult:
        """
        Evaluate speech fluency from transcript.
        
        Args:
            transcript: Speech transcript text
            duration_seconds: Optional speaking duration
            context: Context of the speech (interview, presentation, etc.)
            
        Returns:
            FluencyEvaluationResult with scores and feedback
        """
        logger.info(f"[FluencyEvaluator] Evaluating transcript ({len(transcript)} chars)")
        
        # Handle empty or very short transcripts
        if not transcript or len(transcript.strip()) < 10:
            logger.warning("[FluencyEvaluator] Transcript too short")
            return FluencyEvaluationResult(
                score=0.0,
                wpm_score=0.0,
                filler_score=0.0,
                coherence_score=0.0,
                feedback="No speech detected or transcript too short.",
                filler_words_found=[],
                suggestions=["Speak more to get fluency feedback"]
            )
        
        # Detect fillers locally first
        filler_count, fillers = self._detect_fillers(transcript)
        
        # Try LLM evaluation
        if self.groq_client:
            try:
                result = await self._evaluate_with_llm(
                    transcript, duration_seconds, context, filler_count, fillers
                )
                logger.info(f"[FluencyEvaluator] LLM score: {result.score}")
                return result
            except Exception as e:
                logger.error(f"[FluencyEvaluator] LLM evaluation failed: {e}")
        
        # Fallback to rule-based evaluation
        logger.info("[FluencyEvaluator] Using rule-based fallback")
        return self._evaluate_rule_based(transcript, duration_seconds, filler_count, fillers)
    
    async def _evaluate_with_llm(
        self,
        transcript: str,
        duration_seconds: float,
        context: str,
        filler_count: int,
        fillers: List[str]
    ) -> FluencyEvaluationResult:
        """Evaluate using Groq LLM."""
        
        word_count = len(transcript.split())
        estimated_wpm = (word_count / duration_seconds * 60) if duration_seconds > 0 else 0
        
        prompt = f"""You are an expert speech fluency evaluator. Analyze this speech transcript for fluency.

Context: {context}
Word count: {word_count}
{f"Estimated WPM: {estimated_wpm:.0f}" if estimated_wpm > 0 else "Duration unknown"}
Detected filler words: {', '.join(fillers) if fillers else 'None detected'}
Filler count: {filler_count}

Transcript:
"{transcript}"

Evaluate the speech fluency on these criteria (score each 0-100):

1. WPM_SCORE: Rate the speaking pace (optimal is 120-160 WPM). If duration unknown, estimate from sentence structure and complexity. Fast/rushed or too slow gets lower score.

2. FILLER_SCORE: Rate based on filler word usage. Few fillers = high score, many fillers = low score. {filler_count} fillers detected.

3. COHERENCE_SCORE: Rate how well the ideas flow, sentence structure, and overall clarity of expression.

Provide scores and feedback in this exact format:
WPM_SCORE: [number 0-100]
FILLER_SCORE: [number 0-100]
COHERENCE_SCORE: [number 0-100]
FEEDBACK: [one sentence overall fluency feedback]
SUGGESTIONS: [suggestion1] | [suggestion2] | [suggestion3]"""

        response = self.groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=400
        )
        
        text = response.choices[0].message.content
        logger.debug(f"[FluencyEvaluator] LLM response: {text[:200]}...")
        
        return self._parse_llm_response(text, fillers)
    
    def _parse_llm_response(self, text: str, fillers: List[str]) -> FluencyEvaluationResult:
        """Parse LLM response into FluencyEvaluationResult."""
        
        def extract_score(pattern: str, default: float = 70.0) -> float:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                try:
                    return min(100, max(0, float(match.group(1))))
                except:
                    pass
            return default
        
        def extract_value(pattern: str, default: str = "") -> str:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE | re.DOTALL)
            return match.group(1).strip() if match else default
        
        wpm_score = extract_score(r"WPM_SCORE:\s*(\d+(?:\.\d+)?)")
        filler_score = extract_score(r"FILLER_SCORE:\s*(\d+(?:\.\d+)?)")
        coherence_score = extract_score(r"COHERENCE_SCORE:\s*(\d+(?:\.\d+)?)")
        
        # Overall score (weighted)
        overall = wpm_score * 0.3 + filler_score * 0.35 + coherence_score * 0.35
        
        feedback = extract_value(r"FEEDBACK:\s*(.+?)(?=SUGGESTIONS|$)", "Good effort. Keep practicing!")
        
        suggestions_str = extract_value(r"SUGGESTIONS:\s*(.+?)$")
        suggestions = [s.strip() for s in suggestions_str.split("|") if s.strip()][:3]
        
        return FluencyEvaluationResult(
            score=overall,
            wpm_score=wpm_score,
            filler_score=filler_score,
            coherence_score=coherence_score,
            feedback=feedback.strip(),
            filler_words_found=fillers,
            suggestions=suggestions if suggestions else ["Practice speaking more naturally"]
        )
    
    def _evaluate_rule_based(
        self,
        transcript: str,
        duration_seconds: float,
        filler_count: int,
        fillers: List[str]
    ) -> FluencyEvaluationResult:
        """Fallback rule-based evaluation."""
        
        words = transcript.split()
        word_count = len(words)
        
        # WPM score
        if duration_seconds > 0:
            wpm = (word_count / duration_seconds) * 60
            if 120 <= wpm <= 160:
                wpm_score = 100.0
            elif wpm < 120:
                wpm_score = max(0, 100 - (120 - wpm) * 1.5)
            else:
                wpm_score = max(0, 100 - (wpm - 160) * 1.0)
        else:
            # Estimate based on word count
            wpm_score = 70.0 if word_count > 20 else 50.0
        
        # Filler score
        filler_penalty = min(50, filler_count * 5)
        filler_score = max(0, 100 - filler_penalty)
        
        # Coherence score (based on sentence structure)
        sentences = re.split(r'[.!?]+', transcript)
        sentences = [s.strip() for s in sentences if s.strip()]
        avg_sentence_len = word_count / max(len(sentences), 1)
        
        if 10 <= avg_sentence_len <= 20:
            coherence_score = 80.0
        elif avg_sentence_len < 10:
            coherence_score = 60.0
        else:
            coherence_score = 70.0
        
        # Check for connective words
        connectives = ["because", "therefore", "however", "also", "furthermore", "then"]
        has_connectives = any(w in transcript.lower() for w in connectives)
        if has_connectives:
            coherence_score += 10
        
        coherence_score = min(100, coherence_score)
        
        # Overall
        overall = wpm_score * 0.3 + filler_score * 0.35 + coherence_score * 0.35
        
        # Feedback
        if overall >= 80:
            feedback = "Excellent fluency! Clear and well-paced speech."
        elif overall >= 60:
            feedback = "Good fluency. Try reducing filler words for better flow."
        elif overall >= 40:
            feedback = "Average fluency. Practice speaking more naturally."
        else:
            feedback = "Work on speaking pace and reducing hesitations."
        
        suggestions = []
        if filler_count > 3:
            suggestions.append(f"Reduce filler words like '{', '.join(fillers[:2])}'")
        if wpm_score < 60:
            suggestions.append("Work on your speaking pace (aim for 120-160 WPM)")
        if coherence_score < 70:
            suggestions.append("Use connecting words for better flow")
        if not suggestions:
            suggestions.append("Keep practicing to maintain your good fluency")
        
        return FluencyEvaluationResult(
            score=overall,
            wpm_score=wpm_score,
            filler_score=filler_score,
            coherence_score=coherence_score,
            feedback=feedback,
            filler_words_found=fillers,
            suggestions=suggestions[:3]
        )


# Singleton instance
_evaluator = None


def get_fluency_evaluator() -> FluencyEvaluator:
    """Get singleton FluencyEvaluator instance."""
    global _evaluator
    if _evaluator is None:
        _evaluator = FluencyEvaluator()
    return _evaluator
