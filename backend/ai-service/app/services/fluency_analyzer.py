"""
Fluency Analyzer Service

Provides real speech fluency analysis including:
- Words per minute (WPM) calculation
- Filler word detection
- Pause analysis (from audio timestamps if available)
- Fluency scoring (0-100)

Uses the utility functions from ml/softskills/training/fluency_utils.py
"""

import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

# Common filler words and phrases
FILLER_PATTERNS = [
    r"\bum\b",
    r"\buh\b",
    r"\blike\b",
    r"\byou know\b",
    r"\bbasically\b",
    r"\bactually\b",
    r"\bliterally\b",
    r"\bso\b",
    r"\bwell\b",
    r"\bi mean\b",
    r"\bkind of\b",
    r"\bsort of\b",
    r"\bright\b",
]

# Optimal WPM range for interviews/presentations
OPTIMAL_WPM_MIN = float(120)
OPTIMAL_WPM_MAX = float(160)

# Score weights
WPM_WEIGHT = 0.40
FILLER_WEIGHT = 0.40
PAUSE_WEIGHT = 0.20

# Filler penalty per word
FILLER_PENALTY = 0.05


@dataclass
class FluencyResult:
    """Result from fluency analysis."""
    score: float
    wpm: float
    wpm_score: float
    word_count: int
    filler_count: int
    filler_score: float
    fillers_detected: List[str]
    pause_ratio: float
    pause_score: float
    
    def to_dict(self) -> Dict:
        return {
            "score": round(self.score, 1),
            "wpm": round(self.wpm, 1),
            "wpm_score": round(self.wpm_score, 1),
            "word_count": self.word_count,
            "filler_count": self.filler_count,
            "filler_score": round(self.filler_score, 1),
            "fillers_detected": self.fillers_detected,
            "pause_ratio": round(self.pause_ratio, 3),
            "pause_score": round(self.pause_score, 1),
        }


def calculate_wpm(text: str, duration_seconds: float) -> Tuple[float, int]:
    """
    Calculate words per minute from text and duration.
    
    Args:
        text: Transcript text
        duration_seconds: Audio duration in seconds
    
    Returns:
        (wpm, word_count) tuple
    """
    if duration_seconds <= 0:
        return 0.0, 0
    
    # Count words (split by whitespace)
    words = [w for w in text.split() if w.strip()]
    word_count = len(words)
    
    # Calculate WPM
    minutes = duration_seconds / 60.0
    wpm = word_count / minutes if minutes > 0 else 0
    
    return wpm, word_count


def detect_fillers(text: str) -> Tuple[int, List[str]]:
    """
    Detect filler words in text.
    
    Args:
        text: Input transcript
    
    Returns:
        (count, list of unique fillers) tuple
    """
    text_lower = text.lower()
    
    detected = []
    total_count = 0
    
    for pattern in FILLER_PATTERNS:
        matches = re.findall(pattern, text_lower)
        if matches:
            detected.extend(matches)
            total_count += len(matches)
    
    return total_count, list(set(detected))


def score_wpm(wpm: float) -> float:
    """
    Score WPM on 0-100 scale.
    
    Optimal range: 120-160 WPM
    """
    if OPTIMAL_WPM_MIN <= wpm <= OPTIMAL_WPM_MAX:
        return 100.0
    elif wpm < OPTIMAL_WPM_MIN:
        # Penalty for slow speech
        return max(0, 100 - (OPTIMAL_WPM_MIN - wpm) * 1.5)
    else:
        # Penalty for fast speech  
        return max(0, 100 - (wpm - OPTIMAL_WPM_MAX) * 1.0)


def score_fillers(filler_count: int) -> float:
    """
    Score based on filler word count.
    """
    return max(0, 100 - filler_count * FILLER_PENALTY * 100)


def score_pauses(pause_ratio: float) -> float:
    """
    Score based on pause ratio.
    
    pause_ratio: pause_time / total_time (0-1)
    """
    # 50% pause ratio = 0 score
    return max(0, 100 - pause_ratio * 200)


def analyze_fluency(
    transcript: str,
    duration_seconds: float,
    pause_ratio: float = 0.0
) -> FluencyResult:
    """
    Analyze speech fluency from transcript.
    
    Args:
        transcript: Speech transcript text
        duration_seconds: Audio duration in seconds
        pause_ratio: Ratio of pause time to total time (0-1)
    
    Returns:
        FluencyResult with score breakdown
    """
    if not transcript or duration_seconds <= 0:
        return FluencyResult(
            score=0,
            wpm=0,
            wpm_score=0,
            word_count=0,
            filler_count=0,
            filler_score=0,
            fillers_detected=[],
            pause_ratio=0,
            pause_score=0
        )
    
    # Calculate WPM
    wpm, word_count = calculate_wpm(transcript, duration_seconds)
    wpm_score = score_wpm(wpm)
    
    # Detect fillers
    filler_count, fillers = detect_fillers(transcript)
    filler_score = score_fillers(filler_count)
    
    # Score pauses
    pause_score = score_pauses(pause_ratio)
    
    # Weighted overall score
    overall_score = (
        wpm_score * WPM_WEIGHT +
        filler_score * FILLER_WEIGHT +
        pause_score * PAUSE_WEIGHT
    )
    
    return FluencyResult(
        score=overall_score,
        wpm=wpm,
        wpm_score=wpm_score,
        word_count=word_count,
        filler_count=filler_count,
        filler_score=filler_score,
        fillers_detected=fillers,
        pause_ratio=pause_ratio,
        pause_score=pause_score
    )


# Singleton instance
_fluency_analyzer = None


def get_fluency_analyzer():
    """Get singleton FluencyAnalyzer instance."""
    global _fluency_analyzer
    if _fluency_analyzer is None:
        _fluency_analyzer = FluencyAnalyzer()
    return _fluency_analyzer


class FluencyAnalyzer:
    """
    Fluency analyzer service class.
    
    Provides stateless analysis methods.
    """
    
    def __init__(self):
        self.optimal_wpm_min = OPTIMAL_WPM_MIN
        self.optimal_wpm_max = OPTIMAL_WPM_MAX
    
    def analyze(
        self,
        transcript: str,
        duration_seconds: float,
        pause_ratio: float = 0.0
    ) -> FluencyResult:
        """
        Analyze fluency from transcript.
        """
        return analyze_fluency(transcript, duration_seconds, pause_ratio)
    
    def get_feedback(self, result: FluencyResult) -> List[str]:
        """
        Generate feedback based on fluency result.
        """
        feedback = []
        
        # WPM feedback
        if result.wpm < 100:
            feedback.append("Try speaking a bit faster - your pace is quite slow.")
        elif result.wpm < 120:
            feedback.append("Your speaking pace is slightly below optimal. Aim for 120-160 WPM.")
        elif result.wpm > 180:
            feedback.append("You're speaking quite fast. Try to slow down for clarity.")
        elif result.wpm > 160:
            feedback.append("Your pace is slightly fast. Consider slowing down a bit.")
        else:
            feedback.append("Great speaking pace!")
        
        # Filler feedback
        if result.filler_count > 10:
            feedback.append(f"Try to reduce filler words like '{', '.join(result.fillers_detected[:3])}'.")
        elif result.filler_count > 5:
            feedback.append("You use some filler words. Practice pausing instead.")
        elif result.filler_count > 0:
            feedback.append("Minimal filler words - good job!")
        else:
            feedback.append("Excellent - no filler words detected!")
        
        return feedback
