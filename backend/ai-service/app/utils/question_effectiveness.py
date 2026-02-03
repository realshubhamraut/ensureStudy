"""
Question Effectiveness Scoring Utilities for Type 5 Learning Agent

Calculates psychometric metrics to assess question quality:
- Discrimination Index: How well the question separates strong/weak students
- Difficulty Index: Percentage of students who answered correctly
- Distractor Analysis: Quality of wrong answer options
- Combined Effectiveness Score: Weighted average for overall quality
"""
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


def calculate_discrimination_index(
    top_27_correct: int,
    top_27_total: int,
    bottom_27_correct: int,
    bottom_27_total: int
) -> float:
    """
    Calculate the Discrimination Index (D).
    
    Formula: D = (P_upper - P_lower)
    Where P_upper = proportion of top 27% students who answered correctly
    And P_lower = proportion of bottom 27% students who answered correctly
    
    Interpretation:
    - D >= 0.40: Excellent discrimination
    - 0.30 <= D < 0.40: Good discrimination
    - 0.20 <= D < 0.30: Marginal discrimination
    - D < 0.20: Poor discrimination (consider revising)
    - D < 0: Negative discrimination (question is problematic)
    
    Returns:
        float: Discrimination index between -1 and 1
    """
    if top_27_total == 0 or bottom_27_total == 0:
        return 0.0
    
    p_upper = top_27_correct / top_27_total
    p_lower = bottom_27_correct / bottom_27_total
    
    return round(p_upper - p_lower, 3)


def calculate_difficulty_index(correct_count: int, total_count: int) -> float:
    """
    Calculate the Difficulty Index (P).
    
    Formula: P = (Number of correct responses) / (Total number of responses)
    
    Interpretation:
    - P >= 0.90: Too easy (consider harder questions)
    - 0.60 <= P < 0.90: Good range for most assessments
    - 0.30 <= P < 0.60: Moderate difficulty
    - P < 0.30: Very difficult (may need review)
    
    Returns:
        float: Difficulty index between 0 and 1
    """
    if total_count == 0:
        return 0.5  # Default to medium difficulty
    
    return round(correct_count / total_count, 3)


def analyze_distractors(option_selections: Dict[str, int]) -> Dict[str, float]:
    """
    Analyze the quality of MCQ distractors (wrong answers).
    
    Good distractors:
    - Are selected by some students (not ignored)
    - Don't attract more students than the correct answer
    
    Args:
        option_selections: {"A": 25, "B": 10, "C": 5, "D": 60}
        
    Returns:
        Dictionary with selection proportions for each option
    """
    total = sum(option_selections.values())
    if total == 0:
        return {}
    
    return {
        option: round(count / total, 3)
        for option, count in option_selections.items()
    }


def calculate_distractor_quality(
    option_proportions: Dict[str, float],
    correct_option: str
) -> float:
    """
    Calculate overall distractor quality score.
    
    Good distractors:
    - Each wrong option selected by at least 5% of students
    - No single wrong option selected more than correct option
    
    Returns:
        float: Quality score between 0 and 1
    """
    if not option_proportions or correct_option not in option_proportions:
        return 0.5
    
    correct_prop = option_proportions[correct_option]
    wrong_options = {k: v for k, v in option_proportions.items() if k != correct_option}
    
    if not wrong_options:
        return 0.5
    
    # Score factors
    quality_score = 1.0
    
    # Penalty: wrong options never selected (bad distractor)
    unused_distractors = sum(1 for v in wrong_options.values() if v < 0.05)
    quality_score -= 0.15 * unused_distractors
    
    # Penalty: wrong option more popular than correct (confusing question)
    confusing_distractors = sum(1 for v in wrong_options.values() if v > correct_prop)
    quality_score -= 0.25 * confusing_distractors
    
    # Bonus: even distribution among distractors (well-crafted)
    wrong_values = list(wrong_options.values())
    if len(wrong_values) > 1:
        variance = sum((v - sum(wrong_values)/len(wrong_values))**2 for v in wrong_values) / len(wrong_values)
        if variance < 0.01:  # Low variance = good distribution
            quality_score += 0.1
    
    return max(0.0, min(1.0, round(quality_score, 3)))


def compute_effectiveness_score(
    discrimination_index: float,
    difficulty_index: float,
    distractor_quality: float,
    sample_size: int
) -> float:
    """
    Compute overall question effectiveness score.
    
    Weights:
    - Discrimination: 40% (most important for assessment quality)
    - Difficulty: 30% (optimal around 0.5-0.7)
    - Distractor Quality: 20%
    - Sample Size Confidence: 10%
    
    Returns:
        float: Effectiveness score between 0 and 1
    """
    # Normalize discrimination to 0-1 scale (from -1 to 1)
    disc_normalized = (discrimination_index + 1) / 2
    
    # Optimal difficulty is around 0.6-0.7, penalize extremes
    difficulty_score = 1.0 - abs(difficulty_index - 0.65) * 2
    difficulty_score = max(0.0, min(1.0, difficulty_score))
    
    # Sample size confidence (minimum 10 samples for confidence)
    sample_confidence = min(1.0, sample_size / 50)
    
    # Weighted average
    effectiveness = (
        0.40 * disc_normalized +
        0.30 * difficulty_score +
        0.20 * distractor_quality +
        0.10 * sample_confidence
    )
    
    return round(effectiveness, 3)


def get_effectiveness_rating(score: float) -> str:
    """Get human-readable rating for effectiveness score."""
    if score >= 0.8:
        return "excellent"
    elif score >= 0.6:
        return "good"
    elif score >= 0.4:
        return "fair"
    elif score >= 0.2:
        return "poor"
    else:
        return "critical"


def should_regenerate_question(effectiveness: Dict) -> Tuple[bool, List[str]]:
    """
    Determine if a question should be regenerated by the Learning Agent.
    
    Returns:
        Tuple of (should_regenerate, list of reasons)
    """
    reasons = []
    
    # Check discrimination
    if effectiveness.get("discrimination_index", 0) < 0.15:
        reasons.append("Low discrimination - doesn't separate strong/weak students")
    
    if effectiveness.get("discrimination_index", 0) < 0:
        reasons.append("Negative discrimination - question may be confusing")
    
    # Check difficulty
    difficulty = effectiveness.get("difficulty_index", 0.5)
    if difficulty > 0.95:
        reasons.append("Too easy - almost everyone gets it right")
    elif difficulty < 0.15:
        reasons.append("Too hard - almost everyone gets it wrong")
    
    # Check distractors
    if effectiveness.get("distractor_quality", {}).get("unused_count", 0) > 1:
        reasons.append("Weak distractors - some options never selected")
    
    # Overall score
    if effectiveness.get("effectiveness_score", 0.5) < 0.3:
        reasons.append("Overall effectiveness too low")
    
    return len(reasons) > 0, reasons


async def update_question_effectiveness_from_response(
    question_id: str,
    is_correct: bool,
    selected_option: Optional[str],
    response_time_ms: int,
    student_performance_percentile: float,
    db_session
) -> Dict:
    """
    Update question effectiveness metrics after a student response.
    
    This is called by the Learning Agent after each question is answered.
    """
    from app.models.curriculum import QuestionEffectiveness, TopicQuestion
    
    # Get or create effectiveness record
    effectiveness = db_session.query(QuestionEffectiveness).get(question_id)
    if not effectiveness:
        effectiveness = QuestionEffectiveness(question_id=question_id)
        db_session.add(effectiveness)
    
    # Update basic counts
    effectiveness.total_attempts = (effectiveness.total_attempts or 0) + 1
    effectiveness.sample_size = effectiveness.total_attempts
    
    if is_correct:
        effectiveness.correct_attempts = (effectiveness.correct_attempts or 0) + 1
    
    # Update difficulty index
    effectiveness.difficulty_index = calculate_difficulty_index(
        effectiveness.correct_attempts or 0,
        effectiveness.total_attempts
    )
    
    # Update average response time
    if effectiveness.avg_response_time_ms:
        effectiveness.avg_response_time_ms = int(
            (effectiveness.avg_response_time_ms * (effectiveness.total_attempts - 1) + response_time_ms) 
            / effectiveness.total_attempts
        )
    else:
        effectiveness.avg_response_time_ms = response_time_ms
    
    # Update distractor quality if MCQ
    if selected_option and effectiveness.distractor_quality:
        distractors = effectiveness.distractor_quality or {}
        distractors[selected_option] = distractors.get(selected_option, 0) + 1
        effectiveness.distractor_quality = distractors
    
    # Recalculate effectiveness score
    distractor_quality = calculate_distractor_quality(
        analyze_distractors(effectiveness.distractor_quality or {}),
        "correct"  # This would need to be fetched from the question
    )
    
    effectiveness.effectiveness_score = compute_effectiveness_score(
        effectiveness.discrimination_index or 0,
        effectiveness.difficulty_index or 0.5,
        distractor_quality,
        effectiveness.sample_size or 0
    )
    
    db_session.commit()
    
    return effectiveness.to_dict()
