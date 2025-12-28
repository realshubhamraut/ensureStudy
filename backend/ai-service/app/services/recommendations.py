"""
Action Recommendation Engine

Generates non-automatic learning recommendations based on:
- Confidence score from LLM
- Question difficulty (heuristic)
- Student's weak topics (from profile/history)

Outputs suggestions only - no automatic actions.

Example Outputs:
- "Review prerequisite: Newton's Laws basics"
- "Practice 5 problems on this topic"
- "Schedule a revision session for tomorrow"
"""
from typing import List, Optional


# ============================================================================
# Difficulty Assessment (Heuristic)
# ============================================================================

COMPLEX_KEYWORDS = [
    "derive", "prove", "analyze", "compare", "evaluate", "synthesize",
    "integration", "differentiation", "theorem", "postulate", "advanced"
]

SIMPLE_KEYWORDS = [
    "what is", "define", "list", "name", "basic", "simple", "introduction"
]


def assess_difficulty(question: str) -> str:
    """
    Assess question difficulty using keyword heuristics.
    
    Returns: 'easy', 'medium', or 'hard'
    """
    question_lower = question.lower()
    
    hard_score = sum(1 for kw in COMPLEX_KEYWORDS if kw in question_lower)
    easy_score = sum(1 for kw in SIMPLE_KEYWORDS if kw in question_lower)
    
    if hard_score >= 2:
        return "hard"
    elif easy_score >= 2 or (len(question) < 50 and hard_score == 0):
        return "easy"
    else:
        return "medium"


# ============================================================================
# Mock Weak Topics (would come from student profile)
# ============================================================================

def get_weak_topics(user_id: str, subject: Optional[str] = None) -> List[str]:
    """
    Get student's weak topics from profile.
    
    In production, this would query student history/assessments.
    Returns mock data for development.
    """
    # Mock weak topics by subject
    mock_weak_topics = {
        "physics": ["Vectors and Scalars", "Projectile Motion", "Rotational Dynamics"],
        "math": ["Quadratic Equations", "Trigonometric Identities", "Limits"],
        "chemistry": ["Balancing Equations", "Organic Reactions", "Electrochemistry"],
        "biology": ["Cellular Respiration", "Genetics", "Enzyme Kinetics"],
        "general": ["Study Techniques", "Time Management"]
    }
    
    if subject and subject in mock_weak_topics:
        return mock_weak_topics[subject][:2]  # Return top 2 weak topics
    
    return ["General study skills"]


# ============================================================================
# Recommendation Generator
# ============================================================================

def generate_recommendations(
    confidence_score: float,
    question: str,
    subject: Optional[str] = None,
    user_id: Optional[str] = None,
    suggested_topics: Optional[List[str]] = None
) -> List[str]:
    """
    Generate learning recommendations based on context.
    
    Args:
        confidence_score: LLM's confidence in the answer (0.0-1.0)
        question: Original student question
        subject: Academic subject
        user_id: Student ID for personalization
        suggested_topics: Topics suggested by LLM
        
    Returns:
        List of actionable recommendations (3-5 items)
        
    Example:
        >>> recs = generate_recommendations(
        ...     confidence_score=0.65,
        ...     question="Derive the quadratic formula",
        ...     subject="math",
        ...     user_id="usr_123"
        ... )
        >>> recs[0]
        "Review prerequisite: Completing the Square"
    """
    recommendations = []
    difficulty = assess_difficulty(question)
    weak_topics = get_weak_topics(user_id, subject) if user_id else []
    
    # ========================================
    # Based on Confidence Score
    # ========================================
    
    if confidence_score < 0.5:
        # Low confidence - need more foundational work
        recommendations.append(
            "📚 Review the chapter on this topic in your textbook"
        )
        recommendations.append(
            "❓ Try rephrasing your question for better results"
        )
        if weak_topics:
            recommendations.append(
                f"🔍 Consider reviewing prerequisite: {weak_topics[0]}"
            )
    
    elif confidence_score < 0.75:
        # Medium confidence - some reinforcement needed
        recommendations.append(
            "📝 Take notes on the key points from this explanation"
        )
        recommendations.append(
            "✏️ Try 3-5 practice problems on this topic"
        )
    
    else:
        # High confidence - move forward
        recommendations.append(
            "✅ Great! You seem to understand this concept"
        )
        if suggested_topics:
            recommendations.append(
                f"➡️ Next topic to explore: {suggested_topics[0]}"
            )
    
    # ========================================
    # Based on Difficulty
    # ========================================
    
    if difficulty == "hard":
        recommendations.append(
            "🎥 Watch a video explanation for visual learning"
        )
        if confidence_score < 0.7:
            recommendations.append(
                "👨‍🏫 Consider asking your teacher for clarification"
            )
    
    elif difficulty == "easy" and confidence_score >= 0.8:
        recommendations.append(
            "🚀 Ready for more challenging problems on this topic"
        )
    
    # ========================================
    # Based on Weak Topics
    # ========================================
    
    if weak_topics and len(recommendations) < 5:
        # Check if question relates to weak topic
        for topic in weak_topics:
            if topic.lower() in question.lower():
                recommendations.append(
                    f"⚠️ This is a weak area for you. Schedule extra revision for: {topic}"
                )
                break
    
    # ========================================
    # Generic Recommendations
    # ========================================
    
    if len(recommendations) < 3:
        generic = [
            "📅 Add this topic to your revision schedule",
            "💡 Try explaining this concept to a friend",
            "📊 Take a quick quiz to test your understanding"
        ]
        for rec in generic:
            if rec not in recommendations:
                recommendations.append(rec)
                if len(recommendations) >= 3:
                    break
    
    # Return limited recommendations
    return recommendations[:5]


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Test cases
    test_cases = [
        (0.92, "What is photosynthesis?", "biology"),
        (0.65, "Derive the quadratic formula", "math"),
        (0.45, "Explain quantum entanglement", "physics"),
    ]
    
    for conf, question, subject in test_cases:
        print(f"\nQ: {question}")
        print(f"Confidence: {conf}, Difficulty: {assess_difficulty(question)}")
        recs = generate_recommendations(conf, question, subject, "usr_test")
        for rec in recs:
            print(f"  → {rec}")
