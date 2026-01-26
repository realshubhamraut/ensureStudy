"""
Mock Interview API Routes

Provides endpoints for AI-powered mock interviews:
- Start interview session
- Submit answers and get evaluation
- Get interview questions by topic
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import random
import uuid
from datetime import datetime

router = APIRouter(prefix="/mock-interview", tags=["Mock Interview"])


# ============================================
# Request/Response Schemas
# ============================================

class StartInterviewRequest(BaseModel):
    """Request to start a mock interview session."""
    user_id: str
    subject: str = Field(..., description="Subject: math, physics, chemistry")
    chapter: str = Field(..., description="Chapter/topic within the subject")
    avatar: str = Field(default="female", description="Avatar type: male or female")


class InterviewQuestion(BaseModel):
    """A single interview question."""
    id: str
    question: str
    topic: str
    difficulty: str = "medium"
    reference_answer: str
    key_concepts: List[str]


class StartInterviewResponse(BaseModel):
    """Response when starting an interview."""
    session_id: str
    question: InterviewQuestion
    total_questions: int
    message: str


class SubmitAnswerRequest(BaseModel):
    """Request to submit an answer."""
    session_id: str
    question_id: str
    answer_text: str
    audio_duration_seconds: Optional[float] = None


class AnswerEvaluation(BaseModel):
    """Evaluation of a single answer."""
    question_id: str
    score: float = Field(..., ge=0, le=100)
    concept_scores: dict
    feedback: str
    weak_concepts: List[str]


class SubmitAnswerResponse(BaseModel):
    """Response after submitting an answer."""
    evaluation: AnswerEvaluation
    next_question: Optional[InterviewQuestion] = None
    is_complete: bool
    progress: dict


class InterviewSummary(BaseModel):
    """Summary of completed interview."""
    session_id: str
    subject: str
    chapter: str
    total_questions: int
    average_score: float
    concept_mastery: dict
    weak_topics: List[str]
    recommendations: List[str]
    duration_minutes: float


# ============================================
# Question Banks (In production, use Qdrant)
# ============================================

QUESTION_BANKS = {
    "physics": {
        "Mechanics": [
            {
                "id": "phys_mech_1",
                "question": "Can you explain Newton's first law of motion and provide an example from everyday life?",
                "topic": "Newton's Laws",
                "reference_answer": "Newton's first law states that an object at rest stays at rest and an object in motion stays in motion with the same speed and direction unless acted upon by an unbalanced force. An example is a book on a table remaining at rest until someone pushes it.",
                "key_concepts": ["inertia", "rest", "motion", "unbalanced force"]
            },
            {
                "id": "phys_mech_2",
                "question": "What is the relationship between force, mass, and acceleration according to Newton's second law?",
                "topic": "Newton's Laws",
                "reference_answer": "Newton's second law states that Force equals mass times acceleration (F=ma). This means the acceleration of an object is directly proportional to the net force and inversely proportional to its mass.",
                "key_concepts": ["force", "mass", "acceleration", "F=ma", "proportional"]
            },
            {
                "id": "phys_mech_3",
                "question": "Describe Newton's third law and give an example of action-reaction pairs.",
                "topic": "Newton's Laws",
                "reference_answer": "Newton's third law states that for every action there is an equal and opposite reaction. Examples include a rocket pushing exhaust gases down while the gases push the rocket up, or when you push against a wall, the wall pushes back on you.",
                "key_concepts": ["action", "reaction", "equal", "opposite", "force pairs"]
            }
        ],
        "Thermodynamics": [
            {
                "id": "phys_thermo_1",
                "question": "What is the first law of thermodynamics and how does it relate to energy conservation?",
                "topic": "Laws of Thermodynamics",
                "reference_answer": "The first law of thermodynamics states that energy cannot be created or destroyed, only transferred or converted from one form to another. It is essentially the law of conservation of energy applied to thermodynamic systems.",
                "key_concepts": ["energy conservation", "heat", "work", "internal energy"]
            }
        ]
    },
    "math": {
        "Calculus": [
            {
                "id": "math_calc_1",
                "question": "What is a derivative and what does it represent geometrically?",
                "topic": "Differentiation",
                "reference_answer": "A derivative represents the rate of change of a function. Geometrically, it is the slope of the tangent line to the curve at a given point.",
                "key_concepts": ["rate of change", "slope", "tangent line", "limit"]
            }
        ],
        "Algebra": [
            {
                "id": "math_alg_1",
                "question": "What is the quadratic formula and when would you use it?",
                "topic": "Quadratic Equations",
                "reference_answer": "The quadratic formula is x = (-b ± √(b²-4ac)) / 2a. It is used to find the roots of any quadratic equation in the form ax² + bx + c = 0.",
                "key_concepts": ["roots", "quadratic equation", "discriminant", "solutions"]
            }
        ]
    },
    "chemistry": {
        "Organic Chemistry": [
            {
                "id": "chem_org_1",
                "question": "What is the difference between alkanes, alkenes, and alkynes?",
                "topic": "Hydrocarbons",
                "reference_answer": "Alkanes have single bonds only (saturated). Alkenes have at least one carbon-carbon double bond. Alkynes have at least one carbon-carbon triple bond.",
                "key_concepts": ["single bond", "double bond", "triple bond", "saturated", "unsaturated"]
            }
        ]
    }
}

# Session storage (in production, use Redis or database)
ACTIVE_SESSIONS: dict = {}


# ============================================
# Helper Functions
# ============================================

def get_questions_for_topic(subject: str, chapter: str, count: int = 5) -> List[dict]:
    """Get questions for a specific subject and chapter."""
    subject_bank = QUESTION_BANKS.get(subject.lower(), {})
    chapter_questions = subject_bank.get(chapter, [])
    
    if not chapter_questions:
        # Fallback to any available questions in the subject
        all_questions = []
        for ch_questions in subject_bank.values():
            all_questions.extend(ch_questions)
        chapter_questions = all_questions
    
    # Shuffle and limit
    random.shuffle(chapter_questions)
    return chapter_questions[:count]


def calculate_similarity_score(answer: str, reference: str, key_concepts: List[str]) -> tuple:
    """
    Calculate semantic similarity between answer and reference.
    In production, use sentence-transformers.
    """
    answer_lower = answer.lower()
    reference_lower = reference.lower()
    
    # Check concept coverage
    concept_scores = {}
    for concept in key_concepts:
        if concept.lower() in answer_lower:
            concept_scores[concept] = 100
        elif any(word in answer_lower for word in concept.lower().split()):
            concept_scores[concept] = 60
        else:
            concept_scores[concept] = 0
    
    # Overall score based on concept coverage and length
    avg_concept = sum(concept_scores.values()) / len(concept_scores) if concept_scores else 0
    length_bonus = min(20, len(answer.split()) * 0.5)  # Up to 20 points for length
    
    total_score = min(100, avg_concept * 0.8 + length_bonus)
    
    weak_concepts = [c for c, s in concept_scores.items() if s < 50]
    
    return total_score, concept_scores, weak_concepts


# ============================================
# API Endpoints
# ============================================

@router.post("/start", response_model=StartInterviewResponse)
async def start_interview(request: StartInterviewRequest):
    """Start a new mock interview session."""
    
    # Get questions for the topic
    questions = get_questions_for_topic(request.subject, request.chapter)
    
    if not questions:
        raise HTTPException(
            status_code=404,
            detail=f"No questions found for {request.subject}/{request.chapter}"
        )
    
    # Create session
    session_id = str(uuid.uuid4())
    first_question = questions[0]
    
    ACTIVE_SESSIONS[session_id] = {
        "user_id": request.user_id,
        "subject": request.subject,
        "chapter": request.chapter,
        "avatar": request.avatar,
        "questions": questions,
        "current_index": 0,
        "evaluations": [],
        "started_at": datetime.now(),
        "weak_concepts": []
    }
    
    return StartInterviewResponse(
        session_id=session_id,
        question=InterviewQuestion(
            id=first_question["id"],
            question=first_question["question"],
            topic=first_question["topic"],
            reference_answer=first_question["reference_answer"],
            key_concepts=first_question["key_concepts"]
        ),
        total_questions=len(questions),
        message="Interview started. Answer the question verbally."
    )


@router.post("/submit", response_model=SubmitAnswerResponse)
async def submit_answer(request: SubmitAnswerRequest):
    """Submit an answer and get evaluation."""
    
    session = ACTIVE_SESSIONS.get(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    questions = session["questions"]
    current_idx = session["current_index"]
    current_question = questions[current_idx]
    
    # Evaluate answer
    score, concept_scores, weak_concepts = calculate_similarity_score(
        request.answer_text,
        current_question["reference_answer"],
        current_question["key_concepts"]
    )
    
    # Generate feedback
    if score >= 80:
        feedback = "Excellent answer! You covered the key concepts well."
    elif score >= 60:
        feedback = "Good answer, but you missed some important concepts."
    elif score >= 40:
        feedback = "Partial understanding shown. Review the missing concepts."
    else:
        feedback = "Needs improvement. Focus on understanding the core concepts."
    
    evaluation = AnswerEvaluation(
        question_id=current_question["id"],
        score=round(score, 1),
        concept_scores=concept_scores,
        feedback=feedback,
        weak_concepts=weak_concepts
    )
    
    session["evaluations"].append(evaluation.dict())
    session["weak_concepts"].extend(weak_concepts)
    session["current_index"] += 1
    
    # Check if more questions
    next_idx = session["current_index"]
    is_complete = next_idx >= len(questions)
    next_question = None
    
    if not is_complete:
        next_q = questions[next_idx]
        next_question = InterviewQuestion(
            id=next_q["id"],
            question=next_q["question"],
            topic=next_q["topic"],
            reference_answer=next_q["reference_answer"],
            key_concepts=next_q["key_concepts"]
        )
    
    return SubmitAnswerResponse(
        evaluation=evaluation,
        next_question=next_question,
        is_complete=is_complete,
        progress={
            "current": next_idx,
            "total": len(questions),
            "percentage": round((next_idx / len(questions)) * 100)
        }
    )


@router.get("/summary/{session_id}", response_model=InterviewSummary)
async def get_interview_summary(session_id: str):
    """Get summary of a completed interview."""
    
    session = ACTIVE_SESSIONS.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    evaluations = session["evaluations"]
    if not evaluations:
        raise HTTPException(status_code=400, detail="Interview not complete")
    
    # Calculate stats
    scores = [e["score"] for e in evaluations]
    avg_score = sum(scores) / len(scores)
    
    # Aggregate concept mastery
    concept_mastery = {}
    for eval in evaluations:
        for concept, score in eval["concept_scores"].items():
            if concept not in concept_mastery:
                concept_mastery[concept] = []
            concept_mastery[concept].append(score)
    
    concept_mastery = {k: round(sum(v)/len(v), 1) for k, v in concept_mastery.items()}
    
    # Get unique weak topics
    weak_topics = list(set(session["weak_concepts"]))
    
    # Generate recommendations
    recommendations = []
    for topic in weak_topics[:3]:
        recommendations.append(f"Review the concept of '{topic}' in your study materials")
    if avg_score < 70:
        recommendations.append("Consider revisiting the chapter fundamentals")
    
    duration = (datetime.now() - session["started_at"]).total_seconds() / 60
    
    # ========================================================================
    # Record score for curriculum tracking
    # ========================================================================
    try:
        from app.api.routes.topic_scores import record_interview_score
        user_id = session.get("user_id", "demo-user")
        topic_name = session.get("chapter", session.get("subject", ""))
        if topic_name:
            await record_interview_score(
                user_id=user_id,
                topic_name=topic_name,
                score=avg_score
            )
            logger.info(f"Recorded mock interview score {avg_score} for topic '{topic_name}'")
    except Exception as e:
        logger.warning(f"Failed to record interview score: {e}")
    # ========================================================================
    
    return InterviewSummary(
        session_id=session_id,
        subject=session["subject"],
        chapter=session["chapter"],
        total_questions=len(session["questions"]),
        average_score=round(avg_score, 1),
        concept_mastery=concept_mastery,
        weak_topics=weak_topics,
        recommendations=recommendations,
        duration_minutes=round(duration, 1)
    )
