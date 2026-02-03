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

router = APIRouter(prefix="/api/mock-interview", tags=["Mock Interview"])


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


# ============================================
# New Topic-Based Interview System (DB-backed)
# ============================================

import httpx
import os
import logging

logger = logging.getLogger(__name__)

# Core service URL
CORE_SERVICE_URL = os.getenv("CORE_SERVICE_URL", "http://localhost:5000")


class StartTopicInterviewRequest(BaseModel):
    """Request to start interview with specific topic IDs."""
    user_id: str
    topic_ids: List[str] = Field(..., description="List of ClassroomTopic IDs")
    avatar: str = Field(default="female", description="Avatar type: male or female")
    questions_per_topic: int = Field(default=3, ge=1, le=10)
    token: str = Field(..., description="Auth token for API calls")


class TopicInterviewQuestion(BaseModel):
    """Question from database for topic-based interview."""
    id: str
    question: str
    topic_id: str
    topic_name: str
    difficulty: str = "medium"
    # Note: reference_answer is NOT exposed to student
    

class StartTopicInterviewResponse(BaseModel):
    """Response when starting topic-based interview."""
    session_id: str
    question: TopicInterviewQuestion
    total_questions: int
    topics: List[dict]
    message: str


class SubmitTopicAnswerRequest(BaseModel):
    """Request to submit answer for topic-based interview."""
    session_id: str
    question_id: str
    answer_text: str
    audio_duration_seconds: Optional[float] = None
    response_time_seconds: Optional[int] = None
    token: str


class TopicAnswerEvaluation(BaseModel):
    """Enhanced evaluation with concept-level feedback."""
    question_id: str
    score: float = Field(..., ge=0, le=100)
    concept_scores: dict
    covered_concepts: List[str]
    missed_concepts: List[str]
    feedback: str
    expected_answer_summary: str  # Brief hint, not full answer


class SubmitTopicAnswerResponse(BaseModel):
    """Response after submitting answer."""
    evaluation: TopicAnswerEvaluation
    next_question: Optional[TopicInterviewQuestion] = None
    is_complete: bool
    progress: dict


# Session storage for topic-based interviews
TOPIC_INTERVIEW_SESSIONS: dict = {}


async def fetch_questions_from_db(topic_ids: List[str], questions_per_topic: int, token: str, session_id: str) -> List[dict]:
    """Fetch questions from core-service database."""
    try:
        async with httpx.AsyncClient(verify=False) as client:
            response = await client.post(
                f"{CORE_SERVICE_URL}/api/interview-questions/topics/batch",
                json={
                    "topic_ids": topic_ids,
                    "questions_per_topic": questions_per_topic,
                    "session_id": session_id
                },
                headers={"Authorization": f"Bearer {token}"},
                timeout=30.0
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get("questions", []), data.get("topics", [])
            else:
                logger.warning(f"Failed to fetch questions: {response.status_code}")
                return [], []
    except Exception as e:
        logger.error(f"Error fetching questions: {e}")
        return [], []


async def fetch_topic_info(topic_ids: List[str], token: str) -> List[dict]:
    """Fetch topic information from core-service for LLM generation."""
    topics = []
    try:
        async with httpx.AsyncClient(verify=False) as client:
            for topic_id in topic_ids:
                response = await client.get(
                    f"{CORE_SERVICE_URL}/api/curriculum/classroom-topics/{topic_id}",
                    headers={"Authorization": f"Bearer {token}"},
                    timeout=10.0
                )
                
                if response.status_code == 200:
                    data = response.json()
                    topic = data.get("topic", data)
                    topics.append({
                        "id": topic.get("id", topic_id),
                        "name": topic.get("name", "Unknown Topic"),
                        "description": topic.get("description", "")
                    })
                else:
                    logger.warning(f"Failed to fetch topic {topic_id}: {response.status_code}")
                    topics.append({"id": topic_id, "name": f"Topic {topic_id}", "description": ""})
    except Exception as e:
        logger.error(f"Error fetching topic info: {e}")
        # Fallback - use IDs as names
        for topic_id in topic_ids:
            topics.append({"id": topic_id, "name": f"Topic {topic_id}", "description": ""})
    
    return topics


async def get_question_evaluation_data(question_id: str, token: str) -> dict:
    """Fetch full question data (including expected answer) for evaluation."""
    try:
        async with httpx.AsyncClient(verify=False) as client:
            # Get question with expected answer from AI service's cache or generate
            # For now, fetch from session's question store
            pass
    except Exception as e:
        logger.error(f"Error fetching question data: {e}")
    return {}


async def evaluate_answer_with_llm(student_answer: str, expected_answer: str, key_concepts: List[str]) -> dict:
    """
    Enhanced answer evaluation using LLM for semantic similarity.
    """
    try:
        from app.services.llm_provider import get_llm
        
        llm = get_llm()
        
        prompt = f"""Evaluate this student's answer against the expected answer.

QUESTION CONTEXT:
Expected Answer: {expected_answer}
Key Concepts that should be covered: {', '.join(key_concepts)}

STUDENT'S ANSWER:
{student_answer}

Provide evaluation in JSON format:
{{
    "score": <0-100 overall score>,
    "concept_scores": {{"<concept>": <0-100>, ...}},
    "covered_concepts": ["concepts the student mentioned correctly"],
    "missed_concepts": ["important concepts the student missed"],
    "feedback": "Brief constructive feedback for the student",
    "answer_quality": "excellent|good|partial|needs_improvement"
}}

Return ONLY the JSON."""

        response = llm.invoke(prompt)
        
        # Parse response
        text = response.strip()
        if "```" in text:
            text = text.split("```")[1].replace("json", "").strip()
        
        import json
        result = json.loads(text)
        return result
        
    except Exception as e:
        logger.warning(f"LLM evaluation failed, using fallback: {e}")
        # Fallback to simple keyword matching
        return fallback_evaluation(student_answer, expected_answer, key_concepts)


def fallback_evaluation(student_answer: str, expected_answer: str, key_concepts: List[str]) -> dict:
    """Fallback evaluation using keyword matching."""
    answer_lower = student_answer.lower()
    
    concept_scores = {}
    covered = []
    missed = []
    
    for concept in key_concepts:
        if concept.lower() in answer_lower:
            concept_scores[concept] = 100
            covered.append(concept)
        elif any(word in answer_lower for word in concept.lower().split()):
            concept_scores[concept] = 60
            covered.append(concept)
        else:
            concept_scores[concept] = 0
            missed.append(concept)
    
    avg_score = sum(concept_scores.values()) / len(concept_scores) if concept_scores else 50
    length_factor = min(1.0, len(student_answer.split()) / 50)  # Expect ~50 words
    score = avg_score * 0.7 + length_factor * 30
    
    if score >= 80:
        feedback = "Excellent answer! You covered the key concepts well."
    elif score >= 60:
        feedback = f"Good answer, but consider elaborating on: {', '.join(missed[:2])}"
    else:
        feedback = f"Review these concepts: {', '.join(missed[:3])}"
    
    return {
        "score": round(score, 1),
        "concept_scores": concept_scores,
        "covered_concepts": covered,
        "missed_concepts": missed,
        "feedback": feedback,
        "answer_quality": "good" if score >= 60 else "needs_improvement"
    }


async def update_question_stats(question_id: str, score: float, data: dict, token: str):
    """Update question statistics in core-service."""
    try:
        async with httpx.AsyncClient(verify=False) as client:
            await client.put(
                f"{CORE_SERVICE_URL}/api/interview-questions/{question_id}/stats",
                json={
                    "score": score,
                    "session_id": data.get("session_id"),
                    "student_answer": data.get("student_answer"),
                    "feedback": data.get("feedback"),
                    "concept_scores": data.get("concept_scores"),
                    "response_time_seconds": data.get("response_time_seconds"),
                    "audio_duration_seconds": data.get("audio_duration_seconds")
                },
                headers={"Authorization": f"Bearer {token}"},
                timeout=10.0
            )
    except Exception as e:
        logger.warning(f"Failed to update question stats: {e}")


async def check_and_generate_more_questions(topic_ids: List[str], token: str):
    """Check if question pool is low and trigger generation."""
    try:
        async with httpx.AsyncClient(verify=False) as client:
            for topic_id in topic_ids:
                response = await client.get(
                    f"{CORE_SERVICE_URL}/api/interview-questions/topic/{topic_id}/count",
                    headers={"Authorization": f"Bearer {token}"},
                    timeout=10.0
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get("needs_generation", False):
                        logger.info(f"Topic {topic_id} needs more questions, triggering generation")
                        # This could be done async in background
                        # For now, we'll let the core-service handle it
    except Exception as e:
        logger.warning(f"Error checking question pool: {e}")


@router.post("/start-topic-interview", response_model=StartTopicInterviewResponse)
async def start_topic_interview(request: StartTopicInterviewRequest):
    """
    Start a mock interview session using specific topic IDs.
    Fetches questions from database instead of hardcoded banks.
    """
    session_id = str(uuid.uuid4())
    
    # Fetch questions from database
    questions, topics = await fetch_questions_from_db(
        request.topic_ids, 
        request.questions_per_topic, 
        request.token,
        session_id
    )
    
    if not questions:
        # No questions in DB - generate them on-the-fly using LLM
        logger.info("No questions found in DB, generating with LLM...")
        
        # If topics is empty, fetch topic info separately
        if not topics:
            topics = await fetch_topic_info(request.topic_ids, request.token)
            logger.info(f"Fetched topic info: {[t.get('name') for t in topics]}")
        
        # If still no topics, create fallback based on request
        if not topics:
            logger.warning("No topic info available, using fallback topic names")
            topics = [{"id": tid, "name": f"Topic {i+1}", "description": ""} 
                      for i, tid in enumerate(request.topic_ids)]
        
        logger.info(f"Topics to generate for: {topics}")
        
        from app.agents.interview_question_agent import get_interview_question_agent
        
        try:
            agent = get_interview_question_agent()
            logger.info(f"Agent initialized: {agent}")
            
            # Generate questions for each topic
            all_generated = []
            for topic_info in topics:
                logger.info(f"Generating for topic: {topic_info.get('name')}")
                topic_questions = await agent.generate_for_single_topic(
                    topic_id=topic_info.get("id", ""),
                    topic_name=topic_info.get("name", "Unknown Topic"),
                    description=topic_info.get("description", ""),
                    count=request.questions_per_topic,
                    difficulty="medium"
                )
                
                logger.info(f"Generated {len(topic_questions)} questions for {topic_info.get('name')}")
                
                # Format questions for session
                for q in topic_questions:
                    all_generated.append({
                        "id": str(uuid.uuid4()),  # Generate unique ID
                        "question": q.get("question", ""),
                        "topic_id": topic_info.get("id", ""),
                        "topic_name": topic_info.get("name", ""),
                        "difficulty": q.get("difficulty", "medium"),
                        "expected_answer": q.get("expected_answer", ""),
                        "key_concepts": q.get("key_concepts", [])
                    })
            
            logger.info(f"Total generated: {len(all_generated)}")
            
            if all_generated:
                questions = all_generated
                logger.info(f"Generated {len(questions)} questions via LLM for topics")
            else:
                logger.error("No questions were generated - all_generated is empty")
                raise HTTPException(
                    status_code=404,
                    detail="Failed to generate questions. Please try again."
                )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"LLM question generation failed: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to generate questions: {str(e)}"
            )
    
    first_question = questions[0]
    
    # Store session with full question data (including expected answers for evaluation)
    TOPIC_INTERVIEW_SESSIONS[session_id] = {
        "user_id": request.user_id,
        "topic_ids": request.topic_ids,
        "topics": topics,
        "avatar": request.avatar,
        "questions": questions,  # Contains full data
        "current_index": 0,
        "evaluations": [],
        "weak_concepts": [],
        "token": request.token,
        "started_at": datetime.now()
    }
    
    return StartTopicInterviewResponse(
        session_id=session_id,
        question=TopicInterviewQuestion(
            id=first_question["id"],
            question=first_question["question"],
            topic_id=first_question.get("topic_id", ""),
            topic_name=first_question.get("topic_name", ""),
            difficulty=first_question.get("difficulty", "medium")
        ),
        total_questions=len(questions),
        topics=[{"id": t["id"], "name": t["name"]} for t in topics],
        message="Interview started. Answer the questions verbally."
    )


@router.post("/submit-topic-answer", response_model=SubmitTopicAnswerResponse)
async def submit_topic_answer(request: SubmitTopicAnswerRequest):
    """
    Submit answer for topic-based interview and get evaluation.
    Uses LLM for semantic similarity evaluation.
    """
    session = TOPIC_INTERVIEW_SESSIONS.get(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    questions = session["questions"]
    current_idx = session["current_index"]
    
    if current_idx >= len(questions):
        raise HTTPException(status_code=400, detail="Interview already complete")
    
    current_question = questions[current_idx]
    
    # Get expected answer and key concepts (stored in session, not exposed to student)
    expected_answer = current_question.get("reference_answer", current_question.get("expected_answer", ""))
    key_concepts = current_question.get("key_concepts", [])
    
    # Evaluate using LLM
    eval_result = await evaluate_answer_with_llm(
        request.answer_text,
        expected_answer,
        key_concepts
    )
    
    score = eval_result.get("score", 50)
    
    # Create evaluation response
    evaluation = TopicAnswerEvaluation(
        question_id=current_question["id"],
        score=score,
        concept_scores=eval_result.get("concept_scores", {}),
        covered_concepts=eval_result.get("covered_concepts", []),
        missed_concepts=eval_result.get("missed_concepts", []),
        feedback=eval_result.get("feedback", "Answer recorded."),
        expected_answer_summary=expected_answer[:100] + "..." if len(expected_answer) > 100 else expected_answer
    )
    
    # Store evaluation
    session["evaluations"].append(evaluation.dict())
    session["weak_concepts"].extend(eval_result.get("missed_concepts", []))
    session["current_index"] += 1
    
    # Update question stats in background
    await update_question_stats(
        current_question["id"],
        score,
        {
            "session_id": request.session_id,
            "student_answer": request.answer_text,
            "feedback": evaluation.feedback,
            "concept_scores": evaluation.concept_scores,
            "response_time_seconds": request.response_time_seconds,
            "audio_duration_seconds": request.audio_duration_seconds
        },
        request.token
    )
    
    # Check if more questions
    next_idx = session["current_index"]
    is_complete = next_idx >= len(questions)
    next_question = None
    
    if not is_complete:
        next_q = questions[next_idx]
        next_question = TopicInterviewQuestion(
            id=next_q["id"],
            question=next_q["question"],
            topic_id=next_q.get("topic_id", ""),
            topic_name=next_q.get("topic_name", ""),
            difficulty=next_q.get("difficulty", "medium")
        )
    else:
        # Interview complete - check if we need more questions
        await check_and_generate_more_questions(session["topic_ids"], session["token"])
    
    return SubmitTopicAnswerResponse(
        evaluation=evaluation,
        next_question=next_question,
        is_complete=is_complete,
        progress={
            "current": next_idx,
            "total": len(questions),
            "percentage": round((next_idx / len(questions)) * 100)
        }
    )


@router.get("/topic-interview-summary/{session_id}")
async def get_topic_interview_summary(session_id: str):
    """Get summary of a topic-based interview."""
    session = TOPIC_INTERVIEW_SESSIONS.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    evaluations = session["evaluations"]
    if not evaluations:
        raise HTTPException(status_code=400, detail="No answers submitted yet")
    
    scores = [e["score"] for e in evaluations]
    avg_score = sum(scores) / len(scores)
    
    # Concept mastery
    concept_mastery = {}
    for eval in evaluations:
        for concept, score in eval.get("concept_scores", {}).items():
            if concept not in concept_mastery:
                concept_mastery[concept] = []
            concept_mastery[concept].append(score)
    concept_mastery = {k: round(sum(v)/len(v), 1) for k, v in concept_mastery.items()}
    
    weak_topics = list(set(session["weak_concepts"]))
    
    duration = (datetime.now() - session["started_at"]).total_seconds() / 60
    
    # Record scores for mastery tracking
    try:
        from app.api.routes.topic_scores import record_interview_score
        for topic in session.get("topics", []):
            await record_interview_score(
                user_id=session["user_id"],
                topic_name=topic["name"],
                score=avg_score
            )
    except Exception as e:
        logger.warning(f"Failed to record topic scores: {e}")
    
    return {
        "session_id": session_id,
        "topics": session["topics"],
        "total_questions": len(session["questions"]),
        "questions_answered": len(evaluations),
        "average_score": round(avg_score, 1),
        "concept_mastery": concept_mastery,
        "weak_concepts": weak_topics[:10],
        "strong_concepts": [c for c, s in concept_mastery.items() if s >= 80],
        "duration_minutes": round(duration, 1),
        "recommendations": [
            f"Review: {concept}" for concept in weak_topics[:3]
        ] if weak_topics else ["Great job! All concepts covered well."]
    }


@router.post("/generate-topic-questions")
async def generate_topic_questions(
    topics: List[dict],
    questions_per_topic: int = 5,
    difficulty: str = "medium"
):
    """
    Generate new questions for topics using AI.
    Called when question pool is low.
    """
    try:
        from app.agents.interview_question_agent import get_interview_question_agent
        
        agent = get_interview_question_agent()
        result = await agent.generate({
            "topics": topics,
            "questions_per_topic": questions_per_topic,
            "difficulty": difficulty
        })
        
        return result
        
    except Exception as e:
        logger.error(f"Question generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
