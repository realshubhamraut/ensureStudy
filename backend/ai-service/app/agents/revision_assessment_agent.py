"""
Revision Assessment Agent - Type 5 AI Agent for Daily MCQ Generation

Automatically generates MCQ assessments based on the AI Revision Schedule.
- Fetches topics scheduled for revision today
- Creates or appends questions to existing revision assessments
- Uses LangGraph for workflow orchestration
"""
import logging
import json
import httpx
from typing import Dict, Any, List, TypedDict, Optional
from datetime import datetime, date

from langgraph.graph import StateGraph, END

logger = logging.getLogger(__name__)


# ============================================================================
# State Definition
# ============================================================================

class RevisionAssessmentState(TypedDict):
    """State for the Revision Assessment Agent"""
    # Input
    user_id: str
    target_date: str  # ISO format date
    auth_token: Optional[str]
    
    # Fetched data
    revision_topics: List[Dict]  # Topics scheduled for revision today
    existing_assessment_id: Optional[str]
    existing_questions: List[Dict]
    
    # Generation
    topics_to_generate: List[Dict]  # Topics that need new questions
    generated_questions: List[Dict]
    
    # Output
    assessment_id: Optional[str]
    total_questions: int
    new_questions_added: int
    error: Optional[str]


# ============================================================================
# Node Functions
# ============================================================================

def fetch_revision_topics(state: RevisionAssessmentState) -> RevisionAssessmentState:
    """Fetch topics scheduled for revision on target date from Core Service."""
    try:
        import os
        core_service_url = os.getenv("CORE_SERVICE_URL", "http://localhost:8000")
        
        # For internal service calls, we use internal endpoint
        # The endpoint /api/curriculum/revision-schedule returns the weekly schedule
        # We need to filter for the target date
        
        target_date = state["target_date"]
        
        # Internal call - we'll create a dedicated internal endpoint
        # For now, simulate with direct database query via internal API
        
        # Get week offset (0 = current week)
        target = datetime.fromisoformat(target_date).date()
        today = date.today()
        week_offset = (target - today).days // 7
        
        response = httpx.get(
            f"{core_service_url}/api/curriculum/revision-schedule",
            params={"week_offset": week_offset},
            headers={"Authorization": f"Bearer {state.get('auth_token', '')}"},
            timeout=30.0
        )
        
        if response.status_code != 200:
            state["error"] = f"Failed to fetch revision schedule: {response.status_code}"
            state["revision_topics"] = []
            return state
        
        data = response.json()
        schedule = data.get("schedule", {})
        
        # Get topics for target date
        topics_for_date = schedule.get(target_date, [])
        
        state["revision_topics"] = topics_for_date
        logger.info(f"Found {len(topics_for_date)} topics scheduled for revision on {target_date}")
        
    except Exception as e:
        logger.error(f"Error fetching revision topics: {e}")
        state["error"] = str(e)
        state["revision_topics"] = []
    
    return state


def check_existing_assessment(state: RevisionAssessmentState) -> RevisionAssessmentState:
    """Check if a revision assessment already exists for this date."""
    if state.get("error") or not state["revision_topics"]:
        return state
    
    try:
        import os
        core_service_url = os.getenv("CORE_SERVICE_URL", "http://localhost:8000")
        
        target_date = state["target_date"]
        
        # Check for existing revision assessment via API
        response = httpx.get(
            f"{core_service_url}/api/assessments/daily-revision",
            params={"date": target_date},
            headers={"Authorization": f"Bearer {state.get('auth_token', '')}"},
            timeout=30.0
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get("assessment"):
                state["existing_assessment_id"] = data["assessment"]["id"]
                state["existing_questions"] = data["assessment"].get("questions", [])
                logger.info(f"Found existing revision assessment: {state['existing_assessment_id']}")
            else:
                state["existing_assessment_id"] = None
                state["existing_questions"] = []
        else:
            state["existing_assessment_id"] = None
            state["existing_questions"] = []
            
    except Exception as e:
        logger.warning(f"Error checking existing assessment: {e}")
        state["existing_assessment_id"] = None
        state["existing_questions"] = []
    
    return state


def determine_topics_to_generate(state: RevisionAssessmentState) -> RevisionAssessmentState:
    """Determine which topics need new questions generated."""
    if state.get("error"):
        return state
    
    revision_topics = state["revision_topics"]
    existing_questions = state.get("existing_questions", [])
    
    # Extract topic IDs from existing questions
    existing_topic_ids = set()
    for q in existing_questions:
        if q.get("topic_id"):
            existing_topic_ids.add(q["topic_id"])
    
    # Filter topics that don't have questions yet
    topics_to_generate = []
    for topic in revision_topics:
        topic_id = topic.get("topic_id")
        if topic_id and topic_id not in existing_topic_ids:
            topics_to_generate.append(topic)
    
    state["topics_to_generate"] = topics_to_generate
    logger.info(f"Topics needing new questions: {len(topics_to_generate)}")
    
    return state


def generate_questions(state: RevisionAssessmentState) -> RevisionAssessmentState:
    """Generate MCQ questions for revision topics using LLM."""
    if state.get("error"):
        return state
    
    topics_to_generate = state.get("topics_to_generate", [])
    
    if not topics_to_generate:
        state["generated_questions"] = []
        logger.info("No topics need question generation")
        return state
    
    try:
        from app.services.llm_provider import get_llm
        
        llm = get_llm()
        all_questions = []
        questions_per_topic = 3  # Generate 3 questions per topic
        
        for topic in topics_to_generate:
            topic_name = topic.get("topic_name", "General")
            subject = topic.get("subject_name", "")
            mastery = topic.get("mastery_percentage", 50)
            
            # Determine difficulty based on mastery
            if mastery < 40:
                difficulty = "easy"
                guidance = "Basic recall and understanding questions to build foundation."
            elif mastery < 70:
                difficulty = "medium"
                guidance = "Application and analysis questions to deepen understanding."
            else:
                difficulty = "hard"
                guidance = "Synthesis and evaluation - complex scenarios to challenge mastery."
            
            prompt = f"""Generate {questions_per_topic} multiple choice questions for revision on the topic "{topic_name}" in {subject}.

Student's current mastery: {mastery}%
Difficulty level: {difficulty}
{guidance}

These questions are for spaced repetition review to reinforce learning.

Return a JSON array with EXACTLY this format:
[
  {{
    "question": "Clear, focused question text?",
    "options": ["Option A", "Option B", "Option C", "Option D"],
    "correct_answer": "A",
    "explanation": "Brief explanation of why this is correct",
    "difficulty": "{difficulty}",
    "topic": "{topic_name}"
  }}
]

Return ONLY the JSON array, no other text."""

            try:
                response = llm.invoke(prompt)
                text = response.strip()
                
                # Parse JSON from response
                if "```" in text:
                    text = text.split("```")[1].replace("json", "").strip()
                
                questions = json.loads(text)
                
                # Validate and add topic_id
                for q in questions:
                    if all(k in q for k in ["question", "options", "correct_answer"]):
                        if len(q["correct_answer"]) > 1:
                            q["correct_answer"] = q["correct_answer"][0].upper()
                        q["topic_id"] = topic.get("topic_id")
                        q["topic_name"] = topic_name
                        q["subject"] = subject
                        q["generated_at"] = datetime.utcnow().isoformat()
                        all_questions.append(q)
                        
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse questions for {topic_name}")
            except Exception as e:
                logger.warning(f"Error generating questions for {topic_name}: {e}")
        
        state["generated_questions"] = all_questions
        logger.info(f"Generated {len(all_questions)} new questions")
        
    except Exception as e:
        logger.error(f"Question generation error: {e}")
        state["error"] = str(e)
        state["generated_questions"] = []
    
    return state


def save_assessment(state: RevisionAssessmentState) -> RevisionAssessmentState:
    """Save or update the revision assessment."""
    if state.get("error"):
        return state
    
    generated_questions = state.get("generated_questions", [])
    existing_questions = state.get("existing_questions", [])
    existing_id = state.get("existing_assessment_id")
    
    all_questions = existing_questions + generated_questions
    
    if not all_questions:
        state["error"] = "No questions to save"
        return state
    
    try:
        import os
        core_service_url = os.getenv("CORE_SERVICE_URL", "http://localhost:8000")
        target_date = state["target_date"]
        
        if existing_id:
            # Update existing assessment with new questions
            response = httpx.patch(
                f"{core_service_url}/api/assessments/{existing_id}/append-questions",
                json={"questions": generated_questions},
                headers={
                    "Authorization": f"Bearer {state.get('auth_token', '')}",
                    "Content-Type": "application/json"
                },
                timeout=30.0
            )
        else:
            # Create new revision assessment
            # Get topic names for title
            topic_names = [t.get("topic_name", "") for t in state["revision_topics"][:3]]
            topics_str = ", ".join(topic_names)
            if len(state["revision_topics"]) > 3:
                topics_str += f" +{len(state['revision_topics']) - 3} more"
            
            response = httpx.post(
                f"{core_service_url}/api/assessments/",
                json={
                    "title": f"Daily Revision - {target_date}",
                    "topic": topics_str,
                    "subject": "Revision",
                    "description": f"Auto-generated revision quiz for {target_date}",
                    "questions": all_questions,
                    "difficulty": "mixed",
                    "time_limit_minutes": len(all_questions) * 2,  # 2 min per question
                    "assessment_type": "self_practice",
                    "use_ai_questions": True,
                    "is_revision_assessment": True,
                    "revision_date": target_date
                },
                headers={
                    "Authorization": f"Bearer {state.get('auth_token', '')}",
                    "Content-Type": "application/json"
                },
                timeout=30.0
            )
        
        if response.status_code in [200, 201]:
            data = response.json()
            state["assessment_id"] = data.get("id") or data.get("assessment", {}).get("id")
            state["total_questions"] = len(all_questions)
            state["new_questions_added"] = len(generated_questions)
            logger.info(f"Saved revision assessment: {state['assessment_id']}")
        else:
            state["error"] = f"Failed to save assessment: {response.status_code}"
            logger.error(f"Save failed: {response.text}")
            
    except Exception as e:
        logger.error(f"Error saving assessment: {e}")
        state["error"] = str(e)
    
    return state


def format_output(state: RevisionAssessmentState) -> RevisionAssessmentState:
    """Format the final output."""
    # Just pass through - the state already contains all needed info
    return state


# ============================================================================
# Graph Builder
# ============================================================================

def build_revision_assessment_graph():
    """Build LangGraph workflow for revision assessment generation."""
    workflow = StateGraph(RevisionAssessmentState)
    
    # Add nodes
    workflow.add_node("fetch_topics", fetch_revision_topics)
    workflow.add_node("check_existing", check_existing_assessment)
    workflow.add_node("determine_topics", determine_topics_to_generate)
    workflow.add_node("generate", generate_questions)
    workflow.add_node("save", save_assessment)
    workflow.add_node("format", format_output)
    
    # Define edges
    workflow.set_entry_point("fetch_topics")
    workflow.add_edge("fetch_topics", "check_existing")
    workflow.add_edge("check_existing", "determine_topics")
    workflow.add_edge("determine_topics", "generate")
    workflow.add_edge("generate", "save")
    workflow.add_edge("save", "format")
    workflow.add_edge("format", END)
    
    return workflow.compile()


# ============================================================================
# Agent Class
# ============================================================================

class RevisionAssessmentAgent:
    """
    Type 5 AI Agent for Daily Revision Assessment Generation
    
    Features:
    - Fetches topics from AI Revision Schedule
    - Creates or appends to daily revision assessments
    - Adaptive difficulty based on mastery level
    - Deduplication of questions
    """
    
    def __init__(self):
        self.graph = build_revision_assessment_graph()
        logger.info("Initialized Revision Assessment Agent")
    
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate revision assessment for a given date.
        
        Args:
            input_data: {
                user_id: str,
                date: str (ISO format, optional - defaults to today),
                auth_token: str (for API calls)
            }
        
        Returns:
            {
                success: bool,
                assessment_id: str,
                total_questions: int,
                new_questions_added: int,
                topics_covered: List[str],
                error: str (if any)
            }
        """
        target_date = input_data.get("date", date.today().isoformat())
        
        initial_state: RevisionAssessmentState = {
            "user_id": input_data.get("user_id", ""),
            "target_date": target_date,
            "auth_token": input_data.get("auth_token"),
            "revision_topics": [],
            "existing_assessment_id": None,
            "existing_questions": [],
            "topics_to_generate": [],
            "generated_questions": [],
            "assessment_id": None,
            "total_questions": 0,
            "new_questions_added": 0,
            "error": None
        }
        
        try:
            final_state = await self.graph.ainvoke(initial_state)
            
            return {
                "success": final_state.get("error") is None,
                "assessment_id": final_state.get("assessment_id"),
                "total_questions": final_state.get("total_questions", 0),
                "new_questions_added": final_state.get("new_questions_added", 0),
                "topics_covered": [t.get("topic_name") for t in final_state.get("revision_topics", [])],
                "error": final_state.get("error"),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Revision Assessment Agent error: {e}")
            return {
                "success": False,
                "assessment_id": None,
                "total_questions": 0,
                "new_questions_added": 0,
                "topics_covered": [],
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def execute_sync(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Synchronous wrapper for execute."""
        import asyncio
        return asyncio.run(self.execute(input_data))


# ============================================================================
# Singleton
# ============================================================================

_revision_agent = None

def get_revision_assessment_agent() -> RevisionAssessmentAgent:
    """Get or create the Revision Assessment Agent singleton."""
    global _revision_agent
    if _revision_agent is None:
        _revision_agent = RevisionAssessmentAgent()
    return _revision_agent
