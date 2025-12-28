"""
Study Planner Agent with LangGraph
Generates personalized study plans using Hugging Face LLM
"""
import logging
import json
from typing import Dict, Any, List, TypedDict
from datetime import datetime, timedelta

from langgraph.graph import StateGraph, END

logger = logging.getLogger(__name__)


# ============================================================================
# State Definition
# ============================================================================

class StudyPlanState(TypedDict):
    """State for study plan generation"""
    # Input
    student_id: str
    weak_topics: List[Dict]
    available_hours: int
    deadline_days: int
    
    # Processing
    topic_priorities: List[Dict]
    
    # Output
    study_plan: Dict
    error: str


# ============================================================================
# Node Functions
# ============================================================================

def analyze_topics(state: StudyPlanState) -> StudyPlanState:
    """Analyze and prioritize topics based on weakness scores"""
    if not state["weak_topics"]:
        state["error"] = "No topics to plan"
        return state
    
    # Sort by weakness score (lower = weaker = higher priority)
    sorted_topics = sorted(
        state["weak_topics"],
        key=lambda x: x.get("mastery", 50)
    )
    
    # Assign priorities
    priorities = []
    for i, topic in enumerate(sorted_topics):
        priority = "high" if i < len(sorted_topics) // 3 else (
            "medium" if i < 2 * len(sorted_topics) // 3 else "low"
        )
        priorities.append({
            **topic,
            "priority": priority,
            "rank": i + 1
        })
    
    state["topic_priorities"] = priorities
    logger.info(f"Prioritized {len(priorities)} topics")
    return state


def generate_plan(state: StudyPlanState) -> StudyPlanState:
    """Generate study plan using Hugging Face LLM"""
    if state.get("error"):
        return state
    
    try:
        from app.services.llm_provider import get_llm
        
        llm = get_llm()
        
        # Build topic summary
        topic_summary = "\n".join([
            f"- {t['topic']} ({t['priority']} priority, mastery: {t.get('mastery', 0)}%)"
            for t in state["topic_priorities"][:5]
        ])
        
        prompt = f"""Create a study plan for a student with these weak topics:

{topic_summary}

Available study time: {state['available_hours']} hours total
Deadline: {state['deadline_days']} days

Generate a JSON study plan:
{{
  "daily_schedule": [
    {{
      "day": 1,
      "topics": ["topic1", "topic2"],
      "hours": 2,
      "activities": ["Review concepts", "Practice problems"]
    }}
  ],
  "recommendations": [
    "Focus on X first because...",
    "Practice Y daily..."
  ],
  "milestones": [
    {{"day": 3, "goal": "Complete X basics"}}
  ]
}}

Return ONLY valid JSON."""

        response = llm.invoke(prompt)
        
        # Parse JSON
        try:
            text = response.strip()
            if "```" in text:
                text = text.split("```")[1].replace("json", "").strip()
            
            plan_data = json.loads(text)
            
            state["study_plan"] = {
                "student_id": state["student_id"],
                "topics": state["topic_priorities"],
                "total_hours": state["available_hours"],
                "duration_days": state["deadline_days"],
                **plan_data
            }
            
        except json.JSONDecodeError:
            # Fallback to simple plan
            state["study_plan"] = create_fallback_plan(state)
            
        logger.info("Generated study plan")
        
    except Exception as e:
        logger.error(f"Plan generation error: {e}")
        state["study_plan"] = create_fallback_plan(state)
        state["error"] = str(e)
    
    return state


def create_fallback_plan(state: StudyPlanState) -> Dict:
    """Create simple fallback plan when LLM fails"""
    hours_per_day = max(1, state["available_hours"] // state["deadline_days"])
    topics = state["topic_priorities"]
    
    daily_schedule = []
    topic_idx = 0
    
    for day in range(1, state["deadline_days"] + 1):
        if topic_idx < len(topics):
            daily_schedule.append({
                "day": day,
                "topics": [topics[topic_idx]["topic"]],
                "hours": hours_per_day,
                "activities": ["Review concepts", "Practice exercises"]
            })
            topic_idx += 1
    
    return {
        "student_id": state["student_id"],
        "topics": topics,
        "total_hours": state["available_hours"],
        "duration_days": state["deadline_days"],
        "daily_schedule": daily_schedule,
        "recommendations": [
            "Focus on high priority topics first",
            "Take regular breaks every 45 minutes",
            "Review previous day's material each morning"
        ],
        "milestones": []
    }


# ============================================================================
# Graph Builder
# ============================================================================

def build_study_planner_graph():
    """Build LangGraph workflow for study planning"""
    workflow = StateGraph(StudyPlanState)
    
    workflow.add_node("analyze", analyze_topics)
    workflow.add_node("plan", generate_plan)
    
    workflow.set_entry_point("analyze")
    workflow.add_edge("analyze", "plan")
    workflow.add_edge("plan", END)
    
    return workflow.compile()


# ============================================================================
# Agent Class
# ============================================================================

class StudyPlannerAgent:
    """
    LangGraph-based Study Planner Agent
    
    Creates personalized study plans using:
    - Topic priority analysis
    - Hugging Face LLM for plan generation
    - Adaptive scheduling
    """
    
    def __init__(self):
        self.graph = build_study_planner_graph()
        logger.info("Initialized LangGraph Study Planner Agent")
    
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate study plan
        
        Args:
            input_data: {student_id, weak_topics, available_hours?, deadline_days?}
        
        Returns:
            {study_plan: {...}}
        """
        initial_state: StudyPlanState = {
            "student_id": input_data.get("student_id", ""),
            "weak_topics": input_data.get("weak_topics", []),
            "available_hours": input_data.get("available_hours", 10),
            "deadline_days": input_data.get("deadline_days", 7),
            "topic_priorities": [],
            "study_plan": {},
            "error": ""
        }
        
        try:
            final_state = await self.graph.ainvoke(initial_state)
            
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "agent": "study_planner",
                "data": final_state["study_plan"]
            }
            
        except Exception as e:
            logger.error(f"Study planner error: {e}")
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "agent": "study_planner",
                "data": {
                    "study_plan": create_fallback_plan(initial_state),
                    "error": str(e)
                }
            }
