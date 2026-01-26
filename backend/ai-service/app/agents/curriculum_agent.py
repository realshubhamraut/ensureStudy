"""
Curriculum Agent - LangGraph Agent for Personalized Learning Paths

This agent processes a syllabus and creates a personalized curriculum:
1. Takes extracted topics from syllabus_extractor
2. Analyzes topic dependencies using LLM
3. Assesses student's current knowledge (optional diagnostic)
4. Builds optimized learning path
5. Creates daily/weekly schedule with milestones

Flow:
┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
│ Load     │──▶│ Analyze  │──▶│ Knowledge│──▶│ Build    │──▶│ Schedule │
│ Topics   │   │ Deps     │   │ Assess   │   │ Path     │   │ Generator│
└──────────┘   └──────────┘   └──────────┘   └──────────┘   └──────────┘
"""
import logging
import json
import uuid
from typing import Dict, Any, List, TypedDict, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict

from langgraph.graph import StateGraph, END

logger = logging.getLogger(__name__)


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class CurriculumTopic:
    """Topic with curriculum metadata"""
    id: str
    name: str
    description: str
    difficulty: str  # "easy", "medium", "hard"
    estimated_hours: float
    prerequisites: List[str]  # List of topic IDs
    subtopics: List[str]
    order: int  # Position in learning path
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class DailyGoal:
    """Daily learning goal"""
    day: int
    date: str
    topics: List[str]  # Topic names
    activities: List[Dict]  # [{type, description, duration_min}]
    total_hours: float
    milestone: Optional[str] = None


@dataclass
class Curriculum:
    """Complete curriculum for a student"""
    id: str
    user_id: str
    syllabus_id: str
    subject_name: str
    created_at: str
    
    # Topics organized for learning
    topics: List[CurriculumTopic]
    topic_order: List[str]  # Ordered list of topic IDs
    
    # Schedule
    start_date: str
    end_date: str
    total_days: int
    hours_per_day: float
    daily_goals: List[DailyGoal]
    milestones: List[Dict]
    
    # Progress tracking
    current_topic_index: int = 0
    completed_topics: List[str] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)


# ============================================================================
# State Definition
# ============================================================================

class CurriculumState(TypedDict):
    """State passed through curriculum generation graph"""
    # Input
    syllabus_id: str
    user_id: str
    classroom_id: str
    subject_name: str
    
    # Configuration
    hours_per_day: float
    deadline_days: int
    start_date: str
    
    # Topics from syllabus
    raw_topics: List[Dict]  # From syllabus_extractor
    
    # Dependency Analysis
    topic_dependencies: Dict[str, List[str]]  # topic_name -> [prereq_names]
    dependency_graph_built: bool
    
    # Knowledge Assessment
    student_knowledge: Dict[str, float]  # topic_name -> mastery (0-1)
    diagnostic_complete: bool
    
    # Learning Path
    ordered_topics: List[CurriculumTopic]
    topic_order: List[str]
    
    # Schedule
    daily_goals: List[Dict]
    milestones: List[Dict]
    
    # Output
    curriculum: Optional[Dict]
    error: Optional[str]


# ============================================================================
# Node Functions
# ============================================================================

async def load_syllabus_topics(state: CurriculumState) -> CurriculumState:
    """Load topics from syllabus extractor or database"""
    
    if state.get("raw_topics"):
        # Topics already provided
        logger.info(f"[CURRICULUM] Using {len(state['raw_topics'])} provided topics")
        return state
    
    # Fetch from database via Core Service
    try:
        import httpx
        from app.config import settings
        
        core_url = getattr(settings, 'CORE_SERVICE_URL', 'http://localhost:8000')
        
        async with httpx.AsyncClient() as client:
            # Get topics for this syllabus
            response = await client.get(
                f"{core_url}/api/topics/syllabus/{state['syllabus_id']}",
                headers={"X-Service-Key": "internal-ai-service"},
                timeout=30.0
            )
            
            if response.status_code == 200:
                data = response.json()
                state["raw_topics"] = data.get("topics", [])
                logger.info(f"[CURRICULUM] Loaded {len(state['raw_topics'])} topics from syllabus")
            else:
                state["error"] = f"Failed to load topics: {response.status_code}"
                
    except Exception as e:
        logger.error(f"[CURRICULUM] Error loading topics: {e}")
        state["error"] = str(e)
    
    return state


async def analyze_dependencies(state: CurriculumState) -> CurriculumState:
    """Use LLM to analyze topic dependencies and prerequisites"""
    
    if state.get("error") or not state.get("raw_topics"):
        return state
    
    try:
        from app.agents.tools import invoke_tool
        
        topic_names = [t.get("name", t.get("topic", "")) for t in state["raw_topics"]]
        
        prompt = f"""Analyze these academic topics and identify their prerequisites/dependencies.
For each topic, list which other topics from this list should be learned BEFORE it.

Topics:
{json.dumps(topic_names, indent=2)}

Subject: {state['subject_name']}

Return a JSON object where keys are topic names and values are arrays of prerequisite topic names.
Only include prerequisites that are IN this list. If a topic has no prerequisites, use empty array.

Example:
{{
  "Integration": ["Derivatives", "Limits"],
  "Derivatives": ["Limits"],
  "Limits": []
}}

Return ONLY valid JSON, no explanation."""

        result = await invoke_tool(
            "llm_generate",
            prompt=prompt,
            max_tokens=2000
        )
        
        if result.success:
            response_text = result.data.get("response", "{}")
            
            # Parse JSON from response
            try:
                # Clean up response
                text = response_text.strip()
                if "```" in text:
                    text = text.split("```")[1].replace("json", "").strip()
                
                dependencies = json.loads(text)
                state["topic_dependencies"] = dependencies
                state["dependency_graph_built"] = True
                
                logger.info(f"[CURRICULUM] Analyzed dependencies for {len(dependencies)} topics")
                
            except json.JSONDecodeError as e:
                logger.warning(f"[CURRICULUM] Could not parse dependencies: {e}")
                # Create empty dependencies as fallback
                state["topic_dependencies"] = {t: [] for t in topic_names}
                state["dependency_graph_built"] = True
        else:
            state["error"] = f"LLM call failed: {result.error}"
            
    except Exception as e:
        logger.error(f"[CURRICULUM] Dependency analysis error: {e}")
        # Fallback: no dependencies
        topic_names = [t.get("name", t.get("topic", "")) for t in state["raw_topics"]]
        state["topic_dependencies"] = {t: [] for t in topic_names}
        state["dependency_graph_built"] = True
    
    return state


async def assess_knowledge(state: CurriculumState) -> CurriculumState:
    """
    Assess student's current knowledge using the assessment service.
    
    Phase 3 Implementation:
    - Uses KnowledgeAssessmentService for quick assessment
    - Can generate diagnostic quiz (async flow for frontend)
    - Adjusts estimated hours based on mastery
    """
    
    if state.get("error"):
        return state
    
    try:
        from app.services.assessment_service import get_assessment_service
        
        assessment_service = get_assessment_service()
        topics = state.get("raw_topics", [])
        
        # Build topic list for assessment
        topic_list = [
            {
                "id": t.get("id", t.get("name", "")),
                "name": t.get("name", t.get("topic", "")),
                "description": t.get("description", "")
            }
            for t in topics
        ]
        
        # Check if we have pre-supplied mastery scores
        if state.get("student_knowledge") and any(v > 0 for v in state["student_knowledge"].values()):
            # Mastery already provided (from diagnostic quiz results)
            logger.info("[CURRICULUM] Using pre-supplied mastery scores")
        else:
            # Quick assessment - assume beginner level
            # Frontend can trigger full diagnostic quiz separately
            mastery_scores = await assessment_service.quick_assessment(
                topics=topic_list,
                assumed_level="beginner"  # Default: no prior knowledge
            )
            state["student_knowledge"] = mastery_scores
        
        # Adjust estimated hours based on mastery
        for topic in topics:
            topic_name = topic.get("name", topic.get("topic", ""))
            mastery = state["student_knowledge"].get(topic_name, 0.0)
            base_hours = topic.get("estimated_hours", 2.0)
            
            # Reduce hours for topics student already knows
            adjusted_hours = assessment_service.calculate_adjusted_hours(
                base_hours=base_hours,
                mastery_score=mastery
            )
            topic["estimated_hours"] = adjusted_hours
            topic["original_hours"] = base_hours
            topic["mastery"] = mastery
        
        state["diagnostic_complete"] = True
        
        # Calculate average mastery for logging
        avg_mastery = sum(state["student_knowledge"].values()) / max(len(state["student_knowledge"]), 1)
        
        logger.info(
            f"[CURRICULUM] Knowledge assessment complete. "
            f"Average mastery: {avg_mastery:.0%}"
        )
        
    except ImportError as e:
        logger.warning(f"[CURRICULUM] Assessment service not available: {e}")
        # Fallback: assume no prior knowledge
        topic_names = [t.get("name", t.get("topic", "")) for t in state.get("raw_topics", [])]
        state["student_knowledge"] = {t: 0.0 for t in topic_names}
        state["diagnostic_complete"] = True
        
    except Exception as e:
        logger.error(f"[CURRICULUM] Assessment error: {e}")
        # Don't fail - just continue with no mastery data
        topic_names = [t.get("name", t.get("topic", "")) for t in state.get("raw_topics", [])]
        state["student_knowledge"] = {t: 0.0 for t in topic_names}
        state["diagnostic_complete"] = True
    
    return state


async def build_learning_path(state: CurriculumState) -> CurriculumState:
    """Build optimized learning path using topological sort"""
    
    if state.get("error"):
        return state
    
    try:
        # Build topic objects
        topic_map = {}
        for t in state["raw_topics"]:
            name = t.get("name", t.get("topic", ""))
            topic_id = t.get("id", str(uuid.uuid4())[:8])
            
            # Get prerequisites for this topic
            prereqs = state["topic_dependencies"].get(name, [])
            
            topic_map[name] = CurriculumTopic(
                id=topic_id,
                name=name,
                description=t.get("description", ""),
                difficulty=t.get("difficulty", "medium"),
                estimated_hours=t.get("estimated_hours", 2.0),
                prerequisites=prereqs,
                subtopics=t.get("subtopics", []),
                order=0
            )
        
        # Topological sort for optimal learning order
        ordered = topological_sort(topic_map, state["topic_dependencies"])
        
        # Assign order numbers
        for i, name in enumerate(ordered):
            if name in topic_map:
                topic_map[name].order = i + 1
        
        # Build ordered list of topics
        state["ordered_topics"] = [topic_map[name] for name in ordered if name in topic_map]
        state["topic_order"] = ordered
        
        logger.info(f"[CURRICULUM] Built learning path with {len(ordered)} topics")
        
    except Exception as e:
        logger.error(f"[CURRICULUM] Error building path: {e}")
        # Fallback: use original order
        state["ordered_topics"] = []
        for i, t in enumerate(state["raw_topics"]):
            name = t.get("name", t.get("topic", ""))
            state["ordered_topics"].append(CurriculumTopic(
                id=t.get("id", str(uuid.uuid4())[:8]),
                name=name,
                description=t.get("description", ""),
                difficulty=t.get("difficulty", "medium"),
                estimated_hours=t.get("estimated_hours", 2.0),
                prerequisites=[],
                subtopics=t.get("subtopics", []),
                order=i + 1
            ))
        state["topic_order"] = [t.name for t in state["ordered_topics"]]
    
    return state


def topological_sort(topic_map: Dict, dependencies: Dict) -> List[str]:
    """Topological sort with cycle detection"""
    
    # Build adjacency list (topic -> topics that depend on it)
    in_degree = {name: 0 for name in topic_map}
    
    for topic, prereqs in dependencies.items():
        if topic in in_degree:
            in_degree[topic] = len([p for p in prereqs if p in topic_map])
    
    # Start with topics that have no prerequisites
    queue = [t for t, degree in in_degree.items() if degree == 0]
    result = []
    
    while queue:
        # Sort queue by difficulty for better ordering
        queue.sort(key=lambda t: {"easy": 0, "medium": 1, "hard": 2}.get(
            topic_map[t].difficulty if t in topic_map else "medium", 1
        ))
        
        current = queue.pop(0)
        result.append(current)
        
        # Decrease in-degree for dependent topics
        for topic, prereqs in dependencies.items():
            if current in prereqs and topic in in_degree:
                in_degree[topic] -= 1
                if in_degree[topic] == 0:
                    queue.append(topic)
    
    # Add remaining topics (in case of cycles)
    for topic in topic_map:
        if topic not in result:
            result.append(topic)
    
    return result


async def generate_schedule(state: CurriculumState) -> CurriculumState:
    """Generate daily goals and schedule"""
    
    if state.get("error") or not state.get("ordered_topics"):
        return state
    
    try:
        hours_per_day = state.get("hours_per_day", 2.0)
        deadline_days = state.get("deadline_days", 14)
        start_date = datetime.strptime(
            state.get("start_date", datetime.now().strftime("%Y-%m-%d")),
            "%Y-%m-%d"
        )
        
        topics = state["ordered_topics"]
        total_hours = sum(t.estimated_hours for t in topics)
        
        # Calculate days needed
        days_needed = max(deadline_days, int(total_hours / hours_per_day) + 1)
        
        # Distribute topics across days
        daily_goals = []
        current_day = 1
        current_day_hours = 0
        current_topics = []
        
        for topic in topics:
            if current_day_hours + topic.estimated_hours > hours_per_day and current_topics:
                # Finish current day
                daily_goals.append({
                    "day": current_day,
                    "date": (start_date + timedelta(days=current_day - 1)).strftime("%Y-%m-%d"),
                    "topics": [t.name for t in current_topics],
                    "activities": generate_activities(current_topics),
                    "total_hours": current_day_hours,
                    "milestone": None
                })
                current_day += 1
                current_day_hours = 0
                current_topics = []
            
            current_topics.append(topic)
            current_day_hours += topic.estimated_hours
        
        # Add remaining topics
        if current_topics:
            daily_goals.append({
                "day": current_day,
                "date": (start_date + timedelta(days=current_day - 1)).strftime("%Y-%m-%d"),
                "topics": [t.name for t in current_topics],
                "activities": generate_activities(current_topics),
                "total_hours": current_day_hours,
                "milestone": None
            })
        
        # Add milestones (every 25% of progress)
        milestones = []
        total_topics = len(topics)
        for pct in [25, 50, 75, 100]:
            topic_idx = min(int(total_topics * pct / 100), total_topics - 1)
            milestone_day = next(
                (g["day"] for g in daily_goals if topics[topic_idx].name in g["topics"]),
                len(daily_goals)
            )
            milestones.append({
                "day": milestone_day,
                "percentage": pct,
                "goal": f"Complete {pct}% of curriculum",
                "topic": topics[topic_idx].name if topic_idx < len(topics) else ""
            })
            
            # Add milestone to corresponding day
            for goal in daily_goals:
                if goal["day"] == milestone_day:
                    goal["milestone"] = f"🎯 {pct}% Complete"
                    break
        
        state["daily_goals"] = daily_goals
        state["milestones"] = milestones
        
        logger.info(f"[CURRICULUM] Generated {len(daily_goals)} day schedule with {len(milestones)} milestones")
        
    except Exception as e:
        logger.error(f"[CURRICULUM] Schedule generation error: {e}")
        state["error"] = str(e)
    
    return state


def generate_activities(topics: List[CurriculumTopic]) -> List[Dict]:
    """Generate learning activities for topics"""
    activities = []
    
    for topic in topics:
        # Reading/Review activity
        activities.append({
            "type": "read",
            "topic": topic.name,
            "description": f"Study: {topic.name}",
            "duration_min": int(topic.estimated_hours * 30)  # 50% of time
        })
        
        # Practice activity
        if topic.subtopics:
            activities.append({
                "type": "practice",
                "topic": topic.name,
                "description": f"Practice: {', '.join(topic.subtopics[:3])}",
                "duration_min": int(topic.estimated_hours * 20)  # 33% of time
            })
        
        # Review/Quiz activity
        activities.append({
            "type": "quiz",
            "topic": topic.name,
            "description": f"Self-test: {topic.name}",
            "duration_min": int(topic.estimated_hours * 10)  # 17% of time
        })
    
    return activities


async def compile_curriculum(state: CurriculumState) -> CurriculumState:
    """Compile final curriculum object"""
    
    if state.get("error"):
        state["curriculum"] = None
        return state
    
    try:
        start_date = datetime.strptime(
            state.get("start_date", datetime.now().strftime("%Y-%m-%d")),
            "%Y-%m-%d"
        )
        
        end_date = start_date + timedelta(days=len(state.get("daily_goals", [])))
        
        curriculum = Curriculum(
            id=str(uuid.uuid4()),
            user_id=state["user_id"],
            syllabus_id=state["syllabus_id"],
            subject_name=state["subject_name"],
            created_at=datetime.utcnow().isoformat(),
            topics=[t for t in state.get("ordered_topics", [])],
            topic_order=state.get("topic_order", []),
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
            total_days=len(state.get("daily_goals", [])),
            hours_per_day=state.get("hours_per_day", 2.0),
            daily_goals=[DailyGoal(**g) if isinstance(g, dict) else g for g in state.get("daily_goals", [])],
            milestones=state.get("milestones", []),
            current_topic_index=0,
            completed_topics=[]
        )
        
        state["curriculum"] = curriculum.to_dict()
        
        logger.info(f"[CURRICULUM] Compiled curriculum: {len(curriculum.topics)} topics, {curriculum.total_days} days")
        
    except Exception as e:
        logger.error(f"[CURRICULUM] Compile error: {e}")
        state["error"] = str(e)
        state["curriculum"] = None
    
    return state


# ============================================================================
# Graph Builder
# ============================================================================

def build_curriculum_graph():
    """Build LangGraph workflow for curriculum generation"""
    workflow = StateGraph(CurriculumState)
    
    # Add nodes
    workflow.add_node("load_topics", load_syllabus_topics)
    workflow.add_node("analyze_deps", analyze_dependencies)
    workflow.add_node("assess_knowledge", assess_knowledge)
    workflow.add_node("build_path", build_learning_path)
    workflow.add_node("schedule", generate_schedule)
    workflow.add_node("compile", compile_curriculum)
    
    # Set entry point
    workflow.set_entry_point("load_topics")
    
    # Add edges (linear flow for Phase 1)
    workflow.add_edge("load_topics", "analyze_deps")
    workflow.add_edge("analyze_deps", "assess_knowledge")
    workflow.add_edge("assess_knowledge", "build_path")
    workflow.add_edge("build_path", "schedule")
    workflow.add_edge("schedule", "compile")
    workflow.add_edge("compile", END)
    
    return workflow.compile()


# ============================================================================
# Agent Class
# ============================================================================

class CurriculumAgent:
    """
    LangGraph-based Curriculum Agent
    
    Creates personalized learning paths from syllabus:
    - Analyzes topic dependencies
    - Builds optimal learning order
    - Generates daily schedule
    - Tracks milestones
    
    Usage:
        agent = CurriculumAgent()
        result = await agent.generate({
            "syllabus_id": "xxx",
            "user_id": "yyy",
            "subject_name": "Mathematics",
            "hours_per_day": 2,
            "deadline_days": 14
        })
    """
    
    def __init__(self):
        self.graph = build_curriculum_graph()
        logger.info("[CURRICULUM] Initialized Curriculum Agent")
    
    async def generate(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate personalized curriculum from syllabus
        
        Args:
            input_data: {
                syllabus_id: str - ID of processed syllabus
                user_id: str - Student ID
                classroom_id: str - Classroom ID
                subject_name: str - Subject name
                raw_topics: List[Dict] - Optional: pre-loaded topics
                hours_per_day: float - Available study hours (default: 2)
                deadline_days: int - Days until deadline (default: 14)
                start_date: str - Start date YYYY-MM-DD (default: today)
            }
        
        Returns:
            {curriculum: {...}, error: str?}
        """
        start_date = input_data.get("start_date") or datetime.now().strftime("%Y-%m-%d")
        
        initial_state: CurriculumState = {
            "syllabus_id": input_data.get("syllabus_id", ""),
            "user_id": input_data.get("user_id", ""),
            "classroom_id": input_data.get("classroom_id", ""),
            "subject_name": input_data.get("subject_name", ""),
            
            "hours_per_day": input_data.get("hours_per_day", 2.0),
            "deadline_days": input_data.get("deadline_days", 14),
            "start_date": start_date,
            
            "raw_topics": input_data.get("raw_topics", []),
            
            "topic_dependencies": {},
            "dependency_graph_built": False,
            
            "student_knowledge": {},
            "diagnostic_complete": False,
            
            "ordered_topics": [],
            "topic_order": [],
            
            "daily_goals": [],
            "milestones": [],
            
            "curriculum": None,
            "error": None
        }
        
        try:
            final_state = await self.graph.ainvoke(initial_state)
            
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "agent": "curriculum",
                "success": final_state.get("curriculum") is not None,
                "curriculum": final_state.get("curriculum"),
                "error": final_state.get("error")
            }
            
        except Exception as e:
            logger.error(f"[CURRICULUM] Agent error: {e}", exc_info=True)
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "agent": "curriculum",
                "success": False,
                "curriculum": None,
                "error": str(e)
            }


# Singleton
_curriculum_agent: Optional[CurriculumAgent] = None


def get_curriculum_agent() -> CurriculumAgent:
    """Get or create curriculum agent singleton"""
    global _curriculum_agent
    if _curriculum_agent is None:
        _curriculum_agent = CurriculumAgent()
    return _curriculum_agent
