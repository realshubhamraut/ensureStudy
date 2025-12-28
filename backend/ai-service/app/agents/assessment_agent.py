"""
Assessment Agent with LangGraph - Adaptive Quiz Generation
Uses Hugging Face LLM instead of GPT-4
"""
import logging
import json
from typing import Dict, Any, List, TypedDict
from datetime import datetime

from langgraph.graph import StateGraph, END

logger = logging.getLogger(__name__)


# ============================================================================
# State Definition
# ============================================================================

class AssessmentState(TypedDict):
    """State for assessment generation"""
    # Input
    weak_topics: List[Dict]
    num_questions: int
    difficulty: str
    
    # Processing
    current_topic_idx: int
    generated_questions: List[Dict]
    
    # Output
    assessment: Dict
    error: str


# ============================================================================
# Node Functions
# ============================================================================

def parse_topics(state: AssessmentState) -> AssessmentState:
    """Parse and validate input topics"""
    if not state["weak_topics"]:
        state["error"] = "No topics provided"
        state["assessment"] = {"questions": [], "error": "No topics"}
        return state
    
    # Limit to 3 topics max
    state["weak_topics"] = state["weak_topics"][:3]
    state["current_topic_idx"] = 0
    state["generated_questions"] = []
    
    logger.info(f"Processing {len(state['weak_topics'])} topics")
    return state


def generate_questions(state: AssessmentState) -> AssessmentState:
    """Generate MCQs using Hugging Face LLM"""
    if state.get("error"):
        return state
    
    try:
        from app.services.llm_provider import get_llm
        
        llm = get_llm()
        all_questions = []
        questions_per_topic = max(1, state["num_questions"] // len(state["weak_topics"]))
        
        difficulty_guidance = {
            "easy": "Basic recall and understanding questions.",
            "medium": "Application and analysis questions.",
            "hard": "Synthesis and evaluation - complex scenarios."
        }
        
        for topic in state["weak_topics"]:
            topic_name = topic.get("topic", "General")
            subject = topic.get("subject", "")
            
            prompt = f"""Generate {questions_per_topic} multiple choice questions about "{topic_name}" in {subject}.

Difficulty: {state["difficulty"]}
{difficulty_guidance.get(state["difficulty"], difficulty_guidance["medium"])}

Return a JSON array:
[
  {{
    "question": "Your question here?",
    "options": ["Option A", "Option B", "Option C", "Option D"],
    "correct_answer": "A",
    "explanation": "Why A is correct...",
    "topic": "{topic_name}"
  }}
]

Return ONLY the JSON array, no other text."""

            response = llm.invoke(prompt)
            
            # Parse JSON from response
            try:
                text = response.strip()
                if "```" in text:
                    text = text.split("```")[1].replace("json", "").strip()
                
                questions = json.loads(text)
                
                # Validate and clean
                for q in questions:
                    if all(k in q for k in ["question", "options", "correct_answer"]):
                        if len(q["correct_answer"]) > 1:
                            q["correct_answer"] = q["correct_answer"][0].upper()
                        all_questions.append(q)
                        
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse questions for {topic_name}")
        
        state["generated_questions"] = all_questions[:state["num_questions"]]
        logger.info(f"Generated {len(state['generated_questions'])} questions")
        
    except Exception as e:
        logger.error(f"Question generation error: {e}")
        state["error"] = str(e)
    
    return state


def format_assessment(state: AssessmentState) -> AssessmentState:
    """Format final assessment output"""
    state["assessment"] = {
        "questions": state["generated_questions"],
        "num_questions": len(state["generated_questions"]),
        "difficulty": state["difficulty"],
        "topics_covered": [t.get("topic") for t in state["weak_topics"]]
    }
    return state


# ============================================================================
# Graph Builder
# ============================================================================

def build_assessment_graph():
    """Build LangGraph workflow for assessment generation"""
    workflow = StateGraph(AssessmentState)
    
    workflow.add_node("parse", parse_topics)
    workflow.add_node("generate", generate_questions)
    workflow.add_node("format", format_assessment)
    
    workflow.set_entry_point("parse")
    workflow.add_edge("parse", "generate")
    workflow.add_edge("generate", "format")
    workflow.add_edge("format", END)
    
    return workflow.compile()


# ============================================================================
# Agent Class
# ============================================================================

class AssessmentAgent:
    """
    LangGraph-based Assessment Agent
    
    Generates adaptive MCQ assessments using:
    - Hugging Face LLM (Mistral-7B)
    - Multi-topic coverage
    - Difficulty adjustment
    """
    
    def __init__(self):
        self.graph = build_assessment_graph()
        logger.info("Initialized LangGraph Assessment Agent")
    
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate assessment
        
        Args:
            input_data: {weak_topics, num_questions?, difficulty?}
        
        Returns:
            {assessment: {questions, num_questions, difficulty}}
        """
        initial_state: AssessmentState = {
            "weak_topics": input_data.get("weak_topics", []),
            "num_questions": input_data.get("num_questions", 10),
            "difficulty": input_data.get("difficulty", "medium"),
            "current_topic_idx": 0,
            "generated_questions": [],
            "assessment": {},
            "error": ""
        }
        
        try:
            final_state = await self.graph.ainvoke(initial_state)
            
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "agent": "assessment",
                "data": final_state["assessment"]
            }
            
        except Exception as e:
            logger.error(f"Assessment agent error: {e}")
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "agent": "assessment",
                "data": {
                    "questions": [],
                    "error": str(e)
                }
            }
