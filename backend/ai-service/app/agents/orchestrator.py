"""
Orchestrator Agent - LangGraph Supervisor Pattern

The Orchestrator is a meta-agent that:
1. Receives user query
2. Analyzes intent (learn, research, create, evaluate)
3. Routes to appropriate sub-agent(s)
4. Synthesizes multi-agent responses

This provides a single entry point for all AI operations.

Architecture:
┌─────────────────────────────────────────────────────────────────┐
│                      OrchestratorAgent                          │
│                                                                  │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│  │ Intent   │───▶│ Route    │───▶│ Execute  │───▶│ Synth-   │  │
│  │ Analysis │    │ Decision │    │ Agent(s) │    │ esize    │  │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘  │
│                         │                                        │
│         ┌───────────────┼───────────────┐                       │
│         ▼               ▼               ▼                       │
│    ┌─────────┐    ┌─────────┐    ┌─────────┐                   │
│    │ Tutor   │    │Research │    │Content  │                   │
│    │ Agent   │    │ Agent   │    │ Agent   │                   │
│    └─────────┘    └─────────┘    └─────────┘                   │
└─────────────────────────────────────────────────────────────────┘
"""
import logging
import uuid
from typing import Dict, Any, List, TypedDict, Optional, Literal
from datetime import datetime
from enum import Enum

from langgraph.graph import StateGraph, END

logger = logging.getLogger(__name__)


# ============================================================================
# Intent Types
# ============================================================================

class Intent(str, Enum):
    """User intent categories"""
    LEARN = "learn"          # Q&A, explanations → TutorAgent
    RESEARCH = "research"    # Find content, PDFs → ResearchAgent
    CREATE = "create"        # Generate notes, quizzes → ContentAgent
    EVALUATE = "evaluate"    # Grade, feedback → EvaluationAgent
    MIXED = "mixed"          # Multiple intents


INTENT_KEYWORDS = {
    Intent.LEARN: [
        "what is", "explain", "how does", "why", "define", "describe",
        "tell me about", "help me understand", "concept", "meaning"
    ],
    Intent.RESEARCH: [
        "find", "search", "look up", "resources", "pdf", "document",
        "articles", "videos", "download", "sources", "materials"
    ],
    Intent.CREATE: [
        "create", "generate", "make", "write", "notes", "summary",
        "quiz", "questions", "flashcards", "flowchart", "outline"
    ],
    Intent.EVALUATE: [
        "grade", "check", "evaluate", "score", "feedback", "review",
        "correct", "assess", "is this right", "answer correct"
    ]
}


# ============================================================================
# State Definition
# ============================================================================

class OrchestratorState(TypedDict):
    """State passed through the orchestrator graph"""
    # Input
    query: str
    user_id: str
    session_id: str
    request_id: str
    classroom_id: Optional[str]
    
    # Intent Analysis
    primary_intent: str
    secondary_intents: List[str]
    intent_confidence: float
    extracted_topic: str
    
    # Agent Selection
    selected_agents: List[str]
    agent_execution_order: List[str]
    
    # Agent Results
    tutor_result: Optional[Dict]
    research_result: Optional[Dict]
    content_result: Optional[Dict]
    evaluation_result: Optional[Dict]
    
    # Output
    final_response: str
    sources: List[Dict]
    actions_taken: List[str]
    error: Optional[str]


# ============================================================================
# Intent Classification
# ============================================================================

def classify_intent(query: str) -> tuple[Intent, float, List[Intent]]:
    """
    Classify user intent from query.
    
    Returns:
        (primary_intent, confidence, secondary_intents)
    """
    query_lower = query.lower()
    scores = {intent: 0.0 for intent in Intent if intent != Intent.MIXED}
    
    # Score each intent based on keyword matches
    for intent, keywords in INTENT_KEYWORDS.items():
        for keyword in keywords:
            if keyword in query_lower:
                scores[intent] += 1.0
    
    # Normalize scores
    total = sum(scores.values())
    if total > 0:
        for intent in scores:
            scores[intent] /= total
    else:
        # Default to LEARN if no keywords match
        scores[Intent.LEARN] = 0.6
    
    # Get primary and secondary intents
    sorted_intents = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    primary = sorted_intents[0][0]
    confidence = sorted_intents[0][1]
    
    # Secondary intents with score > 0.2
    secondary = [i for i, s in sorted_intents[1:] if s > 0.2]
    
    # If multiple high scores, it's mixed
    high_scores = [i for i, s in sorted_intents if s > 0.3]
    if len(high_scores) > 1:
        primary = Intent.MIXED
        secondary = high_scores
    
    return primary, confidence, secondary


def extract_topic(query: str) -> str:
    """Extract the main topic from a query"""
    # Simple extraction - in production, use NER or LLM
    # Remove common prefixes
    prefixes = [
        "what is", "explain", "how does", "why", "tell me about",
        "find", "search for", "create", "generate", "help me with"
    ]
    
    topic = query.lower()
    for prefix in prefixes:
        if topic.startswith(prefix):
            topic = topic[len(prefix):].strip()
            break
    
    # Clean up
    topic = topic.strip("?!.")
    
    return topic or query[:50]


# ============================================================================
# Node Functions
# ============================================================================

async def analyze_intent_node(state: OrchestratorState) -> OrchestratorState:
    """Analyze user intent and extract topic"""
    query = state["query"]
    
    # Classify intent
    primary, confidence, secondary = classify_intent(query)
    
    state["primary_intent"] = primary.value
    state["intent_confidence"] = confidence
    state["secondary_intents"] = [i.value for i in secondary]
    
    # Extract topic
    state["extracted_topic"] = extract_topic(query)
    
    logger.info(
        f"[ORCHESTRATOR] Intent: {primary.value} (conf={confidence:.2f}), "
        f"topic='{state['extracted_topic'][:30]}...'"
    )
    
    return state


async def select_agents_node(state: OrchestratorState) -> OrchestratorState:
    """Select which agents to invoke based on intent"""
    primary = state["primary_intent"]
    secondary = state["secondary_intents"]
    
    agents = []
    
    # Map intents to agents
    intent_agent_map = {
        "learn": "tutor",
        "research": "research",
        "create": "content",
        "evaluate": "evaluation"
    }
    
    # Add primary agent
    if primary in intent_agent_map:
        agents.append(intent_agent_map[primary])
    elif primary == "mixed":
        # Add all secondary intent agents
        for intent in secondary:
            if intent in intent_agent_map:
                agents.append(intent_agent_map[intent])
    
    # If no agents selected, default to tutor
    if not agents:
        agents = ["tutor"]
    
    state["selected_agents"] = agents
    state["agent_execution_order"] = agents  # Could reorder for dependencies
    
    logger.info(f"[ORCHESTRATOR] Selected agents: {agents}")
    
    return state


async def execute_tutor_node(state: OrchestratorState) -> OrchestratorState:
    """Execute the Tutor Agent"""
    if "tutor" not in state["selected_agents"]:
        return state
    
    try:
        from app.agents.tutor_agent import TutorAgent
        
        agent = TutorAgent()
        result = await agent.execute({
            "query": state["query"],
            "user_id": state["user_id"],
            "session_id": state["session_id"],
            "classroom_id": state.get("classroom_id", "")
        })
        
        state["tutor_result"] = result
        state["actions_taken"].append("tutor_response")
        
        logger.info("[ORCHESTRATOR] Tutor agent completed")
        
    except Exception as e:
        logger.error(f"[ORCHESTRATOR] Tutor agent error: {e}")
        state["tutor_result"] = {"error": str(e)}
    
    return state


async def execute_research_node(state: OrchestratorState) -> OrchestratorState:
    """Execute the Research Agent"""
    if "research" not in state["selected_agents"]:
        return state
    
    try:
        from app.agents.research_agent import get_research_agent
        
        agent = get_research_agent()
        result = await agent.execute({
            "query": state["query"],
            "user_id": state["user_id"],
            "session_id": state["session_id"],
            "search_pdfs": True,
            "download_pdfs": True,
            "search_youtube": True
        })
        
        state["research_result"] = result
        state["actions_taken"].append("research_completed")
        
        # Add sources from research
        if result.get("data", {}).get("web_results"):
            for r in result["data"]["web_results"][:3]:
                state["sources"].append({
                    "type": "web",
                    "title": r.get("title", ""),
                    "url": r.get("url", "")
                })
        
        logger.info("[ORCHESTRATOR] Research agent completed")
        
    except Exception as e:
        logger.error(f"[ORCHESTRATOR] Research agent error: {e}")
        state["research_result"] = {"error": str(e)}
    
    return state


async def execute_content_node(state: OrchestratorState) -> OrchestratorState:
    """Execute content generation based on intent"""
    if "content" not in state["selected_agents"]:
        return state
    
    try:
        from app.agents.tools import invoke_tool
        
        query_lower = state["query"].lower()
        topic = state["extracted_topic"]
        
        result = {}
        
        # Determine what to generate
        if "quiz" in query_lower or "question" in query_lower:
            gen_result = await invoke_tool(
                "generate_questions",
                topic=topic,
                num_questions=5
            )
            result["questions"] = gen_result.data if gen_result.success else None
            state["actions_taken"].append("generated_questions")
            
        elif "notes" in query_lower or "summary" in query_lower:
            gen_result = await invoke_tool(
                "generate_notes",
                topic=topic,
                style="structured"
            )
            result["notes"] = gen_result.data if gen_result.success else None
            state["actions_taken"].append("generated_notes")
            
        elif "flowchart" in query_lower or "diagram" in query_lower:
            gen_result = await invoke_tool(
                "generate_flowchart",
                topic=topic
            )
            result["flowchart"] = gen_result.data if gen_result.success else None
            state["actions_taken"].append("generated_flowchart")
            
        else:
            # Default: generate summary
            gen_result = await invoke_tool(
                "summarize_content",
                content=state["query"],
                style="bullet_points"
            )
            result["summary"] = gen_result.data if gen_result.success else None
            state["actions_taken"].append("generated_summary")
        
        state["content_result"] = result
        
        logger.info("[ORCHESTRATOR] Content generation completed")
        
    except Exception as e:
        logger.error(f"[ORCHESTRATOR] Content generation error: {e}")
        state["content_result"] = {"error": str(e)}
    
    return state


async def synthesize_response_node(state: OrchestratorState) -> OrchestratorState:
    """Synthesize final response from all agent results"""
    parts = []
    
    # Add tutor response
    if state.get("tutor_result"):
        tutor_data = state["tutor_result"].get("data", {})
        if tutor_data.get("answer"):
            parts.append(tutor_data["answer"])
    
    # Add research summary
    if state.get("research_result"):
        research_data = state["research_result"].get("data", {})
        if research_data.get("summary"):
            parts.append(f"\n\n**Research Summary:**\n{research_data['summary']}")
    
    # Add content result
    if state.get("content_result"):
        content = state["content_result"]
        if content.get("notes", {}).get("notes"):
            parts.append(f"\n\n**Generated Notes:**\n{content['notes']['notes']}")
        if content.get("questions", {}).get("questions"):
            q_count = len(content["questions"]["questions"])
            parts.append(f"\n\n**Generated {q_count} Quiz Questions**")
    
    # Combine
    if parts:
        state["final_response"] = "\n".join(parts)
    else:
        state["final_response"] = "I couldn't process your request. Please try again."
    
    logger.info(f"[ORCHESTRATOR] Synthesized response: {len(state['final_response'])} chars")
    
    return state


# ============================================================================
# Routing Functions
# ============================================================================

def route_to_agents(state: OrchestratorState) -> str:
    """Route to first selected agent"""
    agents = state.get("selected_agents", [])
    
    if "tutor" in agents:
        return "tutor"
    elif "research" in agents:
        return "research"
    elif "content" in agents:
        return "content"
    else:
        return "synthesize"


def route_after_tutor(state: OrchestratorState) -> str:
    """Route after tutor agent"""
    agents = state["selected_agents"]
    
    if "research" in agents:
        return "research"
    elif "content" in agents:
        return "content"
    else:
        return "synthesize"


def route_after_research(state: OrchestratorState) -> str:
    """Route after research agent"""
    agents = state["selected_agents"]
    
    if "content" in agents:
        return "content"
    else:
        return "synthesize"


# ============================================================================
# Graph Builder
# ============================================================================

def build_orchestrator_graph():
    """Build the LangGraph orchestrator workflow"""
    workflow = StateGraph(OrchestratorState)
    
    # Add nodes
    workflow.add_node("analyze_intent", analyze_intent_node)
    workflow.add_node("select_agents", select_agents_node)
    workflow.add_node("tutor", execute_tutor_node)
    workflow.add_node("research", execute_research_node)
    workflow.add_node("content", execute_content_node)
    workflow.add_node("synthesize", synthesize_response_node)
    
    # Set entry point
    workflow.set_entry_point("analyze_intent")
    
    # Linear flow for intent analysis
    workflow.add_edge("analyze_intent", "select_agents")
    
    # Conditional routing to agents
    workflow.add_conditional_edges(
        "select_agents",
        route_to_agents,
        {
            "tutor": "tutor",
            "research": "research",
            "content": "content",
            "synthesize": "synthesize"
        }
    )
    
    # Routing after each agent
    workflow.add_conditional_edges(
        "tutor",
        route_after_tutor,
        {
            "research": "research",
            "content": "content",
            "synthesize": "synthesize"
        }
    )
    
    workflow.add_conditional_edges(
        "research",
        route_after_research,
        {
            "content": "content",
            "synthesize": "synthesize"
        }
    )
    
    workflow.add_edge("content", "synthesize")
    workflow.add_edge("synthesize", END)
    
    return workflow.compile()


# ============================================================================
# Orchestrator Agent Class
# ============================================================================

class OrchestratorAgent:
    """
    Central Orchestrator Agent using Supervisor Pattern
    
    Routes requests to appropriate sub-agents:
    - TutorAgent: Q&A, explanations
    - ResearchAgent: Web search, PDF download
    - ContentAgent: Notes, quizzes, flowcharts
    
    Usage:
        orchestrator = OrchestratorAgent()
        result = await orchestrator.chat("explain photosynthesis")
    """
    
    def __init__(self):
        self.graph = build_orchestrator_graph()
        logger.info("[ORCHESTRATOR] Initialized Orchestrator Agent")
    
    async def chat(
        self,
        query: str,
        user_id: str = "",
        session_id: str = "",
        classroom_id: str = ""
    ) -> Dict[str, Any]:
        """
        Process a user query through the orchestrator.
        
        This is the main entry point for all AI interactions.
        
        Args:
            query: User's question or request
            user_id: User identifier
            session_id: Session for context
            classroom_id: Optional classroom context
            
        Returns:
            Orchestrated response with sources and actions
        """
        if not session_id:
            session_id = str(uuid.uuid4())
        
        request_id = str(uuid.uuid4())[:8]
        
        # Initialize state
        initial_state: OrchestratorState = {
            "query": query,
            "user_id": user_id,
            "session_id": session_id,
            "request_id": request_id,
            "classroom_id": classroom_id,
            
            "primary_intent": "",
            "secondary_intents": [],
            "intent_confidence": 0.0,
            "extracted_topic": "",
            
            "selected_agents": [],
            "agent_execution_order": [],
            
            "tutor_result": None,
            "research_result": None,
            "content_result": None,
            "evaluation_result": None,
            
            "final_response": "",
            "sources": [],
            "actions_taken": [],
            "error": None
        }
        
        try:
            # Run the orchestrator graph
            final_state = await self.graph.ainvoke(initial_state)
            
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "session_id": session_id,
                "request_id": request_id,
                "response": final_state["final_response"],
                "intent": {
                    "primary": final_state["primary_intent"],
                    "confidence": final_state["intent_confidence"],
                    "topic": final_state["extracted_topic"]
                },
                "agents_used": final_state["selected_agents"],
                "actions": final_state["actions_taken"],
                "sources": final_state["sources"],
                "details": {
                    "tutor": final_state.get("tutor_result"),
                    "research": final_state.get("research_result"),
                    "content": final_state.get("content_result")
                }
            }
            
        except Exception as e:
            logger.error(f"[ORCHESTRATOR] Error: {e}", exc_info=True)
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "session_id": session_id,
                "request_id": request_id,
                "response": "An error occurred. Please try again.",
                "error": str(e)
            }


# Singleton
_orchestrator = None


def get_orchestrator() -> OrchestratorAgent:
    """Get or create the orchestrator singleton"""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = OrchestratorAgent()
    return _orchestrator
