"""
Tutor Agent with LangGraph - Industry Standard Implementation
Uses Hugging Face LLM instead of GPT-4
"""
import logging
from typing import Dict, Any, List, TypedDict, Annotated
from datetime import datetime

from langgraph.graph import StateGraph, END

logger = logging.getLogger(__name__)


# ============================================================================
# State Definition
# ============================================================================

class TutorState(TypedDict):
    """State passed between nodes in the tutor graph"""
    # Input
    query: str
    user_id: str
    session_id: str
    
    # Processing
    is_academic: bool
    moderation_score: float
    retrieved_context: List[str]
    retrieved_sources: List[Dict]
    
    # Output
    answer: str
    sources: List[Dict]
    blocked: bool
    error: str


# ============================================================================
# Node Functions
# ============================================================================

def moderate_query(state: TutorState) -> TutorState:
    """Check if query is academic-related"""
    from app.services.llm_provider import get_classifier
    
    try:
        classifier = get_classifier()
        labels = ["academic", "homework", "study", "off-topic", "inappropriate"]
        
        scores = classifier.classify(state["query"], labels)
        
        # Check if academic-related
        academic_score = (
            scores.get("academic", 0) + 
            scores.get("homework", 0) + 
            scores.get("study", 0)
        ) / 3
        
        is_academic = academic_score > 0.3
        
        state["is_academic"] = is_academic
        state["moderation_score"] = academic_score
        state["blocked"] = not is_academic
        
        logger.info(f"Moderation: academic={is_academic}, score={academic_score:.2f}")
        
    except Exception as e:
        logger.error(f"Moderation error: {e}")
        # Default to allowing if moderation fails
        state["is_academic"] = True
        state["moderation_score"] = 0.5
        state["blocked"] = False
    
    return state


def retrieve_context(state: TutorState) -> TutorState:
    """Retrieve relevant context from RAG and enrich with web sources"""
    if state["blocked"]:
        return state
    
    try:
        from app.rag.retriever import get_retriever
        
        retriever = get_retriever()
        results = retriever.retrieve(state["query"], top_k=5)
        
        state["retrieved_context"] = [r.get("content", "") for r in results]
        state["retrieved_sources"] = [
            {
                "title": r.get("title", ""),
                "source": r.get("source", ""),
                "relevance": r.get("relevance_score", 0),
                "type": "document"
            }
            for r in results
        ]
        
        logger.info(f"Retrieved {len(results)} context chunks from RAG")
        
    except Exception as e:
        logger.error(f"Retrieval error: {e}")
        state["retrieved_context"] = []
        state["retrieved_sources"] = []
    
    # Enrich with web sources (async in background)
    try:
        import asyncio
        from app.agents.web_enrichment_agent import WebEnrichmentAgent
        
        web_agent = WebEnrichmentAgent()
        
        # Run web enrichment
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Create task for async context
            web_result = asyncio.create_task(
                web_agent.enrich(state["query"], student_id=state["user_id"])
            )
        else:
            web_result = loop.run_until_complete(
                web_agent.enrich(state["query"], student_id=state["user_id"])
            )
        
        if web_result and web_result.get("sources"):
            # Add web sources to retrieved sources
            for source in web_result["sources"][:3]:  # Top 3 web sources
                state["retrieved_sources"].append({
                    "title": source.get("title", ""),
                    "source": source.get("url", ""),
                    "relevance": source.get("relevance_score", 0.7),
                    "type": source.get("source_type", "web"),
                    "domain": source.get("domain", "")
                })
            
            logger.info(f"Added {len(web_result['sources'][:3])} web sources")
            
    except Exception as e:
        logger.warning(f"Web enrichment error (non-fatal): {e}")
    
    return state


def generate_answer(state: TutorState) -> TutorState:
    """Generate answer using Hugging Face LLM"""
    if state["blocked"]:
        state["answer"] = "I'm here to help with academic questions. Please ask about your studies, homework, or course material."
        return state
    
    try:
        from app.services.llm_provider import get_llm
        
        llm = get_llm()
        
        # Build context
        context = "\n\n".join(state["retrieved_context"][:3])
        
        prompt = f"""You are a helpful academic tutor. Answer the student's question using the provided context.

Context:
{context if context else "No specific context available."}

Student Question: {state["query"]}

Instructions:
- Give a clear, educational answer
- Explain concepts step by step
- If the context doesn't contain the answer, use your knowledge but mention it
- Keep the answer concise but complete

Answer:"""

        answer = llm.invoke(prompt)
        state["answer"] = answer.strip()
        state["sources"] = state["retrieved_sources"]
        
        logger.info(f"Generated answer ({len(answer)} chars)")
        
    except Exception as e:
        logger.error(f"Generation error: {e}")
        state["answer"] = "I'm having trouble generating a response. Please try again."
        state["error"] = str(e)
    
    return state


def route_moderation(state: TutorState) -> str:
    """Route based on moderation result"""
    if state["blocked"]:
        return "blocked"
    return "continue"


# ============================================================================
# Graph Builder
# ============================================================================

def build_tutor_graph():
    """Build the LangGraph workflow for tutoring"""
    workflow = StateGraph(TutorState)
    
    # Add nodes
    workflow.add_node("moderate", moderate_query)
    workflow.add_node("retrieve", retrieve_context)
    workflow.add_node("generate", generate_answer)
    
    # Set entry point
    workflow.set_entry_point("moderate")
    
    # Add conditional routing from moderation
    workflow.add_conditional_edges(
        "moderate",
        route_moderation,
        {
            "blocked": "generate",  # Skip retrieval if blocked
            "continue": "retrieve"
        }
    )
    
    # Add remaining edges
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)
    
    return workflow.compile()


# ============================================================================
# Agent Class (for compatibility)
# ============================================================================

class TutorAgent:
    """
    LangGraph-based Tutor Agent
    
    Uses industry-standard agentic framework with:
    - State machine for flow control
    - Hugging Face LLM (Mistral-7B)
    - Zero-shot classifier for moderation
    - RAG for context retrieval
    """
    
    def __init__(self):
        self.graph = build_tutor_graph()
        logger.info("Initialized LangGraph Tutor Agent")
    
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a student question
        
        Args:
            input_data: {query, user_id, session_id?}
        
        Returns:
            {answer, sources, blocked, timestamp}
        """
        # Initialize state
        initial_state: TutorState = {
            "query": input_data.get("query", ""),
            "user_id": input_data.get("user_id", ""),
            "session_id": input_data.get("session_id", ""),
            "is_academic": True,
            "moderation_score": 0.0,
            "retrieved_context": [],
            "retrieved_sources": [],
            "answer": "",
            "sources": [],
            "blocked": False,
            "error": ""
        }
        
        try:
            # Run the graph
            final_state = await self.graph.ainvoke(initial_state)
            
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "agent": "tutor",
                "data": {
                    "answer": final_state["answer"],
                    "sources": final_state["sources"],
                    "blocked": final_state["blocked"],
                    "moderation_score": final_state["moderation_score"]
                }
            }
            
        except Exception as e:
            logger.error(f"Tutor agent error: {e}")
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "agent": "tutor",
                "data": {
                    "answer": "An error occurred. Please try again.",
                    "sources": [],
                    "blocked": False,
                    "error": str(e)
                }
            }
