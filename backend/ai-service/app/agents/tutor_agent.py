"""
Tutor Agent with LangGraph - Production Architecture

Integrates:
- ABCR (Attention-Based Context Routing) for follow-up detection
- TAL (Topic Anchor Layer) for topic management
- MCP (Memory Context Processor) for web isolation

Flow:
moderate → context_routing (ABCR/TAL) → retrieve_with_mcp → generate
"""
import logging
import uuid
from typing import Dict, Any, List, TypedDict, Optional
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
    request_id: str
    classroom_id: str
    clicked_suggestion: bool
    
    # Session state (persisted across queries)
    turn_texts: List[str]
    last_abcr_decision: str
    consecutive_borderline: int
    
    # Processing - Moderation
    is_academic: bool
    moderation_score: float
    
    # Processing - ABCR
    abcr_decision: str  # "related" or "new_topic"
    abcr_confidence: float
    is_followup: bool
    matched_turn_index: Optional[int]
    
    # Processing - TAL
    topic_anchor_id: Optional[str]
    topic_anchor_title: Optional[str]
    confirm_new_topic: bool
    
    # Processing - RAG/MCP
    raw_chunks: List[Dict]
    mcp_chunks: List[Dict]
    mcp_reason: str
    anchor_hits: int
    web_filtered_count: int
    context_sources: List[str]
    
    # Output
    answer: str
    sources: List[Dict]
    blocked: bool
    error: str


# ============================================================================
# Session Store (in-memory for now, should be Redis/DB in production)
# ============================================================================

_session_states: Dict[str, Dict] = {}

def get_session_state(session_id: str) -> Dict:
    """Get or create session state."""
    if session_id not in _session_states:
        _session_states[session_id] = {
            "turn_texts": [],
            "last_abcr_decision": "new_topic",
            "consecutive_borderline": 0,
            "topic_anchor_id": None,
            "topic_anchor_title": None,
        }
    return _session_states[session_id]

def update_session_state(session_id: str, updates: Dict):
    """Update session state."""
    state = get_session_state(session_id)
    state.update(updates)


# ============================================================================
# Node Functions
# ============================================================================

def moderate_query(state: TutorState) -> TutorState:
    """Check if query is academic-related"""
    import os
    
    # Fast path: skip moderation if disabled
    if os.getenv("SKIP_MODERATION", "false").lower() == "true":
        logger.info("[TUTOR] Moderation skipped (SKIP_MODERATION=true)")
        state["is_academic"] = True
        state["moderation_score"] = 1.0
        state["blocked"] = False
        return state
    
    try:
        from app.services.llm_provider import get_classifier
        
        logger.info("[TUTOR] Loading classifier for moderation...")
        classifier = get_classifier()
        labels = ["academic", "homework", "study", "off-topic", "inappropriate"]
        
        logger.info(f"[TUTOR] Classifying query: {state['query'][:50]}...")
        scores = classifier.classify(state["query"], labels)
        
        academic_score = (
            scores.get("academic", 0) + 
            scores.get("homework", 0) + 
            scores.get("study", 0)
        ) / 3
        
        is_academic = academic_score > 0.3
        
        state["is_academic"] = is_academic
        state["moderation_score"] = academic_score
        state["blocked"] = not is_academic
        
        logger.info(f"[TUTOR] Moderation: academic={is_academic}, score={academic_score:.2f}")
        
    except Exception as e:
        logger.error(f"[TUTOR] Moderation error: {e}")
        state["is_academic"] = True
        state["moderation_score"] = 0.5
        state["blocked"] = False
    
    return state


def context_routing(state: TutorState) -> TutorState:
    """
    ABCR + TAL integration:
    1. Run ABCR to detect if query is follow-up or new topic
    2. If follow-up -> keep existing anchor, no confirmation needed
    3. If new topic -> create new anchor, may need confirmation
    """
    if state["blocked"]:
        return state
    
    session_id = state["session_id"]
    request_id = state["request_id"]
    query = state["query"]
    
    try:
        from app.services.abcr_service import get_abcr_service
        from app.services.topic_anchor_service import (
            get_topic_anchor_service, 
            extract_canonical_title
        )
        
        abcr = get_abcr_service()
        tal = get_topic_anchor_service()
        
        # Get session state
        sess = get_session_state(session_id)
        turn_texts = sess.get("turn_texts", [])
        
        # Run ABCR
        abcr_result, abcr_updated = abcr.compute_relatedness(
            session_id=session_id,
            query_text=query,
            request_id=request_id,
            turn_texts=turn_texts,
            last_decision=sess.get("last_abcr_decision", "new_topic"),
            consecutive_borderline=sess.get("consecutive_borderline", 0),
            clicked_suggestion=state.get("clicked_suggestion", False)
        )
        
        # Update state with ABCR results
        state["abcr_decision"] = abcr_result.decision
        state["abcr_confidence"] = abcr_result.max_relatedness
        state["is_followup"] = abcr_result.decision == "related"
        state["matched_turn_index"] = abcr_result.matched_turn_index
        
        # Update session with ABCR state
        update_session_state(session_id, abcr_updated)
        
        # TAL logic
        current_anchor = tal.get_anchor(session_id)
        
        if abcr_result.decision == "related" and current_anchor:
            # Follow-up: keep existing anchor
            state["topic_anchor_id"] = current_anchor.id
            state["topic_anchor_title"] = current_anchor.canonical_title
            state["confirm_new_topic"] = False
            
            logger.info(
                f"[TUTOR] Follow-up detected: anchor='{current_anchor.canonical_title}' "
                f"confidence={abcr_result.max_relatedness:.2f}"
            )
            
        else:
            # New topic
            canonical_title = extract_canonical_title(query)
            
            if current_anchor:
                # Switching topics - may need confirmation
                state["confirm_new_topic"] = True
                logger.info(
                    f"[TUTOR] New topic detected: '{canonical_title}' "
                    f"(was: '{current_anchor.canonical_title}')"
                )
            else:
                # First topic in session
                state["confirm_new_topic"] = False
            
            # Create new anchor (or update if confirmed)
            # For now, auto-create - in production, frontend may confirm first
            new_anchor = tal.create_anchor(
                session_id=session_id,
                request_id=request_id,
                canonical_title=canonical_title,
                source="user_query"
            )
            
            state["topic_anchor_id"] = new_anchor.id
            state["topic_anchor_title"] = new_anchor.canonical_title
            
            # Update session
            update_session_state(session_id, {
                "topic_anchor_id": new_anchor.id,
                "topic_anchor_title": new_anchor.canonical_title,
            })
        
        # Add current query to turn history
        turn_texts.append(query)
        update_session_state(session_id, {"turn_texts": turn_texts[-10:]})  # Keep last 10
        
        logger.info(
            f"[TUTOR] Context routing: decision={abcr_result.decision} "
            f"anchor='{state.get('topic_anchor_title')}' "
            f"followup={state['is_followup']}"
        )
        
    except Exception as e:
        logger.error(f"[TUTOR] Context routing error: {e}", exc_info=True)
        state["abcr_decision"] = "new_topic"
        state["abcr_confidence"] = 0.0
        state["is_followup"] = False
        state["confirm_new_topic"] = False
    
    return state


def retrieve_with_mcp(state: TutorState) -> TutorState:
    """
    RAG retrieval with MCP isolation:
    1. Retrieve chunks from Qdrant
    2. Apply MCP rules based on active anchor
    3. Filter web chunks when anchor is active
    """
    if state["blocked"]:
        return state
    
    try:
        from app.rag.retriever import get_retriever
        from app.services.topic_anchor_service import get_topic_anchor_service
        from app.services.mcp_context import assemble_mcp_context
        
        retriever = get_retriever()
        tal = get_topic_anchor_service()
        
        # Get active anchor
        active_anchor = tal.get_anchor(state["session_id"])
        
        # Retrieve raw chunks
        raw_results = retriever.retrieve(
            state["query"], 
            top_k=10,
            classroom_id=state.get("classroom_id")
        )
        
        # Convert to chunk dicts
        raw_chunks = [
            {
                "id": r.get("id", f"chunk_{i}"),
                "content": r.get("content", ""),
                "text": r.get("content", ""),
                "similarity": r.get("relevance_score", 0.5),
                "source_type": r.get("source_type", "web"),
                "document_id": r.get("document_id"),
                "classroom_id": r.get("classroom_id"),
                "url": r.get("url"),
                "title": r.get("title", ""),
            }
            for i, r in enumerate(raw_results)
        ]
        
        state["raw_chunks"] = raw_chunks
        
        # Apply MCP rules
        mcp_result = assemble_mcp_context(
            chunks=raw_chunks,
            active_anchor=active_anchor,
            session_id=state["session_id"],
            request_id=state["request_id"]
        )
        
        # Convert MCPChunk objects to dicts
        state["mcp_chunks"] = [
            {"id": c.id, "text": c.text, "source": c.source, "similarity": c.similarity}
            for c in mcp_result.chunks
        ]
        state["mcp_reason"] = mcp_result.reason
        state["anchor_hits"] = mcp_result.anchor_hits
        state["web_filtered_count"] = mcp_result.web_filtered_count
        state["context_sources"] = list(set(c.source for c in mcp_result.chunks))
        
        # Build sources for response
        state["sources"] = [
            {
                "title": r.get("title", ""),
                "source": r.get("url", r.get("source", "")),
                "relevance": r.get("similarity", 0),
                "type": "anchor" if r in state["mcp_chunks"] else "document"
            }
            for r in raw_results[:5]
        ]
        
        logger.info(
            f"[TUTOR] MCP: retrieved={len(raw_chunks)} "
            f"selected={len(mcp_result.chunks)} "
            f"reason={mcp_result.reason} "
            f"web_filtered={mcp_result.web_filtered_count}"
        )
        
    except Exception as e:
        logger.error(f"[TUTOR] Retrieval error: {e}", exc_info=True)
        state["mcp_chunks"] = []
        state["raw_chunks"] = []
        state["sources"] = []
        state["mcp_reason"] = "error"
    
    return state


def generate_answer(state: TutorState) -> TutorState:
    """Generate answer using LLM with anchor constraints and learning enhancement"""
    if state["blocked"]:
        state["answer"] = (
            "I'm here to help with academic questions. "
            "Please ask about your studies, homework, or course material."
        )
        return state
    
    try:
        from app.services.llm_provider import get_llm
        from app.services.topic_anchor_service import get_topic_anchor_service
        
        llm = get_llm()
        tal = get_topic_anchor_service()
        
        # Build context from MCP chunks
        context_texts = [c.get("text", "") for c in state.get("mcp_chunks", [])]
        context = "\n\n".join(context_texts[:5])
        
        # Get anchor prompt fragment
        anchor = tal.get_anchor(state["session_id"])
        anchor_prompt = ""
        topic = state.get("topic_anchor_title", "general")
        if anchor:
            anchor_prompt = anchor.to_prompt_fragment()
            topic = anchor.canonical_title
        
        # Check for insufficient context
        if not context and anchor:
            from app.services.mcp_context import get_insufficient_anchor_response
            state["answer"] = get_insufficient_anchor_response(anchor.canonical_title)
            return state
        
        # === LEARNING ELEMENT: Fetch few-shot examples ===
        few_shot_section = ""
        try:
            from app.learning.learning_element import get_learning_element
            import asyncio
            
            learning = get_learning_element()
            # Get examples for this topic (run sync in async context)
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Already in async context, create task
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run,
                        learning.get_examples(topic, limit=2)
                    )
                    examples = future.result(timeout=3)
            else:
                examples = asyncio.run(learning.get_examples(topic, limit=2))
            
            if examples:
                few_shot_section = learning.build_few_shot_prompt(examples)
                logger.info(f"[TUTOR/LEARNING] Injected {len(examples)} few-shot examples for '{topic}'")
        except Exception as le:
            logger.warning(f"[TUTOR/LEARNING] Could not fetch examples: {le}")
        # === END LEARNING ELEMENT ===
        
        # Build prompt with few-shot enhancement
        system_prompt = f"""You are a helpful academic tutor.

{anchor_prompt}
{few_shot_section}
Instructions:
- Give a clear, educational answer
- Explain concepts step by step
- If the context doesn't contain the answer, say so clearly
- Stay within the topic scope
- Keep the answer concise but complete"""

        user_prompt = f"""Context:
{context if context else "No specific context available."}

Student Question: {state["query"]}

Answer:"""

        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        
        answer = llm.invoke(full_prompt)
        state["answer"] = answer.strip()
        
        logger.info(f"[TUTOR] Generated answer: {len(answer)} chars")
        
        # === EXPERIENCE REPLAY: Log interaction ===
        try:
            from app.learning.learning_element import get_experience_replay
            import asyncio
            
            replay = get_experience_replay()
            asyncio.create_task(replay.add_experience(
                agent_type="tutor",
                session_id=state["session_id"],
                query=state["query"],
                response=answer,
                metadata={
                    "topic": topic,
                    "is_followup": state.get("is_followup", False),
                    "sources_count": len(state.get("sources", [])),
                    "few_shot_used": len(few_shot_section) > 0
                }
            ))
        except Exception as re:
            logger.warning(f"[TUTOR/REPLAY] Could not log experience: {re}")
        # === END EXPERIENCE REPLAY ===
        
    except Exception as e:
        logger.error(f"[TUTOR] Generation error: {e}", exc_info=True)
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
    """Build the LangGraph workflow for tutoring with TAL/ABCR/MCP"""
    workflow = StateGraph(TutorState)
    
    # Add nodes
    workflow.add_node("moderate", moderate_query)
    workflow.add_node("context_routing", context_routing)
    workflow.add_node("retrieve", retrieve_with_mcp)
    workflow.add_node("generate", generate_answer)
    
    # Set entry point
    workflow.set_entry_point("moderate")
    
    # Add conditional routing from moderation
    workflow.add_conditional_edges(
        "moderate",
        route_moderation,
        {
            "blocked": "generate",  # Skip to generate blocked response
            "continue": "context_routing"
        }
    )
    
    # Add remaining edges
    workflow.add_edge("context_routing", "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)
    
    return workflow.compile()


# ============================================================================
# Agent Class
# ============================================================================

class TutorAgent:
    """
    LangGraph-based Tutor Agent with TAL/ABCR/MCP Integration
    
    Features:
    - ABCR for follow-up detection
    - TAL for topic anchoring
    - MCP for web isolation
    - Hugging Face LLM (Mistral-7B)
    """
    
    def __init__(self):
        self.graph = build_tutor_graph()
        logger.info("[TUTOR] Initialized LangGraph Tutor Agent with TAL/ABCR/MCP")
    
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a student question
        
        Args:
            input_data: {
                query: str,
                user_id: str,
                session_id: str (optional),
                classroom_id: str (optional),
                clicked_suggestion: bool (optional)
            }
        
        Returns:
            {
                answer: str,
                sources: List[Dict],
                topic_anchor: {id, title},
                is_followup: bool,
                abcr_confidence: float,
                context_sources: List[str],
                blocked: bool,
                confirm_new_topic: bool
            }
        """
        # Generate session ID if not provided
        session_id = input_data.get("session_id") or str(uuid.uuid4())
        request_id = str(uuid.uuid4())[:8]
        
        # Initialize state
        initial_state: TutorState = {
            "query": input_data.get("query", ""),
            "user_id": input_data.get("user_id", ""),
            "session_id": session_id,
            "request_id": request_id,
            "classroom_id": input_data.get("classroom_id", ""),
            "clicked_suggestion": input_data.get("clicked_suggestion", False),
            
            # Session state
            "turn_texts": [],
            "last_abcr_decision": "new_topic",
            "consecutive_borderline": 0,
            
            # Moderation
            "is_academic": True,
            "moderation_score": 0.0,
            
            # ABCR
            "abcr_decision": "new_topic",
            "abcr_confidence": 0.0,
            "is_followup": False,
            "matched_turn_index": None,
            
            # TAL
            "topic_anchor_id": None,
            "topic_anchor_title": None,
            "confirm_new_topic": False,
            
            # RAG/MCP
            "raw_chunks": [],
            "mcp_chunks": [],
            "mcp_reason": "",
            "anchor_hits": 0,
            "web_filtered_count": 0,
            "context_sources": [],
            
            # Output
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
                "session_id": session_id,
                "request_id": request_id,
                "data": {
                    "answer": final_state["answer"],
                    "sources": final_state["sources"],
                    "blocked": final_state["blocked"],
                    "moderation_score": final_state["moderation_score"],
                    
                    # TAL
                    "topic_anchor": {
                        "id": final_state.get("topic_anchor_id"),
                        "title": final_state.get("topic_anchor_title"),
                    } if final_state.get("topic_anchor_id") else None,
                    
                    # ABCR
                    "is_followup": final_state.get("is_followup", False),
                    "abcr_confidence": final_state.get("abcr_confidence", 0.0),
                    "confirm_new_topic": final_state.get("confirm_new_topic", False),
                    
                    # MCP
                    "context_sources": final_state.get("context_sources", []),
                    "anchor_hits": final_state.get("anchor_hits", 0),
                    "web_filtered_count": final_state.get("web_filtered_count", 0),
                }
            }
            
        except Exception as e:
            logger.error(f"[TUTOR] Agent error: {e}", exc_info=True)
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "agent": "tutor",
                "session_id": session_id,
                "data": {
                    "answer": "An error occurred. Please try again.",
                    "sources": [],
                    "blocked": False,
                    "error": str(e)
                }
            }
