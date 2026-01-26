"""
Agent API Routes - Universal Agent Endpoint

Provides a single entry point for all AI agent operations.
The orchestrator automatically routes to the appropriate agent.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import logging

router = APIRouter(prefix="/api/agent", tags=["AI Agent"])

logger = logging.getLogger(__name__)


# ============================================================================
# Request/Response Models
# ============================================================================

class AgentChatRequest(BaseModel):
    """Request for agent chat"""
    query: str = Field(..., description="User's question or request")
    user_id: str = Field(default="", description="User identifier")
    session_id: str = Field(default="", description="Session ID for context continuity")
    classroom_id: str = Field(default="", description="Classroom context")
    
    # Optional hints to guide agent selection
    preferred_agent: Optional[str] = Field(
        default=None,
        description="Preferred agent: 'tutor', 'research', 'content'"
    )
    
    # Research options
    search_pdfs: bool = Field(default=False, description="Search for PDFs")
    download_pdfs: bool = Field(default=False, description="Download found PDFs")
    search_youtube: bool = Field(default=False, description="Search YouTube")


class AgentChatResponse(BaseModel):
    """Response from agent chat"""
    timestamp: str
    session_id: str
    request_id: str
    response: str
    intent: Dict[str, Any]
    agents_used: List[str]
    actions: List[str]
    sources: List[Dict]
    details: Optional[Dict] = None
    error: Optional[str] = None


class ToolInvokeRequest(BaseModel):
    """Request to invoke a specific tool"""
    tool_name: str = Field(..., description="Name of the tool to invoke")
    parameters: Dict[str, Any] = Field(default={}, description="Tool parameters")


class ToolInvokeResponse(BaseModel):
    """Response from tool invocation"""
    success: bool
    data: Any
    error: Optional[str] = None
    execution_time_ms: float = 0.0


class ToolListResponse(BaseModel):
    """List of available tools"""
    tools: List[Dict[str, Any]]
    count: int


# ============================================================================
# Endpoints
# ============================================================================

@router.post("/chat", response_model=AgentChatResponse)
async def agent_chat(request: AgentChatRequest):
    """
    Universal AI Agent Chat Endpoint
    
    The orchestrator analyzes your query and routes it to the appropriate agent(s):
    
    - **"learn" intent** → TutorAgent (Q&A, explanations)
    - **"research" intent** → ResearchAgent (web search, PDFs)
    - **"create" intent** → ContentAgent (notes, quizzes)
    - **"mixed" intent** → Multiple agents
    
    Examples:
    - "Explain photosynthesis" → TutorAgent
    - "Find resources about statistics" → ResearchAgent
    - "Generate notes on French Revolution" → ContentAgent
    - "Research and explain quantum physics" → Research + Tutor
    """
    try:
        from app.agents.orchestrator import get_orchestrator
        
        orchestrator = get_orchestrator()
        
        result = await orchestrator.chat(
            query=request.query,
            user_id=request.user_id,
            session_id=request.session_id,
            classroom_id=request.classroom_id
        )
        
        return AgentChatResponse(**result)
        
    except Exception as e:
        logger.error(f"[AGENT-API] Chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/research")
async def research_endpoint(request: AgentChatRequest):
    """
    Direct access to Research Agent
    
    Searches web, PDFs, and YouTube for educational content.
    """
    try:
        from app.agents.research_agent import get_research_agent
        
        agent = get_research_agent()
        
        result = await agent.execute({
            "query": request.query,
            "user_id": request.user_id,
            "session_id": request.session_id,
            "search_pdfs": request.search_pdfs,
            "download_pdfs": request.download_pdfs,
            "search_youtube": request.search_youtube
        })
        
        return result
        
    except Exception as e:
        logger.error(f"[AGENT-API] Research error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tool/invoke", response_model=ToolInvokeResponse)
async def invoke_tool_endpoint(request: ToolInvokeRequest):
    """
    Invoke a specific tool directly
    
    Use GET /tools to see available tools.
    """
    try:
        from app.agents.tools import invoke_tool
        
        result = await invoke_tool(request.tool_name, **request.parameters)
        
        return ToolInvokeResponse(
            success=result.success,
            data=result.data,
            error=result.error,
            execution_time_ms=result.execution_time_ms
        )
        
    except Exception as e:
        logger.error(f"[AGENT-API] Tool invoke error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tools", response_model=ToolListResponse)
async def list_tools(category: Optional[str] = None):
    """
    List all available agent tools
    
    Categories: web, rag, content, media
    """
    try:
        from app.agents.tools import get_tool_registry
        
        registry = get_tool_registry()
        tools = registry.list_tools(category)
        
        return ToolListResponse(
            tools=[t.to_schema() for t in tools],
            count=len(tools)
        )
        
    except Exception as e:
        logger.error(f"[AGENT-API] List tools error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def agent_health():
    """Check agent system health"""
    try:
        from app.agents.tools import get_tool_registry
        
        registry = get_tool_registry()
        tool_count = len(registry.list_tools())
        
        return {
            "status": "healthy",
            "service": "agent-orchestrator",
            "tools_registered": tool_count,
            "agents": ["orchestrator", "tutor", "research"]
        }
        
    except Exception as e:
        return {
            "status": "degraded",
            "error": str(e)
        }
