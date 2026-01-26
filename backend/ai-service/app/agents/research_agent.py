"""
Research Agent - LangGraph Agent for Web Research & Content Discovery

This agent handles:
- Web search for educational content
- PDF discovery and download
- YouTube video search
- Content indexing into Qdrant

Graph Flow:
┌──────────┐    ┌───────────┐    ┌──────────┐    ┌───────────┐    ┌───────────┐
│ Analyze  │───▶│ Web Search│───▶│ PDF DL   │───▶│ YouTube   │───▶│ Index     │
│ Query    │    │           │    │ (if PDF) │    │ Search    │    │ Content   │
└──────────┘    └───────────┘    └──────────┘    └───────────┘    └───────────┘
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

class ResearchState(TypedDict):
    """State passed between nodes in the research graph"""
    # Input
    query: str
    user_id: str
    session_id: str
    request_id: str
    
    # Configuration
    search_web: bool
    search_pdfs: bool
    search_youtube: bool
    download_pdfs: bool
    index_content: bool
    max_results: int
    
    # Web Search Results
    web_results: List[Dict]
    web_search_complete: bool
    
    # PDF Results
    pdf_results: List[Dict]
    downloaded_pdfs: List[Dict]
    pdf_search_complete: bool
    
    # YouTube Results
    youtube_results: List[Dict]
    youtube_search_complete: bool
    
    # Indexed Content
    indexed_documents: List[Dict]
    
    # Output
    summary: str
    total_sources: int
    error: Optional[str]


# ============================================================================
# Node Functions
# ============================================================================

async def analyze_query(state: ResearchState) -> ResearchState:
    """Analyze the query to determine research strategy"""
    query = state["query"]
    
    # Enhance query for educational content
    # In production, could use LLM to extract key concepts
    state["request_id"] = str(uuid.uuid4())[:8]
    
    logger.info(f"[RESEARCH] Analyzing query: '{query[:50]}...'")
    
    # Determine what types of content to search for
    query_lower = query.lower()
    
    # Auto-detect if user wants PDFs
    if "pdf" in query_lower or "document" in query_lower or "notes" in query_lower:
        state["search_pdfs"] = True
        state["download_pdfs"] = True
    
    # Auto-detect if user wants videos
    if "video" in query_lower or "watch" in query_lower or "explain" in query_lower:
        state["search_youtube"] = True
    
    return state


async def web_search_node(state: ResearchState) -> ResearchState:
    """Search the web for educational content"""
    if not state.get("search_web", True):
        state["web_search_complete"] = True
        return state
    
    try:
        from app.agents.tools import invoke_tool
        
        result = await invoke_tool(
            "web_search",
            query=state["query"],
            num_results=state.get("max_results", 5)
        )
        
        if result.success:
            state["web_results"] = result.data.get("results", [])
            logger.info(f"[RESEARCH] Web search found {len(state['web_results'])} results")
        else:
            state["web_results"] = []
            logger.warning(f"[RESEARCH] Web search failed: {result.error}")
        
    except Exception as e:
        logger.error(f"[RESEARCH] Web search error: {e}")
        state["web_results"] = []
    
    state["web_search_complete"] = True
    return state


async def pdf_search_node(state: ResearchState) -> ResearchState:
    """Search for and optionally download PDFs"""
    if not state.get("search_pdfs", False):
        state["pdf_search_complete"] = True
        return state
    
    try:
        from app.agents.tools import invoke_tool
        
        # Search for PDFs
        result = await invoke_tool(
            "pdf_search",
            query=state["query"],
            num_results=state.get("max_results", 3)
        )
        
        if result.success:
            state["pdf_results"] = result.data.get("results", [])
            logger.info(f"[RESEARCH] PDF search found {len(state['pdf_results'])} results")
            
            # Download PDFs if enabled
            if state.get("download_pdfs", False) and state["pdf_results"]:
                pdf_urls = [r["url"] for r in state["pdf_results"]]
                
                dl_result = await invoke_tool(
                    "download_pdfs_batch",
                    urls=pdf_urls,
                    user_id=state["user_id"],
                    topic=state["query"][:50],
                    max_downloads=3
                )
                
                if dl_result.success:
                    state["downloaded_pdfs"] = dl_result.data.get("downloaded", [])
                    logger.info(f"[RESEARCH] Downloaded {len(state['downloaded_pdfs'])} PDFs")
                else:
                    state["downloaded_pdfs"] = []
        else:
            state["pdf_results"] = []
            
    except Exception as e:
        logger.error(f"[RESEARCH] PDF search error: {e}")
        state["pdf_results"] = []
        state["downloaded_pdfs"] = []
    
    state["pdf_search_complete"] = True
    return state


async def youtube_search_node(state: ResearchState) -> ResearchState:
    """Search YouTube for educational videos"""
    if not state.get("search_youtube", False):
        state["youtube_search_complete"] = True
        return state
    
    try:
        from app.agents.tools import invoke_tool
        
        result = await invoke_tool(
            "youtube_search",
            query=state["query"],
            num_results=3
        )
        
        if result.success:
            state["youtube_results"] = result.data.get("videos", [])
            logger.info(f"[RESEARCH] YouTube search found {len(state['youtube_results'])} videos")
        else:
            state["youtube_results"] = []
            
    except Exception as e:
        logger.error(f"[RESEARCH] YouTube search error: {e}")
        state["youtube_results"] = []
    
    state["youtube_search_complete"] = True
    return state


async def index_content_node(state: ResearchState) -> ResearchState:
    """Index downloaded content into vector database"""
    if not state.get("index_content", True):
        return state
    
    indexed = []
    
    try:
        from app.agents.tools import invoke_tool
        
        # Index downloaded PDFs
        for pdf in state.get("downloaded_pdfs", []):
            # Extract text from PDF
            extract_result = await invoke_tool(
                "extract_pdf_text",
                pdf_path=pdf["file_path"]
            )
            
            if extract_result.success and extract_result.data.get("text"):
                # Index the extracted text
                index_result = await invoke_tool(
                    "index_content",
                    content=extract_result.data["text"],
                    document_id=pdf["file_name"],
                    title=pdf["file_name"],
                    source_type="web_pdf",
                    metadata={
                        "source_url": pdf["source_url"],
                        "user_id": state["user_id"]
                    }
                )
                
                if index_result.success:
                    indexed.append({
                        "document_id": pdf["file_name"],
                        "chunks": index_result.data.get("chunks_indexed", 0)
                    })
                    logger.info(f"[RESEARCH] Indexed PDF: {pdf['file_name']}")
        
        state["indexed_documents"] = indexed
        
    except Exception as e:
        logger.error(f"[RESEARCH] Indexing error: {e}")
    
    return state


async def compile_results(state: ResearchState) -> ResearchState:
    """Compile all results into a summary"""
    total = 0
    summary_parts = []
    
    # Web results
    web_count = len(state.get("web_results", []))
    if web_count > 0:
        total += web_count
        summary_parts.append(f"{web_count} web articles")
    
    # PDF results
    pdf_count = len(state.get("downloaded_pdfs", []))
    if pdf_count > 0:
        total += pdf_count
        summary_parts.append(f"{pdf_count} PDFs downloaded")
    
    # YouTube results
    yt_count = len(state.get("youtube_results", []))
    if yt_count > 0:
        total += yt_count
        summary_parts.append(f"{yt_count} videos")
    
    # Indexed documents
    indexed_count = len(state.get("indexed_documents", []))
    if indexed_count > 0:
        summary_parts.append(f"{indexed_count} documents indexed")
    
    state["total_sources"] = total
    state["summary"] = f"Found {', '.join(summary_parts)}" if summary_parts else "No results found"
    
    logger.info(f"[RESEARCH] Complete: {state['summary']}")
    
    return state


# ============================================================================
# Routing Functions
# ============================================================================

def route_after_web_search(state: ResearchState) -> str:
    """Route after web search based on configuration"""
    if state.get("search_pdfs", False):
        return "pdf_search"
    elif state.get("search_youtube", False):
        return "youtube_search"
    else:
        return "compile"


def route_after_pdf_search(state: ResearchState) -> str:
    """Route after PDF search"""
    if state.get("search_youtube", False):
        return "youtube_search"
    elif state.get("index_content", True) and state.get("downloaded_pdfs"):
        return "index"
    else:
        return "compile"


def route_after_youtube(state: ResearchState) -> str:
    """Route after YouTube search"""
    if state.get("index_content", True) and state.get("downloaded_pdfs"):
        return "index"
    else:
        return "compile"


# ============================================================================
# Graph Builder
# ============================================================================

def build_research_graph():
    """Build the LangGraph workflow for research"""
    workflow = StateGraph(ResearchState)
    
    # Add nodes
    workflow.add_node("analyze", analyze_query)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("pdf_search", pdf_search_node)
    workflow.add_node("youtube_search", youtube_search_node)
    workflow.add_node("index", index_content_node)
    workflow.add_node("compile", compile_results)
    
    # Set entry point
    workflow.set_entry_point("analyze")
    
    # Add edges
    workflow.add_edge("analyze", "web_search")
    
    # Conditional routing after web search
    workflow.add_conditional_edges(
        "web_search",
        route_after_web_search,
        {
            "pdf_search": "pdf_search",
            "youtube_search": "youtube_search",
            "compile": "compile"
        }
    )
    
    # Conditional routing after PDF search
    workflow.add_conditional_edges(
        "pdf_search",
        route_after_pdf_search,
        {
            "youtube_search": "youtube_search",
            "index": "index",
            "compile": "compile"
        }
    )
    
    # Conditional routing after YouTube
    workflow.add_conditional_edges(
        "youtube_search",
        route_after_youtube,
        {
            "index": "index",
            "compile": "compile"
        }
    )
    
    # Index always goes to compile
    workflow.add_edge("index", "compile")
    
    # Compile goes to END
    workflow.add_edge("compile", END)
    
    return workflow.compile()


# ============================================================================
# Agent Class
# ============================================================================

class ResearchAgent:
    """
    LangGraph-based Research Agent
    
    Capabilities:
    - Web search for educational content
    - PDF search and download
    - YouTube video search
    - Content indexing into Qdrant
    
    Usage:
        agent = ResearchAgent()
        result = await agent.execute({
            "query": "learn about photosynthesis",
            "user_id": "user123",
            "search_pdfs": True,
            "download_pdfs": True
        })
    """
    
    def __init__(self):
        self.graph = build_research_graph()
        logger.info("[RESEARCH] Initialized Research Agent")
    
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute research for a query
        
        Args:
            input_data: {
                query: str - Research topic
                user_id: str - User ID
                search_web: bool - Search web (default True)
                search_pdfs: bool - Search for PDFs (default False)
                download_pdfs: bool - Download found PDFs (default False)
                search_youtube: bool - Search YouTube (default False)
                index_content: bool - Index to vector DB (default True)
                max_results: int - Max results per source (default 5)
            }
        
        Returns:
            Research results with sources and summary
        """
        session_id = input_data.get("session_id") or str(uuid.uuid4())
        
        # Initialize state
        initial_state: ResearchState = {
            "query": input_data.get("query", ""),
            "user_id": input_data.get("user_id", ""),
            "session_id": session_id,
            "request_id": "",
            
            # Configuration
            "search_web": input_data.get("search_web", True),
            "search_pdfs": input_data.get("search_pdfs", False),
            "search_youtube": input_data.get("search_youtube", False),
            "download_pdfs": input_data.get("download_pdfs", False),
            "index_content": input_data.get("index_content", True),
            "max_results": input_data.get("max_results", 5),
            
            # Results
            "web_results": [],
            "web_search_complete": False,
            "pdf_results": [],
            "downloaded_pdfs": [],
            "pdf_search_complete": False,
            "youtube_results": [],
            "youtube_search_complete": False,
            "indexed_documents": [],
            
            # Output
            "summary": "",
            "total_sources": 0,
            "error": None
        }
        
        try:
            # Run the graph
            final_state = await self.graph.ainvoke(initial_state)
            
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "agent": "research",
                "session_id": session_id,
                "request_id": final_state.get("request_id", ""),
                "data": {
                    "query": final_state["query"],
                    "summary": final_state["summary"],
                    "total_sources": final_state["total_sources"],
                    
                    "web_results": final_state.get("web_results", []),
                    "pdf_results": final_state.get("pdf_results", []),
                    "downloaded_pdfs": final_state.get("downloaded_pdfs", []),
                    "youtube_results": final_state.get("youtube_results", []),
                    "indexed_documents": final_state.get("indexed_documents", [])
                }
            }
            
        except Exception as e:
            logger.error(f"[RESEARCH] Agent error: {e}", exc_info=True)
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "agent": "research",
                "session_id": session_id,
                "data": {
                    "query": input_data.get("query", ""),
                    "summary": "Research failed",
                    "total_sources": 0,
                    "error": str(e)
                }
            }


# Singleton instance
_research_agent = None


def get_research_agent() -> ResearchAgent:
    """Get or create the research agent singleton"""
    global _research_agent
    if _research_agent is None:
        _research_agent = ResearchAgent()
    return _research_agent
