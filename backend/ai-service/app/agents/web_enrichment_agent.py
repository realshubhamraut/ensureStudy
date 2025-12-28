"""
Web Enrichment Agent - LangGraph Query-Time Academic Content Fetcher

Triggered when a student asks a question, this agent:
1. Searches academic sources in parallel
2. Filters by quality and relevance
3. Caches results in Redis
4. Returns enriched sources for the AI Tutor

Uses LangGraph StateGraph for orchestration.
"""
import logging
import asyncio
import os
from typing import Dict, Any, List, TypedDict, Optional
from datetime import datetime
from dataclasses import dataclass

from langgraph.graph import StateGraph, END

logger = logging.getLogger(__name__)


# ============================================================================
# State Definition
# ============================================================================

@dataclass
class WebSource:
    """A web source with metadata"""
    id: str
    title: str
    url: str
    source_type: str  # 'wikipedia', 'khan_academy', 'video', 'article'
    snippet: str
    relevance_score: float
    domain: str
    cached_content: Optional[str] = None


class WebEnrichmentState(TypedDict):
    """State for web enrichment graph"""
    # Input
    query: str
    subject: Optional[str]
    student_id: str
    
    # Processing
    cache_hit: bool
    raw_results: Dict[str, List[Dict]]
    filtered_results: List[Dict]
    
    # Output
    sources: List[Dict]
    error: Optional[str]


# ============================================================================
# Source Fetchers
# ============================================================================

async def fetch_wikipedia(query: str, max_results: int = 3) -> List[Dict]:
    """Fetch Wikipedia articles using DuckDuckGo"""
    try:
        from duckduckgo_search import DDGS
        
        with DDGS() as ddgs:
            results = list(ddgs.text(
                f"site:wikipedia.org {query}",
                max_results=max_results
            ))
        
        return [
            {
                "id": f"wiki_{i}",
                "title": r.get("title", ""),
                "url": r.get("href", ""),
                "snippet": r.get("body", ""),
                "source_type": "wikipedia",
                "domain": "wikipedia.org",
                "relevance_score": 0.9 - (i * 0.1)
            }
            for i, r in enumerate(results)
        ]
        
    except Exception as e:
        logger.warning(f"Wikipedia search error: {e}")
        return []


async def fetch_khan_academy(query: str, subject: str = None, max_results: int = 3) -> List[Dict]:
    """Fetch Khan Academy content"""
    try:
        from duckduckgo_search import DDGS
        
        search_query = f"site:khanacademy.org {query}"
        if subject:
            search_query = f"site:khanacademy.org {subject} {query}"
        
        with DDGS() as ddgs:
            results = list(ddgs.text(search_query, max_results=max_results))
        
        return [
            {
                "id": f"khan_{i}",
                "title": r.get("title", ""),
                "url": r.get("href", ""),
                "snippet": r.get("body", ""),
                "source_type": "khan_academy",
                "domain": "khanacademy.org",
                "relevance_score": 0.92 - (i * 0.1)
            }
            for i, r in enumerate(results)
        ]
        
    except Exception as e:
        logger.warning(f"Khan Academy search error: {e}")
        return []


async def fetch_educational_videos(query: str, max_results: int = 3) -> List[Dict]:
    """Fetch educational YouTube videos"""
    try:
        from duckduckgo_search import DDGS
        
        with DDGS() as ddgs:
            results = list(ddgs.videos(
                f"{query} tutorial explanation",
                max_results=max_results
            ))
        
        videos = []
        for i, r in enumerate(results):
            # Filter for educational channels
            publisher = r.get("publisher", "").lower()
            title = r.get("title", "").lower()
            
            # Prioritize educational content
            is_educational = any(x in publisher or x in title for x in [
                "academy", "edu", "learn", "tutorial", "khan", "crash course"
            ])
            
            videos.append({
                "id": f"video_{i}",
                "title": r.get("title", ""),
                "url": r.get("content", ""),
                "snippet": r.get("description", ""),
                "source_type": "video",
                "domain": "youtube.com",
                "thumbnail": r.get("image", ""),
                "duration": r.get("duration", ""),
                "relevance_score": (0.85 if is_educational else 0.7) - (i * 0.05)
            })
        
        return sorted(videos, key=lambda x: x["relevance_score"], reverse=True)[:max_results]
        
    except Exception as e:
        logger.warning(f"Video search error: {e}")
        return []


async def fetch_academic_articles(query: str, max_results: int = 3) -> List[Dict]:
    """Fetch academic articles from general web"""
    try:
        from duckduckgo_search import DDGS
        
        # Prioritize educational domains
        educational_sites = "site:edu OR site:coursera.org OR site:edx.org OR site:mit.edu"
        
        with DDGS() as ddgs:
            results = list(ddgs.text(
                f"{educational_sites} {query}",
                max_results=max_results
            ))
        
        return [
            {
                "id": f"article_{i}",
                "title": r.get("title", ""),
                "url": r.get("href", ""),
                "snippet": r.get("body", ""),
                "source_type": "article",
                "domain": r.get("href", "").split("/")[2] if "/" in r.get("href", "") else "",
                "relevance_score": 0.8 - (i * 0.1)
            }
            for i, r in enumerate(results)
        ]
        
    except Exception as e:
        logger.warning(f"Article search error: {e}")
        return []


# ============================================================================
# Node Functions
# ============================================================================

async def check_cache(state: WebEnrichmentState) -> WebEnrichmentState:
    """Check Redis cache for existing results"""
    try:
        from app.services.response_cache import get_response_cache
        
        cache = get_response_cache()
        cached = cache.get_web_resources(state["query"])
        
        if cached:
            state["cache_hit"] = True
            state["sources"] = cached.get("sources", [])
            logger.info(f"Web enrichment cache hit for: {state['query'][:50]}")
        else:
            state["cache_hit"] = False
            
    except Exception as e:
        logger.warning(f"Cache check error: {e}")
        state["cache_hit"] = False
    
    return state


async def search_sources(state: WebEnrichmentState) -> WebEnrichmentState:
    """Search all sources in parallel"""
    if state["cache_hit"]:
        return state
    
    try:
        # Parallel fetch from all sources
        results = await asyncio.gather(
            fetch_wikipedia(state["query"]),
            fetch_khan_academy(state["query"], state.get("subject")),
            fetch_educational_videos(state["query"]),
            fetch_academic_articles(state["query"]),
            return_exceptions=True
        )
        
        state["raw_results"] = {
            "wikipedia": results[0] if not isinstance(results[0], Exception) else [],
            "khan_academy": results[1] if not isinstance(results[1], Exception) else [],
            "videos": results[2] if not isinstance(results[2], Exception) else [],
            "articles": results[3] if not isinstance(results[3], Exception) else []
        }
        
        logger.info(f"Fetched: {sum(len(v) for v in state['raw_results'].values())} raw sources")
        
    except Exception as e:
        logger.error(f"Source search error: {e}")
        state["error"] = str(e)
    
    return state


async def filter_and_rank(state: WebEnrichmentState) -> WebEnrichmentState:
    """Filter and rank sources by quality"""
    if state["cache_hit"]:
        return state
    
    try:
        all_sources = []
        
        for source_type, sources in state["raw_results"].items():
            for source in sources:
                # Quality scoring
                score = source.get("relevance_score", 0.5)
                
                # Boost for educational domains
                domain = source.get("domain", "")
                if any(edu in domain for edu in ["wikipedia", "khanacademy", ".edu", "mit", "coursera"]):
                    score += 0.1
                
                # Penalize empty snippets
                if not source.get("snippet"):
                    score -= 0.2
                
                source["relevance_score"] = min(1.0, max(0.0, score))
                all_sources.append(source)
        
        # Sort by relevance and deduplicate
        seen_urls = set()
        filtered = []
        
        for source in sorted(all_sources, key=lambda x: x["relevance_score"], reverse=True):
            if source["url"] not in seen_urls:
                seen_urls.add(source["url"])
                filtered.append(source)
        
        # Keep top 8 sources
        state["filtered_results"] = filtered[:8]
        
        logger.info(f"Filtered to {len(state['filtered_results'])} sources")
        
    except Exception as e:
        logger.error(f"Filter error: {e}")
        state["error"] = str(e)
    
    return state


async def cache_and_return(state: WebEnrichmentState) -> WebEnrichmentState:
    """Cache results and format output"""
    if state["cache_hit"]:
        return state
    
    try:
        from app.services.response_cache import get_response_cache
        
        state["sources"] = state.get("filtered_results", [])
        
        # Cache for 24 hours
        cache = get_response_cache()
        cache.set_web_resources(
            state["query"],
            {"sources": state["sources"]},
            ttl=86400  # 24 hours
        )
        
        logger.info(f"Cached {len(state['sources'])} sources for query")
        
    except Exception as e:
        logger.warning(f"Cache error: {e}")
        state["sources"] = state.get("filtered_results", [])
    
    return state


def route_after_cache(state: WebEnrichmentState) -> str:
    """Skip search if cache hit"""
    if state["cache_hit"]:
        return "done"
    return "search"


# ============================================================================
# Graph Builder
# ============================================================================

def build_web_enrichment_graph():
    """Build the LangGraph workflow for web enrichment"""
    workflow = StateGraph(WebEnrichmentState)
    
    # Add nodes
    workflow.add_node("check_cache", check_cache)
    workflow.add_node("search", search_sources)
    workflow.add_node("filter", filter_and_rank)
    workflow.add_node("cache", cache_and_return)
    
    # Entry point
    workflow.set_entry_point("check_cache")
    
    # Conditional routing from cache check
    workflow.add_conditional_edges("check_cache", route_after_cache, {
        "search": "search",
        "done": END
    })
    
    # Sequential flow after search
    workflow.add_edge("search", "filter")
    workflow.add_edge("filter", "cache")
    workflow.add_edge("cache", END)
    
    return workflow.compile()


# ============================================================================
# Agent Class
# ============================================================================

class WebEnrichmentAgent:
    """
    LangGraph-based Web Enrichment Agent
    
    Fetches academic content from:
    - Wikipedia
    - Khan Academy
    - Educational YouTube videos
    - Academic articles (.edu, Coursera, etc.)
    
    Features:
    - Parallel fetching for speed
    - Quality filtering and ranking
    - Redis caching (24h TTL)
    - No API keys required
    """
    
    def __init__(self):
        self.graph = build_web_enrichment_graph()
        logger.info("Initialized WebEnrichmentAgent with LangGraph")
    
    async def enrich(
        self,
        query: str,
        subject: Optional[str] = None,
        student_id: str = ""
    ) -> Dict[str, Any]:
        """
        Enrich a query with web sources
        
        Args:
            query: Student's question
            subject: Optional subject filter
            student_id: For logging
        
        Returns:
            {sources: [...], cache_hit: bool}
        """
        initial_state: WebEnrichmentState = {
            "query": query,
            "subject": subject,
            "student_id": student_id,
            "cache_hit": False,
            "raw_results": {},
            "filtered_results": [],
            "sources": [],
            "error": None
        }
        
        try:
            final_state = await self.graph.ainvoke(initial_state)
            
            return {
                "success": True,
                "sources": final_state["sources"],
                "cache_hit": final_state["cache_hit"],
                "source_count": len(final_state["sources"])
            }
            
        except Exception as e:
            logger.error(f"Web enrichment error: {e}")
            return {
                "success": False,
                "sources": [],
                "error": str(e)
            }


# ============================================================================
# Convenience Function
# ============================================================================

async def enrich_with_web_sources(
    query: str,
    subject: Optional[str] = None,
    student_id: str = ""
) -> Dict[str, Any]:
    """
    Enrich a query with academic web sources
    
    Args:
        query: The student's question
        subject: Optional subject filter
        student_id: For analytics
    
    Returns:
        {sources: [...], cache_hit: bool, source_count: int}
    """
    agent = WebEnrichmentAgent()
    return await agent.enrich(query, subject, student_id)
