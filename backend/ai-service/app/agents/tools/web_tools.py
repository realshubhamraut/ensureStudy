"""
Web & Search Tools - LangGraph Agent Tools

Tools for web searching, PDF discovery, and content downloading.
Wraps existing services: search_api.py, pdf_downloader.py, youtube_video_service.py
"""
from typing import List, Dict, Any, Optional
import logging

from .base_tool import AgentTool, ToolParameter, ToolResult, get_tool_registry

logger = logging.getLogger(__name__)


# ============================================================================
# Web Search Tool
# ============================================================================

async def _web_search(query: str, num_results: int = 5) -> Dict[str, Any]:
    """Execute web search using Serper API"""
    from app.services.search_api import search_web
    
    results = await search_web(query, num_results)
    
    return {
        "query": query,
        "results": results,
        "count": len(results)
    }


web_search_tool = AgentTool(
    name="web_search",
    description="Search the web for educational content using Serper API. Returns URLs, titles, and snippets from trusted educational sources.",
    func=_web_search,
    parameters=[
        ToolParameter(
            name="query",
            type="string",
            description="Search query - will be enhanced with educational context",
            required=True
        ),
        ToolParameter(
            name="num_results",
            type="integer",
            description="Number of results to return (1-10)",
            required=False,
            default=5
        )
    ],
    category="web"
)


# ============================================================================
# PDF Search Tool
# ============================================================================

async def _pdf_search(query: str, num_results: int = 3) -> Dict[str, Any]:
    """Search specifically for PDF documents"""
    from app.services.search_api import search_pdfs
    
    results = await search_pdfs(query, num_results)
    
    return {
        "query": query,
        "results": results,
        "count": len(results)
    }


pdf_search_tool = AgentTool(
    name="pdf_search",
    description="Search for educational PDF documents. Uses filetype:pdf to find downloadable PDFs from trusted sources.",
    func=_pdf_search,
    parameters=[
        ToolParameter(
            name="query",
            type="string",
            description="Topic to search for PDFs (e.g., 'inferential statistics')",
            required=True
        ),
        ToolParameter(
            name="num_results",
            type="integer",
            description="Number of PDF results to return (1-5)",
            required=False,
            default=3
        )
    ],
    category="web"
)


# ============================================================================
# PDF Download Tool
# ============================================================================

async def _download_pdf(url: str, user_id: str, topic: str = "") -> Dict[str, Any]:
    """Download a PDF from URL and store locally"""
    from app.services.pdf_downloader import WebPDFDownloader
    
    downloader = WebPDFDownloader()
    result = await downloader.download_pdf(url, user_id, topic)
    
    if result.success:
        return {
            "success": True,
            "file_name": result.file_name,
            "file_path": result.file_path,
            "file_size": result.file_size,
            "source_url": result.source_url
        }
    else:
        return {
            "success": False,
            "error": result.error,
            "source_url": result.source_url
        }


download_pdf_tool = AgentTool(
    name="download_pdf",
    description="Download a PDF from a URL and save it locally. Returns file path and metadata.",
    func=_download_pdf,
    parameters=[
        ToolParameter(
            name="url",
            type="string",
            description="URL of the PDF to download",
            required=True
        ),
        ToolParameter(
            name="user_id",
            type="string",
            description="User ID for tracking the download",
            required=True
        ),
        ToolParameter(
            name="topic",
            type="string",
            description="Topic for naming the file",
            required=False,
            default=""
        )
    ],
    category="web"
)


# ============================================================================
# Batch PDF Download Tool
# ============================================================================

async def _download_pdfs_batch(
    urls: List[str],
    user_id: str,
    topic: str = "",
    max_downloads: int = 3
) -> Dict[str, Any]:
    """Download multiple PDFs concurrently"""
    from app.services.pdf_downloader import WebPDFDownloader
    
    downloader = WebPDFDownloader()
    results = await downloader.download_multiple(urls, user_id, topic, max_downloads)
    
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]
    
    return {
        "downloaded": [
            {
                "file_name": r.file_name,
                "file_path": r.file_path,
                "file_size": r.file_size,
                "source_url": r.source_url
            }
            for r in successful
        ],
        "failed": [
            {"url": r.source_url, "error": r.error}
            for r in failed
        ],
        "total_downloaded": len(successful),
        "total_failed": len(failed)
    }


download_pdfs_batch_tool = AgentTool(
    name="download_pdfs_batch",
    description="Download multiple PDFs concurrently from a list of URLs.",
    func=_download_pdfs_batch,
    parameters=[
        ToolParameter(
            name="urls",
            type="array",
            description="List of PDF URLs to download",
            required=True
        ),
        ToolParameter(
            name="user_id",
            type="string",
            description="User ID for tracking",
            required=True
        ),
        ToolParameter(
            name="topic",
            type="string",
            description="Topic for naming files",
            required=False,
            default=""
        ),
        ToolParameter(
            name="max_downloads",
            type="integer",
            description="Maximum PDFs to download",
            required=False,
            default=3
        )
    ],
    category="web"
)


# ============================================================================
# YouTube Search Tool
# ============================================================================

async def _youtube_search(query: str, num_results: int = 3) -> Dict[str, Any]:
    """Search YouTube for educational videos"""
    try:
        from app.services.youtube_video_service import search_youtube_videos
        
        results = await search_youtube_videos(query, num_results)
        
        return {
            "query": query,
            "videos": results,
            "count": len(results)
        }
    except ImportError:
        return {
            "query": query,
            "videos": [],
            "count": 0,
            "error": "YouTube service not available"
        }
    except Exception as e:
        return {
            "query": query,
            "videos": [],
            "count": 0,
            "error": str(e)
        }


youtube_search_tool = AgentTool(
    name="youtube_search",
    description="Search YouTube for educational videos on a topic. Returns video titles, URLs, and metadata.",
    func=_youtube_search,
    parameters=[
        ToolParameter(
            name="query",
            type="string",
            description="Topic to search for videos",
            required=True
        ),
        ToolParameter(
            name="num_results",
            type="integer",
            description="Number of videos to return",
            required=False,
            default=3
        )
    ],
    category="web"
)


# ============================================================================
# Web Content Crawl Tool
# ============================================================================

async def _crawl_url(url: str, extract_text: bool = True) -> Dict[str, Any]:
    """Crawl a URL and extract content"""
    try:
        from app.services.content_crawler import crawl_url
        
        content = await crawl_url(url)
        
        return {
            "url": url,
            "title": content.get("title", ""),
            "content": content.get("text", "") if extract_text else None,
            "html_length": len(content.get("html", "")),
            "text_length": len(content.get("text", "")),
            "success": True
        }
    except Exception as e:
        return {
            "url": url,
            "success": False,
            "error": str(e)
        }


crawl_url_tool = AgentTool(
    name="crawl_url",
    description="Crawl a web URL and extract its text content.",
    func=_crawl_url,
    parameters=[
        ToolParameter(
            name="url",
            type="string",
            description="URL to crawl",
            required=True
        ),
        ToolParameter(
            name="extract_text",
            type="boolean",
            description="Whether to extract and return text content",
            required=False,
            default=True
        )
    ],
    category="web"
)


# ============================================================================
# Register All Tools
# ============================================================================

def register_web_tools():
    """Register all web tools with the global registry"""
    registry = get_tool_registry()
    
    registry.register(web_search_tool)
    registry.register(pdf_search_tool)
    registry.register(download_pdf_tool)
    registry.register(download_pdfs_batch_tool)
    registry.register(youtube_search_tool)
    registry.register(crawl_url_tool)
    
    logger.info(f"[WEB-TOOLS] Registered {len(registry.list_tools('web'))} web tools")


# Auto-register on import
register_web_tools()
