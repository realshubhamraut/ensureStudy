"""
Web Resources Fetching Service

Fetches educational resources from the web:
- YouTube videos (using DuckDuckGo video search)
- Images (using DuckDuckGo)
- Articles (using Fetch + Readability + Cache)

All FREE, no API keys required.
"""
import hashlib
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
import time

# DuckDuckGo search (used for videos and images)
try:
    from duckduckgo_search import DDGS
    DDG_AVAILABLE = True
except ImportError:
    DDG_AVAILABLE = False

# Content crawler for article extraction
try:
    from .content_crawler import fetch_and_extract, search_and_extract
    CRAWLER_AVAILABLE = True
except ImportError:
    CRAWLER_AVAILABLE = False


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class VideoResource:
    """YouTube video resource."""
    id: str
    title: str
    url: str
    thumbnail_url: str
    embed_url: str
    duration: Optional[str]
    channel: str
    relevance: int = 85


@dataclass
class ImageResource:
    """Web image resource."""
    id: str
    title: str
    url: str
    thumbnail_url: str
    source: str
    relevance: int = 80


@dataclass
class ArticleResource:
    """Web article resource with cached content."""
    id: str
    title: str
    url: str
    snippet: Optional[str]
    source: str
    resource_type: str = "article"  # article or webpage
    relevance: int = 80
    # Cached content from Readability extraction
    cached_content: Optional[str] = None   # Full extracted text
    cached_summary: Optional[str] = None   # First ~500 chars
    cached_images: List[str] = field(default_factory=list)  # Image URLs from article


@dataclass
class WebResourcesResult:
    """Combined web resources result."""
    videos: List[VideoResource]
    images: List[ImageResource]
    articles: List[ArticleResource]


# ============================================================================
# Cache (Simple in-memory cache)
# ============================================================================

_cache: Dict[str, Any] = {}
_cache_ttl = 300  # 5 minutes


def _get_cache_key(query: str, resource_type: str) -> str:
    """Generate cache key from query and type."""
    return hashlib.md5(f"{query}:{resource_type}".lower().encode()).hexdigest()


def _get_from_cache(key: str) -> Optional[Any]:
    """Get value from cache if not expired."""
    if key in _cache:
        data, timestamp = _cache[key]
        if time.time() - timestamp < _cache_ttl:
            return data
        del _cache[key]
    return None


def _set_cache(key: str, value: Any) -> None:
    """Store value in cache."""
    _cache[key] = (value, time.time())


# ============================================================================
# YouTube Video Search (using DuckDuckGo video search)
# ============================================================================

def fetch_youtube_videos(query: str, max_results: int = 3) -> List[VideoResource]:
    """
    Fetch YouTube videos for a query using DuckDuckGo video search.
    
    Args:
        query: Search query (e.g., "photosynthesis explained")
        max_results: Maximum number of videos to return
        
    Returns:
        List of VideoResource objects
    """
    if not DDG_AVAILABLE:
        print("Warning: duckduckgo-search not installed")
        return []
    
    # Check cache
    cache_key = _get_cache_key(query, "youtube")
    cached = _get_from_cache(cache_key)
    if cached:
        return cached[:max_results]
    
    try:
        # Add educational keywords and site restriction for YouTube
        search_query = f"site:youtube.com {query} explained tutorial"
        
        with DDGS() as ddgs:
            results = list(ddgs.videos(
                search_query,
                max_results=max_results + 2,
                safesearch="on"
            ))
        
        videos = []
        for i, video in enumerate(results[:max_results]):
            # Extract video ID from YouTube URL
            url = video.get("content", "") or video.get("href", "")
            video_id = ""
            if "youtube.com/watch?v=" in url:
                video_id = url.split("watch?v=")[-1].split("&")[0]
            elif "youtu.be/" in url:
                video_id = url.split("youtu.be/")[-1].split("?")[0]
            
            if not video_id:
                continue
                
            videos.append(VideoResource(
                id=f"yt_{video_id}",
                title=video.get("title", "Untitled Video"),
                url=f"https://www.youtube.com/watch?v={video_id}",
                thumbnail_url=video.get("image", "") or f"https://img.youtube.com/vi/{video_id}/mqdefault.jpg",
                embed_url=f"https://www.youtube.com/embed/{video_id}",
                duration=video.get("duration", ""),
                channel=video.get("publisher", "YouTube"),
                relevance=95 - (i * 5)  # Decrease relevance for later results
            ))
        
        # Cache results
        _set_cache(cache_key, videos)
        return videos
        
    except Exception as e:
        print(f"YouTube search error: {e}")
        return []


# ============================================================================
# Web Image Search
# ============================================================================

def fetch_web_images(query: str, max_results: int = 3) -> List[ImageResource]:
    """
    Fetch images from the web using DuckDuckGo.
    
    Args:
        query: Search query (e.g., "photosynthesis diagram")
        max_results: Maximum number of images to return
        
    Returns:
        List of ImageResource objects
    """
    if not DDG_AVAILABLE:
        print("Warning: duckduckgo-search not installed")
        return []
    
    # Check cache
    cache_key = _get_cache_key(query, "images")
    cached = _get_from_cache(cache_key)
    if cached:
        return cached[:max_results]
    
    try:
        # Add diagram/educational keywords
        search_query = f"{query} diagram educational"
        
        with DDGS() as ddgs:
            results = list(ddgs.images(
                search_query,
                max_results=max_results + 2,
                safesearch="on"
            ))
        
        images = []
        for i, img in enumerate(results[:max_results]):
            images.append(ImageResource(
                id=f"img_{i}_{hash(img.get('image', '')) % 10000}",
                title=img.get("title", "Image"),
                url=img.get("image", ""),
                thumbnail_url=img.get("thumbnail", img.get("image", "")),
                source=img.get("source", "Web"),
                relevance=90 - (i * 4)
            ))
        
        # Cache results
        _set_cache(cache_key, images)
        return images
        
    except Exception as e:
        print(f"Image search error: {e}")
        return []


# ============================================================================
# Web Article Search
# ============================================================================

def fetch_web_articles(query: str, max_results: int = 3) -> List[ArticleResource]:
    """
    Fetch articles from the web using DuckDuckGo.
    
    Prioritizes educational sources like Wikipedia, Khan Academy, etc.
    
    Args:
        query: Search query
        max_results: Maximum number of articles to return
        
    Returns:
        List of ArticleResource objects with cached content
    """
    if not DDG_AVAILABLE:
        print("Warning: duckduckgo-search not installed")
        return []
    
    # Check cache for this query
    cache_key = _get_cache_key(query, "articles")
    cached = _get_from_cache(cache_key)
    if cached:
        return cached[:max_results]
    
    try:
        articles = []
        
        # Search for Wikipedia and educational sources
        search_query = f"{query} wikipedia OR britannica"
        with DDGS() as ddgs:
            results = list(ddgs.text(
                search_query,
                max_results=max_results + 2,
                safesearch="on"
            ))
        
        # Filter to only reliable sources
        reliable_domains = ["wikipedia.org", "britannica.com", "khanacademy.org", "bbc.co.uk", "wikiwand.com"]
        
        for i, result in enumerate(results):
            url = result.get("href", "")
            
            # Only include if from a reliable domain
            if not any(domain in url for domain in reliable_domains):
                continue
            
            # Skip wiktionary (dictionary, not articles)
            if "wiktionary.org" in url:
                continue
            
            # Determine source name
            source = "Web"
            resource_type = "article"
            if "wikipedia.org" in url or "wikiwand.com" in url:
                source = "Wikipedia"
            elif "khanacademy.org" in url:
                source = "Khan Academy"
                resource_type = "webpage"
            elif "britannica.com" in url:
                source = "Britannica"
            elif "bbc.co.uk" in url:
                source = "BBC Bitesize"
            
            # Try to extract full content using content crawler
            cached_content = None
            cached_summary = None
            cached_images = []
            
            if CRAWLER_AVAILABLE:
                try:
                    extracted = fetch_and_extract(url)
                    if extracted:
                        cached_content = extracted.text_content
                        cached_summary = extracted.summary
                        cached_images = extracted.images
                        # Use extracted title if available
                        if extracted.title:
                            result["title"] = extracted.title
                except Exception as e:
                    print(f"Content extraction failed for {url}: {e}")
            
            articles.append(ArticleResource(
                id=f"art_{i}_{hash(url) % 10000}",
                title=result.get("title", "Article"),
                url=url,
                snippet=cached_summary or (result.get("body", "")[:200] if result.get("body") else None),
                source=source,
                resource_type=resource_type,
                relevance=95 - (len(articles) * 3),
                cached_content=cached_content,
                cached_summary=cached_summary,
                cached_images=cached_images
            ))
            
            if len(articles) >= max_results:
                break
            
            # Small delay between fetches to avoid rate limiting
            time.sleep(0.3)
        
        # Cache results
        _set_cache(cache_key, articles)
        return articles
        
    except Exception as e:
        print(f"Article search error: {e}")
        return []


# ============================================================================
# Combined Fetch
# ============================================================================

def fetch_all_web_resources(
    query: str,
    max_videos: int = 3,
    max_images: int = 3,
    max_articles: int = 3
) -> WebResourcesResult:
    """
    Fetch all web resources for a query.
    
    Args:
        query: Search query
        max_videos: Max YouTube videos
        max_images: Max images
        max_articles: Max articles
        
    Returns:
        WebResourcesResult with all resources
    """
    # Run searches with small delays to avoid DDG rate limits
    videos = fetch_youtube_videos(query, max_videos)
    time.sleep(0.5)  # Small delay to avoid rate limiting
    
    images = fetch_web_images(query, max_images)
    time.sleep(0.5)  # Small delay to avoid rate limiting
    
    articles = fetch_web_articles(query, max_articles)
    
    return WebResourcesResult(
        videos=videos,
        images=images,
        articles=articles
    )


# ============================================================================
# Conversion to Dict (for API response)
# ============================================================================

def web_resources_to_dict(result: WebResourcesResult) -> Dict[str, List[Dict]]:
    """Convert WebResourcesResult to dictionary for JSON response."""
    return {
        "videos": [
            {
                "id": v.id,
                "type": "video",
                "title": v.title,
                "url": v.url,
                "thumbnailUrl": v.thumbnail_url,
                "embedUrl": v.embed_url,
                "duration": v.duration,
                "source": v.channel or "YouTube",
                "relevance": v.relevance
            }
            for v in result.videos
        ],
        "images": [
            {
                "id": i.id,
                "type": "image",
                "title": i.title,
                "url": i.url,
                "thumbnailUrl": i.thumbnail_url,
                "source": i.source,
                "relevance": i.relevance
            }
            for i in result.images
        ],
        "articles": [
            {
                "id": a.id,
                "type": a.resource_type,
                "title": a.title,
                "url": a.url,
                "snippet": a.snippet,
                "source": a.source,
                "relevance": a.relevance,
                # Cached content from Readability extraction
                "cachedContent": a.cached_content,
                "cachedSummary": a.cached_summary,
                "cachedImages": a.cached_images
            }
            for a in result.articles
        ]
    }
