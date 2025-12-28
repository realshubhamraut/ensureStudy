"""
Content Crawler Service

Fetches web pages, extracts clean content using Readability,
and caches for reliable display without iframe embedding.

Approach: Fetch + Readability + Cache (Best practice)
"""
import hashlib
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any
import re

import httpx
from bs4 import BeautifulSoup

# Readability for clean extraction
try:
    from readability import Document
    READABILITY_AVAILABLE = True
except ImportError:
    READABILITY_AVAILABLE = False

# DuckDuckGo for web search
try:
    from duckduckgo_search import DDGS
    DDG_AVAILABLE = True
except ImportError:
    DDG_AVAILABLE = False


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class ExtractedContent:
    """Extracted and cleaned web content."""
    url: str
    title: str
    text_content: str           # Full clean text
    summary: str                # First ~500 chars
    images: List[str] = field(default_factory=list)  # Image URLs
    metadata: Dict[str, Any] = field(default_factory=dict)
    cached_at: datetime = field(default_factory=datetime.now)
    content_hash: str = ""      # For deduplication
    source_domain: str = ""     # e.g., "wikipedia.org"
    
    def __post_init__(self):
        if not self.content_hash:
            self.content_hash = hashlib.md5(self.text_content.encode()).hexdigest()[:12]
        if not self.source_domain and self.url:
            # Extract domain from URL
            match = re.search(r'https?://(?:www\.)?([^/]+)', self.url)
            if match:
                self.source_domain = match.group(1)


# ============================================================================
# Simple In-Memory Cache
# ============================================================================

_content_cache: Dict[str, ExtractedContent] = {}
_cache_ttl = 3600  # 1 hour


def _get_cache_key(url: str) -> str:
    """Generate cache key from URL."""
    return hashlib.md5(url.lower().encode()).hexdigest()


def _get_from_cache(url: str) -> Optional[ExtractedContent]:
    """Get content from cache if not expired."""
    key = _get_cache_key(url)
    if key in _content_cache:
        content = _content_cache[key]
        age = (datetime.now() - content.cached_at).total_seconds()
        if age < _cache_ttl:
            return content
        del _content_cache[key]
    return None


def _set_cache(url: str, content: ExtractedContent) -> None:
    """Store content in cache."""
    key = _get_cache_key(url)
    content.cached_at = datetime.now()
    _content_cache[key] = content


# ============================================================================
# Content Fetching
# ============================================================================

def fetch_page(url: str, timeout: float = 10.0) -> Optional[str]:
    """
    Fetch a web page's HTML content.
    
    Args:
        url: Page URL to fetch
        timeout: Request timeout in seconds
        
    Returns:
        HTML content string or None if failed
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }
    
    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            response = client.get(url, headers=headers)
            response.raise_for_status()
            return response.text
    except Exception as e:
        print(f"Failed to fetch {url}: {e}")
        return None


# ============================================================================
# Content Extraction
# ============================================================================

def extract_content(html: str, url: str) -> Optional[ExtractedContent]:
    """
    Extract clean content from HTML using Readability.
    
    Args:
        html: Raw HTML content
        url: Original URL (for metadata)
        
    Returns:
        ExtractedContent or None if extraction failed
    """
    if not READABILITY_AVAILABLE:
        print("Warning: readability-lxml not installed")
        return None
    
    try:
        # Use Readability to extract main content
        doc = Document(html)
        title = doc.title()
        content_html = doc.summary()
        
        # Parse the cleaned HTML with BeautifulSoup
        soup = BeautifulSoup(content_html, 'lxml')
        
        # Extract plain text
        text_content = soup.get_text(separator='\n', strip=True)
        
        # Clean up excessive whitespace
        text_content = re.sub(r'\n{3,}', '\n\n', text_content)
        text_content = re.sub(r' {2,}', ' ', text_content)
        
        # Extract images
        images = []
        for img in soup.find_all('img'):
            src = img.get('src', '')
            if src:
                # Convert relative URLs to absolute
                if src.startswith('//'):
                    src = 'https:' + src
                elif src.startswith('/'):
                    # Extract base URL
                    match = re.match(r'(https?://[^/]+)', url)
                    if match:
                        src = match.group(1) + src
                if src.startswith('http'):
                    images.append(src)
        
        # Take unique images (max 5)
        images = list(dict.fromkeys(images))[:5]
        
        # Create summary (first 500 chars)
        summary = text_content[:500].strip()
        if len(text_content) > 500:
            # Try to end at a sentence
            last_period = summary.rfind('.')
            if last_period > 300:
                summary = summary[:last_period + 1]
            else:
                summary += "..."
        
        # Parse original HTML for metadata
        original_soup = BeautifulSoup(html, 'lxml')
        metadata = {}
        
        # Try to get author
        author_meta = original_soup.find('meta', attrs={'name': 'author'})
        if author_meta:
            metadata['author'] = author_meta.get('content', '')
        
        # Try to get description
        desc_meta = original_soup.find('meta', attrs={'name': 'description'})
        if desc_meta:
            metadata['description'] = desc_meta.get('content', '')
        
        # Try to get site name
        site_meta = original_soup.find('meta', attrs={'property': 'og:site_name'})
        if site_meta:
            metadata['site_name'] = site_meta.get('content', '')
        
        return ExtractedContent(
            url=url,
            title=title,
            text_content=text_content,
            summary=summary,
            images=images,
            metadata=metadata
        )
        
    except Exception as e:
        print(f"Failed to extract content from {url}: {e}")
        return None


def fetch_and_extract(url: str) -> Optional[ExtractedContent]:
    """
    Fetch a URL and extract clean content.
    
    Checks cache first, fetches and extracts if not cached.
    
    Args:
        url: Web page URL
        
    Returns:
        ExtractedContent or None if failed
    """
    # Check cache first
    cached = _get_from_cache(url)
    if cached:
        print(f"Cache hit for {url}")
        return cached
    
    # Fetch page
    html = fetch_page(url)
    if not html:
        return None
    
    # Extract content
    content = extract_content(html, url)
    if not content:
        return None
    
    # Cache and return
    _set_cache(url, content)
    print(f"Cached content for {url} ({len(content.text_content)} chars)")
    return content


# ============================================================================
# Search and Extract
# ============================================================================

def search_and_extract(
    query: str,
    max_results: int = 3,
    prioritize_sources: List[str] = None
) -> List[ExtractedContent]:
    """
    Search the web and extract content from results.
    
    Args:
        query: Search query
        max_results: Maximum number of extracted pages
        prioritize_sources: List of domains to prioritize (e.g., ["wikipedia.org"])
        
    Returns:
        List of ExtractedContent objects
    """
    if not DDG_AVAILABLE:
        print("Warning: duckduckgo-search not installed")
        return []
    
    if prioritize_sources is None:
        prioritize_sources = ["wikipedia.org", "britannica.com"]
    
    try:
        # Search with educational focus
        search_query = f"{query} {' OR '.join(prioritize_sources)}"
        
        with DDGS() as ddgs:
            results = list(ddgs.text(
                search_query,
                max_results=max_results + 2,
                safesearch="on"
            ))
        
        extracted = []
        for result in results:
            url = result.get("href", "")
            
            # Skip non-educational sources for cleaner results
            if not any(source in url for source in prioritize_sources + ["edu"]):
                continue
            
            # Skip dictionary/wiktionary
            if "wiktionary" in url:
                continue
            
            # Extract content
            content = fetch_and_extract(url)
            if content and len(content.text_content) > 200:
                extracted.append(content)
                
                if len(extracted) >= max_results:
                    break
            
            # Small delay between fetches
            time.sleep(0.3)
        
        return extracted
        
    except Exception as e:
        print(f"Search and extract error: {e}")
        return []


# ============================================================================
# Conversion to Dict (for API response)
# ============================================================================

def content_to_dict(content: ExtractedContent) -> Dict[str, Any]:
    """Convert ExtractedContent to dictionary for JSON response."""
    return {
        "url": content.url,
        "title": content.title,
        "text_content": content.text_content,
        "summary": content.summary,
        "images": content.images,
        "source_domain": content.source_domain,
        "metadata": content.metadata,
        "content_hash": content.content_hash
    }
