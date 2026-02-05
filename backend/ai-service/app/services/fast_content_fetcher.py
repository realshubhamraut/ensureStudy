"""
Fast Content Fetcher

Parallel URL fetcher with aggressive timeouts for the AI Tutor.
Fetches actual content from web pages to provide richer LLM context.
"""

import asyncio
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
import re

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


@dataclass
class FetchedContent:
    """Result of fetching a URL"""
    url: str
    title: str
    content: str
    success: bool
    error: Optional[str] = None


def extract_main_content(html: str, max_chars: int = 2500) -> str:
    """
    Extract main text content from HTML, removing boilerplate.
    """
    try:
        soup = BeautifulSoup(html, 'html.parser')
        
        # Remove script, style, nav, header, footer, aside elements
        for tag in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 
                         'form', 'button', 'input', 'iframe', 'noscript']):
            tag.decompose()
        
        # Try to find main content area
        main_content = None
        for selector in ['article', 'main', '[role="main"]', '.content', '#content', 
                         '.post-content', '.entry-content', '.article-body']:
            main_content = soup.select_one(selector)
            if main_content:
                break
        
        if not main_content:
            main_content = soup.body if soup.body else soup
        
        # Get text
        text = main_content.get_text(separator=' ', strip=True)
        
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Truncate
        if len(text) > max_chars:
            # Try to cut at sentence boundary
            cut_point = text.rfind('.', 0, max_chars)
            if cut_point > max_chars * 0.7:
                text = text[:cut_point + 1]
            else:
                text = text[:max_chars] + '...'
        
        return text
        
    except Exception as e:
        logger.debug(f"[FAST-FETCH] Content extraction error: {e}")
        return ""


async def fetch_url_fast(url: str, timeout: float = 3.0, max_chars: int = 2500) -> FetchedContent:
    """
    Fetch a single URL with timeout.
    """
    try:
        async with httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=True,
            verify=False  # Skip SSL verification for speed
        ) as client:
            response = await client.get(url, headers={
                'User-Agent': 'Mozilla/5.0 (compatible; EnsureStudyBot/1.0)'
            })
            
            if response.status_code == 200:
                content = extract_main_content(response.text, max_chars)
                
                # Extract title
                soup = BeautifulSoup(response.text, 'html.parser')
                title = soup.title.string if soup.title else url
                
                return FetchedContent(
                    url=url,
                    title=str(title)[:100] if title else url,
                    content=content,
                    success=True
                )
            else:
                return FetchedContent(
                    url=url,
                    title="",
                    content="",
                    success=False,
                    error=f"HTTP {response.status_code}"
                )
                
    except asyncio.TimeoutError:
        return FetchedContent(url=url, title="", content="", success=False, error="Timeout")
    except Exception as e:
        return FetchedContent(url=url, title="", content="", success=False, error=str(e))


async def fetch_articles_fast(
    urls: List[str], 
    timeout_per_url: float = 3.0,
    max_chars_per_article: int = 2500,
    max_total_chars: int = 8000
) -> Dict[str, FetchedContent]:
    """
    Parallel fetch multiple URLs with timeout.
    
    Args:
        urls: List of URLs to fetch
        timeout_per_url: Timeout per individual URL
        max_chars_per_article: Max chars to extract per article
        max_total_chars: Total chars limit across all articles
        
    Returns:
        Dict mapping URL to FetchedContent
    """
    if not urls:
        return {}
    
    # Limit to 5 URLs max
    urls = urls[:5]
    
    logger.info(f"[FAST-FETCH] Fetching {len(urls)} URLs in parallel...")
    
    # Fetch all in parallel
    tasks = [
        fetch_url_fast(url, timeout_per_url, max_chars_per_article)
        for url in urls
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Build result dict, respecting total char limit
    output = {}
    total_chars = 0
    success_count = 0
    
    for result in results:
        if isinstance(result, Exception):
            continue
            
        if result.success and result.content:
            # Check if we have room for this content
            if total_chars + len(result.content) > max_total_chars:
                # Truncate to fit
                remaining = max_total_chars - total_chars
                if remaining > 500:
                    result.content = result.content[:remaining] + '...'
                else:
                    continue  # Skip if too little room
            
            output[result.url] = result
            total_chars += len(result.content)
            success_count += 1
    
    logger.info(f"[FAST-FETCH] ✅ Fetched {success_count}/{len(urls)} articles ({total_chars} chars)")
    
    return output
