"""
Image Search Service (DuckDuckGo)

Fetches educational images using DuckDuckGo Image Search.
NO API KEY REQUIRED - Free and unlimited!

Uses the ddgs library (duckduckgo-search) which is already installed.
"""
import logging
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


def search_images_duckduckgo(
    query: str, 
    count: int = 3,
    safe_search: str = "on"
) -> List[Dict[str, Any]]:
    """
    Search for images using DuckDuckGo Image Search (FREE, no API key!).
    
    Args:
        query: Search query
        count: Number of images to return (default 3)
        safe_search: "on", "moderate", or "off" (default "on" for educational)
        
    Returns:
        List of image dicts with: id, title, url, thumbnailUrl, source, width, height
    """
    try:
        from ddgs import DDGS
        
        print(f"[IMAGES] 🔍 Searching DuckDuckGo images for: {query[:50]}...")
        
        with DDGS() as ddgs:
            # Search for educational images
            results = list(ddgs.images(
                f"{query} educational diagram",
                max_results=min(count + 2, 8),  # Get extra for filtering
                safesearch=safe_search
            ))
        
        if not results:
            print("[IMAGES] ⚠ No images found")
            return []
        
        images = []
        for i, result in enumerate(results[:count]):
            # Extract full-size URL (not thumbnail!)
            full_url = result.get("image", "")
            thumb_url = result.get("thumbnail", "")
            
            # Log URLs for debugging
            print(f"[IMAGES] Image {i}: full={full_url[:80]}... thumb={thumb_url[:60] if thumb_url else 'N/A'}...")
            
            # Prefer high-resolution images
            width = result.get("width") or 0
            height = result.get("height") or 0
            
            image = {
                "id": f"ddg_img_{i}",
                "title": result.get("title", "Image"),
                "url": full_url,  # ALWAYS use full-size image URL
                "thumbnailUrl": thumb_url or full_url,  # Fallback to full URL for thumbnail
                "source": result.get("source", "Web"),
                "width": width,
                "height": height,
                "relevance": 95 - (i * 5)
            }
            
            # Only add if we have a valid URL
            if image["url"]:
                images.append(image)
        
        print(f"[IMAGES] ✅ Found {len(images)} images (using full-size URLs)")
        return images
        
    except Exception as e:
        logger.error(f"[IMAGES] DuckDuckGo image search error: {e}")
        print(f"[IMAGES] ❌ Error: {e}")
        return []


async def search_images_brave(
    query: str, 
    count: int = 3,
    safe_search: str = "strict"
) -> List[Dict[str, Any]]:
    """
    Async wrapper - uses DuckDuckGo (free, no API key needed!).
    Kept the function name for backwards compatibility with tutor.py imports.
    
    Args:
        query: Search query
        count: Number of images to return
        safe_search: Safety level
        
    Returns:
        List of image dicts
    """
    # Run sync function in thread pool to not block async loop
    try:
        loop = __import__('asyncio').get_event_loop()
        with ThreadPoolExecutor() as executor:
            result = await loop.run_in_executor(
                executor, 
                lambda: search_images_duckduckgo(query, count, "on" if safe_search == "strict" else safe_search)
            )
            return result
    except Exception as e:
        logger.error(f"[IMAGES] Async wrapper error: {e}")
        return search_images_duckduckgo(query, count)


def search_images_brave_sync(query: str, count: int = 3) -> List[Dict[str, Any]]:
    """Synchronous wrapper - directly calls DuckDuckGo."""
    return search_images_duckduckgo(query, count)
