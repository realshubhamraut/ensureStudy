"""
YouTube Video Search API Service

Fetches educational videos using YouTube Data API v3.
Requires: YOUTUBE_API_KEY in environment

Endpoint: https://www.googleapis.com/youtube/v3/search
"""
import os
import logging
from typing import List, Dict, Any, Optional
import httpx
import re

logger = logging.getLogger(__name__)

# YouTube Data API v3 endpoints
YOUTUBE_SEARCH_API = "https://www.googleapis.com/youtube/v3/search"
YOUTUBE_VIDEOS_API = "https://www.googleapis.com/youtube/v3/videos"


def _format_duration(iso_duration: str) -> str:
    """Convert ISO 8601 duration (PT1H30M45S) to readable format (1:30:45)."""
    if not iso_duration:
        return ""
    
    # Parse ISO 8601 duration
    match = re.match(r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?', iso_duration)
    if not match:
        return ""
    
    hours = int(match.group(1) or 0)
    minutes = int(match.group(2) or 0)
    seconds = int(match.group(3) or 0)
    
    if hours > 0:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    else:
        return f"{minutes}:{seconds:02d}"


async def search_videos_youtube(
    query: str,
    max_results: int = 3,
    educational_filter: bool = True
) -> List[Dict[str, Any]]:
    """
    Search for videos using YouTube Data API v3.
    
    Args:
        query: Search query
        max_results: Maximum number of videos to return (default 3)
        educational_filter: If True, adds educational keywords to query
        
    Returns:
        List of video dicts with: id, title, url, thumbnailUrl, embedUrl, duration, source
    """
    api_key = os.getenv("YOUTUBE_API_KEY")
    
    if not api_key:
        logger.warning("[YOUTUBE] No YOUTUBE_API_KEY found in environment, skipping video search")
        return []
    
    try:
        # Add educational keywords for better results
        search_query = query
        if educational_filter:
            search_query = f"{query} tutorial explanation educational"
        
        # Step 1: Search for videos
        search_params = {
            "part": "snippet",
            "q": search_query,
            "type": "video",
            "maxResults": min(max_results * 2, 10),  # Get more, filter later
            "key": api_key,
            "relevanceLanguage": "en",
            "safeSearch": "strict",
            "videoEmbeddable": "true",  # Only embeddable videos
            "order": "relevance"
        }
        
        logger.info(f"[YOUTUBE] Searching videos for: {query[:50]}...")
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            # Search for videos
            search_response = await client.get(YOUTUBE_SEARCH_API, params=search_params)
            
            if search_response.status_code != 200:
                logger.error(f"[YOUTUBE] Search API returned {search_response.status_code}")
                return []
            
            search_data = search_response.json()
            items = search_data.get("items", [])
            
            if not items:
                logger.info("[YOUTUBE] No videos found")
                return []
            
            # Get video IDs for duration lookup
            video_ids = [item["id"]["videoId"] for item in items[:max_results]]
            
            # Step 2: Get video details (for duration)
            details_params = {
                "part": "contentDetails,statistics",
                "id": ",".join(video_ids),
                "key": api_key
            }
            
            details_response = await client.get(YOUTUBE_VIDEOS_API, params=details_params)
            
            durations = {}
            view_counts = {}
            if details_response.status_code == 200:
                details_data = details_response.json()
                for detail in details_data.get("items", []):
                    vid_id = detail["id"]
                    durations[vid_id] = _format_duration(
                        detail.get("contentDetails", {}).get("duration", "")
                    )
                    view_counts[vid_id] = int(
                        detail.get("statistics", {}).get("viewCount", 0)
                    )
            
            # Build video list
            videos = []
            for i, item in enumerate(items[:max_results]):
                video_id = item["id"]["videoId"]
                snippet = item.get("snippet", {})
                
                video = {
                    "id": f"yt_{video_id}",
                    "title": snippet.get("title", "Video"),
                    "url": f"https://www.youtube.com/watch?v={video_id}",
                    "thumbnailUrl": snippet.get("thumbnails", {}).get("high", {}).get("url", 
                                    snippet.get("thumbnails", {}).get("default", {}).get("url", "")),
                    "embedUrl": f"https://www.youtube.com/embed/{video_id}",
                    "duration": durations.get(video_id, ""),
                    "source": snippet.get("channelTitle", "YouTube"),
                    "relevance": 95 - (i * 5),  # Decreasing relevance
                    "viewCount": view_counts.get(video_id, 0)
                }
                videos.append(video)
            
            # Sort by view count (popular = more educational usually)
            videos.sort(key=lambda x: x.get("viewCount", 0), reverse=True)
            
            logger.info(f"[YOUTUBE] ✅ Found {len(videos)} videos")
            return videos[:max_results]
            
    except httpx.TimeoutException:
        logger.warning("[YOUTUBE] ⚠ Video search timed out")
        return []
    except Exception as e:
        logger.error(f"[YOUTUBE] Video search error: {e}")
        return []


def search_videos_youtube_sync(query: str, max_results: int = 3) -> List[Dict[str, Any]]:
    """Synchronous wrapper for search_videos_youtube."""
    import asyncio
    
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If already in async context
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    search_videos_youtube(query, max_results)
                )
                return future.result(timeout=15)
        else:
            return loop.run_until_complete(search_videos_youtube(query, max_results))
    except Exception as e:
        logger.error(f"[YOUTUBE] Sync wrapper error: {e}")
        return []
