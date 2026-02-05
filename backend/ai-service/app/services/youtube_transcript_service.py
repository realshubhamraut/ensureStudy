"""
YouTube Transcript Service

Fetches transcripts from YouTube videos for LLM context.
Uses youtube-transcript-api for text extraction.
"""

import asyncio
import logging
from typing import Optional
import re

logger = logging.getLogger(__name__)


async def get_youtube_transcript(video_id: str, max_chars: int = 2000) -> Optional[str]:
    """
    Fetch YouTube video transcript for LLM context.
    
    Args:
        video_id: YouTube video ID (e.g. 'dQw4w9WgXcQ')
        max_chars: Maximum characters to return
        
    Returns:
        Transcript text or None if unavailable
    """
    try:
        # Run in thread pool since youtube_transcript_api is synchronous
        from youtube_transcript_api import YouTubeTranscriptApi
        
        def fetch_transcript():
            try:
                # Try to get English transcript first
                transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
                
                # Prefer manual transcripts over auto-generated
                transcript = None
                try:
                    transcript = transcript_list.find_manually_created_transcript(['en'])
                except:
                    try:
                        transcript = transcript_list.find_generated_transcript(['en'])
                    except:
                        # Try any available transcript
                        for t in transcript_list:
                            transcript = t
                            break
                
                if transcript:
                    # Fetch the actual transcript data
                    transcript_data = transcript.fetch()
                    
                    # Combine text segments
                    full_text = " ".join([
                        segment.get('text', '').strip() 
                        for segment in transcript_data
                    ])
                    
                    # Clean up the text
                    full_text = re.sub(r'\s+', ' ', full_text).strip()
                    
                    return full_text[:max_chars] if len(full_text) > max_chars else full_text
                    
            except Exception as e:
                logger.debug(f"[TRANSCRIPT] Could not fetch transcript for {video_id}: {e}")
                return None
        
        # Run in thread pool with timeout
        loop = asyncio.get_event_loop()
        result = await asyncio.wait_for(
            loop.run_in_executor(None, fetch_transcript),
            timeout=5.0  # 5 second timeout
        )
        
        if result:
            logger.info(f"[TRANSCRIPT] ✅ Fetched {len(result)} chars for video {video_id}")
        
        return result
        
    except asyncio.TimeoutError:
        logger.warning(f"[TRANSCRIPT] Timeout fetching transcript for {video_id}")
        return None
    except Exception as e:
        logger.debug(f"[TRANSCRIPT] Error: {e}")
        return None


def extract_video_id(url: str) -> Optional[str]:
    """
    Extract video ID from YouTube URL.
    
    Supports:
    - youtube.com/watch?v=VIDEO_ID
    - youtu.be/VIDEO_ID
    - youtube.com/embed/VIDEO_ID
    """
    patterns = [
        r'(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([a-zA-Z0-9_-]{11})',
        r'^([a-zA-Z0-9_-]{11})$'  # Just the ID
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    
    return None
