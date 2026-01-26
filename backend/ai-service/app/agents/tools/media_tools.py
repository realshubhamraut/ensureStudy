"""
Media Processing Tools - LangGraph Agent Tools

Tools for processing media: OCR, transcription, video analysis.
Wraps existing services: ocr_service.py, transcription_service.py, video_processor.py
"""
from typing import List, Dict, Any, Optional
import logging

from .base_tool import AgentTool, ToolParameter, get_tool_registry

logger = logging.getLogger(__name__)


# ============================================================================
# OCR Text Extraction Tool
# ============================================================================

async def _ocr_extract(
    image_path: str,
    language: str = "en"
) -> Dict[str, Any]:
    """Extract text from an image using OCR"""
    try:
        from app.services.ocr_service import extract_text_from_image
        
        result = await extract_text_from_image(image_path, language)
        
        return {
            "success": True,
            "text": result.get("text", ""),
            "confidence": result.get("confidence", 0.0),
            "word_count": len(result.get("text", "").split())
        }
    except Exception as e:
        logger.error(f"[OCR] Error: {e}")
        return {
            "success": False,
            "text": "",
            "error": str(e)
        }


ocr_extract_tool = AgentTool(
    name="ocr_extract",
    description="Extract text from an image using OCR (Optical Character Recognition). Supports handwritten and printed text.",
    func=_ocr_extract,
    parameters=[
        ToolParameter(
            name="image_path",
            type="string",
            description="Path to the image file",
            required=True
        ),
        ToolParameter(
            name="language",
            type="string",
            description="Language code (en, hi, etc.)",
            required=False,
            default="en"
        )
    ],
    category="media"
)


# ============================================================================
# Transcribe Audio Tool
# ============================================================================

async def _transcribe_audio(
    audio_path: str,
    language: str = "en"
) -> Dict[str, Any]:
    """Transcribe audio to text using Whisper"""
    try:
        from app.services.transcription_service import transcribe_audio
        
        result = await transcribe_audio(audio_path, language)
        
        return {
            "success": True,
            "transcript": result.get("text", ""),
            "segments": result.get("segments", []),
            "duration_seconds": result.get("duration", 0)
        }
    except Exception as e:
        logger.error(f"[TRANSCRIBE] Error: {e}")
        return {
            "success": False,
            "transcript": "",
            "error": str(e)
        }


transcribe_audio_tool = AgentTool(
    name="transcribe_audio",
    description="Transcribe audio file to text using Whisper. Returns full transcript and segments with timestamps.",
    func=_transcribe_audio,
    parameters=[
        ToolParameter(
            name="audio_path",
            type="string",
            description="Path to audio file (mp3, wav, m4a)",
            required=True
        ),
        ToolParameter(
            name="language",
            type="string",
            description="Language code",
            required=False,
            default="en"
        )
    ],
    category="media"
)


# ============================================================================
# Extract PDF Text Tool
# ============================================================================

async def _extract_pdf_text(
    pdf_path: str,
    use_ocr: bool = False
) -> Dict[str, Any]:
    """Extract text from a PDF file"""
    try:
        from app.services.pdf_processor import PDFProcessor
        
        processor = PDFProcessor()
        result = await processor.extract_text(pdf_path, use_ocr=use_ocr)
        
        return {
            "success": True,
            "text": result.text,
            "pages": result.page_count,
            "word_count": len(result.text.split()),
            "used_ocr": result.used_ocr
        }
    except Exception as e:
        logger.error(f"[PDF-EXTRACT] Error: {e}")
        return {
            "success": False,
            "text": "",
            "error": str(e)
        }


extract_pdf_text_tool = AgentTool(
    name="extract_pdf_text",
    description="Extract text content from a PDF file. Can use OCR for scanned PDFs.",
    func=_extract_pdf_text,
    parameters=[
        ToolParameter(
            name="pdf_path",
            type="string",
            description="Path to the PDF file",
            required=True
        ),
        ToolParameter(
            name="use_ocr",
            type="boolean",
            description="Use OCR for scanned PDFs",
            required=False,
            default=False
        )
    ],
    category="media"
)


# ============================================================================
# Analyze Video Tool
# ============================================================================

async def _analyze_video(
    video_path: str,
    extract_frames: bool = True,
    extract_audio: bool = True
) -> Dict[str, Any]:
    """Analyze a video file - extract frames and/or audio"""
    try:
        from app.services.video_processor import analyze_video
        
        result = await analyze_video(
            video_path,
            extract_frames=extract_frames,
            extract_audio=extract_audio
        )
        
        return {
            "success": True,
            "duration_seconds": result.get("duration", 0),
            "frame_count": result.get("frame_count", 0),
            "audio_path": result.get("audio_path"),
            "frames": result.get("frames", [])[:5]  # Limit frames in response
        }
    except Exception as e:
        logger.error(f"[VIDEO-ANALYZE] Error: {e}")
        return {
            "success": False,
            "error": str(e)
        }


analyze_video_tool = AgentTool(
    name="analyze_video",
    description="Analyze a video file. Extract key frames and/or audio track.",
    func=_analyze_video,
    parameters=[
        ToolParameter(
            name="video_path",
            type="string",
            description="Path to video file",
            required=True
        ),
        ToolParameter(
            name="extract_frames",
            type="boolean",
            description="Extract key frames",
            required=False,
            default=True
        ),
        ToolParameter(
            name="extract_audio",
            type="boolean",
            description="Extract audio track",
            required=False,
            default=True
        )
    ],
    category="media"
)


# ============================================================================
# Register All Tools
# ============================================================================

def register_media_tools():
    """Register all media tools with the global registry"""
    registry = get_tool_registry()
    
    registry.register(ocr_extract_tool)
    registry.register(transcribe_audio_tool)
    registry.register(extract_pdf_text_tool)
    registry.register(analyze_video_tool)
    
    logger.info(f"[MEDIA-TOOLS] Registered {len(registry.list_tools('media'))} media tools")


# Auto-register on import
register_media_tools()
