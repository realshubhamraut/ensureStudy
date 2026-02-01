"""
TTS API Routes
Provides text-to-speech endpoints using AWS Polly with viseme data.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import logging

from app.services.polly_service import get_polly_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/tts", tags=["Text-to-Speech"])


class TTSSynthesizeRequest(BaseModel):
    """Request body for TTS synthesis."""
    text: str = Field(..., min_length=1, max_length=3000, description="Text to synthesize (max 3000 chars)")
    voice: str = Field(default="female", description="Voice type: 'male' or 'female'")


class VisemeData(BaseModel):
    """Viseme timing data for lip sync."""
    time: int = Field(..., description="Time offset in milliseconds")
    value: str = Field(..., description="Oculus viseme ID")


class TTSSynthesizeResponse(BaseModel):
    """Response with audio and viseme data."""
    audio_base64: str = Field(..., description="Base64 encoded MP3 audio")
    visemes: List[VisemeData] = Field(default=[], description="Viseme timing data for lip sync")
    voice: str = Field(..., description="Voice ID used")
    duration_ms: int = Field(default=0, description="Audio duration in milliseconds")


class TTSStatusResponse(BaseModel):
    """TTS service status."""
    available: bool
    provider: str = "aws_polly"
    voices: Dict[str, str]


@router.get("/status", response_model=TTSStatusResponse)
async def get_tts_status():
    """Check if TTS service is available and configured."""
    polly = get_polly_service()
    return TTSStatusResponse(
        available=polly.is_available(),
        provider="aws_polly",
        voices={
            "male": "Matthew (Neural)",
            "female": "Joanna (Neural)"
        }
    )


@router.post("/synthesize", response_model=TTSSynthesizeResponse)
async def synthesize_speech(request: TTSSynthesizeRequest):
    """
    Synthesize speech with viseme data for avatar lip sync.
    
    AWS Polly provides high-quality neural voices with accurate
    viseme timing for realistic lip synchronization.
    
    Returns:
        - audio_base64: MP3 audio as base64 string
        - visemes: List of {time, value} for lip sync
        - duration_ms: Total audio duration
    """
    polly = get_polly_service()
    
    if not polly.is_available():
        raise HTTPException(
            status_code=503,
            detail="TTS service not available. AWS credentials may not be configured."
        )
    
    try:
        # Validate voice type
        voice_type = request.voice.lower()
        if voice_type not in ['male', 'female']:
            voice_type = 'female'
        
        # Synthesize with Polly
        result = await polly.synthesize(
            text=request.text,
            voice_type=voice_type
        )
        
        return TTSSynthesizeResponse(
            audio_base64=result['audio_base64'],
            visemes=[VisemeData(**v) for v in result['visemes']],
            voice=result['voice'],
            duration_ms=result['duration_ms']
        )
        
    except RuntimeError as e:
        logger.error(f"TTS synthesis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected TTS error: {e}")
        raise HTTPException(status_code=500, detail="TTS synthesis failed")
