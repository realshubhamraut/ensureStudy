"""
Speech-to-Text API using local Whisper model.
This provides offline STT capability when browser Web Speech API fails.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
import tempfile
import os
import asyncio

router = APIRouter(prefix="/api/stt", tags=["STT"])


class TranscriptionResponse(BaseModel):
    text: str
    language: str = "en"
    duration_seconds: float = 0.0
    confidence: float = 1.0


class STTStatusResponse(BaseModel):
    available: bool
    provider: str = "whisper_local"
    model: str = "base"


# Cache the whisper model to avoid reloading
_whisper_model = None
_model_loading = False


async def get_whisper_model():
    """Get or load the whisper model (cached singleton)."""
    global _whisper_model, _model_loading
    
    if _whisper_model is not None:
        return _whisper_model
    
    if _model_loading:
        # Wait for model to finish loading
        for _ in range(30):  # Wait up to 30 seconds
            await asyncio.sleep(1)
            if _whisper_model is not None:
                return _whisper_model
        raise RuntimeError("Model loading timeout")
    
    _model_loading = True
    try:
        import whisper
        # Use 'base' model for faster real-time transcription (74MB)
        # Options: tiny (39MB), base (74MB), small (244MB), medium (769MB)
        model_name = os.getenv('WHISPER_STT_MODEL', 'base')
        print(f"[STT] Loading Whisper model: {model_name}")
        
        def load_model():
            return whisper.load_model(model_name)
        
        _whisper_model = await asyncio.to_thread(load_model)
        print(f"[STT] Whisper model loaded successfully")
        return _whisper_model
    except ImportError:
        print("[STT] Whisper not installed. Install with: pip install openai-whisper")
        raise HTTPException(503, "Whisper not installed")
    finally:
        _model_loading = False


@router.get("/status")
async def get_stt_status() -> STTStatusResponse:
    """Check if local STT service is available."""
    try:
        import whisper
        model_name = os.getenv('WHISPER_STT_MODEL', 'base')
        return STTStatusResponse(
            available=True,
            provider="whisper_local",
            model=model_name
        )
    except ImportError:
        return STTStatusResponse(
            available=False,
            provider="whisper_local",
            model="not_installed"
        )


@router.post("/transcribe")
async def transcribe_audio(
    audio: UploadFile = File(...),
    language: str = "en"
) -> TranscriptionResponse:
    """
    Transcribe audio using local Whisper model.
    
    Accepts audio file (WAV, WebM, MP3, etc.)
    Returns transcribed text.
    """
    # Get whisper model
    model = await get_whisper_model()
    
    # Save uploaded audio to temp file
    suffix = os.path.splitext(audio.filename)[1] if audio.filename else ".webm"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await audio.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        # Transcribe with Whisper
        def do_transcribe():
            result = model.transcribe(
                tmp_path,
                language=language,
                temperature=0.0,
                fp16=False  # Use FP32 for better compatibility
            )
            return result
        
        print(f"[STT] Transcribing {len(content)} bytes of audio...")
        result = await asyncio.to_thread(do_transcribe)
        
        text = result.get('text', '').strip()
        duration = result.get('duration', 0) if 'duration' in result else 0
        
        print(f"[STT] Transcribed: '{text[:100]}...' ({duration:.1f}s)")
        
        return TranscriptionResponse(
            text=text,
            language=result.get('language', language),
            duration_seconds=duration,
            confidence=1.0
        )
    finally:
        # Clean up temp file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
