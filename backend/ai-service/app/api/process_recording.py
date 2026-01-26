"""
Recording Processing API
Handles transcription and embedding generation for meeting recordings
Supports both local filesystem and S3 storage
"""
import os
import asyncio
import tempfile
import httpx
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional

from app.services.transcription_service import transcription_service
from app.services.meeting_embedding_service import meeting_embedding_service

router = APIRouter(prefix="/api/process", tags=["recording-processing"])

# Core service URL for status updates
CORE_SERVICE_URL = os.getenv('CORE_SERVICE_URL', 'https://localhost:8000')
RECORDINGS_DIR = os.getenv('RECORDINGS_DIR', '/app/recordings')

# Storage configuration
STORAGE_PROVIDER = os.getenv('STORAGE_PROVIDER', 'local')  # 'local' or 's3'
AWS_S3_BUCKET = os.getenv('AWS_S3_BUCKET', 'ensurestudy-files')
AWS_REGION = os.getenv('AWS_REGION', 'ap-south-1')

# Lazy-loaded S3 client
_s3_client = None


def get_s3_client():
    """Get or create S3 client"""
    global _s3_client
    if _s3_client is None:
        try:
            import boto3
            _s3_client = boto3.client('s3', region_name=AWS_REGION)
        except ImportError:
            print("[Process] boto3 not installed, S3 unavailable")
            return None
    return _s3_client


async def download_from_s3(s3_key: str) -> str:
    """Download file from S3 to temp location, returns local path"""
    s3 = get_s3_client()
    if not s3:
        raise RuntimeError("S3 client not available")
    
    # Get file extension from key
    ext = os.path.splitext(s3_key)[1] or '.webm'
    temp_file = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
    
    print(f"[Process] Downloading from S3: {s3_key}")
    
    def do_download():
        s3.download_file(AWS_S3_BUCKET, s3_key, temp_file.name)
    
    await asyncio.to_thread(do_download)
    print(f"[Process] Downloaded to: {temp_file.name}")
    return temp_file.name


class ProcessRecordingRequest(BaseModel):
    recording_id: str
    meeting_id: str
    classroom_id: str
    video_path: Optional[str] = None  # If not provided, will be inferred


class ProcessingStatus(BaseModel):
    recording_id: str
    status: str
    message: str
    transcript_available: bool = False


async def update_recording_status(
    recording_id: str, 
    status: str, 
    has_transcript: bool = False,
    transcript_text: str = None,
    summary: str = None
):
    """Update recording status and transcript in core service"""
    try:
        data = {
            "status": status,
            "has_transcript": has_transcript
        }
        if transcript_text:
            data["transcript_text"] = transcript_text
        if summary:
            data["summary"] = summary
            
        async with httpx.AsyncClient(verify=False) as client:  # Allow self-signed certs
            await client.patch(
                f"{CORE_SERVICE_URL}/api/recordings/{recording_id}/status",
                json=data,
                timeout=10.0
            )
    except Exception as e:
        print(f"Failed to update recording status: {e}")


async def process_recording_task(
    recording_id: str,
    meeting_id: str,
    classroom_id: str,
    video_path: str
):
    """
    Background task to process a recording:
    1. Transcribe with Whisper
    2. Store in MongoDB
    3. Generate embeddings for Qdrant
    4. Update status to 'ready'
    """
    try:
        print(f"Starting transcription for recording {recording_id}")
        
        # Step 1: Transcribe
        transcript = await transcription_service.transcribe_meeting(
            recording_id=recording_id,
            meeting_id=meeting_id,
            classroom_id=classroom_id,
            video_path=video_path
        )
        
        print(f"Transcription complete: {transcript.word_count} words")
        
        # Step 2: Generate embeddings for vector search (AI tutor retrieval)
        try:
            chunks_embedded = await meeting_embedding_service.embed_transcript(
                recording_id=recording_id,
                meeting_id=meeting_id,
                classroom_id=classroom_id,
                segments=[{
                    'id': seg.id,
                    'start': seg.start,
                    'end': seg.end,
                    'text': seg.text,
                    'speaker_id': seg.speaker_id,
                    'speaker_name': seg.speaker_name or f'Speaker {seg.speaker_id + 1}'
                } for seg in transcript.segments],
                meeting_title=f"Meeting {meeting_id[:8]}"  # Short title for context
            )
            print(f"Embedded {chunks_embedded} chunks to Qdrant")
        except Exception as e:
            print(f"Embedding failed (non-fatal): {e}")
        
        # Step 3: Update status to ready with transcript text
        await update_recording_status(
            recording_id, 
            "ready", 
            has_transcript=True,
            transcript_text=transcript.full_text,
            summary=transcript.summary if hasattr(transcript, 'summary') else None
        )
        
        print(f"Recording {recording_id} processing complete!")
        
    except Exception as e:
        print(f"Recording processing failed: {e}")
        await update_recording_status(recording_id, "failed", has_transcript=False)


@router.post("/recording", response_model=ProcessingStatus)
async def process_recording(
    request: ProcessRecordingRequest,
    background_tasks: BackgroundTasks
):
    """
    Start processing a recording in the background
    Called by core-service after recording is finalized
    
    Supports both local filesystem and S3 storage:
    - STORAGE_PROVIDER=local: Uses local filesystem path
    - STORAGE_PROVIDER=s3: Downloads from S3 to temp file first
    """
    video_path = request.video_path
    s3_key = None  # Track if we need to download from S3
    
    if STORAGE_PROVIDER == 's3':
        # For S3, the video_path is actually an S3 key
        if video_path:
            # If path is provided, use it as S3 key
            s3_key = video_path if not video_path.startswith('/') else f"recordings/{request.recording_id}.webm"
        else:
            # Default S3 key
            s3_key = f"recordings/{request.recording_id}.webm"
        
        # Try to download from S3
        try:
            video_path = await download_from_s3(s3_key)
            print(f"[Process] Downloaded S3 file to: {video_path}")
        except Exception as e:
            raise HTTPException(
                status_code=404,
                detail=f"Failed to download from S3: {s3_key} - {e}"
            )
    else:
        # Local storage
        if not video_path:
            # Default path based on recording ID
            video_path = os.path.join(RECORDINGS_DIR, f"{request.recording_id}.webm")
            
            # Check for MP4 version
            mp4_path = os.path.join(RECORDINGS_DIR, f"{request.recording_id}.mp4")
            if os.path.exists(mp4_path):
                video_path = mp4_path
        
        if not os.path.exists(video_path):
            raise HTTPException(
                status_code=404,
                detail=f"Video file not found: {video_path}"
            )
    
    # Start background processing
    background_tasks.add_task(
        process_recording_task,
        request.recording_id,
        request.meeting_id,
        request.classroom_id,
        video_path
    )
    
    return ProcessingStatus(
        recording_id=request.recording_id,
        status="processing",
        message="Transcription started in background",
        transcript_available=False
    )


@router.get("/recording/{recording_id}/status", response_model=ProcessingStatus)
async def get_processing_status(recording_id: str):
    """Check processing status for a recording"""
    # Check if transcript exists in MongoDB
    transcript = await transcription_service.get_transcript(recording_id)
    
    if transcript:
        return ProcessingStatus(
            recording_id=recording_id,
            status="complete",
            message="Transcription available",
            transcript_available=True
        )
    
    return ProcessingStatus(
        recording_id=recording_id,
        status="pending",
        message="Not yet processed",
        transcript_available=False
    )
