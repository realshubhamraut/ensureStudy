"""
Recording Processing API
Handles transcription and embedding generation for meeting recordings
"""
import os
import asyncio
import httpx
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional

from app.services.transcription_service import transcription_service
from app.services.meeting_embedding_service import meeting_embedding_service

router = APIRouter(prefix="/api/process", tags=["recording-processing"])

# Core service URL for status updates
CORE_SERVICE_URL = os.getenv('CORE_SERVICE_URL', 'http://localhost:9000')
RECORDINGS_DIR = os.getenv('RECORDINGS_DIR', '/Users/proxim/projects/ensureStudy/backend/core-service/recordings')


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


async def update_recording_status(recording_id: str, status: str, has_transcript: bool = False):
    """Update recording status in core service"""
    try:
        async with httpx.AsyncClient() as client:
            await client.patch(
                f"{CORE_SERVICE_URL}/api/recordings/{recording_id}/status",
                json={
                    "status": status,
                    "has_transcript": has_transcript
                },
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
        
        # Step 2: Generate embeddings for vector search
        try:
            chunks_embedded = meeting_embedding_service.embed_transcript(
                recording_id=recording_id,
                meeting_id=meeting_id,
                classroom_id=classroom_id,
                transcript_text=transcript.full_text,
                segments=[{
                    'id': seg.id,
                    'start': seg.start,
                    'end': seg.end,
                    'text': seg.text,
                    'speaker_id': seg.speaker_id
                } for seg in transcript.segments]
            )
            print(f"Embedded {chunks_embedded} chunks to Qdrant")
        except Exception as e:
            print(f"Embedding failed (non-fatal): {e}")
        
        # Step 3: Update status to ready
        await update_recording_status(recording_id, "ready", has_transcript=True)
        
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
    """
    # Determine video path
    video_path = request.video_path
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
