"""
Meeting Q&A API Endpoints
Provides AI-powered Q&A for meeting transcripts
"""
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional

from app.services.meeting_rag import meeting_rag_service, MeetingQAResponse
from app.services.transcription_service import transcription_service
from app.utils.auth import get_current_user

router = APIRouter(prefix="/api/meeting", tags=["meeting-qa"])


class AskQuestionRequest(BaseModel):
    """Request for asking a question about meetings"""
    question: str
    classroom_id: str
    meeting_ids: Optional[List[str]] = None
    

class GetTranscriptRequest(BaseModel):
    """Request for getting transcript"""
    recording_id: str


@router.post("/ask", response_model=MeetingQAResponse)
async def ask_question(
    request: AskQuestionRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Ask a question about meeting content
    
    Uses RAG to search transcripts and generate an answer
    Returns answer with citations and timestamps for video seeking
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    if not request.classroom_id:
        raise HTTPException(status_code=400, detail="classroom_id is required")
    
    response = await meeting_rag_service.ask_question(
        question=request.question,
        classroom_id=request.classroom_id,
        meeting_ids=request.meeting_ids
    )
    
    return response


@router.get("/transcript/{recording_id}")
async def get_transcript(
    recording_id: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Get full transcript for a recording
    """
    transcript = await transcription_service.get_transcript(recording_id)
    
    if not transcript:
        raise HTTPException(status_code=404, detail="Transcript not found")
    
    return {
        "recording_id": transcript.recording_id,
        "meeting_id": transcript.meeting_id,
        "duration_seconds": transcript.duration_seconds,
        "language": transcript.language,
        "full_text": transcript.full_text,
        "summary": transcript.summary,
        "key_topics": transcript.key_topics,
        "word_count": transcript.word_count,
        "speakers": [
            {
                "speaker_id": s.speaker_id,
                "user_name": s.user_name,
                "speaking_time": s.total_speaking_time_seconds,
                "segment_count": s.segment_count
            }
            for s in transcript.speakers
        ],
        "segments": [
            {
                "id": seg.id,
                "start": seg.start,
                "end": seg.end,
                "speaker_id": seg.speaker_id,
                "speaker_name": seg.speaker_name,
                "text": seg.text
            }
            for seg in transcript.segments
        ]
    }


@router.get("/summary/{recording_id}")
async def get_meeting_summary(
    recording_id: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Get summary and key topics for a recording
    """
    summary = await meeting_rag_service.get_meeting_summary(recording_id)
    
    if not summary:
        raise HTTPException(status_code=404, detail="Summary not found")
    
    return summary


@router.post("/search")
async def search_transcripts(
    classroom_id: str,
    query: str,
    limit: int = 10,
    current_user: dict = Depends(get_current_user)
):
    """
    Search across meeting transcripts
    """
    results = await transcription_service.search_transcripts(
        classroom_id=classroom_id,
        query=query,
        limit=limit
    )
    
    return {"results": results}


class ProcessRecordingRequest(BaseModel):
    """Request to trigger recording processing"""
    recording_id: str
    meeting_id: str
    classroom_id: str
    video_path: str
    meeting_title: str = ""
    language: str = "en"


@router.post("/process")
async def process_recording(
    request: ProcessRecordingRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Trigger full processing pipeline for a recording:
    1. Transcribe with Whisper
    2. Store in MongoDB
    3. Generate Qdrant embeddings
    4. Update recording status
    """
    from app.services.recording_pipeline import trigger_processing
    
    result = await trigger_processing(
        recording_id=request.recording_id,
        meeting_id=request.meeting_id,
        classroom_id=request.classroom_id,
        video_path=request.video_path,
        meeting_title=request.meeting_title,
        language=request.language
    )
    
    return result
