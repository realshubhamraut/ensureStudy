"""
Recording Processing Pipeline
Handles the complete flow: transcription → embedding → database updates
Can be triggered via API or Kafka consumer
"""
import os
import asyncio
from datetime import datetime
from typing import Optional, Dict, Any
import httpx

from app.services.transcription_service import transcription_service, MeetingTranscript
from app.services.meeting_embedding_service import meeting_embedding_service

# Core API URL for updating recording status
CORE_API_URL = os.getenv('CORE_API_URL', 'http://localhost:8000')


class RecordingProcessingPipeline:
    """
    Complete pipeline for processing meeting recordings:
    1. Download/locate video file
    2. Run transcription with Whisper
    3. Store transcript in MongoDB
    4. Generate embeddings and store in Qdrant
    5. Update recording status in PostgreSQL
    """
    
    def __init__(self):
        self.transcription = transcription_service
        self.embedding = meeting_embedding_service
    
    async def process_recording(
        self,
        recording_id: str,
        meeting_id: str,
        classroom_id: str,
        video_path: str,
        meeting_title: str = "",
        participant_names: Optional[Dict[int, str]] = None,
        language: str = 'en'
    ) -> Dict[str, Any]:
        """
        Full processing pipeline for a recording
        
        Returns:
            Dict with processing results and statistics
        """
        results = {
            'recording_id': recording_id,
            'success': False,
            'transcript_stored': False,
            'embeddings_count': 0,
            'error': None
        }
        
        try:
            # Step 1: Update status to 'transcribing'
            await self._update_recording_status(recording_id, 'transcribing', 10)
            
            # Step 2: Run transcription
            transcript = await self.transcription.transcribe_meeting(
                recording_id=recording_id,
                meeting_id=meeting_id,
                classroom_id=classroom_id,
                video_path=video_path,
                participant_names=participant_names,
                language=language
            )
            
            results['transcript_stored'] = True
            results['word_count'] = transcript.word_count
            results['speaker_count'] = len(transcript.speakers)
            
            # Step 3: Update status to 'embedding'
            await self._update_recording_status(recording_id, 'embedding', 60)
            
            # Step 4: Generate and store embeddings in Qdrant
            segments_data = [
                {
                    'id': seg.id,
                    'start': seg.start,
                    'end': seg.end,
                    'speaker_id': seg.speaker_id,
                    'speaker_name': seg.speaker_name,
                    'text': seg.text
                }
                for seg in transcript.segments
            ]
            
            embeddings_count = await self.embedding.embed_transcript(
                recording_id=recording_id,
                meeting_id=meeting_id,
                classroom_id=classroom_id,
                segments=segments_data,
                meeting_title=meeting_title
            )
            
            results['embeddings_count'] = embeddings_count
            
            # Step 5: Update recording to 'ready' with all metadata
            await self._update_recording_complete(
                recording_id=recording_id,
                transcript=transcript,
                embeddings_count=embeddings_count
            )
            
            results['success'] = True
            
        except Exception as e:
            results['error'] = str(e)
            await self._update_recording_status(recording_id, 'failed', 0, str(e))
        
        return results
    
    async def _update_recording_status(
        self,
        recording_id: str,
        status: str,
        progress: int,
        error: Optional[str] = None
    ):
        """Update recording status in Core API"""
        try:
            async with httpx.AsyncClient() as client:
                await client.patch(
                    f"{CORE_API_URL}/api/recordings/{recording_id}/status",
                    json={
                        'status': status,
                        'processing_progress': progress,
                        'error_message': error
                    },
                    timeout=10.0
                )
        except Exception as e:
            print(f"Failed to update recording status: {e}")
    
    async def _update_recording_complete(
        self,
        recording_id: str,
        transcript: MeetingTranscript,
        embeddings_count: int
    ):
        """Update recording with final metadata"""
        try:
            async with httpx.AsyncClient() as client:
                await client.patch(
                    f"{CORE_API_URL}/api/recordings/{recording_id}/status",
                    json={
                        'status': 'ready',
                        'processing_progress': 100,
                        'has_transcript': True,
                        'speaker_count': len(transcript.speakers),
                        'word_count': transcript.word_count,
                        'language': transcript.language,
                        'key_topics': transcript.key_topics,
                        'summary_brief': transcript.summary[:500] if transcript.summary else None,
                        'is_indexed': True,
                        'indexed_at': datetime.utcnow().isoformat()
                    },
                    timeout=10.0
                )
        except Exception as e:
            print(f"Failed to update recording complete: {e}")


# Singleton
recording_pipeline = RecordingProcessingPipeline()


# API endpoint function for triggering processing
async def trigger_processing(
    recording_id: str,
    meeting_id: str,
    classroom_id: str,
    video_path: str,
    meeting_title: str = "",
    language: str = 'en'
) -> Dict[str, Any]:
    """
    Trigger the full processing pipeline for a recording
    This can be called from an API endpoint or Kafka consumer
    """
    return await recording_pipeline.process_recording(
        recording_id=recording_id,
        meeting_id=meeting_id,
        classroom_id=classroom_id,
        video_path=video_path,
        meeting_title=meeting_title,
        language=language
    )
