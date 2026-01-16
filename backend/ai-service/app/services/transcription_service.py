"""
Meeting Transcription Service
Uses OpenAI Whisper API for transcription with speaker diarization integration
Stores results in MongoDB for flexible querying
"""
import os
import json
import asyncio
from datetime import datetime
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict
import tempfile
import subprocess

import openai
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel

# MongoDB configuration
MONGO_URI = os.getenv('MONGO_URI', 'mongodb://ensure_study:mongodb_password_123@localhost:27017')
MONGO_DB = os.getenv('MONGO_DB', 'ensure_study_meetings')

# OpenAI configuration
openai.api_key = os.getenv('OPENAI_API_KEY')


class TranscriptSegment(BaseModel):
    """A segment of transcribed speech"""
    id: int
    start: float  # seconds
    end: float
    speaker_id: int  # 0, 1, 2, etc.
    speaker_name: Optional[str] = None
    text: str
    confidence: float = 0.0


class SpeakerInfo(BaseModel):
    """Information about a speaker in the meeting"""
    speaker_id: int
    user_id: Optional[str] = None
    user_name: Optional[str] = None
    total_speaking_time_seconds: float = 0.0
    segment_count: int = 0


class MeetingTranscript(BaseModel):
    """Complete transcript of a meeting"""
    recording_id: str
    meeting_id: str
    classroom_id: str
    language: str = 'en'
    duration_seconds: float
    speakers: List[SpeakerInfo]
    segments: List[TranscriptSegment]
    full_text: str
    summary: Optional[str] = None
    key_topics: List[str] = []
    word_count: int = 0
    created_at: datetime = None
    
    def __init__(self, **data):
        if 'created_at' not in data or data['created_at'] is None:
            data['created_at'] = datetime.utcnow()
        super().__init__(**data)


class TranscriptionService:
    """
    Service for transcribing meeting recordings
    
    Workflow:
    1. Extract audio from video (if needed)
    2. Run Whisper transcription
    3. Run speaker diarization (separate service)
    4. Align speakers with transcript segments
    5. Store in MongoDB
    6. Trigger embedding generation via Kafka
    """
    
    def __init__(self):
        self.mongo_client: Optional[AsyncIOMotorClient] = None
        self.db = None
    
    async def connect_db(self):
        """Connect to MongoDB"""
        if self.mongo_client is None:
            self.mongo_client = AsyncIOMotorClient(MONGO_URI)
            self.db = self.mongo_client[MONGO_DB]
            
            # Create indexes
            await self.db.meeting_transcripts.create_index('recording_id', unique=True)
            await self.db.meeting_transcripts.create_index('meeting_id')
            await self.db.meeting_transcripts.create_index('classroom_id')
    
    async def close_db(self):
        """Close MongoDB connection"""
        if self.mongo_client:
            self.mongo_client.close()
            self.mongo_client = None
    
    async def extract_audio(self, video_path: str) -> str:
        """
        Extract audio from video file using ffmpeg
        Returns path to extracted audio file
        """
        audio_path = tempfile.mktemp(suffix='.mp3')
        
        process = await asyncio.create_subprocess_exec(
            'ffmpeg', '-i', video_path,
            '-vn',  # No video
            '-acodec', 'libmp3lame',
            '-ar', '16000',  # 16kHz for Whisper
            '-ac', '1',  # Mono
            '-y',  # Overwrite
            audio_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        await process.communicate()
        
        if process.returncode != 0:
            raise RuntimeError(f"ffmpeg failed with code {process.returncode}")
        
        return audio_path
    
    async def transcribe_with_whisper(
        self,
        audio_path: str,
        language: str = 'en'
    ) -> Dict[str, Any]:
        """
        Transcribe audio using OpenAI Whisper API with local fallback
        Returns segments with timestamps
        """
        # Try OpenAI API first
        if openai.api_key:
            try:
                with open(audio_path, 'rb') as audio_file:
                    response = await asyncio.to_thread(
                        openai.audio.transcriptions.create,
                        model="whisper-1",
                        file=audio_file,
                        language=language,
                        response_format="verbose_json",
                        timestamp_granularities=["segment"]
                    )
                
                return {
                    'text': response.text,
                    'segments': [
                        {
                            'id': i,
                            'start': seg.start,
                            'end': seg.end,
                            'text': seg.text.strip(),
                            'avg_logprob': getattr(seg, 'avg_logprob', 0.0)
                        }
                        for i, seg in enumerate(response.segments or [])
                    ],
                    'duration': response.duration if hasattr(response, 'duration') else 0,
                    'language': response.language if hasattr(response, 'language') else language
                }
            except Exception as e:
                print(f"OpenAI API failed, falling back to local Whisper: {e}")
        
        # Fallback to local Whisper model
        return await self._transcribe_local_whisper(audio_path, language)
    
    async def _transcribe_local_whisper(
        self,
        audio_path: str,
        language: str = 'en'
    ) -> Dict[str, Any]:
        """
        Transcribe using local Whisper model (openai-whisper package)
        Install with: pip install openai-whisper
        """
        try:
            import whisper
        except ImportError:
            print("Local whisper not installed. Install with: pip install openai-whisper")
            # Return empty result if whisper not available
            return {
                'text': '[Transcription unavailable - Whisper not installed]',
                'segments': [],
                'duration': 0,
                'language': language
            }
        
        def run_whisper():
            # Load model (base is good balance of speed/accuracy)
            model = whisper.load_model("base")
            result = model.transcribe(audio_path, language=language)
            return result
        
        result = await asyncio.to_thread(run_whisper)
        
        return {
            'text': result.get('text', ''),
            'segments': [
                {
                    'id': i,
                    'start': seg.get('start', 0),
                    'end': seg.get('end', 0),
                    'text': seg.get('text', '').strip(),
                    'avg_logprob': seg.get('avg_logprob', 0.0)
                }
                for i, seg in enumerate(result.get('segments', []))
            ],
            'duration': result.get('duration', 0) if 'duration' in result else 
                        (result['segments'][-1]['end'] if result.get('segments') else 0),
            'language': result.get('language', language)
        }
    
    async def run_speaker_diarization(
        self,
        audio_path: str,
        num_speakers: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Run speaker diarization to identify who is speaking when
        
        Note: This is a placeholder - in production, use pyannote.audio
        For now, we'll do simple silence-based splitting
        """
        # TODO: Integrate pyannote.audio for proper speaker diarization
        # pip install pyannote.audio
        # from pyannote.audio import Pipeline
        # pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")
        # diarization = pipeline(audio_path)
        
        # Placeholder: assign all segments to speaker 0 (host)
        # This will be replaced with actual diarization
        return []
    
    def align_speakers_with_transcript(
        self,
        transcript_segments: List[Dict],
        diarization_segments: List[Dict],
        participant_names: Optional[Dict[int, str]] = None
    ) -> List[TranscriptSegment]:
        """
        Align speaker IDs from diarization with transcript segments
        """
        aligned_segments = []
        
        for seg in transcript_segments:
            # Find overlapping diarization segment
            speaker_id = 0  # Default to first speaker (host)
            
            for diar_seg in diarization_segments:
                # Check for overlap
                seg_mid = (seg['start'] + seg['end']) / 2
                if diar_seg['start'] <= seg_mid <= diar_seg['end']:
                    speaker_id = diar_seg.get('speaker_id', 0)
                    break
            
            speaker_name = None
            if participant_names and speaker_id in participant_names:
                speaker_name = participant_names[speaker_id]
            
            aligned_segments.append(TranscriptSegment(
                id=seg['id'],
                start=seg['start'],
                end=seg['end'],
                speaker_id=speaker_id,
                speaker_name=speaker_name,
                text=seg['text'],
                confidence=1.0 - abs(seg.get('avg_logprob', 0))  # Convert logprob to confidence
            ))
        
        return aligned_segments
    
    def calculate_speaker_stats(
        self,
        segments: List[TranscriptSegment]
    ) -> List[SpeakerInfo]:
        """Calculate speaking time and segment count per speaker"""
        speaker_stats: Dict[int, SpeakerInfo] = {}
        
        for seg in segments:
            if seg.speaker_id not in speaker_stats:
                speaker_stats[seg.speaker_id] = SpeakerInfo(
                    speaker_id=seg.speaker_id,
                    speaker_name=seg.speaker_name
                )
            
            info = speaker_stats[seg.speaker_id]
            info.total_speaking_time_seconds += (seg.end - seg.start)
            info.segment_count += 1
            
            # Update name if available
            if seg.speaker_name and not info.user_name:
                info.user_name = seg.speaker_name
        
        return list(speaker_stats.values())
    
    async def generate_summary(self, full_text: str) -> str:
        """Generate a summary of the meeting using GPT-4 with simple fallback"""
        # Try OpenAI first
        if openai.api_key:
            try:
                response = await asyncio.to_thread(
                    openai.chat.completions.create,
                    model="gpt-4",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert at summarizing meeting transcripts. Create a concise summary highlighting key points, decisions, and action items. Use bullet points."
                        },
                        {
                            "role": "user",
                            "content": f"Summarize this meeting transcript:\n\n{full_text[:10000]}"
                        }
                    ],
                    max_tokens=500
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"GPT-4 summary failed, using simple fallback: {e}")
        
        # Simple fallback - first 500 chars summary
        return self._simple_summary(full_text)
    
    def _simple_summary(self, full_text: str) -> str:
        """Generate a basic summary without AI"""
        if not full_text:
            return "No transcript available."
        
        # Get first few sentences
        sentences = full_text.replace('\n', ' ').split('. ')[:5]
        preview = '. '.join(sentences)
        if len(preview) > 500:
            preview = preview[:500] + '...'
        
        word_count = len(full_text.split())
        return f"Meeting transcript ({word_count} words): {preview}"
    
    async def extract_key_topics(self, full_text: str) -> List[str]:
        """Extract key topics from the transcript"""
        try:
            response = await asyncio.to_thread(
                openai.chat.completions.create,
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "Extract 3-7 key topics from this meeting transcript. Return only a JSON array of topic strings, nothing else."
                    },
                    {
                        "role": "user",
                        "content": full_text[:8000]
                    }
                ],
                max_tokens=200
            )
            topics = json.loads(response.choices[0].message.content)
            return topics if isinstance(topics, list) else []
        except Exception as e:
            print(f"Topic extraction failed: {e}")
            return []
    
    async def transcribe_meeting(
        self,
        recording_id: str,
        meeting_id: str,
        classroom_id: str,
        video_path: str,
        participant_names: Optional[Dict[int, str]] = None,
        language: str = 'en'
    ) -> MeetingTranscript:
        """
        Main entry point: transcribe a meeting recording
        
        Args:
            recording_id: ID of the MeetingRecording
            meeting_id: ID of the Meeting
            classroom_id: ID of the Classroom
            video_path: Path to the video/audio file
            participant_names: Optional mapping of speaker_id -> name
            language: Language code for transcription
        
        Returns:
            MeetingTranscript object stored in MongoDB
        """
        await self.connect_db()
        
        audio_path = None
        
        try:
            # Step 1: Extract audio if video
            if video_path.endswith(('.mp4', '.webm', '.mkv', '.avi')):
                audio_path = await self.extract_audio(video_path)
            else:
                audio_path = video_path
            
            # Step 2: Transcribe with Whisper
            whisper_result = await self.transcribe_with_whisper(audio_path, language)
            
            # Step 3: Run speaker diarization
            diarization_segments = await self.run_speaker_diarization(audio_path)
            
            # Step 4: Align speakers with transcript
            if not participant_names:
                participant_names = {0: "Teacher"}  # Default
            
            aligned_segments = self.align_speakers_with_transcript(
                whisper_result['segments'],
                diarization_segments,
                participant_names
            )
            
            # Step 5: Calculate speaker stats
            speakers = self.calculate_speaker_stats(aligned_segments)
            
            # Step 6: Generate summary and extract topics
            full_text = whisper_result['text']
            summary = await self.generate_summary(full_text)
            key_topics = await self.extract_key_topics(full_text)
            
            # Create transcript object
            transcript = MeetingTranscript(
                recording_id=recording_id,
                meeting_id=meeting_id,
                classroom_id=classroom_id,
                language=whisper_result.get('language', language),
                duration_seconds=whisper_result.get('duration', 0),
                speakers=speakers,
                segments=aligned_segments,
                full_text=full_text,
                summary=summary,
                key_topics=key_topics,
                word_count=len(full_text.split())
            )
            
            # Step 7: Store in MongoDB
            await self.db.meeting_transcripts.replace_one(
                {'recording_id': recording_id},
                transcript.model_dump(),
                upsert=True
            )
            
            return transcript
            
        finally:
            # Cleanup temp audio file
            if audio_path and audio_path != video_path and os.path.exists(audio_path):
                os.remove(audio_path)
    
    async def get_transcript(self, recording_id: str) -> Optional[MeetingTranscript]:
        """Get transcript by recording ID"""
        await self.connect_db()
        
        doc = await self.db.meeting_transcripts.find_one({'recording_id': recording_id})
        if doc:
            doc.pop('_id', None)
            return MeetingTranscript(**doc)
        return None
    
    async def search_transcripts(
        self,
        classroom_id: str,
        query: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Full-text search across meeting transcripts
        """
        await self.connect_db()
        
        # Use MongoDB text search
        cursor = self.db.meeting_transcripts.find(
            {
                'classroom_id': classroom_id,
                '$text': {'$search': query}
            },
            {'score': {'$meta': 'textScore'}}
        ).sort([('score', {'$meta': 'textScore'})]).limit(limit)
        
        results = []
        async for doc in cursor:
            doc.pop('_id', None)
            results.append(doc)
        
        return results


# Singleton instance
transcription_service = TranscriptionService()
