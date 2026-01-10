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
MONGO_URI = os.getenv('MONGO_URI', 'mongodb://localhost:27017')
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
        Returns path to extracted audio file (WAV for better quality)
        """
        # Use WAV for lossless audio - better for transcription
        audio_path = tempfile.mktemp(suffix='.wav')
        
        print(f"[Transcription] Extracting audio from {video_path}")
        
        try:
            process = await asyncio.create_subprocess_exec(
                'ffmpeg', '-i', video_path,
                '-vn',  # No video
                '-acodec', 'pcm_s16le',  # WAV format (lossless)
                '-ar', '16000',  # 16kHz for Whisper
                '-ac', '1',  # Mono
                '-y',  # Overwrite
                audio_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                print(f"[Transcription] ffmpeg stderr: {stderr.decode()[:500]}")
                raise RuntimeError(f"ffmpeg failed with code {process.returncode}")
            
            # Verify the file was created and has content
            if not os.path.exists(audio_path):
                raise RuntimeError("Audio extraction failed - no output file")
            
            file_size = os.path.getsize(audio_path)
            print(f"[Transcription] Extracted audio: {file_size / 1024:.1f} KB")
            
            if file_size < 1000:  # Less than 1KB is suspicious
                raise RuntimeError(f"Audio file too small ({file_size} bytes) - may be corrupted")
            
            return audio_path
            
        except FileNotFoundError:
            raise RuntimeError("ffmpeg not found. Install with: brew install ffmpeg")
    
    async def transcribe_with_whisper(
        self,
        audio_path: str,
        language: str = 'en'
    ) -> Dict[str, Any]:
        """
        Transcribe audio using local Whisper model (free, no API key needed)
        Downloads model automatically on first use
        """
        # Use local Whisper directly - no OpenAI API needed
        return await self._transcribe_local_whisper(audio_path, language)
    
    async def _transcribe_local_whisper(
        self,
        audio_path: str,
        language: str = 'en'
    ) -> Dict[str, Any]:
        """
        Transcribe using local Whisper model (openai-whisper package)
        Install with: pip install openai-whisper
        
        Model sizes (download on first use):
        - tiny: 39M params, fastest, lowest quality
        - base: 74M params, fast, decent quality
        - small: 244M params, balanced (RECOMMENDED)
        - medium: 769M params, good quality, slower
        - large: 1550M params, best quality, very slow
        """
        try:
            import whisper
        except ImportError:
            print("Local whisper not installed. Install with: pip install openai-whisper")
            return {
                'text': '[Transcription unavailable - Whisper not installed]',
                'segments': [],
                'duration': 0,
                'language': language
            }
        
        # Use MEDIUM model for best accuracy (downloads ~1.5GB)
        # Takes longer but produces best results
        WHISPER_MODEL = os.getenv('WHISPER_MODEL', 'medium')
        
        def run_whisper():
            print(f"[Transcription] Loading Whisper model: {WHISPER_MODEL}")
            model = whisper.load_model(WHISPER_MODEL)
            
            print(f"[Transcription] Transcribing audio file: {audio_path}")
            result = model.transcribe(
                audio_path, 
                language=language,
                # Best quality settings
                temperature=0.0,  # More deterministic output
                best_of=5,  # Try multiple times and pick best
                beam_size=5,  # Beam search for better accuracy
                condition_on_previous_text=True,  # Context awareness
                verbose=False
            )
            
            print(f"[Transcription] Complete: {len(result.get('text', ''))} chars, {len(result.get('segments', []))} segments")
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
        Run speaker diarization to identify who is speaking when.
        Uses simple_diarizer for local, free diarization.
        """
        try:
            from simple_diarizer.diarizer import Diarizer
            from simple_diarizer.utils import combined_waveplot
            
            print(f"[Diarization] Starting speaker diarization for {audio_path}")
            
            # Run diarization in a thread to not block
            def do_diarize():
                diar = Diarizer(
                    embed_model='xvec',  # or 'ecapa' for better accuracy but slower
                )
                # Cluster speakers
                segments = diar.diarize(
                    audio_path,
                    num_speakers=num_speakers,  # None = auto-detect
                    threshold=0.5
                )
                return segments
            
            segments = await asyncio.to_thread(do_diarize)
            
            # Convert to our format
            diarization_segments = []
            for i, seg in enumerate(segments):
                # Extract speaker ID from label (e.g., "SPEAKER_00" -> 0)
                speaker_label = seg.get('label', 'SPEAKER_00')
                speaker_id = int(speaker_label.replace('SPEAKER_', '')) if 'SPEAKER_' in speaker_label else i
                
                diarization_segments.append({
                    'start': seg['start'],
                    'end': seg['end'],
                    'speaker_id': speaker_id
                })
            
            print(f"[Diarization] Found {len(set(s['speaker_id'] for s in diarization_segments))} speakers, {len(diarization_segments)} segments")
            return diarization_segments
            
        except ImportError:
            print("[Diarization] simple_diarizer not installed. Install with: pip install simple-diarizer")
            return self._fallback_diarization(audio_path)
        except Exception as e:
            print(f"[Diarization] Failed: {e}, using fallback")
            return self._fallback_diarization(audio_path)
    
    def _fallback_diarization(self, audio_path: str) -> List[Dict[str, Any]]:
        """
        Simple fallback diarization.
        Since pydub requires audioop which is removed in Python 3.13+,
        we return empty and let the transcript use Whisper segments directly.
        The align function will assign all to speaker 0.
        
        For multi-speaker detection, use simple_diarizer or pyannote.
        """
        print("[Diarization Fallback] Using single-speaker mode (install simple-diarizer for multi-speaker)")
        # Return empty - all segments will be assigned to Speaker 1
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
    
    def _generate_formatted_transcript(
        self,
        segments: List[TranscriptSegment]
    ) -> str:
        """
        Generate formatted transcript text with speaker labels.
        Groups consecutive segments by same speaker.
        
        Example output:
        Speaker 1: Hello everyone, welcome to the class.
        
        Speaker 2: Thank you, good to be here.
        
        Speaker 1: Today we'll cover...
        """
        if not segments:
            return ""
        
        formatted_parts = []
        current_speaker = None
        current_text = []
        
        for seg in segments:
            speaker_label = seg.speaker_name or f"Speaker {seg.speaker_id + 1}"
            
            if seg.speaker_id != current_speaker:
                # Save previous speaker's text
                if current_text and current_speaker is not None:
                    prev_label = segments[0].speaker_name if current_speaker == 0 and segments[0].speaker_name else f"Speaker {current_speaker + 1}"
                    # Find the actual label for current_speaker
                    for s in segments:
                        if s.speaker_id == current_speaker:
                            prev_label = s.speaker_name or f"Speaker {current_speaker + 1}"
                            break
                    formatted_parts.append(f"{prev_label}: {' '.join(current_text)}")
                
                current_speaker = seg.speaker_id
                current_text = [seg.text.strip()]
            else:
                current_text.append(seg.text.strip())
        
        # Don't forget the last speaker's text
        if current_text and current_speaker is not None:
            speaker_label = segments[-1].speaker_name or f"Speaker {current_speaker + 1}"
            for s in segments:
                if s.speaker_id == current_speaker:
                    speaker_label = s.speaker_name or f"Speaker {current_speaker + 1}"
                    break
            formatted_parts.append(f"{speaker_label}: {' '.join(current_text)}")
        
        return "\n\n".join(formatted_parts)
    
    async def generate_summary(self, full_text: str) -> str:
        """Generate a summary of the meeting using extractive summary"""
        # Skip OpenAI if using test key or no key
        api_key = openai.api_key or ""
        if api_key and not api_key.startswith("sk-test") and len(api_key) > 20:
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
        
        # Use simple extractive summary (no API needed)
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
        # Skip OpenAI if using test key or no key
        api_key = openai.api_key or ""
        if api_key and not api_key.startswith("sk-test") and len(api_key) > 20:
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
        
        # Simple keyword-based topic extraction (no API needed)
        return self._extract_simple_topics(full_text)
    
    def _extract_simple_topics(self, text: str) -> List[str]:
        """Extract topics using simple keyword frequency"""
        if not text:
            return []
        
        # Common stop words to filter
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 
                      'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                      'would', 'could', 'should', 'may', 'might', 'must', 'can',
                      'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
                      'as', 'into', 'through', 'during', 'before', 'after', 'above',
                      'below', 'between', 'under', 'again', 'further', 'then', 'once',
                      'here', 'there', 'when', 'where', 'why', 'how', 'all', 'each',
                      'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
                      'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
                      'just', 'and', 'but', 'or', 'if', 'because', 'about', 'this',
                      'that', 'these', 'those', 'am', 'i', 'you', 'he', 'she', 'it',
                      'we', 'they', 'what', 'which', 'who', 'whom', 'speaker', 'um',
                      'uh', 'like', 'know', 'think', 'yeah', 'okay', 'right', 'well'}
        
        # Count word frequency
        words = text.lower().split()
        word_counts = {}
        for word in words:
            # Clean word
            clean_word = ''.join(c for c in word if c.isalnum())
            if clean_word and len(clean_word) > 3 and clean_word not in stop_words:
                word_counts[clean_word] = word_counts.get(clean_word, 0) + 1
        
        # Get top 5 most frequent meaningful words
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        topics = [word.capitalize() for word, count in sorted_words[:5] if count > 1]
        
        return topics if topics else ["General Discussion"]
    
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
            
            # Step 6: Generate full text WITH speaker labels
            full_text = self._generate_formatted_transcript(aligned_segments)
            
            # Step 7: Generate summary and extract topics
            summary = await self.generate_summary(whisper_result['text'])  # Use original for summary
            key_topics = await self.extract_key_topics(whisper_result['text'])
            
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
                word_count=len(whisper_result['text'].split())
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
