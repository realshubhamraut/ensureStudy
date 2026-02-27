# Page 86: Meeting Recording Pipeline

> End-to-end recording processing: video upload → audio extraction → Whisper transcription → speaker diarization → embedding generation → meeting RAG.

---

## 86.1 Pipeline Architecture

```mermaid\nflowchart TB\n    VU[\"📹 Video Upload<br/>WebM/MP4\"] --> RP[\"recording_pipeline<br/>6.5KB — Orchestrates full flow\"]\n\n    subgraph TS[\"TranscriptionService — 25KB\"]\n        direction TB\n        EA[\"extract_audio()<br/>FFmpeg → WAV 16kHz mono\"]\n        TR[\"transcribe()<br/>Local Whisper model\"]\n        DI[\"diarize()<br/>Speaker identification\"]\n        AL[\"align()<br/>Match speakers to segments\"]\n        EA --> TR --> DI --> AL\n    end\n\n    RP --> TS\n    TS --> MES[\"MeetingEmbedding Service<br/>12.6KB — Chunk + embed transcripts<br/>→ Qdrant 'meeting_transcripts'\"]\n    MES --> MRAG[\"MeetingRAG<br/>8.4KB — Semantic Q&A over meetings<br/>Speaker attribution + timestamps\"]\n\n    style VU fill:#3b82f6,color:#fff\n    style MES fill:#f59e0b,color:#000\n    style MRAG fill:#10b981,color:#fff\n```

### Source Files

| File | Size | Role |
|------|------|------|
| `api/process_recording.py` | 8.4KB | Upload + process endpoint |
| `api/meeting_qa.py` | 4.7KB | Q&A over meeting transcripts |
| `services/recording_pipeline.py` | 6.5KB | Pipeline orchestrator |
| `services/transcription_service.py` | 25KB | Whisper + diarization |
| `services/meeting_embedding_service.py` | 12.6KB | Transcript embedding |
| `services/meeting_rag.py` | 8.4KB | Meeting-aware RAG |
| `core-service/app/routes/recordings.py` | 17KB | Recording CRUD |

---

## 86.2 TranscriptionService

### Data Models

```python
class TranscriptSegment(BaseModel):
    id: int
    start: float            # Start time (seconds)
    end: float              # End time (seconds)
    speaker_id: int
    speaker_name: Optional[str]
    text: str
    confidence: float

class SpeakerInfo(BaseModel):
    speaker_id: int
    user_name: Optional[str]
    total_speaking_time_seconds: float
    segment_count: int

class MeetingTranscript(BaseModel):
    recording_id: str
    meeting_id: str
    classroom_id: str
    language: str = "en"
    duration_seconds: float
    speakers: List[SpeakerInfo]
    segments: List[TranscriptSegment]
    full_text: str
    formatted_transcript: str
    summary: str
    word_count: int
```

### Transcription Flow

```python
class TranscriptionService:
    # Step 1: Extract audio
    async def extract_audio(self, video_path: str) -> str:
        """FFmpeg: video → WAV (16kHz mono for Whisper)"""
    
    # Step 2: Transcribe with local Whisper
    async def transcribe_with_whisper(self, audio_path, language="en"):
        """Uses openai-whisper package (free, local)
        Models: tiny(39M) → base(74M) → small(244M) → medium(769M) → large(1.5B)"""
    
    # Step 3: Speaker diarization
    async def run_speaker_diarization(self, audio_path, num_speakers=None):
        """Uses simple_diarizer for local, free diarization"""
    
    # Step 4: Align speakers with transcript
    def align_speakers_with_transcript(self, transcript_segments, diarization_segments):
        """Match speaker IDs to transcript segments by time overlap"""
    
    # Step 5: Generate formatted output
    def _generate_formatted_transcript(self, segments):
        """Groups consecutive segments by speaker:
        Speaker 1: Hello everyone...
        Speaker 2: Thank you...
        """
    
    # Step 6: Extractive summary
    async def generate_summary(self, full_text):
        """Top sentences by TF-IDF relevance"""
```

### Storage

Transcripts stored in **MongoDB** (`ensure_study_meetings` database) for flexible querying and full-text search.

---

## 86.3 Meeting Embedding & RAG

### Meeting Embedding Service

Chunks transcripts and stores embeddings in Qdrant for semantic search:

```python
class MeetingEmbeddingService:
    def embed_transcript(self, transcript: MeetingTranscript):
        """
        1. Split transcript into ~500-word chunks with speaker context
        2. Generate embeddings via SentenceTransformer
        3. Store in Qdrant 'meeting_transcripts' collection
        4. Metadata: meeting_id, classroom_id, speaker, timestamp range
        """
```

### Meeting RAG

```python
class MeetingRAG:
    def query(self, question: str, classroom_id: str, meeting_id: str = None):
        """
        1. Embed question
        2. Search Qdrant with classroom_id filter
        3. Retrieve top-k chunks with speaker attribution
        4. LLM generates answer citing specific speakers + timestamps
        """
```

---

## 86.4 Recording API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `POST /api/recordings/upload` | POST | Upload recording file |
| `POST /api/recordings/{id}/process` | POST | Trigger processing pipeline |
| `GET /api/recordings/{id}/transcript` | GET | Get full transcript |
| `GET /api/recordings/{id}/summary` | GET | Get meeting summary |
| `POST /api/meetings/{id}/qa` | POST | Ask question about meeting |
| `GET /api/recordings/search` | GET | Full-text search across transcripts |
