# Page 33: AI Service API — Complete Endpoint Reference

---

## 33.1 Overview

The AI Service (FastAPI) exposes **27 router modules** with an estimated **80+ endpoints** covering tutoring, agents, RAG, document processing, speech, proctoring, meetings, and soft skills.

### Base URL: `http://localhost:8001`

---

## 33.2 Endpoint Reference by Router

### Tutor (`api/tutor.py`, `routes/tutor.py`)

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/api/tutor/chat` | Send message to AI tutor (SSE streaming) |
| POST | `/api/tutor/chat/sync` | Synchronous tutor chat |
| GET | `/api/tutor/sessions/<id>` | Get tutor session history |
| POST | `/api/tutor/assess-level` | Assess student TAL level |

### Agent Orchestrator (`api/agents.py`, `routes/agent.py`)

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/api/agents/invoke` | Invoke orchestrator with task |
| POST | `/api/agents/research` | Trigger research agent |
| POST | `/api/agents/curriculum/generate` | Generate curriculum |
| POST | `/api/agents/learning/trigger` | Trigger learning agent cycle |

### RAG & Search (`api/rag.py`, `routes/web_resources.py`)

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/api/rag/query` | RAG query against indexed materials |
| POST | `/api/rag/search` | Semantic search |
| GET | `/api/web-resources/search` | Web search + caching |
| POST | `/api/web-resources/ingest` | Ingest web URL |

### Document Processing (`routes/documents.py`, `routes/indexing.py`)

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/api/documents/process` | Run 7-stage document pipeline |
| POST | `/api/documents/ocr` | OCR a document/image |
| POST | `/api/index/document` | Index document into Qdrant |
| POST | `/api/index/classroom-material` | Index classroom material |
| DELETE | `/api/index/<collection>/<id>` | Remove from index |

### Curriculum (`routes/curriculum.py`, `routes/syllabus.py`)

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/api/curriculum/extract-topics` | Extract topics from syllabus |
| POST | `/api/curriculum/generate-dependencies` | Generate topic dependencies |
| POST | `/api/curriculum/learning-path` | Generate learning path |
| POST | `/api/syllabus/analyze` | Analyze syllabus document |
| POST | `/api/syllabus/extract` | Extract syllabus structure |

### Assessment & Questions (`routes/questions.py`, `routes/questions_scoring.py`)

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/api/questions/generate` | Generate questions for topic |
| POST | `/api/questions/generate-pool` | Generate question pool |
| POST | `/api/questions/score` | Score student answer |
| POST | `/api/questions/score-descriptive` | Score descriptive answer |
| POST | `/api/questions/batch-score` | Batch score multiple answers |

### Chat & Sessions (`routes/chat.py`, `routes/session.py`, `routes/sse.py`)

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/api/chat/message` | Send chat message |
| GET | `/api/chat/stream` | SSE event stream |
| POST | `/api/sessions/create` | Create AI session |
| GET | `/api/sse/events` | Server-Sent Events stream |

### Evaluation & Grading (`routes/evaluation.py`, `routes/grading.py`)

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/api/evaluation/answer` | Evaluate single answer |
| POST | `/api/evaluation/batch` | Batch evaluation |
| POST | `/api/grading/submit` | Submit for AI grading |
| POST | `/api/grading/rubric` | Generate grading rubric |

### Speech (`routes/stt.py`, `routes/tts.py`)

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/api/stt/transcribe` | Speech-to-text (Whisper) |
| POST | `/api/tts/synthesize` | Text-to-speech (AWS Polly) |
| POST | `/api/tts/visemes` | TTS with viseme data (lip-sync) |

### Meetings (`api/meetings.py`, `api/meeting_qa.py`)

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/api/meetings/transcribe` | Transcribe recording (Whisper) |
| POST | `/api/meetings/summarize` | Summarize transcript (Gemini) |
| POST | `/api/meetings/query` | RAG Q&A about meeting |
| GET | `/api/meetings/<id>/transcript` | Get stored transcript |

### Soft Skills (`routes/softskills.py`)

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/api/softskills/analyze-frame` | Analyze video frame |
| POST | `/api/softskills/session/start` | Start evaluation session |
| POST | `/api/softskills/session/end` | End session + get report |
| GET | `/api/softskills/results/<id>` | Get session results |

### Mock Interview (`routes/mock_interview.py`)

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/api/mock-interview/start` | Start mock interview |
| POST | `/api/mock-interview/answer` | Submit interview answer |
| POST | `/api/mock-interview/evaluate` | Get interview evaluation |

### Notes (`api/notes.py`)

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/api/notes/embed` | Embed notes into Qdrant |
| POST | `/api/notes/search` | Search notes semantically |

### Proctoring (`proctor/api.py`)

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/api/proctoring/session/start` | Start proctoring session |
| POST | `/api/proctoring/analyze-frame` | Analyze webcam frame |
| POST | `/api/proctoring/tab-switch` | Record tab switch |
| POST | `/api/proctoring/session/end` | End session + get results |

### Topic Scores (`routes/topic_scores.py`)

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/api/topic-scores/<user_id>` | Get user topic scores |
| POST | `/api/topic-scores/update` | Update topic mastery |

### Web Ingest (`routes/web_ingest.py`)

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/api/web-ingest/crawl` | Crawl and ingest URL |
| POST | `/api/web-ingest/batch` | Batch URL ingestion |

### Classroom Syllabus (`routes/classroom_syllabus.py`)

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/api/classroom-syllabus/analyze` | Analyze uploaded syllabus |
| POST | `/api/classroom-syllabus/extract` | Extract topics from syllabus |

### Anchor Routes (`routes/anchor_routes.py`)

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/api/anchors/generate` | Generate anchor points for content |

---

## 33.3 SSE Streaming Pattern

```python
@router.post("/chat")
async def tutor_chat(request: ChatRequest):
    async def event_generator():
        async for chunk in llm.astream(messages):
            yield f"data: {json.dumps({'content': chunk.content})}\n\n"
        yield "data: [DONE]\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )
```

---

## 33.4 Health Endpoint

```python
@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "ai-tutor", "version": "2.0.0"}
```
