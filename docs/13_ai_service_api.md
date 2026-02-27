# Page 13: AI Service — FastAPI Architecture & Routes

---

## 13.1 Overview

The AI Service is a **FastAPI application** that handles all AI/ML workloads: LLM inference, agent orchestration, RAG retrieval, document processing, proctoring, speech, and soft skills evaluation. It communicates with the Core Service via HTTP and with clients via REST + Server-Sent Events (SSE).

### Source: `backend/ai-service/app/main.py` (231 lines)

| Metric | Value |
|--------|-------|
| Framework | FastAPI (Starlette) |
| Docs | Auto-generated at `/docs` (Swagger) and `/redoc` (ReDoc) |
| CORS | All origins (`*`), credentials disabled |
| Routers | 27 included |
| Middleware | Request logging with timing |
| Startup | Optional model preloading |
| Port | 8001 (default) |

---

## 13.2 Application Setup

```python
app = FastAPI(
    title="ensureStudy AI Tutor Service",
    description="AI-powered tutor with multi-agent orchestration",
    version="2.0.0",
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None
)
```

### Request Logging Middleware

Every request is logged with execution time:

```python
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start
    logger.info(f"{request.method} {request.url.path} → {response.status_code} ({duration:.2f}s)")
    return response
```

---

## 13.3 Complete Router Inventory

### Tutor & Chat (3 routers)

| Router | Prefix | File | Key Endpoints |
|--------|--------|------|---------------|
| `tutor_router` | `/api/ai-tutor` | `routes/tutor.py` | `POST /chat` (legacy streaming endpoint) |
| `tutor_chat_router` | `/api/tutor` | `routes/chat.py` | `POST /chat` (TAL/ABCR/MCP), `GET /session/:id` |
| `pdf_chat_router` | `/api/tutor/chat` | PDF-specific chat | `POST /pdf` (document-grounded Q&A) |

### Agent System (1 router)

| Router | Prefix | File | Key Endpoints |
|--------|--------|------|---------------|
| `agent_router` | `/api/agent` | `routes/agent.py` | `POST /chat` (orchestrator entry), `GET /sessions`, `POST /tool-call` |

### Document & Content (4 routers)

| Router | Prefix | File | Key Endpoints |
|--------|--------|------|---------------|
| `indexing_router` | `/api/index` | `routes/indexing.py` | `POST /document`, `POST /web-content`, `GET /status/:id` |
| `web_ingest_router` | `/api/web-ingest` | `routes/web_ingest.py` | `POST /crawl`, `POST /pdf`, `GET /status/:id` |
| `web_resources_router` | `/api/web-resources` | `routes/web_resources.py` | `GET /search`, `POST /download-pdf`, `GET /sources` |
| `documents_router` | `/api/convert` | `routes/documents.py` | `POST /pptx-to-pdf`, `GET /document/:id` |

### Curriculum & Questions (4 routers)

| Router | Prefix | File | Key Endpoints |
|--------|--------|------|---------------|
| `curriculum_router` | `/api/curriculum` | `routes/curriculum.py` | `POST /generate`, `GET /:id`, `POST /exam-prep` |
| `syllabus_router` | `/api/syllabus` | `routes/syllabus.py` | `POST /extract`, `GET /topics/:id` |
| `questions_router` | `/api/questions` | `routes/questions.py` | `POST /generate`, `GET /topic/:id` |
| `questions_scoring_router` | `/api/questions` | `routes/questions_scoring.py` | `POST /score-descriptive`, `POST /evaluate-interview` |
| `topic_scores_router` | `/api/topic-scores` | `routes/topic_scores.py` | `GET /:topic_id`, `POST /update` |

### Classroom Syllabus (1 router)

| Router | Prefix | File | Key Endpoints |
|--------|--------|------|---------------|
| `classroom_syllabus_router` | `/api/classroom-syllabus` | `routes/classroom_syllabus.py` | `POST /extract`, `GET /topics/:classroom_id`, `POST /hierarchy` |

### Assessment & Evaluation (2 routers)

| Router | Prefix | File | Key Endpoints |
|--------|--------|------|---------------|
| `evaluation_router` | `/api/evaluation` | `routes/evaluation.py` | `POST /exam`, `POST /score`, `GET /session/:id` |
| `mock_interview_router` | `/api/mock-interview` | `routes/mock_interview.py` | `POST /start`, `POST /respond`, `POST /evaluate` |

### Proctoring & Soft Skills (2 routers)

| Router | Prefix | File | Key Endpoints |
|--------|--------|------|---------------|
| `proctor_router` | `/api/proctoring` | Proctoring API | `POST /start`, `POST /analyze-frame`, `POST /end` |
| `softskills_router` | `/api/softskills` | Soft skills API | `POST /evaluate`, `GET /results/:id`, `POST /analyze-frame` |

### Meetings & Notes (3 routers)

| Router | Prefix | File | Key Endpoints |
|--------|--------|------|---------------|
| `meetings_router` | `/api/meetings` | `api/meetings.py` | `POST /process`, `GET /recording/:id` |
| `meeting_qa_router` | `/api/meeting-qa` | `api/meeting_qa.py` | `POST /ask`, `GET /transcript/:id` |
| `notes_router` | `/api/notes` | `api/notes.py` | `POST /digitize`, `POST /ocr`, `GET /search` |

### Speech (2 routers)

| Router | Prefix | File | Key Endpoints |
|--------|--------|------|---------------|
| `tts_router` | `/api/tts` | `routes/tts.py` | `POST /synthesize` (AWS Polly with visemes) |
| `stt_router` | `/api/stt` | `routes/stt.py` | `POST /transcribe` (local Whisper fallback) |

### Grading (1 router)

| Router | Prefix | File | Key Endpoints |
|--------|--------|------|---------------|
| `grading_router` | `/api/grading` | `routes/grading.py` | `POST /grade-assignment`, `POST /grade-submission` |

### Real-Time (2 routers)

| Router | Prefix | File | Key Endpoints |
|--------|--------|------|---------------|
| `sse_router` | `/sse` | `routes/sse.py` | `GET /stream` (Server-Sent Events for real-time updates) |
| `anchor_router` | — | `routes/anchor_routes.py` | Static anchor/utility routes |

---

## 13.4 Streaming Response Pattern

The AI Service uses SSE for real-time tutor responses:

```python
@router.post("/chat")
async def tutor_chat(request: TutorChatRequest):
    async def event_generator():
        async for chunk in orchestrator.stream_response(
            query=request.message,
            user_id=request.user_id,
            session_id=request.session_id,
            classroom_id=request.classroom_id
        ):
            yield f"data: {json.dumps(chunk)}\n\n"
        yield "data: [DONE]\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )
```

### SSE Event Types

| Event | Data Shape | Purpose |
|-------|------------|---------|
| `token` | `{"type": "token", "content": "word"}` | Streaming LLM tokens |
| `sources` | `{"type": "sources", "items": [...]}` | RAG source citations |
| `web_sources` | `{"type": "web_sources", "items": [...]}` | Web enrichment links |
| `agent` | `{"type": "agent", "name": "research"}` | Active agent indicator |
| `tool_call` | `{"type": "tool_call", "tool": "...", "args": {...}}` | MCP tool invocation |
| `error` | `{"type": "error", "message": "..."}` | Error notification |
| `[DONE]` | — | Stream termination signal |

---

## 13.5 Startup & Model Preloading

```python
@app.on_event("startup")
async def startup_event():
    # Conditionally preload ML models
    if os.getenv("PRELOAD_MODELS", "false").lower() == "true":
        from app.services.embedding_service import get_embedding_model
        get_embedding_model()  # Load sentence-transformers
        
    # Initialize Qdrant collections
    from app.services.qdrant_service import get_qdrant_service
    qdrant = get_qdrant_service()
    await qdrant.ensure_collections()
    
    # Log configuration
    logger.info(f"AI Service started | LLM: {settings.LLM_PROVIDER} | Debug: {settings.DEBUG}")
```

---

## 13.6 Services Layer (89 Files)

The AI service has **89 service files** in `app/services/` providing the business logic layer:

| Category | Count | Key Services |
|----------|-------|--------------|
| RAG & Vector | 8 | `qdrant_service.py`, `embedding_service.py`, `chunking_service.py`, `smart_retrieval.py` |
| LLM & Chat | 6 | `llm_service.py`, `chat_service.py`, `response_cache.py`, `prompt_builder.py` |
| Web & Search | 7 | `web_ingest_service.py` (60KB), `search_api.py`, `content_crawler.py`, `pdf_downloader.py` |
| OCR & Image | 6 | `hybrid_ocr.py`, `ocr_service.py`, `image_enhancer.py`, `ocr_adapter.py`, `nanonets_ocr.py` |
| Curriculum | 5 | `curriculum_storage.py`, `syllabus_extractor.py` (33KB), `topic_extractor.py` (37KB), `spaced_repetition.py` |
| Speech | 3 | `tts_service.py` (AWS Polly), `stt_service.py` (Whisper), `youtube_transcript_service.py` |
| Proctoring | 4 | Detector coordination, scoring, model inference |
| Assessment | 3 | `assessment_service.py`, `question_generation.py`, `scoring_service.py` |
| Meetings | 3 | `meeting_processor.py`, `recording_service.py`, `transcript_service.py` |
| Classification | 3 | `subject_classifier.py`, `topic_classifier.py`, `intent_classifier.py` |
| Miscellaneous | 41 | Config, utilities, adapters, caching, monitoring |
