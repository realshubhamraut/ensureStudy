# Page 95: Complete Documentation Master Index — Updated

> All 95 pages organized by category, covering every documented module and feature.

---

## Core Architecture (Pages 1-5)

| # | Page | Description |
|---|------|-------------|
| 1 | System Architecture Overview | Microservices, data flow, deployment topology |
| 2 | AI Service Architecture | FastAPI entrypoint, router registration, middleware |
| 3 | Core Service Architecture | Flask app, blueprints, model registry |
| 4 | Frontend Architecture | Next.js 14, route groups, component hierarchy |
| 5 | Database Architecture | PostgreSQL, Redis, Qdrant, MongoDB, Cassandra |

## AI Agents (Pages 6-13)

| # | Page | Description |
|---|------|-------------|
| 6 | Agent Framework | LangGraph base, state management, types 1-5 |
| 7 | Tutor Agent | Main tutoring pipeline, routing, memory |
| 8 | Assessment Agent | MCQ generation, adaptive difficulty |
| 9 | Research Agent | Web crawling, PDF extraction, resource gathering |
| 10 | Learning Agent | Spaced repetition, mastery tracking |
| 11 | RAG Pipeline | Qdrant retrieval, reranking, context assembly |
| 12 | Web Search Integration | DuckDuckGo, link archival, content ranking |
| 13 | Agent Memory System | Redis memory, context persistence |

## Proctoring & Integrity (Pages 14-15)

| # | Page | Description |
|---|------|-------------|
| 14 | Proctoring System | WebSocket, multi-detector pipeline, static classifier |
| 15 | Browser Event Monitoring | Tab switches, copy/paste, focus tracking |

## API Reference (Pages 16-44)

| # | Page | Description |
|---|------|-------------|
| 16-20 | Authentication & Users | Login, register, JWT, profiles, roles |
| 21-25 | Classrooms | CRUD, membership, settings, invitations |
| 26-30 | Meetings & Recordings | Schedule, join, record, process |
| 31-35 | Assessments | Create, submit, grade, analytics |
| 36-40 | Curriculum & Syllabus | Subjects, topics, progress tracking |
| 41-44 | Documents & Uploads | Upload, process, OCR, embed |

## ML & Analytics (Pages 45-50)

| # | Page | Description |
|---|------|-------------|
| 45 | Recommendation Engine | NCF, content-based, hybrid |
| 46 | Knowledge Tracing | DKT, mastery estimation |
| 47 | Learning Path Optimization | Adaptive sequencing |
| 48 | Difficulty Prediction | Content difficulty estimation |
| 49 | Data Pipelines | Kafka, ETL, analytics |
| 50 | Analytics Dashboard | Metrics, visualizations |

## Frontend Components (Pages 51-60)

| # | Page | Description |
|---|------|-------------|
| 51-55 | Dashboard & Navigation | Layout, sidebar, header, theming |
| 56-58 | Chat & Interaction | Message rendering, avatar, code blocks |
| 59-60 | Document Viewers | PDF viewer, markdown renderer |

## Deployment & Operations (Pages 61-75)

| # | Page | Description |
|---|------|-------------|
| 61-65 | Docker & Compose | Multi-service containers, networking |
| 66-68 | Environment & Secrets | Config management, production setup |
| 69-71 | Monitoring & Logging | Health checks, telemetry |
| 72-73 | Testing | pytest, frontend tests |
| 74 | Makefile & Automation | Development commands |
| 75 | Master Index v1 | Original 75-page index |

## Gap Analysis — Legacy Docs (Pages 76-80)

| # | Page | Description |
|---|------|-------------|
| 76 | Soft Skills Analyzers | WebSocket streaming, TypeScript client, scoring |
| 77 | Feedback Data Models | ER diagrams, learning integration |
| 78 | Agent Roadmap | CAT/IRT, Socratic tutoring, planned features |
| 79 | Proctoring Implementation | StaticProctor, gaze math, PnP, IntegrityScorer |
| 80 | ML Model Architectures | PyTorch NCF, DKT, LPO, Difficulty, MLflow, ONNX |

## Codebase Audit — Gap Fill (Pages 81-95)

| # | Page | Description |
|---|------|-------------|
| 81 | Mock Interview System | LangGraph Type 5 agent, dual interview systems, evaluator |
| 82 | Revision Assessment & Exam Prep | Daily MCQ agent, prep schedule generator |
| 83 | Session Intelligence & Telemetry | Context routing (cosine sim), 3-layer dedup, telemetry |
| 84 | Proctoring Detectors Deep Dive | Blink (EAR), face verify (DeepFace), hand, audio, LSTM, cheat score |
| 85 | OCR Multi-Backend Pipeline | TrOCR, SageMaker, EasyOCR, hybrid approach |
| 86 | Meeting Recording Pipeline | Whisper transcription, speaker diarization, RAG |
| 87 | Syllabus & Topic Extraction | PDF → Qdrant → LLM topics → curriculum population |
| 88 | Suggestion Engine & Follow-ups | 5-stage pipeline, anti-recursion, diversity filtering |
| 89 | LLM Provider & API Key Mgmt | HuggingFace, SageMaker, Groq, rotating key manager |
| 90 | Parent Portal & Admin Panel | 8 parent pages, 6 admin pages, 16+ admin API endpoints |
| 91 | Frontend Curriculum & Assessment | StudyCalendar, ProgressDashboard, CreateAssessmentModal, hooks |
| 92 | Core Infrastructure Services | Authorization (RBAC), rate limiter, 6-layer cache, S3/local storage |
| 93 | YouTube & Video Integration | Data API v3 search, transcript extraction for RAG |
| 94 | Voice Interface & Streaming | TTS (Polly + visemes), STT (Whisper), SSE resource streaming |
| 95 | Complete Documentation Index | This page — full 95-page index |

---

## Statistics

| Metric | Value |
|--------|-------|
| Total pages | 95 |
| Core architecture | 5 pages |
| AI agents | 8 pages |
| API reference | 29 pages |
| Frontend components | 10 pages |
| ML/Analytics | 6 pages |
| Deployment/Ops | 15 pages |
| Legacy gap analysis | 5 pages |
| Codebase audit gap fill | 15 pages |
| Proctoring & integrity | 4 pages (across sections) |
