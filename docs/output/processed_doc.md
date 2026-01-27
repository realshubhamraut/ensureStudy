---
title: "EnsureStudy: AI-First Learning Platform"
subtitle: "Comprehensive Technical Documentation"
author: "EnsureStudy Development Team"
date: "January 2026"
documentclass: report
fontsize: 12pt
geometry: "margin=1in"
mainfont: "Times New Roman"
monofont: "Courier New"
linestretch: 1.3
colorlinks: true
linkcolor: blue
urlcolor: blue
toc: true
toc-depth: 3
numbersections: true
header-includes: |
  \usepackage{fancyhdr}
  \usepackage{graphicx}
  \usepackage{longtable}
  \usepackage{booktabs}
  \usepackage{listings}
  \usepackage{xcolor}
  \pagestyle{fancy}
  \fancyhead[L]{\leftmark}
  \fancyhead[R]{EnsureStudy}
  \fancyfoot[C]{\thepage}
  \definecolor{codegreen}{rgb}{0,0.6,0}
  \definecolor{codegray}{rgb}{0.5,0.5,0.5}
  \definecolor{codepurple}{rgb}{0.58,0,0.82}
  \lstset{
    basicstyle=\ttfamily\small,
    keywordstyle=\color{codepurple}\bfseries,
    stringstyle=\color{codegreen},
    commentstyle=\color{codegray}\itshape,
    numbers=left,
    numberstyle=\tiny\color{codegray},
    breaklines=true,
    frame=single
  }
---

\newpage

# Executive Summary

**EnsureStudy** is an AI-first learning platform that combines intelligent multi-agent tutoring, real-time proctoring, and soft skills evaluation. The platform leverages cutting-edge AI technologies including:

- **LangGraph** for agent orchestration
- **RAG (Retrieval-Augmented Generation)** for context-aware responses
- **Mistral 7B** for language understanding and generation
- **YOLO and MediaPipe** for computer vision
- **Apache Kafka and PySpark** for real-time data pipelines

The system is designed for educational institutions seeking to provide personalized learning experiences at scale while maintaining academic integrity.

---

## Key Statistics

| Metric | Value |
|--------|-------|
| Backend Services | 2 (Core + AI) |
| Frontend Framework | Next.js 14 |
| AI Agents | 8 specialized agents |
| Databases | 5 (PostgreSQL, Qdrant, Redis, MongoDB, Cassandra) |
| ML Models | 4+ (Mistral, YOLO, Whisper, FaceMesh) |
| API Endpoints | 50+ |

---

\newpage

# Part I: System Architecture

## High-Level Architecture

The platform follows a **microservices architecture** with two primary backend services connected to a unified frontend.


*[Diagram 1]*


## Service Responsibilities

| Service | Port | Technology | Responsibilities |
|---------|------|------------|------------------|
| **Core Service** | 8000 | Flask, SQLAlchemy | Authentication, user management, classroom operations, file uploads, assessments |
| **AI Service** | 8001 | FastAPI, LangGraph | RAG queries, tutoring agents, proctoring, soft skills, document indexing |
| **Frontend** | 3000/4000 | Next.js 14, TypeScript | User interface, real-time updates, WebSocket connections |

## Request-Response Flow

A typical user query to the AI tutor follows this sequence:


*[Diagram 2]*


---

\newpage

# Part II: Multi-Agent System

## Agent Classification (Russell & Norvig Taxonomy)

| Agent Type | Description | Implementation |
|------------|-------------|----------------|
| **Type 1: Simple Reflex** | Action based on current percept | Proctoring (YOLO detections) |
| **Type 2: Model-Based** | Maintains internal state | Session memory in Redis |
| **Type 3: Goal-Based** | Plans to achieve goals | Orchestrator, Research, Curriculum |
| **Type 5: Learning** | Improves over time | **Tutor Agent** with feedback loop |

## Agent Hierarchy


![Diagram 3](/Users/proxim/projects/ensureStudy/docs/output/diagrams/diagram_3.png)


---

## Orchestrator Agent

The central coordinator using the **Supervisor Pattern** to route requests to specialized sub-agents.

### Intent Classification


![Diagram 4](/Users/proxim/projects/ensureStudy/docs/output/diagrams/diagram_4.png)


### Intent Keywords

| Intent | Trigger Keywords |
|--------|------------------|
| **LEARN** | what is, explain, how does, why, define, tell me about |
| **RESEARCH** | find, search, resources, pdf, download, look up |
| **CREATE** | create, generate, notes, quiz, summary, flashcards |
| **EVALUATE** | grade, check, evaluate, score, feedback, review |

---

## Tutor Agent (Type 5 Learning Agent)

The primary learning assistant with advanced context management and **continuous improvement through feedback**.

### Core Components

| Component | Full Name | Purpose |
|-----------|-----------|---------|
| **ABCR** | Attention-Based Context Routing | Detects follow-up vs new topic queries |
| **TAL** | Topic Anchor Layer | Maintains topic continuity across turns |
| **MCP** | Memory Context Processor | Isolates web vs classroom content |
| **Learning Element** | Few-Shot Injector | Injects high-rated examples into prompts |

### Processing Pipeline


![Diagram 5](/Users/proxim/projects/ensureStudy/docs/output/diagrams/diagram_5.png)


### Learning Loop Architecture


![Diagram 6](/Users/proxim/projects/ensureStudy/docs/output/diagrams/diagram_6.png)


### ABCR Service (Attention-Based Context Routing)

Determines if a query is a follow-up using **semantic similarity** with hysteresis:

```python
class ABCRService:
    def __init__(self):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.attention_threshold = 0.65
        self.hysteresis_factor = 0.1
        
    def is_followup(self, query: str, history: List[dict]) -> bool:
        """
        Compare query embedding against recent turns.
        Apply hysteresis to prevent rapid topic switching.
        """
        if not history:
            return False
            
        recent_context = [h['content'] for h in history[-3:]]
        scores = self.compute_attention_scores(query, recent_context)
        
        max_score = max(scores)
        adjusted_threshold = self.attention_threshold - self.hysteresis_factor
        
        return max_score > adjusted_threshold
```

---

## Research Agent

Discovers and indexes educational content from multiple sources.

### Content Discovery Pipeline


![Diagram 7](/Users/proxim/projects/ensureStudy/docs/output/diagrams/diagram_7.png)


### Web Ingest Workers

| Worker | Function |
|--------|----------|
| W1: Topic Extractor | Extracts key topics from query using LLM |
| W2: DuckDuckGo | Searches web for articles |
| W3: Wikipedia Search | Finds Wikipedia articles |
| W4: Wikipedia Content | Fetches full article content |
| W5: Parallel Crawler | Downloads pages concurrently (httpx) |
| W6: Content Cleaner | Removes boilerplate HTML |
| W6B: PDF Search | Searches and downloads educational PDFs |
| W7: Chunk & Embed | Splits text, generates embeddings, stores in Qdrant |

---

## Curriculum Agent

Creates personalized learning paths based on syllabus and student progress.

### Learning Path Generation


![Diagram 8](/Users/proxim/projects/ensureStudy/docs/output/diagrams/diagram_8.png)


### Schedule Output

```python
class LearningPath:
    topics: List[TopicSchedule]
    total_hours: int
    daily_hours: int
    milestones: List[Milestone]
    
class TopicSchedule:
    topic: str
    day: int
    duration_hours: float
    resources: List[Resource]
    prerequisites: List[str]
```

---

## Document Processing Agent

Ingests and indexes documents for RAG retrieval.

### 7-Stage Pipeline


![Diagram 9](/Users/proxim/projects/ensureStudy/docs/output/diagrams/diagram_9.png)


### Supported Formats

| Format | Extraction Method |
|--------|-------------------|
| PDF (text) | PyMuPDF |
| PDF (scanned) | PyMuPDF + TrOCR |
| Images | TrOCR / Pytesseract |
| DOCX | python-docx |
| PPTX | python-pptx |
| Markdown | Direct parse |

---

\newpage

# Part III: RAG Pipeline

## Overview

The RAG (Retrieval-Augmented Generation) pipeline combines vector search with LLM generation for context-aware responses.


![Diagram 10](/Users/proxim/projects/ensureStudy/docs/output/diagrams/diagram_10.png)


## Embedding Model

| Property | Value |
|----------|-------|
| Model | sentence-transformers/all-MiniLM-L6-v2 |
| Dimensions | 384 |
| Distance Metric | Cosine Similarity |
| Context Window | 512 tokens |

## Retrieval Parameters

```python
RETRIEVAL_CONFIG = {
    "top_k": 5,
    "score_threshold": 0.4,
    "rerank": True,
    "filters": {
        "classroom_id": optional,
        "document_type": optional
    }
}
```

## Context Assembly

The Model Context Protocol (MCP) assembles context with isolation rules:

| Source Type | Isolation Rule |
|-------------|----------------|
| Classroom Materials | Highest priority, user-specific |
| Meeting Transcripts | Time-bounded, classroom-specific |
| Web Content | Lower trust, fact-check flag |
| General Knowledge | Fallback only |

---

\newpage

# Part IV: Real-Time Features

## Proctoring System

Real-time monitoring during assessments using computer vision.

### Detection Pipeline


![Diagram 11](/Users/proxim/projects/ensureStudy/docs/output/diagrams/diagram_11.png)


### Detection Thresholds

| Detection | Model | Threshold |
|-----------|-------|-----------|
| Multiple faces | YOLO person class | > 1 person |
| Mobile phone | YOLO cell phone class | confidence > 0.5 |
| Gaze deviation | Eye landmarks | > 30 degrees |
| Face absence | FaceLandmarker | > 3 seconds |

### Violation Severity

| Level | Violations | Score Impact |
|-------|------------|--------------|
| **LOW** | Slight gaze deviation | -5 points |
| **MEDIUM** | Looking away, face absence | -15 points |
| **HIGH** | Phone detected, multiple people | -30 points |
| **CRITICAL** | Repeated high violations | Session flagged |

---

## Soft Skills Evaluation

### Metric Weights

| Metric | Weight | Analysis Method |
|--------|--------|-----------------|
| **Fluency** | 25% | Speech rate, filler words, pauses |
| **Grammar** | 20% | LanguageTool analysis |
| **Vocabulary** | 15% | Type-token ratio, word diversity |
| **Eye Contact** | 15% | Iris tracking vs camera position |
| **Expression** | 10% | Facial emotion detection |
| **Posture** | 10% | Body position stability |
| **Confidence** | 5% | Combined delivery metrics |

### Evaluation Pipeline


![Diagram 12](/Users/proxim/projects/ensureStudy/docs/output/diagrams/diagram_12.png)


---

\newpage

# Part V: Database Architecture

## Overview

The platform uses **5 specialized databases**, each optimized for specific data patterns.


![Diagram 13](/Users/proxim/projects/ensureStudy/docs/output/diagrams/diagram_13.png)


## Database Selection Criteria

| Database | Type | Use Case | Consistency |
|----------|------|----------|-------------|
| **PostgreSQL** | Relational | Users, classrooms, assessments | Strong (ACID) |
| **Qdrant** | Vector | Document embeddings, RAG | Eventual |
| **Redis** | Key-Value | Sessions, cache, rate limits | Strong |
| **MongoDB** | Document | Transcripts, proctoring reports | Eventual |
| **Cassandra** | Time-Series | Analytics, event streams | Eventual |

## PostgreSQL Schema

### Core Tables

```sql
-- Users
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    name VARCHAR(100),
    role VARCHAR(20) NOT NULL DEFAULT 'student',
    created_at TIMESTAMP DEFAULT NOW()
);

-- Classrooms
CREATE TABLE classrooms (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(200) NOT NULL,
    teacher_id UUID NOT NULL REFERENCES users(id),
    join_code VARCHAR(8) UNIQUE,
    subject VARCHAR(100),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Chat Conversations
CREATE TABLE chat_conversations (
    id VARCHAR(36) PRIMARY KEY,
    user_id VARCHAR(36) NOT NULL REFERENCES users(id),
    title VARCHAR(200),
    subject VARCHAR(50),
    classroom_id VARCHAR(36),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Chat Messages
CREATE TABLE chat_messages (
    id VARCHAR(36) PRIMARY KEY,
    conversation_id VARCHAR(36) NOT NULL REFERENCES chat_conversations(id),
    type VARCHAR(20) NOT NULL,  -- 'user' or 'assistant'
    content TEXT NOT NULL,
    response_json JSONB,
    timestamp TIMESTAMP DEFAULT NOW()
);
```

## Qdrant Collections

### Documents Collection

```python
{
    "collection_name": "documents",
    "vectors_config": {
        "size": 384,  # all-MiniLM-L6-v2
        "distance": "Cosine"
    },
    "hnsw_config": {
        "m": 16,
        "ef_construct": 128
    }
}

# Point structure
{
    "id": "uuid-string",
    "vector": [0.1, 0.2, ...],  # 384 dimensions
    "payload": {
        "text": "Chunk content...",
        "classroom_id": "uuid",
        "document_id": "uuid",
        "document_type": "material",
        "page_number": 5,
        "chunk_index": 12
    }
}
```

## Redis Data Structures

| Key Pattern | Type | TTL | Purpose |
|-------------|------|-----|---------|
| `session:{id}` | Hash | 1 hour | User sessions |
| `ratelimit:{user}:{endpoint}` | Counter | 1 minute | Rate limiting |
| `chat:{session}:history` | List | 2 hours | Chat history cache |
| `cache:{query_hash}` | String | 1 hour | Response cache |

---

\newpage

# Part VI: Data Pipelines

## Event Streaming Architecture


![Diagram 14](/Users/proxim/projects/ensureStudy/docs/output/diagrams/diagram_14.png)


## Kafka Topics

| Topic | Partitions | Retention | Producers |
|-------|------------|-----------|-----------|
| `user-events` | 6 | 7 days | Core Service |
| `learning-progress` | 3 | 30 days | AI Service |
| `meeting-events` | 3 | 7 days | Core Service |
| `proctoring-events` | 6 | 30 days | AI Service |

## Cassandra Analytics Tables

```cql
-- Page View Statistics
CREATE TABLE analytics.page_view_stats (
    date date,
    hour int,
    page text,
    view_count counter,
    PRIMARY KEY ((date), hour, page)
);

-- Learning Progress
CREATE TABLE analytics.learning_progress (
    user_id uuid,
    subject text,
    week_start date,
    lessons_completed int,
    time_spent_minutes int,
    average_score double,
    PRIMARY KEY ((user_id, subject), week_start)
);
```

---

\newpage

# Part VII: API Reference

## Core Service Endpoints

### Authentication

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/auth/signup` | Register new user |
| POST | `/api/auth/login` | Login and get JWT |
| POST | `/api/auth/logout` | Invalidate session |
| GET | `/api/auth/me` | Get current user |

### Classrooms

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/classrooms` | List user's classrooms |
| POST | `/api/classrooms` | Create classroom |
| POST | `/api/classrooms/join` | Join with code |
| GET | `/api/classrooms/{id}` | Get classroom details |

### Chat History

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/chat/conversations` | List conversations |
| POST | `/api/chat/conversations` | Create conversation |
| GET | `/api/chat/conversations/{id}` | Get with messages |
| POST | `/api/chat/conversations/{id}/messages` | Add message |

## AI Service Endpoints

### Tutor

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/ai-tutor/query` | Ask question |
| POST | `/api/tutor/document-chat` | Chat with PDF |

### Indexing

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/indexing/upload` | Upload document |
| GET | `/api/indexing/status/{id}` | Check status |

### Proctoring

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/proctor/start` | Start session |
| WS | `/api/proctor/ws/{session}` | Frame stream |
| GET | `/api/proctor/result/{session}` | Get results |

### Soft Skills

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/softskills/start` | Start evaluation |
| POST | `/api/softskills/analyze` | Analyze recording |

---

\newpage

# Part VIII: Deployment

## Development Setup

### Prerequisites

- Docker & Docker Compose
- Node.js 20+
- Python 3.11+
- HuggingFace API key

### Quick Start

```bash
# 1. Clone repository
git clone https://github.com/ensurestudy/ensurestudy.git
cd ensurestudy

# 2. Configure environment
cp .env.example .env
# Edit .env with your API keys

# 3. Start infrastructure
docker-compose up -d

# 4. Run development servers
./run-local.sh
```

### Service Ports

| Service | Port | URL |
|---------|------|-----|
| Frontend | 4000 | http://localhost:4000 |
| Core API | 9000 | http://localhost:9000 |
| AI API | 8001 | http://localhost:8001 |
| Qdrant | 6333 | http://localhost:6333 |
| Kafka UI | 8080 | http://localhost:8080 |

## Docker Compose Services

```yaml
services:
  # Databases
  postgres:
    image: postgres:15-alpine
    ports: ["5432:5432"]
    
  qdrant:
    image: qdrant/qdrant:latest
    ports: ["6333:6333", "6334:6334"]
    
  redis:
    image: redis:7-alpine
    ports: ["6379:6379"]
    
  mongodb:
    image: mongo:7
    ports: ["27017:27017"]
    
  cassandra:
    image: cassandra:4
    ports: ["9042:9042"]
    
  # Message Queue
  kafka:
    image: confluentinc/cp-kafka:7.5.0
    ports: ["9092:9092"]
    
  # Applications
  core-api:
    build: ./backend/core-service
    ports: ["8000:8000"]
    
  ai-service:
    build: ./backend/ai-service
    ports: ["8001:8001"]
```

---

\newpage

# Part IX: Configuration Reference

## Environment Variables

### Core Service

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection string | Required |
| `REDIS_URL` | Redis connection string | `redis://localhost:6379` |
| `JWT_SECRET` | JWT signing secret | Required |
| `KAFKA_BOOTSTRAP_SERVERS` | Kafka brokers | `localhost:9092` |

### AI Service

| Variable | Description | Default |
|----------|-------------|---------|
| `HUGGINGFACE_API_KEY` | HuggingFace API key | Required |
| `LLM_MODEL` | LLM model name | `mistralai/Mistral-7B-Instruct-v0.2` |
| `EMBEDDING_MODEL` | Embedding model | `all-MiniLM-L6-v2` |
| `QDRANT_HOST` | Qdrant host | `localhost` |
| `QDRANT_PORT` | Qdrant port | `6333` |
| `CORE_SERVICE_URL` | Core service URL | `http://localhost:9000` |

### Feature Flags

| Variable | Description | Default |
|----------|-------------|---------|
| `SKIP_MODERATION` | Disable content moderation | `false` |
| `ABCR_ENABLED` | Enable ABCR routing | `true` |
| `WEB_CRAWL_ENABLED` | Enable web content ingestion | `true` |

---

\newpage

# Part X: Monitoring & Operations

## Key Metrics

| Metric | Source | Threshold |
|--------|--------|-----------|
| API latency (p99) | FastAPI middleware | < 500ms |
| RAG retrieval time | AI Service | < 200ms |
| LLM response time | AI Service | < 5s |
| Cache hit rate | Redis | > 80% |
| Queue depth | Kafka | < 10,000 messages |
| Error rate | All services | < 1% |

## Health Checks

```bash
# Core Service
curl http://localhost:9000/health

# AI Service
curl http://localhost:8001/health

# Qdrant
curl http://localhost:6333/health
```

## Failure Modes

| Failure | Impact | Mitigation |
|---------|--------|------------|
| PostgreSQL down | Auth fails | Read replicas, connection pooling |
| Qdrant down | RAG disabled | Fallback to keyword search |
| Redis down | No caching | Local cache, circuit breaker |
| Kafka down | Events queued | Local buffer, retry logic |
| LLM API down | No AI responses | Cached responses, fallback prompts |

---

\newpage

# Appendices

## Appendix A: Technology Stack Summary

### Application Layer

| Component | Technology |
|-----------|------------|
| Frontend | Next.js 14, TypeScript, TailwindCSS |
| Core API | Flask, SQLAlchemy, JWT |
| AI API | FastAPI, LangGraph, LangChain |

### AI/ML Layer

| Component | Technology |
|-----------|------------|
| LLM | Mistral 7B via HuggingFace |
| Embeddings | all-MiniLM-L6-v2 |
| Object Detection | YOLOv11 |
| Face Analysis | MediaPipe FaceMesh |
| Speech-to-Text | Whisper |
| OCR | TrOCR, Pytesseract |

### Data Layer

| Database | Purpose |
|----------|---------|
| PostgreSQL | Relational data |
| Qdrant | Vector embeddings |
| Redis | Caching, sessions |
| MongoDB | Documents, logs |
| Cassandra | Time-series analytics |

---

## Appendix B: Project Structure

```
ensureStudy/
├── backend/
│   ├── ai-service/
│   │   ├── app/
│   │   │   ├── agents/          # AI agents
│   │   │   ├── api/             # FastAPI routes
│   │   │   ├── proctor/         # Proctoring system
│   │   │   ├── rag/             # RAG pipeline
│   │   │   └── services/        # Business logic
│   │   └── Dockerfile
│   ├── core-service/
│   │   ├── app/
│   │   │   ├── models/          # SQLAlchemy models
│   │   │   ├── routes/          # Flask blueprints
│   │   │   └── utils/           # Utilities
│   │   └── Dockerfile
│   └── kafka/                   # Kafka consumers
├── frontend/
│   ├── app/                     # Next.js pages
│   ├── components/              # React components
│   └── hooks/                   # Custom hooks
├── docs/                        # Documentation
├── ml/                          # ML models & training
├── scripts/                     # Utility scripts
├── docker-compose.yml
└── README.md
```

---

## Appendix C: Glossary

| Term | Definition |
|------|------------|
| **ABCR** | Attention-Based Context Routing - determines if a query is a follow-up |
| **TAL** | Topic Anchor Layer - maintains topic continuity |
| **MCP** | Memory Context Processor - manages conversation memory |
| **RAG** | Retrieval-Augmented Generation - combining search with LLM |
| **Embedding** | Numerical vector representation of text |
| **Chunk** | A segment of a document for indexing |
| **Proctoring** | Real-time monitoring during assessments |

---

*Document generated on January 2026*

*EnsureStudy Development Team*
