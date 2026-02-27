# Page 36: Dependency Analysis — 152 Python + 80 Node.js Packages

---

## 36.1 Overview

ensureStudy's dependency footprint spans **111 Python packages** (AI Service), **41 Python packages** (Core Service), and **~80 Node.js packages** (Frontend). This page catalogs every dependency by category, its purpose, and licensing considerations.

---

## 36.2 AI Service Dependencies (111 packages)

### Source: `backend/ai-service/requirements.txt`

#### Core Framework

| Package | Version | Purpose |
|---------|---------|---------|
| `fastapi` | ≥0.109.0 | Async web framework |
| `uvicorn` | ≥0.27.0 | ASGI server |
| `pydantic` | ≥2.5.0 | Data validation |
| `pydantic-settings` | ≥2.1.0 | Settings management |
| `sse-starlette` | ≥1.6.0 | Server-Sent Events |

#### AI/ML Models (Local, Free)

| Package | Version | Purpose | Size |
|---------|---------|---------|------|
| `sentence-transformers` | ≥2.3.0 | Text embeddings (all-mpnet-base-v2) | ~400 MB |
| `transformers` | ≥4.45.0 | Hugging Face model hub | ~100 MB |
| `torch` | ≥2.1.0 | PyTorch deep learning | ~2 GB |
| `torchvision` | ≥0.16.0 | Vision models | ~100 MB |
| `qwen-vl-utils` | ≥0.0.8 | Nanonets-OCR2-3B support | ~50 MB |
| `openai-whisper` | ≥20231117 | Speech-to-text (local) | ~140 MB (base) |

#### Agent Framework

| Package | Version | Purpose |
|---------|---------|---------|
| `langchain` | ≥0.1.0 | LLM orchestration |
| `langgraph` | ≥0.0.20 | Agentic workflow graphs |
| `google-generativeai` | ≥0.3.0 | Gemini LLM API |

#### Vector Database

| Package | Version | Purpose |
|---------|---------|---------|
| `qdrant-client` | ≥1.7.0 | Qdrant vector DB client |

#### Document Processing

| Package | Version | Purpose |
|---------|---------|---------|
| `pymupdf` | ≥1.23.0 | PDF text extraction |
| `pypdf` | ≥4.0.0 | PDF parsing |
| `pdf2image` | ≥1.17.0 | PDF → image conversion |
| `python-pptx` | ≥0.6.23 | PowerPoint processing |
| `mammoth` | ≥1.6.0 | Word document processing |
| `pytesseract` | ≥0.3.10 | Tesseract OCR wrapper |

#### Computer Vision (Proctoring)

| Package | Version | Purpose |
|---------|---------|---------|
| `opencv-python` | ≥4.9.0 | Video frame processing |
| `dlib` | ≥19.24.0 | Face detection & landmarks |
| `deepface` | ≥0.0.79 | Face verification |
| `Pillow` | ≥10.0.0 | Image manipulation |
| `scikit-image` | ≥0.22.0 | Advanced image processing |

#### NLP & Content Extraction

| Package | Version | Purpose |
|---------|---------|---------|
| `spacy` | ≥3.7.0 | NLP (noun chunks, cache matching) |
| `beautifulsoup4` | ≥4.12.0 | HTML parsing |
| `readability-lxml` | ≥0.8.1 | Article text extraction |
| `lxml` | ≥5.0.0 | XML/HTML parser |
| `trafilatura` | ≥2.0.0 | Web content extraction |

#### Web Search (Free, No API Keys)

| Package | Version | Purpose |
|---------|---------|---------|
| `youtube-search-python` | ≥1.6.6 | YouTube video search |
| `youtube-transcript-api` | ≥0.6.0 | YouTube transcripts |
| `duckduckgo-search` | ≥4.1.1 | Web search (free) |
| `ddgs` | ≥9.10.0 | DuckDuckGo image search |

#### Audio Processing

| Package | Version | Purpose |
|---------|---------|---------|
| `simple-diarizer` | ≥0.0.13 | Speaker diarization |
| `pydub` | ≥0.25.1 | Audio processing |

#### Infrastructure

| Package | Version | Purpose |
|---------|---------|---------|
| `redis` | ≥5.0.0 | Caching client |
| `httpx` | ≥0.26.0 | Async HTTP |
| `requests` | ≥2.31.0 | Sync HTTP |
| `aiohttp` | ≥3.9.0 | Async HTTP client |
| `websockets` | ≥12.0 | WebSocket support |
| `python-dotenv` | ≥1.0.0 | Environment loading |
| `numpy` | ≥1.24.0 | Numerical computing |

---

## 36.3 Core Service Dependencies (41 packages)

### Source: `backend/core-service/requirements.txt`

#### Framework & ORM

| Package | Version | Purpose |
|---------|---------|---------|
| `flask` | 3.0.0 | Web framework |
| `flask-cors` | 4.0.0 | CORS middleware |
| `flask-sqlalchemy` | 3.1.1 | SQLAlchemy integration |
| `flask-migrate` | 4.0.5 | Alembic migrations |
| `sqlalchemy` | 2.0.23 | ORM |
| `alembic` | 1.13.0 | Migration tool |

#### Database & Storage

| Package | Version | Purpose |
|---------|---------|---------|
| `psycopg[binary]` | ≥3.1.0 | PostgreSQL driver |
| `redis` | 5.0.1 | Redis caching |
| `kafka-python` | 2.0.2 | Kafka producer |
| `boto3` | ≥1.34.0 | AWS S3 SDK |

#### Auth & Security

| Package | Version | Purpose |
|---------|---------|---------|
| `pyjwt` | 2.8.0 | JWT tokens |
| `werkzeug` | 3.0.1 | Password hashing |

#### Integrations

| Package | Version | Purpose |
|---------|---------|---------|
| `livekit-api` | ≥0.7.0 | LiveKit room management |

#### Dev & Testing

| Package | Version | Purpose |
|---------|---------|---------|
| `pytest` | 7.4.3 | Testing |
| `pytest-flask` | 1.3.0 | Flask test fixtures |
| `pytest-cov` | 4.1.0 | Coverage reporting |
| `black` | 23.12.0 | Code formatting |
| `flake8` | 6.1.0 | Linting |

---

## 36.4 Frontend Dependencies (key packages)

### Source: `frontend/package.json`

| Category | Key Packages |
|----------|-------------|
| **Framework** | next 14, react 18, typescript |
| **Auth** | next-auth |
| **State** | zustand |
| **UI** | tailwindcss, lucide-react, framer-motion |
| **Video** | @livekit/components-react, livekit-client |
| **3D** | three, @react-three/fiber, @react-three/drei |
| **Markdown** | react-markdown, remark-math, rehype-katex |
| **Charts** | recharts |
| **HTTP** | axios |

---

## 36.5 System Dependencies

| Dependency | Install Method | Required By |
|-----------|---------------|------------|
| **Tesseract OCR** | `brew install tesseract` | pytesseract |
| **poppler** | `brew install poppler` | pdf2image |
| **ffmpeg** | `brew install ffmpeg` | openai-whisper |
| **portaudio** | `brew install portaudio` | pyaudio (optional) |
| **Docker** | Docker Desktop | All infrastructure |
| **mkcert** | `brew install mkcert` | LAN TLS certificates |
