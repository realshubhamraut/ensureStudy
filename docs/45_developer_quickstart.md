# Page 45: Developer Quick-Start Guide

---

## 45.1 Prerequisites

| Requirement | Version | Install |
|------------|---------|---------|
| Docker Desktop | Latest | [docker.com](https://docker.com) |
| Node.js | 20+ | `brew install node` |
| Python | 3.11+ | `brew install python@3.11` |
| Git | Latest | `brew install git` |
| mkcert | Latest | `brew install mkcert` (optional, for HTTPS) |
| Tesseract | Latest | `brew install tesseract` (for OCR) |
| ffmpeg | Latest | `brew install ffmpeg` (for Whisper) |

---

## 45.2 Clone & Setup

```bash
# 1. Clone repository
git clone https://github.com/realshubhamraut/ensureStudy.git
cd ensureStudy

# 2. Copy environment file
cp .env.production.example .env
# Edit .env with your API keys (OpenAI, Gemini, Groq, etc.)
```

---

## 45.3 Option A: Docker Compose (Recommended)

```bash
# Start all 12 services
make up

# This runs:
# 1. docker-compose up -d          (start containers)
# 2. wait for services to be healthy
# 3. make health-check              (verify everything)
```

### Verify Services

```bash
make health-check
# ✓ PostgreSQL: accepting connections
# ✓ Redis: PONG
# ✓ Qdrant: healthy
# ✓ Kafka: topics available
```

### Access Points

| Service | URL |
|---------|-----|
| Frontend | http://localhost:3000 |
| Core API | http://localhost:8000 |
| AI Service | http://localhost:8001 |
| Kafka UI | http://localhost:8080 |
| MLflow | http://localhost:5000 |
| MinIO Console | http://localhost:9101 |
| Qdrant Dashboard | http://localhost:6333/dashboard |

---

## 45.4 Option B: Local Development (Hybrid)

```bash
# 1. Start infrastructure in Docker
docker-compose up -d postgres redis qdrant zookeeper kafka mongodb cassandra minio

# Wait for services
sleep 15

# 2. Setup Python virtual environments
cd backend/core-service
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 3. Initialize database
flask db upgrade

# 4. Seed database
cd ../..
python seed_database.py
python seed_progress_data.py

# 5. Start Core Service (Terminal 1)
cd backend/core-service
flask run --port 8000

# 6. Start AI Service (Terminal 2)
cd backend/ai-service
pip install -r requirements.txt
uvicorn app.main:app --port 8001 --reload

# 7. Start Frontend (Terminal 3)
cd frontend
npm install
npm run dev
```

---

## 45.5 First Steps After Setup

### 1. Create Admin Account

Access http://localhost:3000/auth/signup and register.

### 2. Create a Classroom (Teacher)

- Navigate to `/teacher/classrooms`
- Click "Create Classroom"
- Upload a syllabus PDF
- Share the join code with students

### 3. Upload Materials

- In the classroom, upload study PDFs
- The AI service automatically processes and indexes them
- Materials become available for RAG queries in the tutor

### 4. Chat with the Tutor

- Navigate to `/chat`
- Select a classroom context
- Ask questions — the tutor uses RAG + ABCR cycle

### 5. Take an Assessment

- Teacher creates assessment from `/teacher/assessments`
- Student takes it at `/assessments/take/[id]`
- Proctoring activates automatically if webcam is available

---

## 45.6 Common Commands

```bash
# Development
make dev-frontend          # npm run dev
make dev-core-service      # flask run --port 8000
make dev-ai-service        # uvicorn --reload --port 8001

# Database
make db-init               # flask db upgrade
make load-docs             # Load sample documents into Qdrant

# Testing
make test                  # pytest all services
make test-ml               # ML model tests

# Kafka
make kafka-topics          # Create required topics

# ML Training
make train-moderation      # Train content moderation model
make train-difficulty      # Train difficulty predictor

# Cleanup
make clean                 # docker-compose down -v + cleanup
```

---

## 45.7 Troubleshooting

| Problem | Solution |
|---------|----------|
| `python: command not found` | Use `python3` instead |
| `No module named 'cv2'` | `pip install opencv-python` |
| Qdrant connection refused | Wait for Docker healthcheck; `docker-compose up -d qdrant` |
| Kafka timeout | Ensure Zookeeper is running first |
| `OPENAI_API_KEY` error | Set key in `.env`; restart service |
| Port already in use | `lsof -i :8000` then `kill <PID>` |
| PostgreSQL auth failed | Check `DATABASE_URL` in `.env` matches docker-compose |
| Frontend can't reach API | API URLs are auto-detected; check both services are running |
| OCR failing | Install tesseract: `brew install tesseract` |
| Whisper import error | Install ffmpeg: `brew install ffmpeg` |

---

## 45.8 Project Structure Quick Reference

```mermaid
flowchart TB
    subgraph MAIN["Project Structure Quick Reference "]
        direction TB
        N0["ensureStudy/"]
        N1["frontend/              # Next.js 14 (TypeScript, TailwindCSS)"]
        N2["app/               # 51 pages across 5 route groups"]
        N3["components/        # 53+ reusable components"]
        N4["backend/"]
        N5["core-service/      # Flask API (29 routes, 40+ models)"]
        N6["ai-service/        # FastAPI AI (27 routers, 89 services)"]
        N7["kafka/             # Kafka configuration"]
        N8["data-pipelines/    # PySpark ETL + Streaming"]
        N9["ml/                    # ML training scripts + notebooks"]
        N10["models/                # Pre-trained model files"]
        N11["datasets/              # Training datasets"]
        N12["docs/                  # This documentation (45 pages)"]
        N13["scripts/               # Migration & demo scripts"]
        N14["docker-compose.yml     # Development (12 services)"]
    end

    style MAIN fill:#3b82f6,color:#fff
```

---

## 45.9 Complete Documentation Index (Pages 1-45)

| Batch | Pages | Focus |
|-------|-------|-------|
| 1 (1-5) | Architecture & Agent Core | Overview, architecture, multi-agent, tutor, RAG |
| 2 (6-10) | Specialized Agents | Research, curriculum, learning, documents, assessments |
| 3 (11-15) | Backend & Frontend | Core Service, routes, AI Service, databases, frontend |
| 4 (16-20) | ML & Streaming | Proctoring, soft skills, meetings, Kafka, ML pipeline |
| 5 (21-25) | Operations | Infrastructure, security, LLM strategy, observability, production |
| 6 (26-30) | Extended | ETL, service catalog, CI/CD, env config, scripts |
| 7 (31-35) | Deep Reference | Frontend pages, Core API, AI API, data models, flow sequences |
| 8 (36-40) | Patterns & Glossary | Dependencies, caching, error handling, components, glossary |
| 9 (41-45) | Advanced Topics | Pre-trained models, Dockerfiles, spaced repetition, gamification, quick-start |

---

*ensureStudy documentation — 45 pages covering 500+ source files, 89 AI services, 40+ database models, 51 frontend pages, 16 pre-trained ML models, and 12 Docker services.*
