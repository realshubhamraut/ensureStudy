# Page 30: Scripts, Utilities & Developer Tooling

---

## 30.1 Overview

The project root contains **test scripts, seed scripts, migration utilities, and demo scripts** that support development, testing, and onboarding.

---

## 30.2 Seed Scripts

### `seed_database.py`

Seeds PostgreSQL with initial data for development and demos:
- Creates default admin, teacher, and student users
- Sets up sample organizations/schools
- Creates demo classrooms with join codes
- Populates subject → topic → subtopic hierarchies

### `seed_progress_data.py`

Seeds student progress records for analytics testing:
- Generates simulated assessment scores across subjects
- Creates progress records with varying confidence levels
- Populates leaderboard with realistic point distributions
- Enables testing of dashboards and ML models

---

## 30.3 Test Scripts (14 files)

| Script | Lines | Test Target |
|--------|-------|-------------|
| `test_agentic_crawl.py` | — | Web crawling agent with URL following |
| `test_cache_api.py` | — | Redis cache API integration |
| `test_cache.py` | — | Redis caching unit tests |
| `test_chunk_only.py` | — | Text chunking (isolated) |
| `test_chunking.py` | — | Advanced chunking strategies |
| `test_full_pipeline.py` | — | End-to-end document → RAG test |
| `test_groq_classifier.py` | — | Groq LLM classification accuracy |
| `test_learning_agent_standalone.py` | — | Learning agent without dependencies |
| `test_ocr_model.py` | — | OCR model accuracy benchmarks |
| `test_qdrant.py` | — | Qdrant CRUD and search operations |
| `test_subject_classifier.py` | — | Subject classification accuracy |
| `test_topic_chaining.py` | — | Topic dependency graph building |
| `test_worker6.py` | — | Kafka worker processing |
| `test_workers.py` | — | Multi-worker concurrent processing |

### pytest Configuration

```ini
# pytest.ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
```

---

## 30.4 Migration & Demo Scripts

### Source: `scripts/` (5 files)

| Script | Purpose |
|--------|---------|
| `demo_session_chaining.py` | Demonstrates session state chaining across tutor conversations |
| `demo_session_intelligence.py` | Shows intelligent session management with context |
| `download_softskills_datasets.py` | Downloads HaGRID gesture dataset and other soft skills training data |
| `migrate_session_intelligence.py` | Database migration for session intelligence tables |
| `migrate_softskills.py` | Database migration for soft skills evaluation tables |

---

## 30.5 Build & Utility Files

### `Makefile` — 14 Targets

```makefile
make up              # Start Docker Compose + health check
make down            # Stop all containers
make logs            # Tail Docker logs
make health-check    # Check PostgreSQL, Redis, Qdrant, Kafka
make db-init         # Run Flask-Migrate upgrades
make load-docs       # Load sample documents into Qdrant
make test            # Run pytest for core, AI, Kafka
make test-ml         # Run ML model tests
make dev-frontend    # npm run dev
make dev-ai-service  # uvicorn --reload
make dev-core-service # flask run
make clean           # docker-compose down -v + purge caches
make kafka-topics    # Create 5 Kafka topics
make dashboards      # Start Streamlit dashboards
make train-moderation # Train content moderation model
make train-difficulty # Train difficulty prediction model
make run-etl         # Run PySpark ETL pipeline
```

### `run-local.sh`

Local development startup script:
1. Starts infrastructure containers (PostgreSQL, Redis, Qdrant, Kafka, MongoDB, Cassandra, MinIO)
2. Waits for health checks
3. Launches Core Service, AI Service, and Frontend in background

### `run-lan.sh`

LAN access script:
1. Uses mkcert-generated TLS certificates
2. Binds to LAN IP instead of localhost
3. Enables testing on mobile devices with valid HTTPS

---

## 30.6 Mermaid Diagram Tooling

| File | Purpose |
|------|---------|
| `mermaid.lua` | Pandoc Lua filter for Mermaid diagram rendering |
| `mermaid-filter.err` | Error log from Mermaid rendering |

Used to convert Mermaid diagrams in Markdown documentation to images when generating PDF/HTML output via Pandoc.

---

## 30.7 Proctoring Resources

### `proctor-requirements.txt`

Dependencies specific to the proctoring system:
- OpenCV, MediaPipe, dlib, DeepFace
- Ultralytics (YOLO), LightGBM
- NumPy, Pillow, imutils

### `proctoring_resources/`

Contains training data, test images, and model evaluation assets for the proctoring detectors.

---

## 30.8 Extended Documentation Index (Pages 1-30)

| Batch | Pages | Topics |
|-------|-------|--------|
| **Batch 1** (1-5) | Project overview, system architecture, multi-agent system, tutor agent, RAG pipeline |
| **Batch 2** (6-10) | Research/web agents, curriculum agent, learning agent, document processing, assessment agents |
| **Batch 3** (11-15) | Core Service (Flask), routes/auth, AI Service (FastAPI), databases (5-db), frontend (Next.js) |
| **Batch 4** (16-20) | Proctoring, soft skills, meetings, Kafka streaming, ML pipeline |
| **Batch 5** (21-25) | Infrastructure/Docker, security, LLM strategy, observability, production readiness |
| **Batch 6** (26-30) | Data pipelines/ETL, AI services catalog, CI/CD, environment config, scripts/tooling |

### Complete File Listing

```mermaid
flowchart TB
    subgraph MAIN["Overview"]
        direction TB
        N0["docs/"]
        N1["00_documentation_plan.md"]
        N2["01_project_overview.md"]
        N3["02_system_architecture.md"]
        N4["03_multi_agent_system.md"]
        N5["04_tutor_agent.md"]
        N6["05_rag_pipeline.md"]
        N7["06_research_web_enrichment_agents.md"]
        N8["07_curriculum_agent.md"]
        N9["08_learning_agent.md"]
        N10["09_document_processing_pipeline.md"]
        N11["10_notes_assessment_question_agents.md"]
        N12["11_core_service_architecture.md"]
        N13["12_core_service_routes.md"]
        N14["13_ai_service_api.md"]
    end

    style MAIN fill:#3b82f6,color:#fff
```
