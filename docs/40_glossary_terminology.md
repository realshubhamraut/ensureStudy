# Page 40: Glossary, Acronyms & Technical Terminology

---

## 40.1 Platform-Specific Terms

| Term | Full Name | Definition |
|------|-----------|-----------|
| **ABCR** | Assess-Build-Challenge-Reflect | 4-phase tutoring cycle used by the Tutor Agent |
| **TAL** | Teaching Adaptation Level | 5-level student proficiency scale (1=Beginner to 5=Expert) |
| **MCP** | Model Context Protocol | Protocol for agents to access contextual tools and data |
| **AutoOEP** | Automated Online Exam Proctoring | ML classification pipeline for proctoring |
| **ensureStudy** | — | The platform name — AI-powered adaptive learning system |
| **Core Service** | — | Flask backend handling auth, CRUD, and business logic |
| **AI Service** | — | FastAPI backend handling all AI/ML operations |
| **Orchestrator** | Agent Orchestrator | Routes incoming tasks to the appropriate specialized agent |
| **BaseAgent** | — | Abstract base class for all AI agents |
| **ABCR Cache** | — | Redis cache storing tutoring cycle state per student per topic |

---

## 40.2 AI/ML Terms

| Term | Definition |
|------|-----------|
| **RAG** | Retrieval-Augmented Generation — augmenting LLM responses with retrieved context chunks |
| **Embedding** | Dense vector representation of text, used for semantic similarity |
| **Vector Search** | Finding similar items by comparing embedding vectors (cosine similarity) |
| **Chunking** | Splitting documents into smaller pieces (~500 chars) for embedding |
| **Fine-tuning** | Adapting a pre-trained model to a specific task |
| **Inference** | Running a trained model on new data to get predictions |
| **LSTM** | Long Short-Term Memory — RNN variant for sequential data |
| **LightGBM** | Light Gradient Boosting Machine — fast tree-based classifier |
| **XGBoost** | Extreme Gradient Boosting — ensemble tree classifier |
| **YOLO** | You Only Look Once — real-time object detection model |
| **dlib** | C++ ML library — face detection and 68-point landmark detection |
| **MediaPipe** | Google's ML framework — pose estimation, hand tracking |
| **Whisper** | OpenAI's speech-to-text model |
| **Gemini** | Google's multimodal LLM |
| **Groq** | Cloud inference provider with fast hardware (LPU) |
| **Ollama** | Local LLM hosting tool |
| **TTFB** | Time To First Byte — latency before first response chunk |
| **SSE** | Server-Sent Events — one-way server → client streaming |
| **HaGRID** | Hand Gesture Recognition Image Dataset — 552K images, 18 categories |

---

## 40.3 Architecture Terms

| Term | Definition |
|------|-----------|
| **Microservices** | Architecture where each service runs independently |
| **Polyglot Persistence** | Using different databases for different data types |
| **Event Streaming** | Asynchronous message passing via Kafka topics |
| **ETL** | Extract-Transform-Load — batch data processing pipeline |
| **JDBC** | Java Database Connectivity — database access protocol (used by PySpark) |
| **CRUD** | Create-Read-Update-Delete — basic data operations |
| **Blueprint** | Flask's modular route grouping mechanism |
| **Router** | FastAPI's route grouping mechanism (equivalent to Flask Blueprint) |
| **Middleware** | Request/response interceptor (logging, auth, CORS) |
| **Factory Pattern** | Application factory — `create_app()` function |
| **Lazy Loading** | Deferring object creation until first use |
| **Connection Pooling** | Reusing database connections to reduce overhead |

---

## 40.4 Database Terms

| Term | Definition |
|------|-----------|
| **PostgreSQL** | Relational database — primary data store (users, classrooms, progress) |
| **Qdrant** | Vector database — stores embeddings for semantic search |
| **Redis** | In-memory key-value store — caching and session state |
| **MongoDB** | Document database — meeting transcripts and unstructured data |
| **Cassandra** | Wide-column store — time-series meeting analytics |
| **SQLAlchemy** | Python ORM for PostgreSQL |
| **Alembic** | Database migration tool for SQLAlchemy |
| **Collection** | Qdrant's equivalent of a table — groups related vectors |
| **Cosine Similarity** | Metric measuring angle between vectors (1.0 = identical) |

---

## 40.5 Infrastructure Terms

| Term | Definition |
|------|-----------|
| **Docker Compose** | Tool for defining multi-container applications |
| **Docker Volume** | Persistent storage mounted into containers |
| **ghcr.io** | GitHub Container Registry — Docker image hosting |
| **Gunicorn** | Production WSGI server for Flask |
| **Uvicorn** | Production ASGI server for FastAPI |
| **Nginx** | Reverse proxy and SSL termination |
| **mkcert** | Tool for generating locally-trusted TLS certificates |
| **AWS RDS** | Amazon Relational Database Service (managed PostgreSQL) |
| **AWS S3** | Amazon Simple Storage Service (file/object storage) |
| **MinIO** | S3-compatible object storage (development replacement) |
| **LiveKit** | Open-source WebRTC platform for video conferencing |
| **GitHub Actions** | CI/CD automation platform |
| **Codecov** | Code coverage reporting service |

---

## 40.6 Frontend Terms

| Term | Definition |
|------|-----------|
| **Next.js** | React framework with SSR and App Router |
| **App Router** | Next.js 14 file-based routing system |
| **Route Group** | Next.js `(group)` directories for organizing without affecting URL |
| **NextAuth** | Authentication library for Next.js (session management) |
| **Zustand** | Lightweight React state management library |
| **TailwindCSS** | Utility-first CSS framework |
| **Lucide** | Icon library (replacing emoji with professional icons) |
| **Recharts** | React charting library |
| **Three.js** | 3D graphics library for WebGL |
| **KaTeX** | Fast LaTeX math rendering library |
| **Framer Motion** | React animation library |

---

## 40.7 Proctoring-Specific Terms

| Term | Definition |
|------|-----------|
| **Integrity Score** | 0-100 score measuring exam fairness (100 = no suspicious behavior) |
| **Flag** | Specific suspicious behavior detected (e.g., "face_not_detected") |
| **EAR** | Eye Aspect Ratio — metric for blink detection |
| **MAR** | Mouth Aspect Ratio — metric for mouth openness |
| **Head Pose** | 3D rotation angles (yaw, pitch, roll) of the head |
| **Gaze Direction** | Estimated eye gaze vector relative to screen |
| **Tab Switch** | Browser tab/window change during exam |
| **Static Classifier** | Per-frame behavior model (LightGBM) |
| **Temporal Predictor** | Sequence-based behavior model (LSTM over 30 frames) |
| **Face Verification** | Confirming that the current person matches the registered student |

---

## 40.8 Complete Documentation Map (Pages 1-40)

| Batch | Pages | Focus Area |
|-------|-------|------------|
| **1** (1-5) | Architecture & Agent Core | Overview, architecture, multi-agent, tutor, RAG |
| **2** (6-10) | Specialized Agents | Research, curriculum, learning, documents, assessments |
| **3** (11-15) | Backend & Frontend | Core Service, routes, AI Service, databases, frontend |
| **4** (16-20) | ML & Streaming | Proctoring, soft skills, meetings, Kafka, ML pipeline |
| **5** (21-25) | Operations | Infrastructure, security, LLM strategy, observability, production |
| **6** (26-30) | Extended | ETL, service catalog, CI/CD, env config, scripts |
| **7** (31-35) | Deep Reference | Frontend pages, Core API, AI API, data models, flow sequences |
| **8** (36-40) | Patterns & Glossary | Dependencies, caching, error handling, components, glossary |

---

*This documentation was generated through comprehensive analysis of the ensureStudy codebase — covering 500+ source files, 89 AI services, 40+ database models, 51 frontend pages, and 12 Docker services.*
