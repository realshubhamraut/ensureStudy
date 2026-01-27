# ensureStudy

An AI-first learning platform combining intelligent multi-agent tutoring, real-time proctoring, and soft skills evaluation. Built with LangGraph orchestration, RAG-powered conversations, Kafka streaming, PyTorch ML models, and PySpark data pipelines.

---

## Platform Overview

```mermaid
graph TB
    subgraph "Student Experience"
        Web[Web App] --> Chat[AI Tutor Chat]
        Web --> Class[Virtual Classrooms]
        Web --> Assess[Assessments]
        Web --> Soft[Soft Skills Practice]
    end
    
    subgraph "Multi-Agent System"
        Chat --> Orchestrator[Orchestrator Agent]
        Orchestrator --> Tutor[Tutor Agent]
        Orchestrator --> Research[Research Agent]
        Orchestrator --> Curriculum[Curriculum Agent]
        Assess --> Proctor[Proctoring System]
        Soft --> Vision[Computer Vision]
    end
    
    subgraph "Intelligence Layer"
        Tutor --> RAG[RAG Engine]
        Research --> WebSearch[Web Crawler]
        Curriculum --> Planner[Learning Paths]
        RAG --> Vector[(Vector DB)]
        Tutor --> LLM[Mistral LLM]
    end
```

---

## Key Features

| Feature | Description | Technology |
|---------|-------------|------------|
| Multi-Agent Tutoring | Orchestrated AI agents for learning, research, and content creation | LangGraph, Mistral, RAG |
| Smart Proctoring | Real-time exam monitoring with violation detection | YOLO, MediaPipe, PyTorch |
| Soft Skills | Fluency, grammar, eye contact, and posture analysis | Whisper, FaceMesh, NLP |
| Virtual Classrooms | Live meetings with recordings and transcripts | WebRTC, MongoDB |
| Learning Analytics | Progress tracking and personalized recommendations | PySpark, Cassandra |
| Document Processing | PDF/image ingestion with OCR and chunking | PyMuPDF, Qdrant |

---

## Multi-Agent Architecture

The platform uses a **Supervisor Pattern** orchestration where a central Orchestrator Agent routes requests to specialized sub-agents based on user intent.

### Agent System Overview

```mermaid
flowchart TB
    subgraph "User Request"
        Query[Student Query]
    end
    
    subgraph "Orchestrator Layer"
        Orch[Orchestrator Agent]
        Intent[Intent Classification]
    end
    
    subgraph "Specialized Agents"
        Tutor[Tutor Agent]
        Research[Research Agent]
        Curriculum[Curriculum Agent]
        Document[Document Agent]
        Notes[Notes Agent]
        Assessment[Assessment Agent]
    end
    
    subgraph "Support Services"
        Moderation[Content Moderation]
        WebEnrich[Web Enrichment]
    end
    
    Query --> Orch
    Orch --> Intent
    Intent --> |Learn| Tutor
    Intent --> |Research| Research
    Intent --> |Create| Curriculum
    Intent --> |Evaluate| Assessment
    
    Tutor --> Moderation
    Research --> WebEnrich
    Document --> Notes
```

---

## Agent Descriptions

### 1. Orchestrator Agent (Supervisor Pattern)

The central coordinator that receives all user queries and routes them to appropriate sub-agents.

**Capabilities:**
- Intent classification (Learn, Research, Create, Evaluate, Mixed)
- Multi-agent coordination and parallel execution
- Response synthesis from multiple agent outputs
- Session state management

**Intent Classification Flow:**

```mermaid
stateDiagram-v2
    [*] --> Analyze: User Query
    Analyze --> LEARN: "What is...", "Explain..."
    Analyze --> RESEARCH: "Find...", "Search..."
    Analyze --> CREATE: "Generate...", "Make..."
    Analyze --> EVALUATE: "Check...", "Assess..."
    
    LEARN --> TutorAgent
    RESEARCH --> ResearchAgent
    CREATE --> ContentGeneration
    EVALUATE --> AssessmentAgent
    
    TutorAgent --> Synthesize
    ResearchAgent --> Synthesize
    ContentGeneration --> Synthesize
    AssessmentAgent --> Synthesize
    
    Synthesize --> [*]: Final Response
```

---

### 2. Tutor Agent (ABCR + TAL + MCP)

The primary learning assistant with advanced context management.

**Core Components:**

| Component | Full Name | Function |
|-----------|-----------|----------|
| ABCR | Attention-Based Context Routing | Detects follow-up vs new topic queries |
| TAL | Topic Anchor Layer | Maintains topic continuity across turns |
| MCP | Memory Context Processor | Isolates web vs classroom content |

**Processing Flow:**

```mermaid
stateDiagram-v2
    [*] --> Receive: User message
    Receive --> Moderate: Content check
    Moderate --> ABCR: Classify query type
    
    ABCR --> FollowUp: Follow-up detected
    ABCR --> NewTopic: New topic
    
    FollowUp --> KeepAnchor: Use existing context
    NewTopic --> TAL: Extract and anchor topic
    
    KeepAnchor --> Retrieve
    TAL --> Retrieve: Vector search with anchor
    
    Retrieve --> MCP: Apply context isolation
    MCP --> Generate: Build prompt with filtered context
    Generate --> [*]: Return response
```

**Features:**
- Session-aware conversation memory
- Topic anchoring for multi-turn coherence
- Web content isolation (MCP rules)
- Confidence scoring for answers
- Source attribution with page numbers

---

### 3. Research Agent (Web + PDF + YouTube)

Discovers and indexes educational content from multiple sources.

**Capabilities:**
- Web search for educational articles
- PDF discovery and download
- YouTube video search
- Automatic content indexing into Qdrant

**Research Pipeline:**

```mermaid
flowchart LR
    Query[User Query] --> Analyze[Analyze Intent]
    
    Analyze --> Web[Web Search]
    Analyze --> PDF[PDF Search]
    Analyze --> YT[YouTube Search]
    
    Web --> Articles[Article Results]
    PDF --> Download[Download PDFs]
    YT --> Videos[Video Results]
    
    Download --> Index[Index in Qdrant]
    
    Articles --> Compile[Compile Results]
    Index --> Compile
    Videos --> Compile
    
    Compile --> Summary[Research Summary]
```

---

### 4. Curriculum Agent (Personalized Learning Paths)

Creates adaptive learning schedules from syllabus documents.

**Pipeline:**

```mermaid
flowchart LR
    Syllabus[Syllabus Topics] --> Dependencies[Analyze Dependencies]
    Dependencies --> Knowledge[Assess Knowledge]
    Knowledge --> Order[Topological Sort]
    Order --> Schedule[Generate Schedule]
    Schedule --> Milestones[Add Milestones]
    Milestones --> Curriculum[Final Curriculum]
```

**Features:**
- Topic dependency analysis using LLM
- Prerequisite chain detection
- Adaptive scheduling based on hours/day
- Milestone generation for progress tracking
- Integration with knowledge assessment service

---

### 5. Document Processing Agent (7-Stage Pipeline)

Ingests and indexes documents for RAG retrieval.

**Pipeline Stages:**

```mermaid
flowchart LR
    Upload[Document] --> Validate[1. Validate]
    Validate --> Preprocess[2. Preprocess]
    Preprocess --> OCR[3. OCR Extract]
    OCR --> Chunk[4. Semantic Chunking]
    Chunk --> Embed[5. Generate Embeddings]
    Embed --> Index[6. Index in Qdrant]
    Index --> Complete[7. Complete]
```

**Supported Formats:**
- PDF (text and scanned with OCR)
- Images (PNG, JPG with OCR)
- Word documents (DOCX)
- PowerPoint (PPTX)

---

### 6. Notes Agent

Generates study notes from classroom materials.

**Output Types:**
- Summary notes
- Key concepts extraction
- Q&A generation
- Flashcard creation

---

### 7. Assessment Agent

Handles evaluation and grading tasks.

**Capabilities:**
- Question generation from content
- Answer evaluation with rubrics
- Feedback generation
- Score calculation

---

### 8. Web Enrichment Agent

Enhances responses with web content.

**Features:**
- Article crawling and summarization
- Image search (Brave API)
- YouTube video discovery
- Trust score calculation for sources

---

## Base Agent Architecture

All agents inherit from `BaseAgent` with Model Context Protocol (MCP) support:

**Standard Agent Interface:**
- `execute(input_data)` - Main execution method
- `validate_input()` - Input validation
- `format_output()` - Standardized MCP output format
- `log_execution()` - Monitoring and logging

**Agent Contexts:**
- TUTOR - Q&A and explanations
- STUDY_PLANNER - Learning paths
- ASSESSMENT - Evaluation
- NOTES_GENERATOR - Content creation
- MODERATION - Safety checks
- SCRAPER - Web content

---

## LangGraph State Machines

Each agent uses LangGraph's StateGraph for workflow orchestration:

```mermaid
graph LR
    subgraph "LangGraph Workflow"
        Start[START] --> N1[Node 1]
        N1 --> Router{Conditional}
        Router --> |Path A| N2[Node 2]
        Router --> |Path B| N3[Node 3]
        N2 --> End[END]
        N3 --> End
    end
```

**Benefits:**
- Visual workflow definition
- Conditional routing
- Parallel execution
- Checkpointing and recovery
- State persistence

---

## System Architecture

```mermaid
flowchart LR
    subgraph Frontend
        Next[Next.js 14]
    end
    
    subgraph Backend
        Core[Core Service<br/>Flask:8000]
        AI[AI Service<br/>FastAPI:8001]
    end
    
    subgraph Databases
        PG[(PostgreSQL)]
        QD[(Qdrant)]
        RD[(Redis)]
        MG[(MongoDB)]
        CS[(Cassandra)]
    end
    
    subgraph ML
        Models[PyTorch Models]
    end
    
    subgraph Streaming
        Kafka[Apache Kafka]
        Spark[PySpark]
    end
    
    Next --> Core
    Next --> AI
    Core --> PG
    Core --> RD
    AI --> QD
    AI --> RD
    AI --> Models
    Core --> Kafka
    Kafka --> Spark
    Spark --> CS
```

---

## Technology Stack

### Application Layer

| Component | Technology | Purpose |
|-----------|------------|---------|
| Frontend | Next.js 14, TypeScript, TailwindCSS | Web application |
| Core API | Flask, SQLAlchemy, JWT | Auth, users, classrooms |
| AI API | FastAPI, LangGraph, LangChain | Agents, RAG, inference |
| Real-time | WebSocket, WebRTC | Live features |

### AI and ML Layer

| Component | Technology | Purpose |
|-----------|------------|---------|
| LLM | Mistral 7B via HuggingFace | Text generation |
| Embeddings | all-MiniLM-L6-v2 | Semantic search |
| Object Detection | YOLOv11 | Proctoring objects |
| Face Analysis | MediaPipe FaceMesh | Gaze, expressions |
| Agent Framework | LangGraph | Workflow orchestration |

### Data Layer

| Database | Type | Use Case |
|----------|------|----------|
| PostgreSQL | Relational | Users, classrooms, assessments |
| Qdrant | Vector | Document embeddings, RAG |
| Redis | Key-Value | Sessions, cache, rate limits |
| MongoDB | Document | Transcripts, logs, reports |
| Cassandra | Time-Series | Analytics, event streams |
| Kafka | Message Queue | Event streaming |

---

## Proctoring System

Real-time monitoring during assessments:

```mermaid
flowchart LR
    Camera[Webcam] --> Frames[Frame Capture]
    Frames --> WS[WebSocket]
    
    WS --> YOLO[YOLO Detector]
    WS --> Face[FaceMesh]
    
    YOLO --> Phone[Phone Detection]
    YOLO --> Person[Person Count]
    
    Face --> Gaze[Gaze Tracking]
    Face --> Head[Head Pose]
    
    Phone --> Score[Integrity Score]
    Person --> Score
    Gaze --> Score
    Head --> Score
    
    Score --> Report[Session Report]
```

| Detection | Model | Threshold |
|-----------|-------|-----------|
| Multiple faces | YOLO person class | > 1 person |
| Mobile phone | YOLO cell phone | confidence > 0.5 |
| Gaze deviation | Eye landmarks | > 30 degrees |
| Face absence | FaceLandmarker | > 3 seconds |

---

## Soft Skills Evaluation

| Metric | Weight | Analysis Method |
|--------|--------|-----------------|
| Fluency | 25% | Speech rate, filler words, pauses |
| Grammar | 20% | Language tool analysis |
| Vocabulary | 15% | Type-token ratio, word diversity |
| Eye Contact | 15% | Iris tracking vs camera |
| Expression | 10% | Facial emotion detection |
| Posture | 10% | Body position stability |
| Confidence | 5% | Combined delivery metrics |

---

## Data Pipeline

```mermaid
flowchart TB
    subgraph Sources
        App[Application Events]
        Meet[Meeting Events]
        Learn[Learning Progress]
    end
    
    subgraph Kafka
        Topics[Kafka Topics]
    end
    
    subgraph Processing
        Stream[Spark Streaming]
        Batch[Spark Batch]
    end
    
    subgraph Storage
        Cassandra[(Cassandra)]
        Analytics[(Analytics DB)]
    end
    
    App --> Topics
    Meet --> Topics
    Learn --> Topics
    
    Topics --> Stream
    Stream --> Cassandra
    
    Cassandra --> Batch
    Batch --> Analytics
```

---

## Quick Start

Prerequisites: Docker, Node.js 20+, Python 3.11+

| Step | Action |
|------|--------|
| 1 | Copy `.env.example` to `.env` |
| 2 | Add HuggingFace API key |
| 3 | Configure database passwords |
| 4 | Run `docker-compose up -d` |
| 5 | Run `make dev` |

| Service | Port | URL |
|---------|------|-----|
| Frontend | 3000 | http://localhost:3000 |
| Core API | 8000 | http://localhost:8000 |
| AI API | 8001 | http://localhost:8001 |
| Qdrant | 6333 | http://localhost:6333 |

---

## Documentation

| Document | Contents |
|----------|----------|
| architecture.md | System design and patterns |
| ai-service.md | Tutor agent and RAG pipeline |
| agent-possibilities.md | Future agent capabilities |
| proctoring.md | Computer vision detection |
| softskills.md | Communication evaluation |
| data-pipelines.md | Kafka and Spark processing |
| databases.md | Schema definitions |
| api-reference.md | Complete API documentation |

---

## License

MIT

---

## Contributing

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

For major changes, please open an issue first.
