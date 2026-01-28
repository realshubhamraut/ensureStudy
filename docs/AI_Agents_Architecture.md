# ensureStudy — AI Agents Architecture & Automation Analysis

 January 2026  
**Project:** ensureStudy — AI-First Learning Platform

---

## Executive Summary

ensureStudy implements a sophisticated **multi-agent AI orchestration system** using **LangGraph StateGraph** patterns. The platform deploys 8 specialized AI agents that work collaboratively to automate critical educational workflows, eliminating the need for human intervention in tutoring, assessment, curriculum planning, research, document processing, and exam proctoring.

---

## 1. Architecture Overview

### 1.1 Supervisor Pattern Implementation

The system follows the **LangGraph Supervisor Pattern** where a central **Orchestrator Agent** acts as a meta-controller, routing requests to specialized sub-agents based on user intent analysis.

**Architecture Flow:**

- ORCHESTRATOR (Intent Analysis & Route Decision)
  - TUTOR AGENT --> ASSESSMENT AGENT
  - RESEARCH AGENT --> CURRICULUM AGENT
  - CONTENT AGENT --> DOCUMENT AGENT

### 1.2 Intent Classification System

The Orchestrator classifies user queries into four primary intent categories using keyword-based scoring:

| Intent | Description | Routed To |
|--------|-------------|-----------|
| **LEARN** | Q&A, explanations, concept clarification | Tutor Agent |
| **RESEARCH** | Find content, PDFs, educational resources | Research Agent |
| **CREATE** | Generate notes, quizzes, flashcards | Content Agent |
| **EVALUATE** | Grade answers, provide feedback | Evaluation Agent |

---

## 2. The Eight AI Agents

### 2.1 Orchestrator Agent

**File:** `backend/ai-service/app/agents/orchestrator.py`

**Human Role Replaced:** Academic Coordinator / Dispatcher

**Key Capabilities:**

- **Intent Analysis:** Classifies user queries using keyword matching and confidence scoring
- **Topic Extraction:** Identifies the main subject matter from natural language queries
- **Agent Selection:** Determines which sub-agents to invoke based on intent
- **Response Synthesis:** Aggregates outputs from multiple agents into coherent responses

**Technical Implementation:**

```python
class Intent(str, Enum):
    LEARN = "learn"          # → TutorAgent
    RESEARCH = "research"    # → ResearchAgent
    CREATE = "create"        # → ContentAgent
    EVALUATE = "evaluate"    # → EvaluationAgent
    MIXED = "mixed"          # → Multiple Agents
```

---

### 2.2 Tutor Agent

**File:** `backend/ai-service/app/agents/tutor_agent.py`

**Human Role Replaced:** Personal Tutor / Subject Matter Expert

**Key Capabilities:**

- **ABCR (Attention-Based Context Routing):** Detects whether a query is a follow-up or new topic
- **TAL (Topic Anchor Layer):** Maintains conversation continuity across sessions
- **MCP (Memory Context Processor):** Isolates web content from classroom content in RAG
- **Content Moderation:** Filters non-academic queries using ML classification

**Processing Flow:**

- Moderate Query
- Context Routing (ABCR/TAL)
- Retrieve with MCP
- Generate Response

**Session State Management:**

```python
_session_states: Dict[str, Dict] = {
    "turn_texts": [],           # Conversation history
    "last_abcr_decision": "",   # "related" or "new_topic"
    "consecutive_borderline": 0,
    "topic_anchor_id": None,
    "topic_anchor_title": None
}
```

---

### 2.3 Research Agent

**File:** `backend/ai-service/app/agents/research_agent.py`

**Human Role Replaced:** Research Assistant / Librarian

**Key Capabilities:**

- **Web Search:** Finds educational content via DuckDuckGo
- **PDF Discovery:** Searches and downloads academic PDFs
- **YouTube Integration:** Discovers relevant educational videos
- **Content Indexing:** Stores discovered content in Qdrant vector database

**Processing Pipeline:**

1. Analyze Query
2. Web Search
3. PDF Download (if PDF)
4. YouTube Search
5. Index Content

---

### 2.4 Assessment Agent

**File:** `backend/ai-service/app/agents/assessment_agent.py`

**Human Role Replaced:** Question Paper Setter / Examiner

**Key Capabilities:**

- **Adaptive MCQ Generation:** Creates questions based on weak topics
- **Difficulty Calibration:** Adjusts complexity (easy/medium/hard)
- **Explanation Generation:** Provides detailed answer explanations
- **Topic Coverage:** Ensures balanced coverage across subjects

**Difficulty Guidance:**

```python
difficulty_guidance = {
    "easy": "Basic recall and understanding questions.",
    "medium": "Application and analysis questions.",
    "hard": "Synthesis and evaluation - complex scenarios."
}
```

---

### 2.5 Curriculum Agent

**File:** `backend/ai-service/app/agents/curriculum_agent.py`

**Human Role Replaced:** Academic Counselor / Curriculum Planner

**Key Capabilities:**

- **Syllabus Processing:** Loads and parses extracted syllabus topics
- **Dependency Analysis:** Uses LLM to identify topic prerequisites
- **Knowledge Assessment:** Evaluates student's current mastery levels
- **Learning Path Generation:** Creates optimized topic sequences
- **Schedule Creation:** Generates daily/weekly study schedules with milestones

**Processing Flow:**

1. Load Topics
2. Analyze Dependencies
3. Knowledge Assessment
4. Build Path
5. Schedule Generator

**Data Structures:**

```python
@dataclass
class CurriculumTopic:
    id: str
    name: str
    difficulty: str          # "easy", "medium", "hard"
    estimated_hours: float
    prerequisites: List[str]  # Topic IDs
    subtopics: List[str]
    order: int               # Position in learning path
```

---

### 2.6 Study Planner Agent

**File:** `backend/ai-service/app/agents/study_planner.py`

**Human Role Replaced:** Study Coach / Academic Advisor

**Key Capabilities:**

- **Topic Prioritization:** Ranks topics by weakness scores
- **Resource Allocation:** Distributes study hours optimally
- **Milestone Setting:** Creates checkpoints for progress tracking
- **Personalized Recommendations:** Provides study tips based on patterns

**Output Structure:**

```python
study_plan = {
    "daily_schedule": [...],
    "recommendations": [
        "Focus on high priority topics first",
        "Take regular breaks every 45 minutes",
        "Review previous day's material each morning"
    ],
    "milestones": [
        {"day": 3, "goal": "Complete X basics"}
    ]
}
```

---

### 2.7 Document Processing Agent

**File:** `backend/ai-service/app/agents/document_agent.py`

**Human Role Replaced:** Document Processor / Data Entry Specialist

**Key Capabilities:**

- **Multi-Format Support:** PDF, DOCX, PPTX processing
- **OCR Integration:** Extracts text from images and scanned documents
- **Intelligent Chunking:** Splits documents for optimal RAG retrieval
- **Vector Embedding:** Generates embeddings for semantic search
- **Qdrant Indexing:** Stores processed content for retrieval

**7-Stage Pipeline:**

```
Stage 1: VALIDATE    → Check file exists and is processable
Stage 2: PREPROCESS  → Convert to standard format
Stage 3: OCR         → Extract text from images
Stage 4: CHUNK       → Split into semantic chunks
Stage 5: EMBED       → Generate vector embeddings
Stage 6: INDEX       → Store in Qdrant vector database
Stage 7: COMPLETE    → Finalize and report status
```

**Processing States:**

```python
class ProcessingStage(str, Enum):
    PENDING = "pending"
    VALIDATING = "validating"
    PREPROCESSING = "preprocessing"
    OCR = "ocr"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    INDEXING = "indexing"
    COMPLETED = "completed"
    FAILED = "failed"
```

---

### 2.8 Web Enrichment Agent

**File:** `backend/ai-service/app/agents/web_enrichment_agent.py`

**Human Role Replaced:** Research Assistant (Real-Time)

**Key Capabilities:**

- **Wikipedia Fetching:** Retrieves relevant encyclopedia articles
- **Khan Academy Integration:** Finds educational content
- **Quality Filtering:** Scores and ranks sources by relevance
- **Redis Caching:** Stores results for faster subsequent retrieval

**Source Types:**

```python
source_types = ['wikipedia', 'khan_academy', 'video', 'article']
```

---

## 3. Proctoring System

**File:** `backend/ai-service/app/proctor/session.py`

**Human Role Replaced:** Exam Invigilator / Proctor

### 3.1 Multi-Modal Detection

The proctoring system runs multiple ML detectors simultaneously:

| Detector | Model | Function |
|----------|-------|----------|
| **Face Detector** | MediaPipe | Detects face presence |
| **Head Pose Estimator** | MediaPipe | Tracks head orientation |
| **Gaze Tracker** | Custom CNN | Monitors eye direction |
| **Object Detector** | YOLOv11 | Identifies phones, books, people |
| **Hand Detector** | MediaPipe | Tracks hand positions |
| **Audio Detector** | Whisper | Detects voice/sounds |
| **Blink Detector** | Custom | Monitors blink patterns |
| **Face Verifier** | FaceNet | Verifies student identity |

### 3.2 Integrity Scoring

```python
class ProctorSession:
    def __init__(self, assessment_id, student_id):
        self.metrics = MetricsAggregator(session_id=self.id)
        self.scorer = IntegrityScorer()
        self.flagger = FlagGenerator()
```

---

## 4. Human Tasks Automated

| Human Role | AI Replacement | Time Saved |
|------------|----------------|------------|
| Personal Tutor | Tutor Agent + ABCR | 24/7 availability |
| Question Setter | Assessment Agent | Instant quiz generation |
| Academic Counselor | Curriculum Agent | Automated path planning |
| Research Assistant | Research Agent | Minutes vs hours |
| Librarian | Document Agent | Automated indexing |
| Exam Invigilator | Proctor Session | Multi-modal monitoring |
| Study Coach | Study Planner Agent | Personalized schedules |
| Coordinator | Orchestrator Agent | Intelligent routing |

---

## 5. Technical Stack

### 5.1 AI/ML Frameworks

- **LangGraph** — Agent orchestration and state management
- **LangChain** — LLM integration and prompt management
- **PyTorch** — Deep learning models (proctoring, temporal analysis)
- **Hugging Face Transformers** — NLP models and embeddings
- **MediaPipe** — Face landmarks and pose estimation
- **YOLOv11** — Real-time object detection
- **Whisper** — Speech-to-text transcription

### 5.2 Data Infrastructure

- **Qdrant** — Vector database for RAG
- **PostgreSQL** — Relational data storage
- **MongoDB** — Document storage (transcripts)
- **Redis** — Session caching and real-time data
- **Apache Kafka** — Event streaming

### 5.3 Backend Services

- **FastAPI** — AI service endpoints
- **Flask** — Core service (auth, CRUD)
- **Docker** — Containerization

---

## 6. Conclusion

The ensureStudy platform demonstrates a production-grade implementation of autonomous AI agents that collectively replace multiple human roles in educational workflows. The LangGraph-based architecture enables:

1. **Intelligent Routing** — Automatic query classification and agent selection
2. **Context Awareness** — Conversation continuity via ABCR/TAL
3. **Adaptive Learning** — Personalized content based on student performance
4. **Automated Assessment** — Dynamic quiz generation and evaluation
5. **Real-Time Monitoring** — Multi-modal exam proctoring
6. **Knowledge Management** — Automated document processing and indexing

This multi-agent system provides 24/7 educational support while maintaining the quality and personalization traditionally requiring human intervention.

---

**Document Generated:** January 2026  
**Technology:** LangGraph, PyTorch, Qdrant, FastAPI  
**Repository:** github.com/realshubhamraut/ensureStudy
