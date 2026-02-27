# Page 35: Agent Interaction Flows & System Sequences

---

## 35.1 Overview

This page documents the **end-to-end interaction sequences** between agents, services, and databases for the most important user flows in ensureStudy.

---

## 35.2 Flow 1: Student Asks Tutor a Question

```mermaid
sequenceDiagram
    participant S as Student
    participant FE as Frontend (ChatInput)
    participant AI as AI Service
    participant ABCR as ABCR Service
    participant RAG as RAG Pipeline
    participant QD as Qdrant
    participant LLM as LLM Provider
    participant CS as Core Service
    participant PG as PostgreSQL

    S->>FE: Type question in /chat
    FE->>AI: POST /api/tutor/chat (SSE)
    AI->>AI: Load student profile (TAL level, classroom)
    AI->>AI: Get chat history
    AI->>ABCR: ABCR cycle
    ABCR->>ABCR: Assess → Build → Challenge → Reflect
    AI->>RAG: RAG query
    RAG->>RAG: Rewrite query
    RAG->>QD: Search top-k chunks
    QD->>RAG: Context chunks
    AI->>AI: Build prompt (system + context + history)
    AI->>LLM: Stream response (GPT-4 / Gemini / Groq)
    LLM-->>FE: SSE chunks (real-time)
    FE->>CS: Save chat session
    CS->>PG: Store in chat_sessions
```

---

## 35.3 Flow 2: Teacher Uploads Material → RAG Indexing

```mermaid
sequenceDiagram
    participant T as Teacher
    participant FE as Frontend
    participant CS as Core Service
    participant K as Kafka
    participant AI as AI Service
    participant QD as Qdrant

    T->>FE: Upload PDF in /teacher/classroom/[id]
    FE->>CS: POST /api/classrooms/<id>/materials
    CS->>CS: Save file to storage (S3/MinIO)
    CS->>CS: Create ClassroomMaterial record
    CS->>K: Publish to "document-processing" topic
    K->>AI: Consumer triggers processing
    AI->>AI: Stage 1: Validate file type/size
    AI->>AI: Stage 2: Preprocess (image enhancement)
    AI->>AI: Stage 3: OCR (if scanned)
    AI->>AI: Stage 4: Text extraction (PyMuPDF)
    AI->>AI: Stage 5: Chunk text (500-char + overlap)
    AI->>AI: Stage 6: Embed chunks (sentence-transformers)
    AI->>QD: Store in classroom_materials collection
    AI->>AI: Stage 7: Update status to "indexed"
    AI->>CS: Callback: indexing_status = "complete"
    Note over T,QD: Material now available for RAG queries
```

---

## 35.4 Flow 3: Student Takes Proctored Assessment

```mermaid
sequenceDiagram
    participant S as Student
    participant FE as Frontend
    participant CS as Core Service
    participant AI as AI Service
    participant DET as 8 Detectors
    participant ML as ML Models

    S->>FE: Navigate to /assessments/take/[id]
    FE->>CS: Load assessment questions
    FE->>FE: Request webcam access
    FE->>AI: POST /api/proctoring/session/start
    AI->>AI: Create ProctorSession (lazy-load detectors)

    loop Every 1 second
        FE->>FE: Capture webcam frame
        FE->>AI: POST /api/proctoring/analyze-frame
        AI->>DET: Run 8 detectors (face, gaze, head, object, hand, audio, blink, verify)
        DET->>ML: Format for AutoOEP
        ML->>ML: Static classifier (LightGBM)
        ML->>ML: Temporal predictor (LSTM, 30-frame)
        ML->>AI: {current_score, active_flags, detections}
        AI->>FE: Live integrity indicator
    end

    opt Tab Switch
        FE->>AI: POST /api/proctoring/tab-switch
    end

    S->>CS: POST /api/assessments/<id>/submit
    CS->>CS: Save responses, calculate score
    FE->>AI: POST /api/proctoring/session/end
    AI->>AI: Finalize → {integrity_score, flags, frame_count}
```

---

## 35.5 Flow 4: Curriculum Generation from Syllabus

```mermaid
sequenceDiagram
    participant T as Teacher
    participant FE as Frontend
    participant CS as Core Service
    participant AI as AI Service
    participant LLM as LLM (Groq/GPT-4)
    participant PG as PostgreSQL

    T->>FE: Upload syllabus PDF
    FE->>CS: POST /api/classrooms/<id>/syllabus
    CS->>CS: Save file, create Syllabus record
    FE->>AI: POST /api/curriculum/extract-topics
    AI->>AI: Extract text (pdf_extractor.py)
    AI->>LLM: Topic extraction → JSON (topics + subtopics)
    AI->>LLM: Dependency analysis → prerequisites
    AI->>AI: Build dependency graph (topological sort)
    AI->>AI: Generate learning path + durations
    AI->>PG: Create Subject → Topic → Subtopic hierarchy
    FE->>AI: POST /api/curriculum/generate-dependencies
    AI->>LLM: Analyze topic pairs → prerequisites
    Note over T,PG: Result: Structured curriculum with learning path
```

---

## 35.6 Flow 5: Learning Agent (Type 5) Self-Improving Cycle

```mermaid
stateDiagram-v2
    [*] --> Trigger: Student completes assessment
    Trigger --> Kafka: Publish to assessment-submissions
    Kafka --> Consumer: Learning Agent Consumer

    state Consumer {
        [*] --> Critic
        Critic --> Learner
        Learner --> Performance
        Performance --> Iterate

        state Critic {
            [*] --> AnalyzeResponses
            AnalyzeResponses --> CompareExpected
            CompareExpected --> ScoreQuality: Score 0-10
            ScoreQuality --> IdentifyGaps
        }

        state Learner {
            [*] --> ReadStrategy: From LearningAgentMemory
            ReadStrategy --> UpdateStrategy
            UpdateStrategy --> AdjustDifficulty
            AdjustDifficulty --> StoreStrategy
        }

        state Performance {
            [*] --> CheckMastery
            CheckMastery --> Advance: mastery > 80%
            CheckMastery --> GenerateNew: mastery <= 80%
            Advance --> UpdateScore
            GenerateNew --> UpdateScore
        }
    }

    Iterate --> [*]: Loop with next assessment
```

---

## 35.7 Flow 6: Meeting → Transcription → Q&A

```mermaid
sequenceDiagram
    participant T as Teacher
    participant CS as Core Service
    participant LK as LiveKit
    participant K as Kafka
    participant SP as Spark Streaming
    participant W as Whisper
    participant GM as Gemini 1.5 Flash
    participant QD as Qdrant
    participant CAS as Cassandra
    participant S as Student

    T->>CS: POST /api/meetings (create)
    T->>LK: Start video room
    T->>LK: End meeting → recording saved
    LK->>K: Event: meeting-recordings
    K->>SP: meeting_processor.py
    SP->>W: POST /api/meetings/transcribe
    W->>SP: Transcript + segments
    SP->>GM: POST /api/meetings/summarize
    GM->>SP: Brief, detailed, key_points, action_items
    SP->>QD: Embed transcript chunks (500-char + timestamps)
    SP->>CAS: Store analytics → meeting_analytics

    S->>CS: POST /api/meetings/query
    CS->>QD: Search meeting_chunks
    QD->>GM: Synthesis
    GM->>S: Answer with sources
```

---

## 35.8 Flow 7: Soft Skills Mock Interview

```mermaid
sequenceDiagram
    participant S as Student
    participant FE as Frontend
    participant AI as AI Service
    participant W as Whisper
    participant LLM as LLM
    participant SS as Soft Skills Analyzers

    S->>FE: Start mock interview
    FE->>AI: POST /api/mock-interview/start
    AI->>LLM: Generate interview questions

    loop Each Question
        FE->>FE: Display question + start recording
        FE->>SS: Video frames (every 1s)
        SS->>SS: Gaze analyzer (eye contact)
        SS->>SS: Posture analyzer (MediaPipe Pose)
        SS->>SS: Gesture analyzer (hand movements)
        SS->>SS: Filler detector (audio analysis)
        S->>FE: Finish answer → stop recording
        FE->>W: POST /api/stt/transcribe → text
        FE->>AI: POST /api/mock-interview/answer
        AI->>LLM: Evaluate answer content quality
        AI->>AI: Combine: content score + delivery score
    end

    FE->>AI: POST /api/mock-interview/evaluate
    AI->>S: Final report: overall score, per-question,<br/>soft skills metrics, improvements
```

---

## 35.9 Cross-Cutting Patterns

| Pattern | Used By | Mechanism |
|---------|---------|-----------|
| **SSE Streaming** | Tutor chat, agent responses | `StreamingResponse` + `text/event-stream` |
| **Async via Kafka** | Document processing, learning agent, meeting transcription | Produce → Topic → Consumer |
| **Redis Caching** | ABCR state, web resources, RAG queries | Cache with TTL (1h-7d) |
| **Lazy Loading** | Proctoring detectors, ML models | `@property` with `_instance is None` check |
| **Fallback Chain** | LLM calls | Try OpenAI → Gemini → Groq → Ollama |
| **Webhook Callbacks** | Grading, indexing status | AI Service → Core Service HTTP callback |
