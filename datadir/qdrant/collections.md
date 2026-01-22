# Qdrant Vector Collections

Qdrant is used for semantic search and RAG (Retrieval Augmented Generation).

## Connection

```python
from qdrant_client import QdrantClient

# Development
client = QdrantClient(host="localhost", port=6333)

# Docker network
client = QdrantClient(host="qdrant", port=6333)
```

---

## Collections

### 1. `ensure_study_documents` (Main RAG Collection)

**Purpose:** Primary collection for document embeddings used in RAG retrieval.

| Property | Value |
|----------|-------|
| Vector Size | 384 (all-MiniLM-L6-v2) or 1536 (OpenAI) |
| Distance | COSINE |
| HNSW M | 16 |
| HNSW ef_construct | 100 |

**Payload Schema:**
```json
{
    "text": "First 1000 chars of chunk text",
    "full_text": "Complete chunk text",
    "source": "textbook|notes|video|web",
    "page": 1,
    "subject": "mathematics|physics|chemistry|history",
    "topic": "Pythagorean Theorem",
    "difficulty": "easy|medium|hard",
    "type": "textbook|notes|video_transcript|web_article",
    "chunk_index": 0,
    "classroom_id": "uuid",
    "student_id": "uuid",
    "document_id": "uuid"
}
```

**Payload Indices:**
- `classroom_id` (KEYWORD)
- `student_id` (KEYWORD)
- `document_id` (KEYWORD)
- `source_type` (KEYWORD)
- `subject` (KEYWORD)
- `contains_formula` (BOOL)

---

### 2. `classroom_materials` (Classroom-specific RAG)

**Purpose:** Classroom-specific document chunks for teacher-uploaded materials.

| Property | Value |
|----------|-------|
| Vector Size | 384 |
| Distance | COSINE |
| Quantization | INT8 (Scalar) |

**Payload Schema:**
```json
{
    "classroom_id": "uuid",
    "document_id": "uuid",
    "student_id": null,
    "title": "Document title",
    "text": "Chunk text content",
    "page_number": 1,
    "chunk_index": 0,
    "source_type": "classroom",
    "subject": "mathematics",
    "contains_formula": false,
    "bbox": [100, 200, 500, 300],
    "created_at": "2024-01-01T00:00:00Z"
}
```

**Payload Indices:**
- `classroom_id` (KEYWORD)
- `document_id` (KEYWORD)
- `source_type` (KEYWORD)
- `subject` (KEYWORD)

---

### 3. `web_cache` (Semantic Web Content Cache)

**Purpose:** Cache for web-crawled content to avoid repeated crawls.

| Property | Value |
|----------|-------|
| Vector Size | 384 |
| Distance | COSINE |

**Payload Schema:**
```json
{
    "query": "Original search query",
    "query_hash": "sha256 hash of normalized query",
    "answer": "Cached answer text",
    "sources": ["url1", "url2"],
    "confidence": 0.95,
    "created_at": "2024-01-01T00:00:00Z",
    "expires_at": "2024-01-08T00:00:00Z"
}
```

---

### 4. `student_notes` (Personal Student Notes)

**Purpose:** Personal student notes for personalized retrieval.

| Property | Value |
|----------|-------|
| Vector Size | 384 |
| Distance | COSINE |

**Payload Schema:**
```json
{
    "student_id": "uuid",
    "note_id": "uuid",
    "title": "Note title",
    "text": "Note content chunk",
    "subject": "mathematics",
    "topics": ["algebra", "equations"],
    "created_at": "2024-01-01T00:00:00Z"
}
```

**Payload Indices:**
- `student_id` (KEYWORD)
- `subject` (KEYWORD)

---

### 5. `youtube_transcripts` (YouTube Video Transcripts)

**Purpose:** Cached YouTube video transcripts for educational content.

| Property | Value |
|----------|-------|
| Vector Size | 384 |
| Distance | COSINE |

**Payload Schema:**
```json
{
    "video_id": "dQw4w9WgXcQ",
    "title": "Video title",
    "channel": "Channel name",
    "text": "Transcript chunk",
    "timestamp_start": 120.5,
    "timestamp_end": 180.0,
    "language": "en"
}
```

---

## Environment Variables

```env
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_API_KEY=<optional>
QDRANT_COLLECTION_NAME=ensure_study_documents
EMBEDDING_DIMENSIONS=384
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

---

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Health check |
| `GET /collections` | List all collections |
| `GET /collections/{name}` | Collection info |
| `POST /collections/{name}/points/search` | Vector search |
| `PUT /collections/{name}/points` | Upsert points |
