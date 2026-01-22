# MongoDB Collections

MongoDB is used for storing meeting transcripts, summaries, and unstructured data.

## Connection

```python
from pymongo import MongoClient

# Development
client = MongoClient("mongodb://ensure_study:mongodb_password_123@localhost:27017/ensure_study_meetings")

# Docker network
client = MongoClient("mongodb://ensure_study:mongodb_password_123@mongodb:27017/ensure_study_meetings")

db = client.ensure_study_meetings
```

---

## Database: `ensure_study_meetings`

### 1. `meetings` Collection

**Purpose:** Store meeting metadata and session info.

```javascript
{
    "_id": ObjectId("..."),
    "meeting_id": "uuid",
    "title": "Physics Study Group",
    "classroom_id": "uuid",
    "host_id": "uuid",
    "participants": [
        { "user_id": "uuid", "name": "John Doe", "role": "student" }
    ],
    "scheduled_at": ISODate("2024-01-01T10:00:00Z"),
    "started_at": ISODate("2024-01-01T10:02:00Z"),
    "ended_at": ISODate("2024-01-01T11:00:00Z"),
    "duration_minutes": 58,
    "status": "completed",  // "scheduled", "active", "completed", "cancelled"
    "recording_url": "s3://bucket/recordings/meeting_id.webm",
    "created_at": ISODate("2024-01-01T09:00:00Z"),
    "updated_at": ISODate("2024-01-01T11:00:00Z")
}
```

**Indexes:**
```javascript
db.meetings.createIndex({ "meeting_id": 1 }, { unique: true })
db.meetings.createIndex({ "classroom_id": 1 })
db.meetings.createIndex({ "host_id": 1 })
db.meetings.createIndex({ "scheduled_at": 1 })
db.meetings.createIndex({ "status": 1 })
```

---

### 2. `transcripts` Collection

**Purpose:** Store full meeting transcripts with speaker diarization.

```javascript
{
    "_id": ObjectId("..."),
    "meeting_id": "uuid",
    "segments": [
        {
            "speaker_id": "uuid",
            "speaker_name": "John Doe",
            "text": "Let's discuss the concept of momentum.",
            "start_time": 120.5,
            "end_time": 125.3,
            "confidence": 0.95
        },
        {
            "speaker_id": "uuid",
            "speaker_name": "Jane Smith",
            "text": "Momentum equals mass times velocity.",
            "start_time": 126.0,
            "end_time": 130.2,
            "confidence": 0.92
        }
    ],
    "full_text": "Let's discuss the concept of momentum. Momentum equals mass times velocity...",
    "word_count": 1250,
    "language": "en",
    "transcription_engine": "whisper",
    "created_at": ISODate("2024-01-01T11:05:00Z")
}
```

**Indexes:**
```javascript
db.transcripts.createIndex({ "meeting_id": 1 }, { unique: true })
db.transcripts.createIndex({ "segments.speaker_id": 1 })
db.transcripts.createIndex({ "$**": "text" })  // Text search
```

---

### 3. `summaries` Collection

**Purpose:** AI-generated meeting summaries.

```javascript
{
    "_id": ObjectId("..."),
    "meeting_id": "uuid",
    "summary_type": "executive",  // "executive", "detailed", "action_items"
    "content": {
        "overview": "Discussion about momentum and kinetic energy...",
        "key_points": [
            "Momentum formula: p = mv",
            "Conservation of momentum in collisions",
            "Examples of elastic vs inelastic collisions"
        ],
        "action_items": [
            { "task": "Complete momentum problem set", "assignee": "All students", "due": "2024-01-05" }
        ],
        "topics_covered": ["momentum", "kinetic energy", "collisions"],
        "questions_raised": [
            { "question": "How does air resistance affect momentum?", "answered": true }
        ]
    },
    "model_used": "gpt-4",
    "generated_at": ISODate("2024-01-01T11:10:00Z")
}
```

**Indexes:**
```javascript
db.summaries.createIndex({ "meeting_id": 1 })
db.summaries.createIndex({ "summary_type": 1 })
```

---

### 4. `meeting_qa` Collection

**Purpose:** Q&A pairs from meetings for RAG.

```javascript
{
    "_id": ObjectId("..."),
    "meeting_id": "uuid",
    "qa_pairs": [
        {
            "question": "What is the formula for momentum?",
            "answer": "Momentum equals mass times velocity (p = mv).",
            "asked_by": "uuid",
            "answered_by": "uuid",
            "timestamp": 245.5,
            "context": "Discussion about basic physics concepts"
        }
    ],
    "indexed_in_qdrant": true,
    "qdrant_point_ids": ["uuid1", "uuid2"],
    "created_at": ISODate("2024-01-01T11:15:00Z")
}
```

---

### 5. `meeting_recordings` Collection

**Purpose:** Recording chunks and processing status.

```javascript
{
    "_id": ObjectId("..."),
    "meeting_id": "uuid",
    "chunk_index": 0,
    "s3_path": "s3://bucket/recordings/meeting_id/chunk_0.webm",
    "duration_seconds": 300,
    "start_time": 0,
    "end_time": 300,
    "file_size_bytes": 15000000,
    "processing_status": "transcribed",  // "uploaded", "processing", "transcribed", "error"
    "created_at": ISODate("2024-01-01T10:10:00Z")
}
```

---

## Environment Variables

```env
MONGODB_URL=mongodb://ensure_study:mongodb_password_123@localhost:27017/ensure_study_meetings
MONGO_INITDB_ROOT_USERNAME=ensure_study
MONGO_INITDB_ROOT_PASSWORD=mongodb_password_123
MONGO_INITDB_DATABASE=ensure_study_meetings
```
