# Kafka Topics & Event Schemas

Kafka is used for event streaming between microservices.

## Connection

```python
from kafka import KafkaProducer, KafkaConsumer

# Development
producer = KafkaProducer(bootstrap_servers='localhost:9092')
consumer = KafkaConsumer('topic', bootstrap_servers='localhost:9092')

# Docker network
producer = KafkaProducer(bootstrap_servers='kafka:29092')
```

---

## Topics

### 1. `document-events`

**Purpose:** Document lifecycle events (upload, processing, indexing).

**Partitions:** 3  
**Retention:** 7 days

**Event Schema:**
```json
{
    "event_id": "uuid",
    "event_type": "document.uploaded|document.processing|document.indexed|document.error",
    "timestamp": "2024-01-01T10:00:00Z",
    "document_id": "uuid",
    "classroom_id": "uuid",
    "user_id": "uuid",
    "payload": {
        "filename": "notes.pdf",
        "file_size": 1024000,
        "status": "uploaded"
    }
}
```

---

### 2. `meeting-events`

**Purpose:** Real-time meeting events.

**Partitions:** 6  
**Retention:** 3 days

**Event Schema:**
```json
{
    "event_id": "uuid",
    "event_type": "meeting.started|meeting.ended|participant.joined|participant.left|recording.started",
    "timestamp": "2024-01-01T10:00:00Z",
    "meeting_id": "uuid",
    "payload": {
        "participant_id": "uuid",
        "participant_name": "John Doe"
    }
}
```

---

### 3. `tutor-queries`

**Purpose:** AI tutor query events for analytics.

**Partitions:** 3  
**Retention:** 30 days

**Event Schema:**
```json
{
    "event_id": "uuid",
    "event_type": "query.submitted|query.answered|query.error",
    "timestamp": "2024-01-01T10:00:00Z",
    "session_id": "uuid",
    "user_id": "uuid",
    "payload": {
        "question": "What is photosynthesis?",
        "answer_latency_ms": 1250,
        "sources_count": 3,
        "related_to_previous": true
    }
}
```

---

### 4. `assessment-events`

**Purpose:** Exam and assessment lifecycle.

**Partitions:** 3  
**Retention:** 90 days

**Event Schema:**
```json
{
    "event_id": "uuid",
    "event_type": "exam.started|exam.submitted|exam.graded|proctor.violation",
    "timestamp": "2024-01-01T10:00:00Z",
    "exam_session_id": "uuid",
    "user_id": "uuid",
    "payload": {
        "exam_id": "uuid",
        "score": 85.5,
        "violations": []
    }
}
```

---

### 5. `analytics-events`

**Purpose:** General analytics and tracking.

**Partitions:** 6  
**Retention:** 30 days

**Event Schema:**
```json
{
    "event_id": "uuid",
    "event_type": "page.view|feature.used|error.occurred",
    "timestamp": "2024-01-01T10:00:00Z",
    "user_id": "uuid",
    "payload": {
        "page": "/dashboard",
        "duration_ms": 5000,
        "metadata": {}
    }
}
```

---

## Environment Variables

```env
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
KAFKA_GROUP_ID=ensure-study-consumers
KAFKA_AUTO_OFFSET_RESET=earliest
```

---

## Topic Creation Commands

```bash
# Using kafka-topics CLI
kafka-topics --create --topic document-events --partitions 3 --replication-factor 1 --bootstrap-server localhost:9092
kafka-topics --create --topic meeting-events --partitions 6 --replication-factor 1 --bootstrap-server localhost:9092
kafka-topics --create --topic tutor-queries --partitions 3 --replication-factor 1 --bootstrap-server localhost:9092
kafka-topics --create --topic assessment-events --partitions 3 --replication-factor 1 --bootstrap-server localhost:9092
kafka-topics --create --topic analytics-events --partitions 6 --replication-factor 1 --bootstrap-server localhost:9092

# List topics
kafka-topics --list --bootstrap-server localhost:9092
```

---

## Consumer Groups

| Group ID | Topics | Purpose |
|----------|--------|---------|
| `document-processor` | document-events | OCR & indexing pipeline |
| `meeting-processor` | meeting-events | Transcription & summary |
| `analytics-aggregator` | all | Cassandra time-series writes |
| `notification-sender` | all | Real-time notifications |
