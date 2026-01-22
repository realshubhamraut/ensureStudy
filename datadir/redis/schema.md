# Redis Key Patterns & Data Structures

Redis is used for caching, rate limiting, and session management.

## Connection

```python
import redis

# Development
r = redis.from_url("redis://localhost:6379/0")

# Docker network
r = redis.from_url("redis://redis:6379")
```

---

## Key Patterns

### 1. Session Cache (AI Tutor Sessions)

**Pattern:** `session:{user_id}`  
**Type:** STRING (JSON serialized)  
**TTL:** 24 hours (86400 seconds)

```json
{
    "session_id": "uuid",
    "user_id": "uuid",
    "classroom_id": "uuid",
    "created_at": "2024-01-01T00:00:00Z",
    "last_active_at": "2024-01-01T12:00:00Z",
    "turn_count": 5,
    "resource_list": ["res_1", "res_2"],
    "topic_anchor": {
        "id": "anchor_xxx",
        "canonical_title": "French Revolution",
        "created_at": "2024-01-01T10:00:00Z"
    },
    "config": {
        "ttl_hours": 24,
        "max_resources": 25,
        "relatedness_threshold": 0.65
    }
}
```

---

### 2. Rate Limiting (Sliding Window)

**Pattern:** `rate:{user_id}:{endpoint}`  
**Type:** SORTED SET (timestamps as scores)  
**TTL:** 1 minute (60 seconds)

**Operations:**
```redis
ZADD rate:user123:tutor_query <timestamp> <timestamp>
ZREMRANGEBYSCORE rate:user123:tutor_query 0 <timestamp-window>
ZCARD rate:user123:tutor_query
EXPIRE rate:user123:tutor_query 60
```

**Rate Limits:**
| Endpoint | Limit |
|----------|-------|
| `tutor_query` | 20/minute |
| `web_search` | 10/minute |
| `file_upload` | 5/minute |

---

### 3. Unified Cache (General Purpose)

**Pattern:** `cache:{namespace}:{key_hash}`  
**Type:** STRING (JSON serialized)  
**TTL:** Varies by namespace

**Namespaces:**

| Namespace | TTL | Purpose |
|-----------|-----|---------|
| `ocr` | 7 days | OCR results |
| `embedding` | 1 day | Vector embeddings |
| `llm` | 1 hour | LLM responses |

**Examples:**
```redis
# OCR results
SET cache:ocr:sha256hash '{"text": "...", "confidence": 0.95}' EX 604800

# Embedding cache
SET cache:embedding:sha256hash '[0.1, 0.2, ...]' EX 86400

# LLM response
SET cache:llm:sha256hash '{"answer": "...", "confidence": 0.9}' EX 3600
```

---

### 4. ABCR Decision Cache (Context Routing)

**Pattern:** `abcr:{session_id}:{question_hash}`  
**Type:** STRING (JSON serialized)  
**TTL:** 1 hour (3600 seconds)

```json
{
    "decision": "related",
    "attention_score": 0.78,
    "similarity_score": 0.65,
    "combined_score": 0.72,
    "computed_at": "2024-01-01T12:00:00Z"
}
```

---

### 5. Topic Anchor Cache

**Pattern:** `anchor:{session_id}`  
**Type:** STRING (JSON serialized)  
**TTL:** 24 hours (86400 seconds)

```json
{
    "id": "anchor_abc123",
    "canonical_title": "French Revolution",
    "canonical_seed": "French Revolution",
    "topic_embedding": [0.1, 0.2, "..."],
    "subject_scope": ["causes", "events", "consequences"],
    "locked_entities": ["1789", "Louis XVI", "Bastille"],
    "source": "user_query",
    "created_at": "2024-01-01T10:00:00Z"
}
```

---

### 6. Topic History

**Pattern:** `topic_history:{session_id}`  
**Type:** LIST (JSON strings)  
**TTL:** 24 hours

```redis
LPUSH topic_history:sess123 '{"canonical_title": "French Revolution", ...}'
EXPIRE topic_history:sess123 86400
```

---

### 7. Suggestion History (Deduplication)

**Pattern:** `suggestions:{session_id}`  
**Type:** SET (hashes of shown suggestions)  
**TTL:** 24 hours

```redis
SADD suggestions:sess123 "sha256hash1" "sha256hash2"
SISMEMBER suggestions:sess123 "sha256hash1"
EXPIRE suggestions:sess123 86400
```

---

### 8. Web Content Cache

**Pattern:** `web:{url_hash}`  
**Type:** STRING (JSON serialized)  
**TTL:** 7 days (604800 seconds)

```json
{
    "url": "https://example.com/article",
    "title": "Article Title",
    "content": "Article content...",
    "crawled_at": "2024-01-01T12:00:00Z"
}
```

---

### 9. YouTube Transcript Cache

**Pattern:** `youtube:{video_id}`  
**Type:** STRING (JSON serialized)  
**TTL:** 30 days (2592000 seconds)

```json
{
    "video_id": "dQw4w9WgXcQ",
    "title": "Video Title",
    "channel": "Channel Name",
    "duration": 212,
    "transcript": ["..."],
    "fetched_at": "2024-01-01T00:00:00Z"
}
```

---

## Environment Variables

```env
REDIS_URL=redis://localhost:6379/0
REDIS_PASSWORD=<optional>
CACHE_TTL_SECONDS=86400
RATE_LIMIT_ENABLED=true
```

---

## Quick Reference

| Pattern | Type | TTL |
|---------|------|-----|
| `session:{user_id}` | STRING | 24h |
| `rate:{user_id}:{endpoint}` | ZSET | 1min |
| `cache:{namespace}:{hash}` | STRING | varies |
| `abcr:{session}:{hash}` | STRING | 1h |
| `anchor:{session}` | STRING | 24h |
| `topic_history:{session}` | LIST | 24h |
| `suggestions:{session}` | SET | 24h |
| `web:{url_hash}` | STRING | 7d |
| `youtube:{video_id}` | STRING | 30d |
