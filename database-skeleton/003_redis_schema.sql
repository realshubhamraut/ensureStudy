-- ============================================================================
-- ensureStudy Redis Schema (Key Patterns & Data Structures)
-- ============================================================================
-- Redis is used for caching, rate limiting, and session management
-- 
-- Database: Redis 7+
-- Default URL: redis://localhost:6379/0
-- ============================================================================

-- ============================================================================
-- SESSION CACHE (AI Tutor Sessions)
-- ============================================================================
-- Purpose: Fast access to active tutor sessions for context retention

-- Key Pattern: session:{user_id}
-- Type: STRING (JSON serialized)
-- TTL: 24 hours (86400 seconds)

/*
SET session:550e8400-e29b-41d4-a716-446655440000 '{
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
}' EX 86400
*/


-- ============================================================================
-- RATE LIMITING (Sliding Window)
-- ============================================================================
-- Purpose: API rate limiting per user/endpoint

-- Key Pattern: rate:{user_id}:{endpoint}
-- Type: SORTED SET (timestamps as scores)
-- TTL: 1 minute (60 seconds) for per-minute limits

/*
Example Operations:
ZADD rate:user123:tutor_query <timestamp> <timestamp>
ZREMRANGEBYSCORE rate:user123:tutor_query 0 <timestamp-window>
ZCARD rate:user123:tutor_query
EXPIRE rate:user123:tutor_query 60
*/

-- Rate limit configurations:
-- - tutor_query: 20 requests/minute per user
-- - web_search: 10 requests/minute per user  
-- - file_upload: 5 requests/minute per user


-- ============================================================================
-- UNIFIED CACHE (General Purpose)
-- ============================================================================
-- Purpose: Caching expensive computations and API responses

-- Key Pattern: cache:{namespace}:{key_hash}
-- Type: STRING (JSON serialized)
-- TTL: Varies by namespace

/*
# OCR results cache
SET cache:ocr:sha256hash '{
    "text": "extracted text",
    "confidence": 0.95,
    "pages": 5
}' EX 604800  # 7 days

# Embedding cache
SET cache:embedding:sha256hash '[0.1, 0.2, ...]' EX 86400  # 1 day

# LLM response cache
SET cache:llm:sha256hash '{
    "answer": "response text",
    "confidence": 0.9
}' EX 3600  # 1 hour
*/


-- ============================================================================
-- ABCR DECISION CACHE (Context Routing)
-- ============================================================================
-- Purpose: Cache ABCR (Attention-Based Context Routing) decisions

-- Key Pattern: abcr:{session_id}:{question_hash}
-- Type: STRING (JSON serialized)
-- TTL: 1 hour (3600 seconds)

/*
SET abcr:sess123:qhash456 '{
    "decision": "related",
    "attention_score": 0.78,
    "similarity_score": 0.65,
    "combined_score": 0.72,
    "computed_at": "2024-01-01T12:00:00Z"
}' EX 3600
*/


-- ============================================================================
-- TOPIC ANCHOR CACHE
-- ============================================================================
-- Purpose: Store active topic anchors for sessions

-- Key Pattern: anchor:{session_id}
-- Type: STRING (JSON serialized)
-- TTL: 24 hours (86400 seconds)

/*
SET anchor:sess123 '{
    "id": "anchor_abc123",
    "canonical_title": "French Revolution",
    "canonical_seed": "French Revolution",
    "topic_embedding": [0.1, 0.2, ...],
    "subject_scope": ["causes", "events", "consequences"],
    "locked_entities": ["1789", "Louis XVI", "Bastille"],
    "source": "user_query",
    "created_at": "2024-01-01T10:00:00Z"
}' EX 86400
*/


-- ============================================================================
-- TOPIC HISTORY
-- ============================================================================
-- Purpose: Track previous topics in a session

-- Key Pattern: topic_history:{session_id}
-- Type: LIST (JSON strings)
-- TTL: 24 hours

/*
LPUSH topic_history:sess123 '{
    "canonical_title": "French Revolution",
    "started_at": "2024-01-01T10:00:00Z",
    "ended_at": "2024-01-01T11:00:00Z",
    "turn_count": 8,
    "end_reason": "new_topic"
}'
EXPIRE topic_history:sess123 86400
*/


-- ============================================================================
-- SUGGESTION HISTORY (Deduplication)
-- ============================================================================
-- Purpose: Track shown suggestions to avoid repetition

-- Key Pattern: suggestions:{session_id}
-- Type: SET (hashes of shown suggestions)
-- TTL: 24 hours

/*
SADD suggestions:sess123 "sha256hash1" "sha256hash2" "sha256hash3"
SISMEMBER suggestions:sess123 "sha256hash1"  # Check if already shown
EXPIRE suggestions:sess123 86400
*/


-- ============================================================================
-- WEB CONTENT CACHE
-- ============================================================================
-- Purpose: Cache crawled web content to reduce API calls

-- Key Pattern: web:{url_hash}
-- Type: STRING (JSON serialized)
-- TTL: 7 days (604800 seconds)

/*
SET web:sha256urlhash '{
    "url": "https://example.com/article",
    "title": "Article Title",
    "content": "Article content...",
    "crawled_at": "2024-01-01T12:00:00Z"
}' EX 604800
*/


-- ============================================================================
-- YOUTUBE TRANSCRIPT CACHE
-- ============================================================================
-- Purpose: Cache YouTube video transcripts

-- Key Pattern: youtube:{video_id}
-- Type: STRING (JSON serialized)
-- TTL: 30 days (2592000 seconds)

/*
SET youtube:dQw4w9WgXcQ '{
    "video_id": "dQw4w9WgXcQ",
    "title": "Video Title",
    "channel": "Channel Name",
    "duration": 212,
    "transcript": [...],
    "fetched_at": "2024-01-01T00:00:00Z"
}' EX 2592000
*/


-- ============================================================================
-- ENVIRONMENT VARIABLES FOR REDIS
-- ============================================================================
-- REDIS_URL=redis://localhost:6379/0
-- REDIS_PASSWORD=<optional>
-- CACHE_TTL_SECONDS=86400
-- RATE_LIMIT_ENABLED=true
