# Page 92: Core Infrastructure Services

> Authorization, rate limiting, caching (6-layer unified + response), and storage abstraction (local + S3) that underpin all backend operations.

---

## 92.1 Authorization Service

### Source: `core-service/app/services/authorization_service.py` (360 lines)

Role-Based Access Control extending JWT authentication:

### Permission Constants

```python
class Permissions:
    # Documents
    DOCUMENT_UPLOAD = "document:upload"
    DOCUMENT_VIEW = "document:view"
    DOCUMENT_DELETE = "document:delete"
    DOCUMENT_VIEW_ALL = "document:view_all"
    
    # Tutor
    TUTOR_QUERY = "tutor:query"
    TUTOR_VIEW_HISTORY = "tutor:view_history"
    TUTOR_VIEW_ALL_HISTORY = "tutor:view_all_history"
    
    # Admin
    ADMIN_VIEW_LOGS = "admin:view_logs"
    ADMIN_REINDEX = "admin:reindex"
    ADMIN_WEB_FETCH = "admin:web_fetch"
```

### Role → Permission Mapping

| Role | Permissions |
|------|-------------|
| `student` | `document:view`, `tutor:query`, `tutor:view_history` |
| `teacher` | All student + `document:upload`, `document:delete`, `document:view_all`, `tutor:view_all_history`, `admin:web_fetch` |
| `admin` | All teacher + `admin:view_logs`, `admin:reindex` |
| `parent` | `tutor:view_history` (child's only) |

### Authorization Methods

```python
class AuthorizationService:
    def has_permission(self, user, permission: str) -> bool
    def check_classroom_access(self, user_id, classroom_id, required_role=None) -> bool
    def check_document_access(self, user_id, document_id, action="read") -> bool
    def check_resource_ownership(self, user_id, resource_id, resource_type) -> bool
    def get_user_classrooms(self, user_id) -> List[str]
```

### Flask Decorators

```python
@require_auth            # JWT validation
@require_role("teacher") # Role check
@require_classroom_access(classroom_id_param="classroom_id")  # Membership check
def some_route():
    ...
```

---

## 92.2 Rate Limiter

### Source: `core-service/app/services/rate_limiter.py` (251 lines)

Redis-based sliding window rate limiting:

### Default Rate Limits

| Action | Max Requests | Window |
|--------|-------------|--------|
| `ai_tutor_query_minute` | 30 | 60s |
| `ai_tutor_query_hour` | 200 | 3,600s |
| `document_upload` | 10 | 3,600s |
| `video_search` | 20 | 60s |
| `web_crawl` | 5 | 60s |
| `assessment_generate` | 10 | 3,600s |
| `login_attempt` | 5 | 300s |
| `password_reset` | 3 | 3,600s |

### Usage

```python
from services.rate_limiter import rate_limit

@rate_limit("ai_tutor_query_minute")
def query_tutor():
    # Rate-limited: 30 requests/minute per user
    ...

# Manual check
limiter = get_rate_limiter()
result = limiter.check_rate_limit(user_id, "document_upload")
# → {allowed: True, remaining: 8, reset_at: timestamp, retry_after: 0}
```

---

## 92.3 Unified Cache Service (6-Layer)

### Source: `core-service/app/services/unified_cache.py` (488 lines)

### Cache Layers

| Layer | TTL | Purpose |
|-------|-----|---------|
| OCR Results | 7 days | Avoid re-processing same images |
| Embeddings | ∞ (no expiry) | Deterministic, never changes |
| Vector Search | 1 hour | Query result caching |
| RAG Responses | 24 hours | LLM answer caching |
| Document Metadata | 1 hour | DB query reduction |
| Web Resources | 7 days | External resource caching |

### API

```python
class UnifiedCacheService:
    # OCR Layer
    get_ocr(image_bytes) / set_ocr(image_bytes, result)
    
    # Embedding Layer (no expiry)
    get_embedding(text, model) / set_embedding(text, model, vector)
    
    # Search Layer
    get_search(query_hash, classroom_id, top_k) / set_search(...)
    
    # RAG Layer
    get_rag(question, classroom_id) / set_rag(question, classroom_id, response)
    
    # Document Layer
    get_document(document_id) / set_document(document_id, meta)
    
    # Invalidation
    invalidate_document(document_id)  # Cascading: doc + search + RAG
    invalidate_pattern("ensure:rag:*")
    
    # Metrics
    get_stats() → {hit_rate, hits, misses, errors, avg_get_time_ms}
```

### Graceful Degradation
- Redis unavailable → in-memory dict fallback
- All operations wrapped in try/except
- Metrics tracking even in fallback mode

---

## 92.4 Response Cache

### Source: `ai-service/app/services/response_cache.py` (272 lines)

Caches expensive AI-service computations:

```python
class ResponseCache:
    # LLM Responses (1 hour TTL)
    get_llm_response(question, context_hash, subject) -> CachedResponse
    set_llm_response(question, context_hash, subject, response, ttl=3600)
    
    # Web Resources (7 day TTL)
    get_web_resources(query) -> Dict
    set_web_resources(query, resources, ttl=604800)
    
    # Pattern Invalidation
    invalidate_pattern("ensure:llm:*") -> int  # Returns keys deleted
```

---

## 92.5 Storage Service

### Source: `core-service/app/services/storage_service.py` (288 lines)

Abstract storage supporting local filesystem and AWS S3:

```python
class StorageService:
    def __init__(self, provider=None):
        # Provider: STORAGE_PROVIDER env var ("local" or "s3")
    
    # Upload
    upload_file(file_data, folder, filename, content_type) -> str  # Returns key
    upload_from_path(local_path, folder, filename) -> str
    
    # Access
    get_url(key, expires_in=3600) -> str    # Pre-signed URL for S3
    get_local_path(key) -> str              # Downloads from S3 if needed
    
    # Management
    delete_file(key) -> bool
    file_exists(key) -> bool
```

### Folders

| Folder | Content |
|--------|---------|
| `recordings/` | Meeting video/audio recordings |
| `materials/` | Uploaded PDFs, documents |
| `syllabus/` | Uploaded syllabi |
| `avatars/` | User profile photos |
| `exports/` | Generated reports/exports |
