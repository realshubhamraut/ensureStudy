# ensureStudy Database Schema Overview

Complete database schema documentation for the ensureStudy web application.

## Databases Used

| Database | Purpose | Port |
|----------|---------|------|
| **PostgreSQL** | Primary relational database | 5432 |
| **Qdrant** | Vector embeddings for RAG | 6333 |
| **Redis** | Caching & rate limiting | 6379 |

---

## Quick Reference

### PostgreSQL Tables

| Table | Purpose | FK Dependencies |
|-------|---------|-----------------|
| `users` | User accounts | - |
| `classrooms` | Teacher-created classes | → users |
| `classroom_enrollments` | Student-class mapping | → users, classrooms |
| `classroom_materials` | Uploaded files | → classrooms, users |
| `documents` | Document metadata | → classrooms |
| `document_pages` | Per-page OCR results | → documents |
| `document_chunks` | Text chunks for RAG | → documents |
| `document_quality_reports` | OCR quality metrics | → documents |
| `tutor_sessions` | AI tutor sessions | → users, classrooms |
| `session_turns` | Query history | → tutor_sessions |
| `session_resources` | Discovered resources | → tutor_sessions |

### Qdrant Collections

| Collection | Purpose | Vector Size |
|------------|---------|-------------|
| `ensure_study_documents` | Main RAG collection | 384/1536 |
| `classroom_materials` | Classroom-specific chunks | 384 |
| `web_cache` | Cached web content | 384 |
| `student_notes` | Personal notes | 384 |

### Redis Key Patterns

| Pattern | Purpose | TTL |
|---------|---------|-----|
| `session:{user_id}` | Tutor session cache | 24h |
| `rate:{user_id}:{endpoint}` | Rate limiting | 1min |
| `anchor:{session_id}` | Topic anchor cache | 24h |
| `abcr:{session_id}:{hash}` | ABCR decision cache | 1h |
| `cache:{namespace}:{hash}` | General cache | varies |

---

## Files in This Directory

| File | Description |
|------|-------------|
| `001_postgresql_schema.sql` | Complete PostgreSQL CREATE TABLE statements |
| `002_qdrant_collections.sql` | Qdrant collection definitions & payload schemas |
| `003_redis_schema.sql` | Redis key patterns & data structures |
| `004_migrations.md` | Migration history and scripts |

---

## Environment Variables

```env
# PostgreSQL
DATABASE_URL=postgresql://ensure_study_user:password@localhost:5432/ensure_study
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=ensure_study
POSTGRES_USER=ensure_study_user
POSTGRES_PASSWORD=secure_password_123

# Qdrant
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION_NAME=ensure_study_documents
EMBEDDING_DIMENSIONS=384

# Redis
REDIS_URL=redis://localhost:6379/0
```

---

## Running Migrations

```bash
# PostgreSQL migrations
psql $DATABASE_URL -f database-skeleton/001_postgresql_schema.sql

# Python migrations
python migrate_add_sessions.py
python migrate_add_indexing_columns.py
python scripts/migrate_session_intelligence.py
```

---

## Entity Relationship Diagram

```
┌─────────────┐     ┌─────────────────┐     ┌───────────────────┐
│   users     │────<│ classrooms      │────<│classroom_materials│
└─────────────┘     └─────────────────┘     └───────────────────┘
       │                    │                        │
       │     ┌──────────────┘                        │
       │     │                                       │
       ▼     ▼                                       ▼
┌─────────────────────┐     ┌─────────────┐     ┌─────────────────┐
│classroom_enrollments│     │  documents  │────<│ document_pages  │
└─────────────────────┘     └─────────────┘     └─────────────────┘
                                   │                    
                                   │────<──────────────────────────┐
                                   │                               │
                                   ▼                               ▼
                            ┌─────────────────┐    ┌──────────────────────┐
                            │ document_chunks │    │document_quality_rpts │
                            └─────────────────┘    └──────────────────────┘

┌─────────────┐     ┌───────────────┐     ┌───────────────────┐
│tutor_sessions│────<│ session_turns │     │ session_resources │
└─────────────┘     └───────────────┘     └───────────────────┘
       ▲                                           │
       └───────────────────────────────────────────┘
```
