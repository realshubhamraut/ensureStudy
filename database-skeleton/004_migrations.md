# Database Migrations History

## Migration Scripts

### 1. Initial Setup
- **File**: `backend/core-service/migrations/init.sql`
- **Purpose**: Enable uuid-ossp extension, grant permissions
- **When**: Container first start

### 2. Document Ingestion Tables
- **File**: `backend/core-service/migrations/003_document_ingestion.sql`
- **Tables Created**:
  - `documents`
  - `document_pages`
  - `document_chunks`
  - `document_quality_reports`

### 3. Session Tables
- **File**: `migrate_add_sessions.py`
- **Tables Created**:
  - `tutor_sessions`
  - `session_turns`
  - `session_resources`
- **Run**: `python migrate_add_sessions.py`
- **Rollback**: `python migrate_add_sessions.py --rollback`

### 4. Indexing Columns
- **File**: `migrate_add_indexing_columns.py`
- **Changes**: Adds to `classroom_materials`:
  - `indexing_status VARCHAR(20)`
  - `indexed_at TIMESTAMP`
  - `chunk_count INTEGER`
  - `indexing_error TEXT`
- **Run**: `python migrate_add_indexing_columns.py`

### 5. Session Intelligence
- **File**: `scripts/migrate_session_intelligence.py`
- **Changes**: Adds to `tutor_sessions`:
  - `last_topic_vector JSONB`
  - `last_decision VARCHAR(20)`
  - `consecutive_borderline INTEGER`
- **Run**: `python scripts/migrate_session_intelligence.py`
- **Rollback**: `python scripts/migrate_session_intelligence.py --rollback`

---

## Running All Migrations

```bash
# 1. Start PostgreSQL
docker-compose up -d postgres

# 2. Run SQL migrations
psql $DATABASE_URL -f backend/core-service/migrations/init.sql
psql $DATABASE_URL -f backend/core-service/migrations/003_document_ingestion.sql

# 3. Run Python migrations
python migrate_add_sessions.py
python migrate_add_indexing_columns.py
python scripts/migrate_session_intelligence.py
```

---

## Checking Current Schema

```sql
-- List all tables
\dt

-- Describe a table
\d users
\d tutor_sessions

-- Check indexes
\di

-- Check foreign keys
SELECT conname, conrelid::regclass, confrelid::regclass
FROM pg_constraint
WHERE contype = 'f';
```
