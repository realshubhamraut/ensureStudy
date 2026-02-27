# Page 56: Database Migrations & Schema Evolution

---

## 56.1 Overview

ensureStudy uses **Flask-Migrate** (Alembic) for SQLAlchemy model migrations and **raw SQL migration files** for complex schema changes. The `migrations/` directory contains 4 migration files that track the database schema from initial setup through feature additions.

---

## 56.2 Migration Tooling

| Tool | Purpose |
|------|---------|
| Flask-Migrate | Auto-generate migrations from SQLAlchemy model changes |
| Alembic | Underlying migration engine |
| Raw SQL | Complex DDL changes, indexes, data migrations |

### Commands

```bash
# Auto-generate migration from model changes
flask db migrate -m "Add new fields to progress"

# Apply all pending migrations
flask db upgrade

# Rollback last migration
flask db downgrade

# Show current migration version
flask db current

# Show migration history
flask db history
```

---

## 56.3 Migration Files

### `init.sql` — Initial Schema

Creates all core tables:

```sql
-- Users
CREATE TABLE users (
    id VARCHAR(36) PRIMARY KEY,
    username VARCHAR(80) UNIQUE NOT NULL,
    email VARCHAR(120) UNIQUE NOT NULL,
    password_hash VARCHAR(256) NOT NULL,
    role VARCHAR(20) DEFAULT 'student',
    first_name VARCHAR(50),
    last_name VARCHAR(50),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Classrooms
CREATE TABLE classrooms (
    id VARCHAR(36) PRIMARY KEY,
    name VARCHAR(200) NOT NULL,
    teacher_id VARCHAR(36) REFERENCES users(id),
    join_code VARCHAR(8) UNIQUE,
    subject VARCHAR(100),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Progress
CREATE TABLE progress (
    id VARCHAR(36) PRIMARY KEY,
    user_id VARCHAR(36) REFERENCES users(id),
    topic VARCHAR(200),
    confidence_score FLOAT DEFAULT 0.0,
    times_studied INTEGER DEFAULT 0,
    is_weak BOOLEAN DEFAULT FALSE,
    tal_level INTEGER DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ... (40+ tables total)
```

### `003_document_ingestion.sql` — Document Processing

```sql
-- Document Intelligence metadata
CREATE TABLE document_intelligence (
    id VARCHAR(36) PRIMARY KEY,
    document_id VARCHAR(36) REFERENCES documents(id),
    total_pages INTEGER,
    has_images BOOLEAN DEFAULT FALSE,
    has_tables BOOLEAN DEFAULT FALSE,
    language VARCHAR(10) DEFAULT 'en',
    ocr_confidence FLOAT,
    processing_time_ms INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Document chunks for vector search
CREATE TABLE document_chunks (
    id VARCHAR(36) PRIMARY KEY,
    document_id VARCHAR(36) REFERENCES documents(id),
    chunk_index INTEGER,
    text TEXT,
    page_number INTEGER,
    qdrant_point_id VARCHAR(36),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### `004_add_ocr_bboxes.sql` — OCR Bounding Boxes

```sql
ALTER TABLE document_intelligence
    ADD COLUMN ocr_bboxes JSONB,
    ADD COLUMN text_regions JSONB,
    ADD COLUMN layout_analysis JSONB;
```

### `add_learning_agent_tables.py` — Learning Agent Memory

```python
def upgrade():
    op.create_table('learning_agent_memory',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('topic_id', sa.String(36)),
        sa.Column('strategy', sa.JSON),
        sa.Column('critic_scores', sa.JSON),
        sa.Column('iteration', sa.Integer, default=0),
        sa.Column('created_at', sa.DateTime, default=datetime.utcnow),
        sa.Column('updated_at', sa.DateTime)
    )
    
    op.create_table('question_effectiveness',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('question_id', sa.String(36)),
        sa.Column('times_asked', sa.Integer, default=0),
        sa.Column('correct_rate', sa.Float, default=0.0),
        sa.Column('discrimination_index', sa.Float)
    )

def downgrade():
    op.drop_table('question_effectiveness')
    op.drop_table('learning_agent_memory')
```

---

## 56.4 Seed Data Scripts

| Script | Purpose |
|--------|---------|
| `seed_database.py` | Create demo users, classrooms, subjects, topics |
| `seed_progress_data.py` | Generate progress records, leaderboard entries |

---

## 56.5 Migration Best Practices

| Practice | Implementation |
|----------|---------------|
| **Atomic migrations** | Each file does one logical change |
| **Reversible** | Every `upgrade()` has a `downgrade()` |
| **Idempotent** | `CREATE TABLE IF NOT EXISTS` where possible |
| **Data-safe** | `ALTER TABLE ADD COLUMN` (never drop in prod) |
| **Ordered** | Numeric prefixes ensure correct sequence |
