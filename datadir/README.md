# ensureStudy - Centralized Database Directory

> Complete database schemas, configurations, and documentation for all datastores.

## 📊 Database Overview

| Database | Purpose | Port | Container | Docker Volume |
|----------|---------|------|-----------|---------------|
| **PostgreSQL 15** | Primary relational DB (users, classrooms, sessions) | 5432 | `ensure-study-postgres` | `postgres_data:/var/lib/postgresql/data` |
| **Qdrant** | Vector embeddings for RAG | 6333, 6334 | `ensure-study-qdrant` | `qdrant_storage:/qdrant/storage` |
| **Redis 7** | Caching, sessions, rate limiting | 6379 | `ensure-study-redis` | `redis_data:/data` |
| **MongoDB 7** | Meeting transcripts & summaries | 27017 | `ensure-study-mongodb` | `mongo_data:/data/db` |
| **Cassandra 4** | Real-time meeting analytics | 9042 | `ensure-study-cassandra` | `cassandra_data:/var/lib/cassandra` |
| **Kafka** | Event streaming | 9092 | `ensure-study-kafka` | `kafka_data:/var/lib/kafka/data` |
| **MinIO** | Object storage (S3-compatible) | 9000 | `ensure-study-minio` | (external) |

---

## 📁 Directory Structure

```
datadir/
├── README.md                          # This file
├── docker-volumes.md                  # Docker volume locations
│
├── postgresql/                        # PostgreSQL schemas
│   ├── 001_core_schema.sql           # Users, classrooms, enrollments
│   ├── 002_documents_schema.sql      # Document ingestion tables
│   ├── 003_tutor_sessions_schema.sql # AI tutor session tables
│   ├── 004_softskills_schema.sql     # Soft skills evaluation
│   └── init.sql                      # Docker init script
│
├── qdrant/                           # Qdrant vector collections
│   ├── collections.md                # Collection definitions
│   └── setup.py                      # Python setup script
│
├── redis/                            # Redis key patterns
│   └── schema.md                     # Key patterns & TTLs
│
├── mongodb/                          # MongoDB schemas
│   ├── collections.md                # Collection definitions
│   └── init.js                       # Init script
│
├── cassandra/                        # Cassandra schemas
│   ├── keyspaces.cql                 # Keyspace definitions
│   └── tables.cql                    # Table schemas
│
└── migrations/                       # Migration scripts
    ├── README.md                     # Migration guide
    └── *.py                          # Python migration scripts
```

---

## 🔌 Connection Strings

### Development (localhost)

```env
# PostgreSQL
DATABASE_URL=postgresql://ensure_study_user:secure_password_123@localhost:5432/ensure_study

# Qdrant
QDRANT_HOST=localhost
QDRANT_PORT=6333

# Redis
REDIS_URL=redis://localhost:6379/0

# MongoDB
MONGODB_URL=mongodb://ensure_study:mongodb_password_123@localhost:27017/ensure_study_meetings

# Cassandra
CASSANDRA_HOST=localhost
CASSANDRA_PORT=9042
```

### Docker Network (internal)

```env
# PostgreSQL
DATABASE_URL=postgresql://ensure_study_user:secure_password_123@postgres:5432/ensure_study

# Qdrant
QDRANT_HOST=qdrant
QDRANT_PORT=6333

# Redis
REDIS_URL=redis://redis:6379

# MongoDB
MONGODB_URL=mongodb://ensure_study:mongodb_password_123@mongodb:27017/ensure_study_meetings

# Cassandra
CASSANDRA_HOST=cassandra
CASSANDRA_PORT=9042
```

---

## 🚀 Quick Commands

### Start All Databases
```bash
docker-compose up -d postgres redis qdrant mongodb cassandra
```

### Check Database Health
```bash
# PostgreSQL
docker exec ensure-study-postgres pg_isready -U ensure_study_user

# Redis
docker exec ensure-study-redis redis-cli ping

# Qdrant
curl http://localhost:6333/health

# MongoDB
docker exec ensure-study-mongodb mongosh --eval "db.adminCommand('ping')"

# Cassandra
docker exec ensure-study-cassandra cqlsh -e "describe keyspaces"
```

### Run Migrations
```bash
# PostgreSQL - Full schema
psql $DATABASE_URL -f datadir/postgresql/001_core_schema.sql
psql $DATABASE_URL -f datadir/postgresql/002_documents_schema.sql
psql $DATABASE_URL -f datadir/postgresql/003_tutor_sessions_schema.sql
psql $DATABASE_URL -f datadir/postgresql/004_softskills_schema.sql

# Python migrations
python datadir/migrations/migrate_session_intelligence.py
```

### Backup Databases
```bash
# PostgreSQL dump
docker exec ensure-study-postgres pg_dump -U ensure_study_user ensure_study > backup.sql

# MongoDB dump
docker exec ensure-study-mongodb mongodump --out /data/backup

# Redis snapshot
docker exec ensure-study-redis redis-cli BGSAVE
```

---

## 📈 Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                          ensureStudy Data Flow                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  User Request                                                        │
│       │                                                              │
│       ▼                                                              │
│  ┌─────────┐     ┌─────────┐     ┌─────────┐                        │
│  │  Redis  │────▶│  Check  │────▶│PostgreSQL│                       │
│  │ (Cache) │     │  Cache  │     │ (Auth)   │                       │
│  └─────────┘     └─────────┘     └─────────┘                        │
│       │                                │                             │
│       ▼                                ▼                             │
│  ┌─────────┐                    ┌─────────┐                         │
│  │ Qdrant  │◀───────────────────│   AI    │                         │
│  │ (RAG)   │                    │ Service │                         │
│  └─────────┘                    └─────────┘                         │
│       │                                │                             │
│       ▼                                ▼                             │
│  ┌─────────┐                    ┌─────────┐                         │
│  │ MongoDB │◀───────────────────│ Kafka   │                         │
│  │(Meetings│                    │(Events) │                         │
│  └─────────┘                    └─────────┘                         │
│       │                                                              │
│       ▼                                                              │
│  ┌─────────┐                                                        │
│  │Cassandra│                                                        │
│  │(Analytics│                                                       │
│  └─────────┘                                                        │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📝 License

MIT - ensureStudy
