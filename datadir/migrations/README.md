# Database Migrations

Scripts for migrating database schemas.

## PostgreSQL Migrations

### Run All Schemas
```bash
# From project root
psql $DATABASE_URL -f datadir/postgresql/001_core_schema.sql
psql $DATABASE_URL -f datadir/postgresql/002_documents_schema.sql
psql $DATABASE_URL -f datadir/postgresql/003_tutor_sessions_schema.sql
psql $DATABASE_URL -f datadir/postgresql/004_softskills_schema.sql
```

### Python Migrations

| Script | Purpose |
|--------|---------|
| `migrate_session_intelligence.py` | Adds session intelligence columns to tutor_sessions |
| `migrate_softskills.py` | Creates soft skills evaluation tables |

```bash
# Run a specific migration
python datadir/migrations/migrate_session_intelligence.py

# Rollback
python datadir/migrations/migrate_session_intelligence.py --rollback
```

---

## Qdrant Setup

```bash
# Create all collections
python datadir/qdrant/setup.py
```

---

## MongoDB Setup

```bash
# Run init script (usually automatic on first container start)
docker exec -it ensure-study-mongodb mongosh -f /docker-entrypoint-initdb.d/init.js
```

---

## Cassandra Setup

```bash
# Create keyspaces
docker exec -it ensure-study-cassandra cqlsh -f /datadir/cassandra/keyspaces.cql

# Create tables
docker exec -it ensure-study-cassandra cqlsh -f /datadir/cassandra/tables.cql
```

---

## Full Setup Sequence

1. Start containers: `docker-compose up -d postgres redis qdrant mongodb cassandra`
2. Wait for health: `sleep 30 && make health-check`
3. PostgreSQL schemas: Run SQL files
4. Qdrant collections: `python datadir/qdrant/setup.py`
5. Cassandra tables: Run CQL files
6. MongoDB indexes: Auto-created on first write

---

## Migration History

| Date | Migration | Description |
|------|-----------|-------------|
| 2024-01-15 | 001_core_schema | Initial user/classroom tables |
| 2024-01-20 | 002_documents_schema | Document ingestion pipeline |
| 2024-02-01 | 003_tutor_sessions_schema | AI tutor sessions |
| 2024-02-15 | 004_softskills_schema | Soft skills evaluation |
| 2024-03-01 | migrate_session_intelligence | Session intelligence fields |
