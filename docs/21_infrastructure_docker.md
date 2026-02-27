# Page 21: Infrastructure & Docker Deployment

---

## 21.1 Overview

ensureStudy runs as a **12-service Docker Compose stack** in development and a streamlined **6-service production stack** with external managed services (AWS RDS, S3). The entire platform can be brought up with a single `make up` command.

---

## 21.2 Development Stack (12 Services)

### Source: `docker-compose.yml` (340 lines)

```mermaid
flowchart TB
    subgraph APP["APPLICATION LAYER"]
        direction LR
        FE["Frontend<br/>:3000<br/>Next.js"]
        CORE["Core API<br/>:8000<br/>Flask"]
        AI["AI Service<br/>:8001<br/>FastAPI"]
        DASH["Dashboards<br/>:8501<br/>Streamlit"]
        MLF["MLflow<br/>:5000"]
    end

    subgraph DATA["DATA LAYER"]
        direction LR
        PG["PostgreSQL<br/>:5432"]
        RD["Redis<br/>:6379"]
        QD["Qdrant<br/>:6333"]
        MDB["MongoDB<br/>:27017"]
        CAS["Cassandra<br/>:9042"]
        MINIO["MinIO<br/>:9100"]
        KUI["Kafka UI<br/>:8080"]
    end

    subgraph STREAM["STREAMING LAYER"]
        direction LR
        ZK["Zookeeper<br/>:2181"]
        KFK["Kafka<br/>:9092"]
    end

    style APP fill:#3b82f6,color:#fff
    style DATA fill:#10b981,color:#fff
    style STREAM fill:#f59e0b,color:#000
```

### Service Configuration Details

| Service | Image | Port | Healthcheck | Depends On |
|---------|-------|------|-------------|------------|
| `postgres` | postgres:15-alpine | 5432 | `pg_isready` | — |
| `redis` | redis:7-alpine | 6379 | `redis-cli ping` | — |
| `qdrant` | qdrant/qdrant:latest | 6333, 6334 | HTTP /health | — |
| `zookeeper` | confluentinc/cp-zookeeper:7.5.0 | 2181 | `nc -z` | — |
| `kafka` | confluentinc/cp-kafka:7.5.0 | 9092, 29092 | broker-api-versions | zookeeper |
| `kafka-ui` | provectuslabs/kafka-ui:latest | 8080 | — | kafka |
| `mongodb` | mongo:7 | 27017 | `mongosh ping` | — |
| `cassandra` | cassandra:4 | 9042 | `cqlsh describe` | — |
| `minio` | minio/minio:latest | 9100, 9101 | HTTP /health/live | — |
| `core-api` | Built from Dockerfile | 8000 | — | postgres, redis |
| `ai-service` | Built from Dockerfile | 8001 | — | postgres, redis, qdrant, kafka |
| `frontend` | Built from Dockerfile | 3000 | — | core-api, ai-service |
| `dashboards` | Built from Dockerfile | 8501 | — | postgres, qdrant |
| `mlflow` | ghcr.io/mlflow/mlflow:v2.9.0 | 5000 | — | postgres |

### Docker Volumes (10)

```yaml
volumes:
  postgres_data:       # PostgreSQL persistent storage
  redis_data:          # Redis AOF/RDB persistence
  qdrant_storage:      # Vector database storage
  zookeeper_data:      # Zookeeper state
  zookeeper_log:       # Zookeeper transaction logs
  kafka_data:          # Kafka message logs
  mlflow_artifacts:    # MLflow experiment artifacts
  mongo_data:          # MongoDB collections
  cassandra_data:      # Cassandra SSTables
  minio_data:          # Object storage (S3-compatible)
```

---

## 21.3 Production Stack (6 Services)

### Source: `docker-compose.prod.yml` (208 lines)

| Service | Image | Key Differences from Dev |
|---------|-------|--------------------------|
| `core-api` | Dockerfile.prod | Gunicorn, `restart: unless-stopped`, AWS S3 storage |
| `ai-service` | Dockerfile.prod | Configurable Whisper model size, Gemini API |
| `mongodb` | mongo:7 | Secret-based auth, Atlas option |
| `redis` | redis:7-alpine | Persistent, `restart: unless-stopped` |
| `qdrant` | qdrant/qdrant:latest | `restart: unless-stopped` |
| `nginx` | nginx:alpine | (Commented) SSL termination, reverse proxy |

**Excluded from prod** (use managed services): PostgreSQL (AWS RDS), Kafka (optional), Cassandra, MinIO (use S3 directly), MLflow, Dashboards, Kafka-UI.

---

## 21.4 Makefile Targets

### Source: `Makefile` (92 lines)

| Target | Command | Purpose |
|--------|---------|---------|
| `make up` | `docker-compose up -d` + wait + health-check | Start everything |
| `make down` | `docker-compose down` | Stop services |
| `make logs` | `docker-compose logs -f` | Tail logs |
| `make health-check` | Check Qdrant, PostgreSQL, Redis, Kafka | Verify all services |
| `make db-init` | `flask db upgrade` | Apply database migrations |
| `make load-docs` | `python scripts/load_documents.py` | Seed Qdrant |
| `make test` | `pytest` (core, ai, kafka) | Run all tests |
| `make test-ml` | `pytest tests/` | Run ML tests |
| `make dev-frontend` | `npm run dev` | Frontend dev server |
| `make dev-ai-service` | `uvicorn --reload` | AI service dev |
| `make dev-core-service` | `flask run` | Core service dev |
| `make clean` | `docker-compose down -v` + cleanup | Full cleanup |
| `make kafka-topics` | Create 5 Kafka topics | Initialize topics |
| `make train-moderation` | Train moderation model | ML training |
| `make train-difficulty` | Train difficulty model | ML training |
| `make run-etl` | PySpark ETL pipeline | Data pipeline |
| `make dashboards` | `streamlit run` | Start dashboards |

---

## 21.5 Network Architecture

All services communicate over a single Docker bridge network: `ensure-study`.

```mermaid
flowchart LR
    subgraph NET["ensure-study network"]
        FE["frontend:3000"] -->|HTTP| CORE["core-api:8000"]
        CORE -->|SQL| PG["postgres:5432"]
        CORE -->|HTTP| AI["ai-service:8001"]
        AI --> QD["qdrant:6333"]
        AI --> RD["redis:6379"]
        AI --> MDB["mongodb:27017"]
        AI --> KFK["kafka:29092"]
        CORE --> RD
        CORE --> KFK
        KFK <--> ZK["zookeeper:2181"]
        KUI["kafka-ui:8080"] --> KFK
        MLF["mlflow:5000"] --> PG
        DASH["dashboards:8501"] --> PG
        DASH --> QD
    end

    style NET fill:#1e293b,color:#fff
```

---

## 21.6 Launch Scripts

### `run-local.sh` — Local Development

```bash
#!/bin/bash
# Start all infrastructure services
docker-compose up -d postgres redis qdrant zookeeper kafka mongodb cassandra minio

# Wait for services to be healthy
sleep 15

# Start application services in separate terminals
# Terminal 1: Flask core service
cd backend/core-service && flask run --port 8000 &

# Terminal 2: FastAPI AI service
cd backend/ai-service && uvicorn app.main:app --port 8001 --reload &

# Terminal 3: Next.js frontend
cd frontend && npm run dev &
```

### `run-lan.sh` — LAN Access (mkcert TLS)

Uses locally-generated TLS certificates from `mkcert` for HTTPS on the local network:

```bash
# Certificate files present in project root
# 192.168.4.60+2-key.pem   / 192.168.4.60+2.pem
# 192.168.4.157+2-key.pem  / 192.168.4.157+2.pem
# localhost+2-key.pem      / localhost+2.pem
```

Enables testing on mobile devices and other machines on the same network with valid HTTPS.

---

## 21.7 AWS Production Deployment

### Recommended AWS Architecture

```mermaid
flowchart TB
    subgraph AWS["☁️ AWS Cloud"]
        subgraph EC2["EC2 Instance (t3.medium)"]
            direction TB
            DC["Docker Compose"]
            DC --> CA["Core API (Gunicorn)"]
            DC --> AIS["AI Service (Uvicorn)"]
            DC --> RD["Redis"]
            DC --> MDB["MongoDB"]
            DC --> QD["Qdrant"]
            DC --> NGX["Nginx (SSL)"]
        end

        subgraph MANAGED["Managed Services"]
            direction LR
            RDS["RDS PostgreSQL<br/>db.t3.micro"]
            S3["S3 Bucket<br/>ensurestudy-files"]
            IAM["IAM User<br/>S3 read/write"]
        end
    end

    EC2 --> RDS & S3

    style EC2 fill:#3b82f6,color:#fff
    style MANAGED fill:#f59e0b,color:#000
```
