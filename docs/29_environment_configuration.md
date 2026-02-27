# Page 29: Environment Configuration & API Key Reference

---

## 29.1 Overview

ensureStudy has an extensive configuration surface spanning **114 environment variables** across `.env` (development) and `.env.production.example` (production template). This page documents every configuration variable, its purpose, and its consumer service.

---

## 29.2 Configuration Files

| File | Lines | Purpose | Git Status |
|------|-------|---------|------------|
| `.env` | 114 | Development configuration | gitignored |
| `.env.production.example` | 100 | Production template | committed |

---

## 29.3 Complete Variable Reference

### LLM & Embedding Configuration

| Variable | Example | Consumer | Purpose |
|----------|---------|----------|---------|
| `LLM_MODEL` | `mistralai/Mistral-7B-Instruct-v0.2` | AI Service | Default LLM model |
| `LLM_USE_API` | `true` | AI Service | Use API vs local model |
| `OPENAI_API_KEY` | `sk-...` | AI Service | OpenAI GPT-4 + Whisper |
| `GEMINI_API_KEY` | `AIzaSy...` | AI Service | Google Gemini (supports comma-separated rotation) |
| `GROQ_API_KEY` | `gsk_...` | AI Service | Groq fast inference |
| `MISTRAL_API_KEY` | `3Jim...` | AI Service | Mistral API |
| `HUGGINGFACE_API_KEY` | `hf_...` | AI Service | HuggingFace (supports comma-separated rotation) |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | AI Service | Text embedding model |
| `EMBEDDING_DIMENSIONS` | `1536` | AI Service | Embedding vector size |
| `WHISPER_MODEL` | `medium` | AI Service | Whisper model size (tiny/base/small/medium/large) |

### Database Configuration

| Variable | Example | Consumer | Purpose |
|----------|---------|----------|---------|
| `DATABASE_URL` | `postgresql://user:pass@localhost:5432/ensure_study` | Core Service, AI Service | Full PostgreSQL connection string |
| `DB_HOST` | `localhost` | Data Pipelines | PostgreSQL host |
| `DB_PORT` | `5432` | Data Pipelines | PostgreSQL port |
| `DB_NAME` | `ensure_study` | Data Pipelines | PostgreSQL database name |
| `DB_USER` | `ensure_study_user` | Data Pipelines | PostgreSQL username |
| `DB_PASSWORD` | `secure_password_123` | Data Pipelines | PostgreSQL password |
| `REDIS_URL` | `redis://localhost:6379` | Core, AI Service | Redis connection URL |
| `QDRANT_HOST` | `localhost` | AI Service | Qdrant host |
| `QDRANT_PORT` | `6333` | AI Service | Qdrant HTTP port |
| `QDRANT_API_KEY` | (empty) | AI Service | Qdrant auth (prod only) |
| `QDRANT_COLLECTION_NAME` | `classroom_materials` | AI Service | Default Qdrant collection |

### Authentication & Security

| Variable | Example | Consumer | Purpose |
|----------|---------|----------|---------|
| `JWT_SECRET` | `your-super-secret-key-min-32-chars-here` | Core Service | JWT signing key |
| `JWT_EXPIRATION_HOURS` | `24` | Core Service | Token lifetime |
| `REFRESH_TOKEN_SECRET` | `your-refresh-secret-min-32-chars` | Core Service | Refresh token signing |
| `NEXTAUTH_SECRET` | `your-nextauth-secret-here` | Frontend | NextAuth session encryption |
| `NEXTAUTH_URL` | `http://localhost:3000` | Frontend | Canonical URL |

### Kafka & Spark

| Variable | Example | Consumer | Purpose |
|----------|---------|----------|---------|
| `KAFKA_BOOTSTRAP_SERVERS` | `localhost:9092` | Core, AI, Kafka | Kafka broker address |
| `KAFKA_GROUP_ID` | `ensure-study-consumers` | Kafka consumers | Consumer group |
| `SPARK_MASTER` | `local[*]` | Data Pipelines | Spark cluster URL |
| `SPARK_DRIVER_MEMORY` | `4g` | Data Pipelines | Spark driver memory |
| `SPARK_EXECUTOR_MEMORY` | `4g` | Data Pipelines | Spark executor memory |

### MLflow

| Variable | Example | Consumer | Purpose |
|----------|---------|----------|---------|
| `MLFLOW_TRACKING_URI` | `http://localhost:5000` | ML Training | MLflow server URL |
| `MLFLOW_EXPERIMENT_NAME` | `ensureStudy` | ML Training | Default experiment |

### AWS Configuration

| Variable | Example | Consumer | Purpose |
|----------|---------|----------|---------|
| `AWS_ACCESS_KEY_ID` | `AKIA...` | Core, AI Service | AWS IAM key |
| `AWS_SECRET_ACCESS_KEY` | `8QYd...` | Core, AI Service | AWS IAM secret |
| `AWS_REGION` | `ap-south-1` | Core, AI Service | AWS region |
| `AWS_S3_BUCKET` | `ensure-study-datalake` | Core Service | S3 bucket name |
| `STORAGE_PROVIDER` | `local` / `s3` | Core Service | Storage backend |

### Web Search & APIs

| Variable | Example | Consumer | Purpose |
|----------|---------|----------|---------|
| `SERPER_API_KEY` | `41a6...` | AI Service | Google SERP search |
| `WEB_SEARCH_PROVIDER` | `serper` | AI Service | Search provider |
| `WEB_SEARCH_MAX_RESULTS` | `5` | AI Service | Max results per query |
| `YOUTUBE_API_KEY` | `AIzaSy...` | AI Service | YouTube Data API v3 |

### LiveKit (Video Conferencing)

| Variable | Example | Consumer | Purpose |
|----------|---------|----------|---------|
| `NEXT_PUBLIC_LIVEKIT_URL` | `wss://ensurestudy-*.livekit.cloud` | Frontend | LiveKit WebSocket URL |
| `LIVEKIT_API_KEY` | `APIQw...` | Core Service | LiveKit API authentication |
| `LIVEKIT_API_SECRET` | `qCbCT...` | Core Service | LiveKit API secret |

### Application

| Variable | Example | Consumer | Purpose |
|----------|---------|----------|---------|
| `LOG_LEVEL` | `INFO` | All services | Logging verbosity |
| `ENVIRONMENT` | `development` | All services | Running environment |
| `FRONTEND_URL` | `https://yourdomain.com` | Core Service | CORS allowed origin |

---

## 29.4 API Key Rotation

Some variables support **comma-separated key rotation**:

```bash
# Multiple keys separated by commas — the service rotates through them
HUGGINGFACE_API_KEY="hf_key1, hf_key2, hf_key3, hf_key4"
GEMINI_API_KEY="AIzaKey1, AIzaKey2"
```

This mechanism distributes API calls across multiple keys to avoid per-key rate limits.

---

## 29.5 Production Environment Template

Key differences from development:

| Setting | Development | Production |
|---------|------------|------------|
| `DATABASE_URL` | localhost PostgreSQL | AWS RDS |
| `STORAGE_PROVIDER` | `local` | `s3` |
| `FRONTEND_URL` | `http://localhost:3000` | `https://yourdomain.com` |
| `WHISPER_MODEL` | `medium` | `small` (CPU) or `medium` (GPU) |
| `MONGO_PASSWORD` | hardcoded | strong password |
| `JWT_SECRET` | placeholder | `openssl rand -base64 32` |

### Recommended AWS Resources

| Service | Spec | Notes |
|---------|------|-------|
| **EC2** | t3.small (2 GB) or t3.medium (4 GB) | Docker hosting |
| **RDS** | db.t3.micro | Free tier eligible |
| **S3** | Standard | File storage |
| **IAM** | Programmatic access | S3 read/write only |
