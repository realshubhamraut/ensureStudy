# Page 24: Observability, Logging & Monitoring

---

## 24.1 Overview

ensureStudy implements observability across **request logging, application logging, ML experiment tracking, data dashboards, and health monitoring**. The system uses structured logging, MLflow for experiment management, Streamlit for dashboards, and Docker healthchecks for service monitoring.

---

## 24.2 Request Logging (AI Service)

### Source: `backend/ai-service/app/main.py`

Every HTTP request is logged with execution time:

```python
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start
    
    logger.info(
        f"{request.method} {request.url.path} → {response.status_code} ({duration:.2f}s)"
    )
    return response
```

### Frontend Action Logging

```python
@app.post("/api/log-action")
async def log_frontend_action(request: Request):
    """Log frontend actions for debugging.
    Body: { action, target, details }
    """
    body = await request.json()
    logger.info(f"[Frontend] {body['action']}: {body['target']} - {body.get('details', '')}")
```

---

## 24.3 Application Logging

### Log Format

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
```

### Per-Module Loggers

| Module | Logger Name | Key Events |
|--------|-------------|------------|
| Orchestrator | `orchestrator` | Agent selection, routing decisions |
| Tutor Agent | `tutor_agent` | TAL level changes, ABCR transitions |
| RAG Pipeline | `rag_service` | Query rewriting, retrieval stats |
| Document Agent | `document_agent` | Processing stages, OCR results |
| Proctoring | `proctor.session` | Frame analysis, flag triggers |
| Web Ingest | `web_ingest` | Crawl status, PDF downloads |
| Curriculum | `curriculum_agent` | Topology sort, scheduling |
| Learning Agent | `learning_agent` | Critic scores, strategy updates |

### Log Levels Usage

| Level | Usage |
|-------|-------|
| `DEBUG` | Detailed processing steps, prompt contents |
| `INFO` | Request flow, agent decisions, timing |
| `WARNING` | Rate limits, fallback triggers, degraded performance |
| `ERROR` | API failures, model errors, database issues |
| `CRITICAL` | Service crashes, data corruption |

---

## 24.4 Docker Container Logging

```bash
# View all service logs
docker-compose logs -f

# View specific service
docker-compose logs -f ai-service

# View with timestamps and tail
docker-compose logs -f --tail=100 --timestamps core-api
```

### Log Directory

Source: `logs/` directory stores persistent log files for debugging.

---

## 24.5 MLflow Experiment Tracking

### Source: `docker-compose.yml` (MLflow service)

```yaml
mlflow:
  image: ghcr.io/mlflow/mlflow:v2.9.0
  ports:
    - "5000:5000"
  environment:
    MLFLOW_BACKEND_STORE_URI: postgresql://...@postgres:5432/ensure_study
    MLFLOW_DEFAULT_ARTIFACT_ROOT: /mlflow/artifacts
```

### Tracked Experiments

| Experiment | Metrics | Artifacts |
|-----------|---------|-----------|
| Engagement Model | MSE, val_loss, epoch | `engagement_model.pth` |
| Content Recommendation | NDCG, hit_rate | Model weights |
| Difficulty Predictor | accuracy, F1 per class | Model weights |
| Proctoring Static | precision, recall, F1 | LightGBM `.pkl` |
| Proctoring Temporal | accuracy, AUC-ROC | LSTM `.pt` |
| Filler Detection | accuracy, per-class F1 | XGBoost `.joblib` |

### MLflow Usage Pattern

```python
import mlflow

with mlflow.start_run(experiment_id="engagement"):
    mlflow.log_param("hidden_dims", [64, 32, 16])
    mlflow.log_param("learning_rate", 0.001)
    mlflow.log_param("epochs", 50)
    
    # Training loop...
    
    mlflow.log_metric("train_loss", train_loss, step=epoch)
    mlflow.log_metric("val_loss", val_loss, step=epoch)
    mlflow.log_metric("best_val_loss", best_val_loss)
    
    mlflow.pytorch.log_model(model, "engagement_model")
```

---

## 24.6 Streamlit Dashboards

### Source: `dashboards/` directory

```yaml
dashboards:
  ports: ["8501:8501"]
  command: streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

Dashboard capabilities:
- **Real-time metrics**: Active users, assessments completed, API latency
- **Qdrant stats**: Collection sizes, query performance
- **Kafka monitoring**: Topic lag, consumer group status
- **Student analytics**: Engagement scores, progress distributions

---

## 24.7 Health Monitoring

### Healthcheck Configuration

| Service | Check Method | Interval | Retries |
|---------|-------------|----------|---------|
| PostgreSQL | `pg_isready` | 10s | 5 |
| Redis | `redis-cli ping` | 10s | 5 |
| Qdrant | HTTP `/health` | 10s | 5 |
| Zookeeper | `nc -z :2181` | 10s | 5 |
| Kafka | `kafka-broker-api-versions` | 10s | 5 |
| MongoDB | `mongosh ping` | 10s | 5 |
| Cassandra | `cqlsh describe` | 30s | 5 |
| MinIO | HTTP `/minio/health/live` | 30s | 3 |
| Core API (prod) | HTTP `/health` | 30s | 3 |
| AI Service (prod) | HTTP `/health` | 30s | 3 |

### Health Endpoints

```python
# Core Service
@app.route('/health')
def health():
    return {'status': 'healthy', 'service': 'core-api'}

# AI Service
@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "ai-tutor", "version": "2.0.0"}
```

### Makefile Health Check

```bash
make health-check
# Checking Qdrant... {"title":"qdrant","version":"1.x"}
# Checking PostgreSQL... /var/run/postgresql:5432 - accepting connections
# Checking Redis... PONG
# Checking Kafka... student-events, chat-messages, ...
```

---

## 24.8 Agent Performance Tracking

### Source: `backend/core-service/app/models/feedback.py`

| Model | Purpose |
|-------|---------|
| `AgentInteraction` | Logs every agent invocation (agent type, duration, tokens used) |
| `InteractionFeedback` | Student feedback on agent responses (helpful/not helpful) |
| `LearningExample` | Stores positive/negative examples for agent improvement |
| `AgentPerformanceMetrics` | Aggregated metrics per agent (avg response time, satisfaction) |

---

## 24.9 Error Tracking (Production)

### Sentry Integration (Optional)

```bash
# .env.production.example
SENTRY_DSN=https://your-sentry-dsn
```

When configured, Sentry captures:
- Unhandled exceptions in both services
- Performance transactions
- Breadcrumbs for debugging
- Release tracking
