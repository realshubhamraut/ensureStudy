# ensureStudy

AI-First Learning Platform with RAG, Kafka, PyTorch, and PySpark

## Quick Start

```bash
# Clone and setup
git clone <repo-url> ensure-study
cd ensure-study
cp .env.example .env
# Edit .env with your API keys

# Start all services
docker-compose up -d

# Wait for services (30s)
sleep 30

# Verify services
make health-check

# Initialize database
make db-init

# Load sample documents
make load-docs
```

## Architecture


- **AI Service**: FastAPI + LangChain + Qdrant RAG
- **Core Service**: Flask + PostgreSQL + JWT Auth
- **Data Pipeline**: PySpark ETL/ELT + Airflow
- **Streaming**: Kafka + Spark Structured Streaming
- **ML Training**: PyTorch + MLflow
- **Dashboards**: Streamlit

## Project Structure

```
ensure-study/
├── frontend/                 # Next.js Web App
├── backend/
│   ├── ai-service/          # FastAPI RAG + Agents
│   ├── core-service/        # Flask Auth + DB
│   ├── ml-training/         # PyTorch Models
│   ├── data-pipelines/      # PySpark ETL
│   └── kafka/               # Kafka Producers/Consumers
├── dashboards/              # Streamlit Evaluation
├── docs/                    # Documentation
└── scripts/                 # Utility Scripts
```

## Development

```bash
# Run tests
make test

# Start dev servers
make dev

# View logs
make logs
```

## License

MIT
