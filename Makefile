.PHONY: help up down logs health-check db-init load-docs test dev clean

help:
	@echo "ensureStudy - Development Commands"
	@echo ""
	@echo "  make up          - Start all Docker services"
	@echo "  make down        - Stop all Docker services"
	@echo "  make logs        - View Docker logs"
	@echo "  make health-check - Check all service health"
	@echo "  make db-init     - Initialize database schema"
	@echo "  make load-docs   - Load sample documents to Qdrant"
	@echo "  make test        - Run all tests"
	@echo "  make dev         - Start development servers"
	@echo "  make clean       - Clean up containers and volumes"

up:
	docker-compose up -d
	@echo "Waiting for services to start..."
	@sleep 30
	@make health-check

down:
	docker-compose down

logs:
	docker-compose logs -f

health-check:
	@echo "Checking Qdrant..."
	@curl -s http://localhost:6333/health | head -c 50 || echo "Qdrant not ready"
	@echo ""
	@echo "Checking PostgreSQL..."
	@docker-compose exec -T postgres pg_isready -U ensure_study_user || echo "PostgreSQL not ready"
	@echo "Checking Redis..."
	@docker-compose exec -T redis redis-cli ping || echo "Redis not ready"
	@echo "Checking Kafka..."
	@docker-compose exec -T kafka kafka-topics.sh --list --bootstrap-server localhost:9092 2>/dev/null | head -1 || echo "Kafka not ready"

db-init:
	cd backend/core-service && flask db upgrade

load-docs:
	cd backend/ai-service && python scripts/load_documents.py

test:
	cd backend/core-service && pytest tests/ -v
	cd backend/ai-service && pytest tests/ -v
	cd backend/kafka && pytest tests/ -v

test-ml:
	cd backend/ml-training && pytest tests/ -v

dev-frontend:
	cd frontend && npm run dev

dev-ai-service:
	cd backend/ai-service && uvicorn app.main:app --reload --port 8001

dev-core-service:
	cd backend/core-service && flask run --port 8000

dev:
	@echo "Start services in separate terminals:"
	@echo "  Terminal 1: make dev-frontend"
	@echo "  Terminal 2: make dev-ai-service"
	@echo "  Terminal 3: make dev-core-service"

clean:
	docker-compose down -v --remove-orphans
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "node_modules" -exec rm -rf {} + 2>/dev/null || true

train-moderation:
	cd backend/ml-training && python training/train_moderation.py

train-difficulty:
	cd backend/ml-training && python training/train_difficulty.py

run-etl:
	cd backend/data-pipelines && python -m pyspark etl/extract/extract_student_data.py

dashboards:
	cd dashboards && streamlit run live_demo.py

kafka-topics:
	docker-compose exec kafka kafka-topics.sh --create --topic student-events --bootstrap-server localhost:9092 --partitions 3 --replication-factor 1
	docker-compose exec kafka kafka-topics.sh --create --topic chat-messages --bootstrap-server localhost:9092 --partitions 3 --replication-factor 1
	docker-compose exec kafka kafka-topics.sh --create --topic assessment-submissions --bootstrap-server localhost:9092 --partitions 3 --replication-factor 1
	docker-compose exec kafka kafka-topics.sh --create --topic moderation-events --bootstrap-server localhost:9092 --partitions 3 --replication-factor 1
	docker-compose exec kafka kafka-topics.sh --create --topic leaderboard-updates --bootstrap-server localhost:9092 --partitions 3 --replication-factor 1
