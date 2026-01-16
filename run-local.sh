#!/bin/bash
# Run ensureStudy locally without Docker
# Uses different ports than run-lan.sh to avoid conflicts
# All logs are stored in ./logs directory

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
VENV_PATH="$PROJECT_ROOT/venv"
LOG_DIR="$PROJECT_ROOT/logs"

# Different ports from run-lan.sh to allow both to run
CORE_PORT=9000
AI_PORT=9001
FRONTEND_PORT=4000
DASH_MAIN_PORT=9501
DASH_NOTES_PORT=9502

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Create logs directory
mkdir -p "$LOG_DIR"

echo -e "${BLUE}=== ensureStudy Local Development ===${NC}"
echo -e "${BLUE}Logs stored in: $LOG_DIR${NC}"

# Kill any stale processes on our ports
echo -e "${YELLOW}Killing stale processes on ports $CORE_PORT, $AI_PORT, $FRONTEND_PORT...${NC}"
lsof -ti:$CORE_PORT | xargs kill -9 2>/dev/null || true
lsof -ti:$AI_PORT | xargs kill -9 2>/dev/null || true
lsof -ti:$FRONTEND_PORT | xargs kill -9 2>/dev/null || true
lsof -ti:$DASH_MAIN_PORT | xargs kill -9 2>/dev/null || true
lsof -ti:$DASH_NOTES_PORT | xargs kill -9 2>/dev/null || true
sleep 1

# Start Qdrant using docker-compose (if docker is available)
echo -e "${YELLOW}Starting Qdrant vector database...${NC}"
if command -v docker &> /dev/null; then
    # Start only qdrant service from docker-compose
    docker-compose up -d qdrant 2>/dev/null || docker compose up -d qdrant 2>/dev/null || {
        echo -e "${YELLOW}Starting Qdrant standalone...${NC}"
        docker run -d --name ensure-study-qdrant -p 6333:6333 -p 6334:6334 qdrant/qdrant 2>/dev/null || true
    }
    echo -e "${GREEN}Qdrant started on http://localhost:6333${NC}"
else
    echo -e "${RED}Docker not found - Qdrant won't be available${NC}"
fi

# Check if venv exists
if [ ! -d "$VENV_PATH" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_PATH"
fi

# Activate venv if not already activated
if [ -z "$VIRTUAL_ENV" ] || [ "$VIRTUAL_ENV" != "$VENV_PATH" ]; then
    echo -e "${YELLOW}Activating virtual environment...${NC}"
    source "$VENV_PATH/bin/activate"
else
    echo -e "${GREEN}Virtual environment already active${NC}"
fi

# Check if flask is installed
if ! python -c "import flask" 2>/dev/null; then
    echo "Installing Python dependencies..."
    pip install flask flask-cors flask-sqlalchemy flask-migrate pyjwt werkzeug redis python-dotenv gunicorn
    pip install fastapi uvicorn pydantic python-jose python-multipart aiohttp httpx openai langchain langchain-openai sentence-transformers
fi

# Check if streamlit is installed
if ! python -c "import streamlit" 2>/dev/null; then
    echo "Installing Streamlit..."
    pip install streamlit plotly pandas
fi

# Load environment variables from .env
if [ -f "$PROJECT_ROOT/.env" ]; then
    export $(grep -v '^#' "$PROJECT_ROOT/.env" | xargs)
fi

# Set environment variables - Use PostgreSQL from .env
export FLASK_APP=app
export FLASK_DEBUG=1
export JWT_SECRET="${JWT_SECRET:-local-dev-jwt-secret-key-32chars}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-sk-test-key}"
export PYTHONUNBUFFERED=1

# Date for log files
DATE=$(date +%Y-%m-%d)

# Start Core API with logging
echo -e "${GREEN}Starting Core API on http://localhost:$CORE_PORT${NC}"
cd "$PROJECT_ROOT/backend/core-service"
python -m flask run --port $CORE_PORT 2>&1 | tee -a "$LOG_DIR/core-service_$DATE.log" &
CORE_PID=$!

sleep 2

# Start AI Service with logging
echo -e "${GREEN}Starting AI Service on http://localhost:$AI_PORT${NC}"
cd "$PROJECT_ROOT/backend/ai-service"
python -m uvicorn app.main:app --port $AI_PORT --reload 2>&1 | tee -a "$LOG_DIR/ai-service_$DATE.log" &
AI_PID=$!

sleep 2

# Start Frontend with logging
echo -e "${GREEN}Starting Frontend on http://localhost:$FRONTEND_PORT${NC}"
cd "$PROJECT_ROOT/frontend"
if [ ! -d "node_modules" ]; then
    echo "Installing npm dependencies..."
    npm install
fi

# Set API URLs to use the local ports
export NEXT_PUBLIC_API_URL="http://localhost:$CORE_PORT"
export NEXT_PUBLIC_AI_URL="http://localhost:$AI_PORT"

NEXTAUTH_SECRET="${JWT_SECRET:-local-dev-jwt-secret-key-32chars}" \
NEXTAUTH_URL="http://localhost:$FRONTEND_PORT" \
npm run dev -- --port $FRONTEND_PORT 2>&1 | tee -a "$LOG_DIR/frontend_$DATE.log" &
FRONTEND_PID=$!

sleep 2

# Start Streamlit Dashboards with logging
echo -e "${YELLOW}Starting Streamlit Dashboards...${NC}"
cd "$PROJECT_ROOT/dashboards"

echo -e "${GREEN}  Main Dashboard on http://localhost:$DASH_MAIN_PORT${NC}"
streamlit run main_dashboard.py --server.port $DASH_MAIN_PORT --server.headless true 2>&1 | tee -a "$LOG_DIR/dashboard_main_$DATE.log" &
DASH_MAIN_PID=$!

echo -e "${GREEN}  Notes Tester on http://localhost:$DASH_NOTES_PORT${NC}"
streamlit run notes_tester.py --server.port $DASH_NOTES_PORT --server.headless true 2>&1 | tee -a "$LOG_DIR/dashboard_notes_$DATE.log" &
DASH_NOTES_PID=$!

# Wait a bit for services to start
sleep 3

echo ""
echo -e "${GREEN}=== All services started (LOCAL) ===${NC}"
echo "┌────────────────────────────────────────────────┐"
echo "│ Core API:       http://localhost:$CORE_PORT          │"
echo "│ AI Service:     http://localhost:$AI_PORT          │"
echo "│ Frontend:       http://localhost:$FRONTEND_PORT          │"
echo "│ Qdrant (Vector):http://localhost:6333          │"
echo "│ Dashboard:      http://localhost:$DASH_MAIN_PORT          │"
echo "│ Notes Tester:   http://localhost:$DASH_NOTES_PORT          │"
echo "├────────────────────────────────────────────────┤"
echo "│ Logs:           $LOG_DIR  │"
echo "└────────────────────────────────────────────────┘"
echo ""
echo -e "${YELLOW}NOTE: These ports differ from run-lan.sh to avoid conflicts${NC}"
echo -e "${YELLOW}Log files:${NC}"
echo "  tail -f $LOG_DIR/core-service_$DATE.log"
echo "  tail -f $LOG_DIR/ai-service_$DATE.log"
echo "  tail -f $LOG_DIR/frontend_$DATE.log"
echo ""
echo "Press Ctrl+C to stop all services"

# Trap Ctrl+C and cleanup
cleanup() {
    echo ""
    echo -e "${YELLOW}Stopping services...${NC}"
    kill $CORE_PID $AI_PID $FRONTEND_PID $DASH_MAIN_PID $DASH_NOTES_PID 2>/dev/null
    echo -e "${GREEN}All services stopped. Logs saved to $LOG_DIR${NC}"
    exit 0
}

trap cleanup SIGINT SIGTERM

# Wait for any process to exit
wait
