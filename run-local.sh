#!/bin/bash
# Run ensureStudy locally without Docker
# All logs are stored in ./logs directory

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
VENV_PATH="$PROJECT_ROOT/venv"
LOG_DIR="$PROJECT_ROOT/logs"

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
echo -e "${YELLOW}Killing stale processes on ports 8000, 8001, 3000...${NC}"
lsof -ti:8000 | xargs kill -9 2>/dev/null || true
lsof -ti:8001 | xargs kill -9 2>/dev/null || true
lsof -ti:3000 | xargs kill -9 2>/dev/null || true
sleep 1

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
# DATABASE_URL is loaded from .env (PostgreSQL)
export JWT_SECRET="${JWT_SECRET:-local-dev-jwt-secret-key-32chars}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-sk-test-key}"
export PYTHONUNBUFFERED=1  # Force unbuffered output for real-time logging

# Date for log files
DATE=$(date +%Y-%m-%d)

# Start Core API with logging
echo -e "${GREEN}Starting Core API on http://localhost:8000${NC}"
cd "$PROJECT_ROOT/backend/core-service"
python -m flask run --port 8000 2>&1 | tee -a "$LOG_DIR/core-service_$DATE.log" &
CORE_PID=$!

sleep 2

# Start AI Service with logging
echo -e "${GREEN}Starting AI Service on http://localhost:8001${NC}"
cd "$PROJECT_ROOT/backend/ai-service"
python -m uvicorn app.main:app --port 8001 --reload 2>&1 | tee -a "$LOG_DIR/ai-service_$DATE.log" &
AI_PID=$!

sleep 2

# Start Frontend with logging
echo -e "${GREEN}Starting Frontend on http://localhost:3000${NC}"
cd "$PROJECT_ROOT/frontend"
if [ ! -d "node_modules" ]; then
    echo "Installing npm dependencies..."
    npm install
fi
NEXTAUTH_SECRET="${JWT_SECRET:-local-dev-jwt-secret-key-32chars}" NEXTAUTH_URL="http://localhost:3000" npm run dev 2>&1 | tee -a "$LOG_DIR/frontend_$DATE.log" &
FRONTEND_PID=$!

sleep 2

# Start Streamlit Dashboards with logging
echo -e "${YELLOW}Starting Streamlit Dashboards...${NC}"
cd "$PROJECT_ROOT/dashboards"

echo -e "${GREEN}  Main Dashboard on http://localhost:8501${NC}"
streamlit run main_dashboard.py --server.port 8501 --server.headless true 2>&1 | tee -a "$LOG_DIR/dashboard_main_$DATE.log" &
DASH_MAIN_PID=$!

echo -e "${GREEN}  Notes Tester on http://localhost:8502${NC}"
streamlit run notes_tester.py --server.port 8502 --server.headless true 2>&1 | tee -a "$LOG_DIR/dashboard_notes_$DATE.log" &
DASH_NOTES_PID=$!

# Wait a bit for services to start
sleep 3

echo ""
echo -e "${GREEN}=== All services started ===${NC}"
echo "┌────────────────────────────────────────────────┐"
echo "│ Core API:       http://localhost:8000          │"
echo "│ AI Service:     http://localhost:8001          │"
echo "│ Frontend:       http://localhost:3000          │"
echo "│ Dashboard:      http://localhost:8501          │"
echo "│ Notes Tester:   http://localhost:8502          │"
echo "├────────────────────────────────────────────────┤"
echo "│ Logs:           $LOG_DIR  │"
echo "└────────────────────────────────────────────────┘"
echo ""
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
