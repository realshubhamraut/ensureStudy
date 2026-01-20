#!/bin/bash
# Run ensureStudy for LAN access (accessible from other devices on same network)
# All logs are stored in ./logs directory

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
VENV_PATH="$PROJECT_ROOT/venv"
LOG_DIR="$PROJECT_ROOT/logs"

# Get local IP address
LOCAL_IP=$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || echo "localhost")

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Create logs directory
mkdir -p "$LOG_DIR"

echo -e "${BLUE}=== ensureStudy LAN Development ===${NC}"
echo -e "${BLUE}Local IP: $LOCAL_IP${NC}"
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

# Load environment variables from .env
if [ -f "$PROJECT_ROOT/.env" ]; then
    export $(grep -v '^#' "$PROJECT_ROOT/.env" | xargs)
fi

# Set environment variables
export FLASK_APP=app
export FLASK_DEBUG=1
export JWT_SECRET="${JWT_SECRET:-local-dev-jwt-secret-key-32chars}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-sk-test-key}"
export PYTHONUNBUFFERED=1

# Date for log files
DATE=$(date +%Y-%m-%d)

# Start Core API with LAN access (0.0.0.0)
echo -e "${GREEN}Starting Core API on http://$LOCAL_IP:8000${NC}"
cd "$PROJECT_ROOT/backend/core-service"
python -m flask run --host=0.0.0.0 --port 8000 2>&1 | tee -a "$LOG_DIR/core-service_$DATE.log" &
CORE_PID=$!

sleep 2

# Start AI Service with LAN access (0.0.0.0)
echo -e "${GREEN}Starting AI Service on http://$LOCAL_IP:8001${NC}"
cd "$PROJECT_ROOT/backend/ai-service"
python -m uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload 2>&1 | tee -a "$LOG_DIR/ai-service_$DATE.log" &
AI_PID=$!

sleep 2

# Start Frontend with LAN access (HTTPS for camera/mic on mobile)
echo -e "${GREEN}Starting Frontend on https://$LOCAL_IP:3000${NC}"
cd "$PROJECT_ROOT/frontend"
if [ ! -d "node_modules" ]; then
    echo "Installing npm dependencies..."
    npm install
fi

# Check if mkcert certs exist in project root
CERT_FILE="$PROJECT_ROOT/192.168.4.60+2.pem"
KEY_FILE="$PROJECT_ROOT/192.168.4.60+2-key.pem"

# Set API URL to use LAN IP (HTTP is fine for API calls)
export NEXT_PUBLIC_API_URL="http://$LOCAL_IP:8000"
export NEXT_PUBLIC_AI_URL="http://$LOCAL_IP:8001"

if [ -f "$CERT_FILE" ] && [ -f "$KEY_FILE" ]; then
    echo -e "${GREEN}Using HTTPS with mkcert certificates${NC}"
    # Create a custom server to use HTTPS
    NEXTAUTH_SECRET="${JWT_SECRET:-local-dev-jwt-secret-key-32chars}" \
    NEXTAUTH_URL="https://$LOCAL_IP:3000" \
    npm run dev -- -H 0.0.0.0 --experimental-https 2>&1 | tee -a "$LOG_DIR/frontend_$DATE.log" &
else
    echo -e "${YELLOW}No mkcert certs found, using HTTP (camera/mic won't work on mobile)${NC}"
    NEXTAUTH_SECRET="${JWT_SECRET:-local-dev-jwt-secret-key-32chars}" \
    NEXTAUTH_URL="http://$LOCAL_IP:3000" \
    npm run dev -- -H 0.0.0.0 2>&1 | tee -a "$LOG_DIR/frontend_$DATE.log" &
fi
FRONTEND_PID=$!

sleep 2

# Start Streamlit Dashboards with LAN access
echo -e "${YELLOW}Starting Streamlit Dashboards...${NC}"
cd "$PROJECT_ROOT/dashboards"

echo -e "${GREEN}  Main Dashboard on http://$LOCAL_IP:8501${NC}"
streamlit run main_dashboard.py --server.port 8501 --server.headless true --server.address 0.0.0.0 2>&1 | tee -a "$LOG_DIR/dashboard_main_$DATE.log" &
DASH_MAIN_PID=$!

echo -e "${GREEN}  Notes Tester on http://$LOCAL_IP:8502${NC}"
streamlit run notes_tester.py --server.port 8502 --server.headless true --server.address 0.0.0.0 2>&1 | tee -a "$LOG_DIR/dashboard_notes_$DATE.log" &
DASH_NOTES_PID=$!

# Wait a bit for services to start
sleep 3

echo ""
echo -e "${GREEN}=== All services started (LAN Accessible) ===${NC}"
echo "┌────────────────────────────────────────────────────────┐"
echo "│ Your Network IP: $LOCAL_IP                             "
echo "├────────────────────────────────────────────────────────┤"
echo "│ Core API:       http://$LOCAL_IP:8000                  "
echo "│ AI Service:     http://$LOCAL_IP:8001                  "
echo "│ Frontend:       https://$LOCAL_IP:3000  (HTTPS!)       "
echo "│ Dashboard:      http://$LOCAL_IP:8501                  "
echo "│ Notes Tester:   http://$LOCAL_IP:8502                  "
echo "├────────────────────────────────────────────────────────┤"
echo "│ For Mobile (with camera/mic):                         │"
echo "│   https://$LOCAL_IP:3000                               "
echo "└────────────────────────────────────────────────────────┘"
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
