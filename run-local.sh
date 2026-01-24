#!/bin/bash
# ============================================================================
# run-local.sh - LOCAL Development Only (Your Machine)
# ============================================================================
# Ports: Frontend 3000, Core API 8000, AI Service 8001
# Access: https://localhost:3000 (YOUR MACHINE ONLY)
# Can run SIMULTANEOUSLY with run-lan.sh (uses different ports)
# ============================================================================

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
VENV_PATH="$PROJECT_ROOT/venv"
LOG_DIR="$PROJECT_ROOT/logs"

# LOCAL PORTS (different from run-lan.sh for simultaneous running)
FRONTEND_PORT=3000
CORE_PORT=8000
AI_PORT=8001

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m'

mkdir -p "$LOG_DIR"

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║       ensureStudy LOCAL Development (HTTPS)                ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo -e "${BLUE}Ports: Frontend=$FRONTEND_PORT, API=$CORE_PORT, AI=$AI_PORT${NC}"
echo ""

# Kill stale processes on our ports only
echo -e "${YELLOW}Killing stale processes on ports $CORE_PORT, $AI_PORT, $FRONTEND_PORT...${NC}"
lsof -ti:$CORE_PORT | xargs kill -9 2>/dev/null || true
lsof -ti:$AI_PORT | xargs kill -9 2>/dev/null || true
lsof -ti:$FRONTEND_PORT | xargs kill -9 2>/dev/null || true
sleep 1

# Generate mkcert certificates for localhost
CERT_FILE="$PROJECT_ROOT/localhost+2.pem"
KEY_FILE="$PROJECT_ROOT/localhost+2-key.pem"

if [ ! -f "$CERT_FILE" ] || [ ! -f "$KEY_FILE" ]; then
    echo -e "${YELLOW}Generating mkcert certificates for localhost...${NC}"
    if ! command -v mkcert &> /dev/null; then
        echo -e "${RED}mkcert not found! Install with: brew install mkcert${NC}"
        exit 1
    fi
    mkcert -install
    cd "$PROJECT_ROOT"
    mkcert localhost 127.0.0.1 ::1
    echo -e "${GREEN}✓ Certificates generated${NC}"
else
    echo -e "${GREEN}✓ Using existing certificates${NC}"
fi

# Start Docker services
if command -v docker &> /dev/null && docker info &> /dev/null 2>&1; then
    docker start ensure-study-qdrant 2>/dev/null || docker run -d --name ensure-study-qdrant -p 6333:6333 -p 6334:6334 qdrant/qdrant 2>/dev/null || true
    docker start mongodb 2>/dev/null || docker run -d --name mongodb -p 27017:27017 mongo:latest 2>/dev/null || true
fi

# Setup Python virtual environment
[ ! -d "$VENV_PATH" ] && python3 -m venv "$VENV_PATH"
[ -z "$VIRTUAL_ENV" ] && source "$VENV_PATH/bin/activate"

# Load environment
[ -f "$PROJECT_ROOT/.env" ] && export $(grep -v '^#' "$PROJECT_ROOT/.env" | xargs)

export FLASK_APP=app FLASK_DEBUG=1 PYTHONUNBUFFERED=1
export JWT_SECRET="${JWT_SECRET:-local-dev-jwt-secret-key-32chars}"
export AI_SERVICE_URL="https://localhost:$AI_PORT"
export CORE_SERVICE_URL="https://localhost:$CORE_PORT"

DATE=$(date +%Y-%m-%d)

# Start Core API (localhost only)
echo -e "${GREEN}Starting Core API on https://localhost:$CORE_PORT${NC}"
cd "$PROJECT_ROOT/backend/core-service"
python run_https.py "$CERT_FILE" "$KEY_FILE" $CORE_PORT 2>&1 | tee -a "$LOG_DIR/core-local_$DATE.log" &
sleep 3

# Start AI Service (localhost only)
echo -e "${GREEN}Starting AI Service on https://localhost:$AI_PORT${NC}"
cd "$PROJECT_ROOT/backend/ai-service"
python -m uvicorn app.main:app --host 127.0.0.1 --port $AI_PORT \
    --ssl-keyfile "$KEY_FILE" --ssl-certfile "$CERT_FILE" \
    --reload 2>&1 | tee -a "$LOG_DIR/ai-local_$DATE.log" &
sleep 3

# Start Frontend (localhost only)
echo -e "${GREEN}Starting Frontend on https://localhost:$FRONTEND_PORT${NC}"
cd "$PROJECT_ROOT/frontend"
[ ! -d "node_modules" ] && npm install

NEXT_PUBLIC_API_URL="https://localhost:$CORE_PORT" \
NEXT_PUBLIC_AI_SERVICE_URL="https://localhost:$AI_PORT" \
NEXTAUTH_SECRET="${JWT_SECRET}" \
NEXTAUTH_URL="https://localhost:$FRONTEND_PORT" \
npm run dev -- --port $FRONTEND_PORT --experimental-https 2>&1 | tee -a "$LOG_DIR/frontend-local_$DATE.log" &

sleep 5

echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║       LOCAL Session Ready! 🔒                              ║${NC}"
echo -e "${GREEN}╠════════════════════════════════════════════════════════════╣${NC}"
echo -e "${GREEN}║  Frontend:     ${YELLOW}https://localhost:$FRONTEND_PORT${NC}                     ${GREEN}║${NC}"
echo -e "${GREEN}║  Core API:     ${YELLOW}https://localhost:$CORE_PORT${NC}                     ${GREEN}║${NC}"
echo -e "${GREEN}║  AI Service:   ${YELLOW}https://localhost:$AI_PORT${NC}                     ${GREEN}║${NC}"
echo -e "${GREEN}╠════════════════════════════════════════════════════════════╣${NC}"
echo -e "${GREEN}║  ${BLUE}This is YOUR session. Uses localhost ONLY.${NC}               ${GREEN}║${NC}"
echo -e "${GREEN}║  ${BLUE}Run run-lan.sh in another terminal for friend access.${NC}   ${GREEN}║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "Press Ctrl+C to stop"

trap 'echo ""; echo -e "${YELLOW}Stopping LOCAL services...${NC}"; pkill -P $$; exit 0' SIGINT SIGTERM
wait
