#!/bin/bash
# ============================================================================
# run-lan.sh - LAN/NETWORK Development (For Your Friend)
# ============================================================================
# Ports: Frontend 4000, Core API 9000, AI Service 9001
# Access: https://<YOUR_IP>:4000 (NETWORK ONLY - NO LOCALHOST)
# Can run SIMULTANEOUSLY with run-local.sh (uses different ports)
# ============================================================================

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
VENV_PATH="$PROJECT_ROOT/venv"
LOG_DIR="$PROJECT_ROOT/logs"

# Get local IP address
LOCAL_IP=$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || echo "0.0.0.0")

# LAN PORTS (different from run-local.sh for simultaneous running)
FRONTEND_PORT=4000
CORE_PORT=9000
AI_PORT=9001

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m'

mkdir -p "$LOG_DIR"

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║       ensureStudy LAN Development (HTTPS)                  ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo -e "${BLUE}Your IP: ${YELLOW}$LOCAL_IP${NC}"
echo -e "${BLUE}Ports: Frontend=$FRONTEND_PORT, API=$CORE_PORT, AI=$AI_PORT${NC}"
echo ""

# Kill stale processes on our ports only
echo -e "${YELLOW}Killing stale processes on ports $CORE_PORT, $AI_PORT, $FRONTEND_PORT...${NC}"
lsof -ti:$CORE_PORT | xargs kill -9 2>/dev/null || true
lsof -ti:$AI_PORT | xargs kill -9 2>/dev/null || true
lsof -ti:$FRONTEND_PORT | xargs kill -9 2>/dev/null || true
sleep 1

# Generate mkcert certificates for IP address
CERT_FILE="$PROJECT_ROOT/${LOCAL_IP}+2.pem"
KEY_FILE="$PROJECT_ROOT/${LOCAL_IP}+2-key.pem"

if [ ! -f "$CERT_FILE" ] || [ ! -f "$KEY_FILE" ]; then
    echo -e "${YELLOW}Generating mkcert certificates for $LOCAL_IP...${NC}"
    if ! command -v mkcert &> /dev/null; then
        echo -e "${RED}mkcert not found! Install with: brew install mkcert${NC}"
        exit 1
    fi
    mkcert -install
    cd "$PROJECT_ROOT"
    mkcert "$LOCAL_IP" localhost 127.0.0.1
    echo -e "${GREEN}✓ Certificates generated${NC}"
else
    echo -e "${GREEN}✓ Using existing certificates for $LOCAL_IP${NC}"
fi

# Start Docker services
if command -v docker &> /dev/null && docker info &> /dev/null 2>&1; then
    docker start ensure-study-qdrant 2>/dev/null || true
    docker start mongodb 2>/dev/null || true
fi

# Setup Python virtual environment
[ ! -d "$VENV_PATH" ] && python3 -m venv "$VENV_PATH"
[ -z "$VIRTUAL_ENV" ] && source "$VENV_PATH/bin/activate"

# Load environment
[ -f "$PROJECT_ROOT/.env" ] && export $(grep -v '^#' "$PROJECT_ROOT/.env" | xargs)

export FLASK_APP=app FLASK_DEBUG=1 PYTHONUNBUFFERED=1
export JWT_SECRET="${JWT_SECRET:-local-dev-jwt-secret-key-32chars}"
# NETWORK URLS ONLY - NO LOCALHOST
export AI_SERVICE_URL="https://$LOCAL_IP:$AI_PORT"
export CORE_SERVICE_URL="https://$LOCAL_IP:$CORE_PORT"

DATE=$(date +%Y-%m-%d)

# Start Core API (bind to 0.0.0.0 for network access)
echo -e "${GREEN}Starting Core API on https://$LOCAL_IP:$CORE_PORT${NC}"
cd "$PROJECT_ROOT/backend/core-service"
python run_https.py "$CERT_FILE" "$KEY_FILE" $CORE_PORT 2>&1 | tee -a "$LOG_DIR/core-lan_$DATE.log" &
sleep 3

# Start AI Service (bind to 0.0.0.0 for network access)
echo -e "${GREEN}Starting AI Service on https://$LOCAL_IP:$AI_PORT${NC}"
cd "$PROJECT_ROOT/backend/ai-service"
python -m uvicorn app.main:app --host 0.0.0.0 --port $AI_PORT \
    --ssl-keyfile "$KEY_FILE" --ssl-certfile "$CERT_FILE" \
    --reload 2>&1 | tee -a "$LOG_DIR/ai-lan_$DATE.log" &
sleep 3

# Start Frontend (bind to 0.0.0.0 for network access)
echo -e "${GREEN}Starting Frontend on https://$LOCAL_IP:$FRONTEND_PORT${NC}"
cd "$PROJECT_ROOT/frontend"
[ ! -d "node_modules" ] && npm install

# NETWORK URLS ONLY - NO LOCALHOST
NEXT_PUBLIC_API_URL="https://$LOCAL_IP:$CORE_PORT" \
NEXT_PUBLIC_AI_SERVICE_URL="https://$LOCAL_IP:$AI_PORT" \
NEXTAUTH_SECRET="${JWT_SECRET}" \
NEXTAUTH_URL="https://$LOCAL_IP:$FRONTEND_PORT" \
npm run dev -- -H 0.0.0.0 -p $FRONTEND_PORT --experimental-https 2>&1 | tee -a "$LOG_DIR/frontend-lan_$DATE.log" &

sleep 5

echo ""
echo -e "${GREEN}╔═══════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║       LAN Session Ready! 🔒                                       ║${NC}"
echo -e "${GREEN}╠═══════════════════════════════════════════════════════════════════╣${NC}"
echo -e "${GREEN}║  ${BLUE}SHARE THESE WITH YOUR FRIEND:${NC}                                    ${GREEN}║${NC}"
echo -e "${GREEN}║    Frontend:   ${YELLOW}https://$LOCAL_IP:$FRONTEND_PORT${NC}                          ${GREEN}║${NC}"
echo -e "${GREEN}║    Core API:   ${YELLOW}https://$LOCAL_IP:$CORE_PORT${NC}                          ${GREEN}║${NC}"
echo -e "${GREEN}║    AI Service: ${YELLOW}https://$LOCAL_IP:$AI_PORT${NC}                          ${GREEN}║${NC}"
echo -e "${GREEN}╠═══════════════════════════════════════════════════════════════════╣${NC}"
echo -e "${GREEN}║  ${RED}⚠️  TELL YOUR FRIEND: Accept certificate warning!${NC}                 ${GREEN}║${NC}"
echo -e "${GREEN}║  ${BLUE}This is FRIEND'S session. Uses IP ONLY.${NC}                           ${GREEN}║${NC}"
echo -e "${GREEN}║  ${BLUE}Use run-local.sh for YOUR localhost session.${NC}                      ${GREEN}║${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "Press Ctrl+C to stop"

trap 'echo ""; echo -e "${YELLOW}Stopping LAN services...${NC}"; pkill -P $$; exit 0' SIGINT SIGTERM
wait
