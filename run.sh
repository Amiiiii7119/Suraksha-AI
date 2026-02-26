#!/bin/bash

# ─────────────────────────────────────────────
# SURAKSHA AI — One Command Startup
# ─────────────────────────────────────────────

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo ""
echo -e "${RED}███████╗██╗   ██╗██████╗  █████╗ ██╗  ██╗███████╗██╗  ██╗ █████╗ ${NC}"
echo -e "${RED}██╔════╝██║   ██║██╔══██╗██╔══██╗██║ ██╔╝██╔════╝██║  ██║██╔══██╗${NC}"
echo -e "${RED}███████╗██║   ██║██████╔╝███████║█████╔╝ ███████╗███████║███████║${NC}"
echo -e "${RED}╚════██║██║   ██║██╔══██╗██╔══██║██╔═██╗ ╚════██║██╔══██║██╔══██║${NC}"
echo -e "${RED}███████║╚██████╔╝██║  ██║██║  ██║██║  ██╗███████║██║  ██║██║  ██║${NC}"
echo -e "${RED}╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝${NC}"
echo -e "${GREEN}              AI-Powered Workplace Safety Platform${NC}"
echo ""

# ── Check we're in the right directory ──
if [ ! -d "backend" ]; then
  echo -e "${RED}[ERROR]${NC} 'backend' folder not found."
  echo -e "        Run this script from your project root:"
  echo -e "        ${YELLOW}cd /path/to/your/project && bash run.sh${NC}"
  exit 1
fi

if [ ! -f "frontend/app.html" ]; then
  echo -e "${RED}[ERROR]${NC} 'frontend/app.html' not found."
  exit 1
fi

# ── Kill anything already on ports 8000 / 3000 ──
echo -e "${YELLOW}[CLEANUP]${NC} Freeing ports 8000 and 3000..."
fuser -k 8000/tcp 2>/dev/null
fuser -k 3000/tcp 2>/dev/null
sleep 1

# ── Activate virtualenv if it exists ──
if [ -d "backend/venv" ]; then
  echo -e "${GREEN}[VENV]${NC} Activating virtual environment..."
  source backend/venv/bin/activate
elif [ -d "venv" ]; then
  echo -e "${GREEN}[VENV]${NC} Activating virtual environment..."
  source venv/bin/activate
else
  echo -e "${YELLOW}[VENV]${NC} No venv found, using system Python."
fi

# ── Start Backend ──
echo ""
echo -e "${GREEN}[BACKEND]${NC} Starting FastAPI on http://localhost:8000 ..."
cd backend
# NOTE: --reload is removed because it conflicts with Pathway's pw.run() threading
uvicorn app.main:app --host 0.0.0.0 --port 8000 > ../backend.log 2>&1 &
BACKEND_PID=$!
cd ..

# Wait for backend to be ready
echo -e "${YELLOW}[WAIT]${NC} Waiting for backend to start..."
for i in {1..20}; do
  sleep 1
  if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo -e "${GREEN}[BACKEND]${NC} ✅ Backend is up!"
    break
  fi
  echo -n "."
done
echo ""

# Show last few lines of backend log to confirm Pathway started
echo -e "${GREEN}[BACKEND LOG]${NC}"
tail -5 backend.log 2>/dev/null
echo ""

# ── Start Frontend ──
echo -e "${GREEN}[FRONTEND]${NC} Starting frontend server on http://localhost:3000 ..."
cd frontend
python3 -m http.server 3000 --bind 127.0.0.1 &
FRONTEND_PID=$!
cd ..
sleep 1

# Auto-open app.html directly
if command -v xdg-open &> /dev/null; then
  sleep 1 && xdg-open "http://127.0.0.1:3000/app.html" &
elif command -v open &> /dev/null; then
  sleep 1 && open "http://127.0.0.1:3000/app.html" &
fi

# ── Done ──
echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}  ✅ SURAKSHA AI IS RUNNING${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "  🌐 Frontend  → ${YELLOW}http://localhost:3000/app.html${NC}"
echo -e "  ⚙️  Backend   → ${YELLOW}http://localhost:8000${NC}"
echo -e "  📖 API Docs  → ${YELLOW}http://localhost:8000/docs${NC}"
echo ""
echo -e "  🔐 Default login:"
echo -e "     Email    : ${YELLOW}amteshwarrajsingh@gmail.com${NC}"
echo -e "     Password : ${YELLOW}admin123${NC}"
echo ""
echo -e "  Press ${RED}Ctrl+C${NC} to stop everything."
echo ""

# ── Wait and cleanup on Ctrl+C ──
trap "echo ''; echo -e '${RED}[SHUTDOWN]${NC} Stopping servers...'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; echo -e '${GREEN}[DONE]${NC} All stopped. Goodbye!'; exit 0" SIGINT SIGTERM

wait