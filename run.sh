echo "Starting Suraksha AI..."


cd "$(dirname "$0")/backend"
source venv/bin/activate
uvicorn app.main:app --reload &
BACKEND_PID=$!
echo "Backend started (PID: $BACKEND_PID)"


sleep 5


cd "$(dirname "$0")/frontend"
python3 -m http.server 3000 &
FRONTEND_PID=$!
echo "Frontend started (PID: $FRONTEND_PID)"

echo ""
echo "Backend:  http://127.0.0.1:8000"
echo "Frontend: http://127.0.0.1:3000/app.html"
echo ""
echo "Press Ctrl+C to stop everything"


trap "kill $BACKEND_PID $FRONTEND_PID; exit 0" SIGINT SIGTERM
wait