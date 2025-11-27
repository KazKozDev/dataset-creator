#!/bin/bash

# Get the directory where this script is located
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "🚀 Restarting Application..."

# Function to kill process on port
kill_port() {
  PORT=$1
  PID=$(lsof -t -i:$PORT)
  if [ -n "$PID" ]; then
    echo "Killing process on port $PORT (PID: $PID)..."
    kill -9 $PID
  else
    echo "No process found on port $PORT."
  fi
}

# 1. Kill existing processes
echo "--------------------------------"
echo "Cleaning up ports..."
kill_port 8000 # Backend
kill_port 3000 # Frontend
echo "--------------------------------"

# 2. Start Backend in new terminal
echo "Starting Backend Server..."
osascript -e "tell application \"Terminal\" to do script \"cd '$PROJECT_DIR/backend' && uvicorn main:app --reload --port 8000\""

# 3. Start Frontend in new terminal
echo "Starting Frontend..."
osascript -e "tell application \"Terminal\" to do script \"cd '$PROJECT_DIR/frontend' && npm start\""

# 4. Wait for servers to initialize
echo "Waiting for servers to initialize (10 seconds)..."
sleep 10

# 5. Check if frontend is running
echo "Checking if frontend started..."
if lsof -i :3000 > /dev/null 2>&1; then
  echo "✅ Frontend is running on port 3000"
else
  echo "⚠️  Warning: Frontend may not have started yet"
fi

# 6. Open Browser
echo "Opening Browser..."
open "http://localhost:3000"

echo "✅ Done! App is restarting."
echo ""
echo "📝 Note: Keep the terminal windows open!"
echo "   - Backend terminal: uvicorn server"
echo "   - Frontend terminal: npm start"
echo ""
echo "Press Ctrl+C in each terminal to stop the servers."

# Don't close this window automatically - let user see the output
