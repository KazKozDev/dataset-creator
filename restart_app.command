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
osascript -e "tell application \"Terminal\" to do script \"cd '$PROJECT_DIR' && python3 backend/main.py\""

# 3. Start Frontend in new terminal
echo "Starting Frontend..."
osascript -e "tell application \"Terminal\" to do script \"cd '$PROJECT_DIR/frontend' && npm start\""

# 4. Wait a bit for servers to initialize
echo "Waiting for servers to initialize (5 seconds)..."
sleep 5

# 5. Open Browser
echo "Opening Browser..."
open "http://localhost:3000"

echo "✅ Done! App is restarting."

# Close this terminal window
osascript -e 'tell application "Terminal" to close front window' &
