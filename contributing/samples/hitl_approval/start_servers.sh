#!/bin/bash
# Copyright 2026 Google LLC

# 1. Kill any lingering local servers from previous runs to free up ports
killall python uvicorn streamlit adk 2>/dev/null || true
sleep 1

# 2. Ensure we're running from the repo root so imports resolve correctly
cd "$(git rev-parse --show-toplevel)"

# 3. Load GOOGLE_GENAI_API_KEY from .env if present
if [ -f .env ]; then
  source .env
fi

echo "Starting FastAPI HITL Backend (:8000)..."
export HITL_DB_PATH="./contributing/samples/hitl_approval/hitl.db"
.venv/bin/uvicorn google.adk_community.services.hitl_approval.api:app --port 8000 &
API_PID=$!

echo "Starting Streamlit Dashboard (:8501)..."
STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
  .venv/bin/streamlit run contributing/samples/hitl_approval/dashboard/app.py \
  --server.headless true &
STREAMLIT_PID=$!

echo "Starting ADK Web Chat (:8080)..."
.venv/bin/adk web contributing/samples/hitl_approval --port 8080 &
ADK_PID=$!

echo ""
echo "All services launched."
echo "=========================================="
echo "Backend API:    http://localhost:8000/docs"
echo "Dashboard UI:   http://localhost:8501"
echo "ADK Agent Chat: http://localhost:8080"
echo "=========================================="
echo "Press Ctrl+C to shut down all servers."

trap "kill $API_PID $STREAMLIT_PID $ADK_PID 2>/dev/null; exit" EXIT

wait
