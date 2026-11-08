#!/bin/bash
set -e

echo "Starting Semantic Router Gateway Service..."

# Ensure data directory exists
mkdir -p data

# Automatically seed the SQLite database
echo "Checking and seeding SQLite revenue database..."
python3 scripts/seed_db.py

# Launch FastAPI gateway via uvicorn
PORT="${PORT:-8000}"
echo "Starting API on port ${PORT}..."
exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT}"
