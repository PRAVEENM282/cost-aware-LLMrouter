FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    sqlite3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install CPU-only PyTorch first to keep image lightweight and fast
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Copy requirements and install python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-cache the embedding model during build time for instant startup
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy application files
COPY app/ app/
COPY data/ data/
COPY scripts/ scripts/
COPY evaluate_router.py .
COPY metrics_report.json .
COPY .env.example .

# Make entrypoint executable
RUN chmod +x scripts/entrypoint.sh

EXPOSE 8000

HEALTHCHECK --interval=5s --timeout=5s --start-period=5s --retries=10 \
    CMD curl -f http://localhost:8000/health || exit 1

ENTRYPOINT ["/app/scripts/entrypoint.sh"]
