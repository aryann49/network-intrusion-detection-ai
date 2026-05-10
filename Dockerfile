# ─────────────────────────────────────────────
# Dockerfile
# AI-Based Network Intrusion Detection System
# ─────────────────────────────────────────────
#
# Build:   docker build -t netguard-ids .
# Run:     docker-compose up
#

# Use official Python slim image — smaller size, faster build
FROM python:3.11-slim

# Metadata
LABEL maintainer="Aryan — MCA, Amity University Noida"
LABEL description="AI-Based Network Intrusion Detection System"
LABEL version="2.0"

# ── System dependencies ───────────────────────
# libgomp1 is required by XGBoost and LightGBM on Linux
RUN apt-get update && apt-get install -y \
    libgomp1 \
    gcc \
    g++ \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────
WORKDIR /app

# ── Install Python dependencies ───────────────
# Copy requirements first — Docker caches this layer
# So if only code changes, pip install is skipped on rebuild
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# ── Copy project files ────────────────────────
COPY app.py        .
COPY train.py      .
COPY database.py   .
COPY templates/    ./templates/
COPY static/       ./static/

# Copy scripts folder if it exists
COPY scripts/      ./scripts/

# ── Create required directories ───────────────
# data/ and models/ are mounted as volumes (see docker-compose.yml)
# These are created here just in case volume mount is missing
RUN mkdir -p data models logs static/reports

# ── Environment variables ─────────────────────
# These can be overridden in docker-compose.yml or at runtime
ENV IDS_PORT=5000
ENV IDS_HOST=0.0.0.0
ENV IDS_DEBUG=0
ENV IDS_SECRET_KEY=netguard_docker_secret_2024
ENV IDS_CORS_ORIGINS=
ENV IDS_SHOW_LOGIN_HINT=0
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# ── Expose port ───────────────────────────────
EXPOSE 5000

# ── Health check ─────────────────────────────
# Docker will check if the server is responding every 30s
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000')" \
  || exit 1

# ── Entrypoint ────────────────────────────────
# Runs app.py — which starts Flask + SocketIO + IDS engine thread
CMD ["python", "app.py"]
