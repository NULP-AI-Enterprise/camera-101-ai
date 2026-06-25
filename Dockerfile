# syntax=docker/dockerfile:1
# ── Stage 1: install Python deps into a venv ─────────────────────────────────
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .

RUN python -m venv /venv

# CPU-only torch first — avoids downloading ~2 GB of CUDA packages
RUN /venv/bin/pip install --no-cache-dir \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

RUN /venv/bin/pip install --no-cache-dir -r requirements.txt

# ── Stage 2: production image ─────────────────────────────────────────────────
FROM python:3.11-slim AS runner

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 libglib2.0-0 libsm6 libxext6 libgl1 ffmpeg wget \
    && rm -rf /var/lib/apt/lists/*

RUN addgroup --gid 1000 appgroup && \
    adduser --uid 1000 --gid 1000 --disabled-password --gecos "" \
            --home /home/appuser appuser

COPY --from=builder /venv /venv

WORKDIR /app
COPY --chown=appuser:appgroup . .

USER appuser

ENV PATH="/venv/bin:$PATH" \
    HOME=/home/appuser \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    # Models download to the PVC on first run and persist across restarts.
    # InsightFace (~100 MB buffalo_s) and YOLO (~6 MB yolov8n.pt) land in /data.
    INSIGHTFACE_HOME=/data/.insightface \
    YOLO_CONFIG_DIR=/data/.ultralytics

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD wget -qO- http://localhost:8501/_stcore/health || exit 1

# CWD=/data (PVC) — relative paths (people.db, raw_events/, models) resolve there
CMD ["sh", "-c", "cd /data && exec streamlit run /app/admin_app.py \
     --server.port=8501 --server.address=0.0.0.0 \
     --server.headless=true --server.fileWatcherType=none"]
