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
    libgomp1 libglib2.0-0 libsm6 libxext6 libgl1 ffmpeg wget tzdata \
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
    TZ=Europe/Kyiv \
    LOG_TZ_OFFSET=3 \
    # Models download to the PVC on first run and persist across restarts.
    INSIGHTFACE_HOME=/data/.insightface \
    YOLO_CONFIG_DIR=/data/.ultralytics

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD wget -qO- http://localhost:8501/health || exit 1

# Runtime CWD = /data (PVC mount) so that relative paths (people.db, logs/,
# raw_events/, snapshots/, models) all resolve to the persistent volume.
# db.py calls Base.metadata.create_all() at import time — tables are created
# automatically on first run without needing a separate alembic step.
# Schema migrations: run manually inside the container when needed:
#   kubectl exec <pod> -- alembic --config /app/alembic.ini upgrade head
WORKDIR /data

CMD ["/venv/bin/uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8501", "--workers", "1", "--app-dir", "/app", "--no-access-log"]
