# syntax=docker/dockerfile:1
# ── Stage 1: install heavy deps (cached between code-only rebuilds) ───────────
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake libgomp1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements.txt .

# Strip macOS-only packages before installing
RUN grep -v "pyobjc" requirements.txt > requirements-linux.txt && \
    pip install --no-cache-dir --prefix=/install -r requirements-linux.txt

# ── Stage 2: download ML models into a layer ─────────────────────────────────
FROM python:3.11-slim AS models

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 libglib2.0-0 libgl1 wget \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local

# Pre-fetch YOLO v8n weights (~6 MB) and InsightFace buffalo_s (~100 MB)
# so the first container start does not need internet access.
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')" && \
    python -c " \
from insightface.app import FaceAnalysis; \
fa = FaceAnalysis(name='buffalo_s', providers=['CPUExecutionProvider']); \
fa.prepare(ctx_id=0, det_size=(320,320)); \
print('InsightFace buffalo_s ready') \
"

# ── Stage 3: final runtime image ─────────────────────────────────────────────
FROM python:3.11-slim AS runner

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 libglib2.0-0 libsm6 libxext6 libgl1 ffmpeg wget \
    && rm -rf /var/lib/apt/lists/*

# Non-root user (matches Kubernetes securityContext runAsUser: 1000)
RUN addgroup --gid 1000 appgroup && \
    adduser  --uid 1000 --gid 1000 --disabled-password --gecos "" \
             --home /home/appuser appuser

# Python packages
COPY --from=builder /install /usr/local

# Pre-downloaded ML model caches (owned by appuser so they remain accessible)
COPY --from=models --chown=appuser:appgroup /root/.config       /home/appuser/.config
COPY --from=models --chown=appuser:appgroup /root/.insightface  /home/appuser/.insightface

WORKDIR /app

# Application code
COPY --chown=appuser:appgroup . .

USER appuser

ENV PYTHONUNBUFFERED=1 \
    HOME=/home/appuser \
    INSIGHTFACE_HOME=/home/appuser/.insightface \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD wget -qO- http://localhost:8501/_stcore/health || exit 1

# CWD = /data (PVC mount) so every relative path (people.db, raw_events/, etc.)
# resolves inside the persistent volume, not the read-only image layer.
CMD ["sh", "-c", "cd /data && exec streamlit run /app/admin_app.py \
     --server.port=8501 \
     --server.address=0.0.0.0 \
     --server.headless=true \
     --server.fileWatcherType=none"]
