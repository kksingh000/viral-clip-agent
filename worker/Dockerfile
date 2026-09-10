# syntax=docker/dockerfile:1.7
# =============================================================================
# Worker image.
#
# Everything the API image has, plus the ML wheels (faster-whisper /
# CTranslate2) and the OpenCV runtime libraries the vision provider needs.
# ffmpeg AND ffprobe are both present here: the fallback stderr probe exists
# for developer machines, not for production.
# =============================================================================
FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    OMP_NUM_THREADS=4 \
    HF_HOME=/data/models

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        curl \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        fonts-dejavu-core \
        fonts-noto-core \
        fonts-noto-devanagari \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY backend/requirements.txt backend/requirements-ml.txt ./
RUN pip install --no-cache-dir -r requirements-ml.txt

COPY backend/alembic.ini ./
COPY backend/alembic ./alembic
COPY backend/app ./app

RUN groupadd --system app && useradd --system --gid app --home /app app \
    && mkdir -p /data/storage /data/work /data/models \
    && chown -R app:app /app /data
USER app

ENV STORAGE_LOCAL_ROOT=/data/storage \
    WORKDIR=/data/work

# A worker is healthy when it answers a broker ping.
HEALTHCHECK --interval=60s --timeout=15s --start-period=45s --retries=3 \
    CMD celery -A app.workers.celery_app.celery_app inspect ping -d celery@$HOSTNAME || exit 1

# Queues are separated so a long render cannot starve short jobs. Scale the
# render queue horizontally; concurrency stays low because ffmpeg is already
# multi-threaded.
CMD ["celery", "-A", "app.workers.celery_app.celery_app", "worker", \
     "--loglevel=info", "--concurrency=2", \
     "-Q", "default,analysis,render,discovery,publishing"]
