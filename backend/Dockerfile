# syntax=docker/dockerfile:1.7
# =============================================================================
# API image.
#
# Ships ffmpeg/ffprobe because the API probes uploads and serves media, but not
# the ML wheels -- transcription runs in the worker image.
# =============================================================================
FROM python:3.11-slim-bookworm AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        curl \
        fonts-dejavu-core \
        fonts-noto-core \
        fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# -----------------------------------------------------------------------------
FROM base AS deps
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# -----------------------------------------------------------------------------
FROM deps AS runtime

# Devanagari coverage for Hindi/Hinglish captions; the caption renderer picks
# fonts per script and warns when it has to fall back to a bitmap font.
RUN apt-get update && apt-get install -y --no-install-recommends \
        fonts-noto-devanagari \
    && rm -rf /var/lib/apt/lists/*

COPY alembic.ini pytest.ini ./
COPY alembic ./alembic
COPY app ./app

RUN groupadd --system app && useradd --system --gid app --home /app app \
    && mkdir -p /data/storage /data/work \
    && chown -R app:app /app /data
USER app

ENV STORAGE_LOCAL_ROOT=/data/storage \
    WORKDIR=/data/work

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
