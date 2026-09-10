"""Object-storage key layout and working directories.

Keys are namespaced by user so a bucket policy can scope access per tenant,
and every key is derived from a UUID rather than a user-supplied filename.
"""

from __future__ import annotations

import re
import uuid
from pathlib import Path

from app.core.config import settings

SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def safe_filename(name: str, *, default: str = "file") -> str:
    """Reduce an uploaded filename to something safe to store.

    Never used as the storage key itself -- only as a trailing, human-readable
    hint after a UUID.
    """
    stem = Path(name or "").name
    cleaned = SAFE_NAME.sub("_", stem).strip("._-")
    cleaned = cleaned[:120]
    return cleaned or default


def source_video_key(user_id: uuid.UUID, video_id: uuid.UUID, filename: str) -> str:
    return f"users/{user_id}/videos/{video_id}/source/{safe_filename(filename, default='source.mp4')}"


def source_audio_key(user_id: uuid.UUID, video_id: uuid.UUID) -> str:
    return f"users/{user_id}/videos/{video_id}/audio/audio.wav"


def clip_video_key(user_id: uuid.UUID, clip_id: uuid.UUID) -> str:
    return f"users/{user_id}/clips/{clip_id}/clip.mp4"


def clip_variant_key(user_id: uuid.UUID, clip_id: uuid.UUID, label: str) -> str:
    return f"users/{user_id}/clips/{clip_id}/variant_{safe_filename(label, default='v')}.mp4"


def clip_thumbnail_key(user_id: uuid.UUID, clip_id: uuid.UUID) -> str:
    return f"users/{user_id}/clips/{clip_id}/thumbnail.jpg"


def clip_subtitle_key(user_id: uuid.UUID, clip_id: uuid.UUID, extension: str) -> str:
    ext = extension.lstrip(".")
    return f"users/{user_id}/clips/{clip_id}/captions.{ext}"


def video_thumbnail_key(user_id: uuid.UUID, video_id: uuid.UUID) -> str:
    return f"users/{user_id}/videos/{video_id}/thumbnail.jpg"


def job_workdir(job_id: uuid.UUID | str) -> Path:
    """A scratch directory for one job. Callers are responsible for cleanup."""
    path = Path(settings.workdir) / str(job_id)
    path.mkdir(parents=True, exist_ok=True)
    return path


def video_workdir(video_id: uuid.UUID | str) -> Path:
    """Cache directory for one video's derived media (audio, frames).

    Kept between jobs so re-analysis does not re-extract audio.
    """
    path = Path(settings.workdir) / "videos" / str(video_id)
    path.mkdir(parents=True, exist_ok=True)
    return path
