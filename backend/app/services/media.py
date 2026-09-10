"""Media resolution and registration.

Bridges the object store and the local filesystem that ffmpeg needs: local
storage is used in place, remote storage is fetched into the video's work
directory and cached there.
"""

from __future__ import annotations

import uuid
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.errors import AuthorizationStatusError, NotFoundError
from app.core.logging import get_logger
from app.models.enums import PROCESSABLE_AUTHORIZATION, MediaAssetKind
from app.models.video import MediaAsset, Video
from app.providers.base import StorageProvider
from app.providers.registry import get_storage_provider
from app.services.storage_paths import video_workdir

logger = get_logger(__name__)


def assert_processable(video: Video) -> None:
    """Gate every media operation on the rights position.

    Discovery records hold metadata only; they never reach this point with a
    media asset attached, and this raises if anything tries.
    """
    if video.authorization_status not in PROCESSABLE_AUTHORIZATION:
        raise AuthorizationStatusError(
            f"Video {video.id} has authorization_status="
            f"{video.authorization_status}, which does not permit processing. "
            "Confirm you own this content or hold a licence, then set the "
            "status before analysing it.",
            details={"authorization_status": str(video.authorization_status)},
        )


def get_asset(
    session: Session, video_id: uuid.UUID, kind: MediaAssetKind
) -> MediaAsset | None:
    """The most recent asset of ``kind`` belonging to a *video*."""
    statement = (
        select(MediaAsset)
        .where(MediaAsset.video_id == video_id, MediaAsset.kind == kind)
        .order_by(MediaAsset.created_at.desc())
        .limit(1)
    )
    return session.execute(statement).scalars().first()


def get_clip_asset(
    session: Session, clip_id: uuid.UUID, kind: MediaAssetKind
) -> MediaAsset | None:
    """The most recent asset of ``kind`` belonging to a *clip*.

    Clip assets are stored with ``clip_id`` set and ``video_id`` NULL, so
    looking one up through :func:`get_asset` can never match.
    """
    statement = (
        select(MediaAsset)
        .where(MediaAsset.clip_id == clip_id, MediaAsset.kind == kind)
        .order_by(MediaAsset.created_at.desc())
        .limit(1)
    )
    return session.execute(statement).scalars().first()


def resolve_source_media(
    session: Session,
    video: Video,
    *,
    storage: StorageProvider | None = None,
) -> Path:
    """Return a local path to the source video, downloading it if necessary."""
    assert_processable(video)
    storage = storage or get_storage_provider()

    asset = get_asset(session, video.id, MediaAssetKind.SOURCE_VIDEO)
    if asset is None:
        raise NotFoundError(
            f"Video {video.id} has no source media. Upload the file or connect "
            "an authorized source before analysing it."
        )

    if (local := storage.local_path(asset.storage_key)) is not None:
        return local

    destination = video_workdir(video.id) / Path(asset.storage_key).name
    if destination.exists() and asset.size_bytes and (
        destination.stat().st_size == asset.size_bytes
    ):
        logger.debug("using cached source media", extra={"path": str(destination)})
        return destination
    logger.info(
        "downloading source media",
        extra={"video_id": str(video.id), "key": asset.storage_key},
    )
    return storage.get_to_path(asset.storage_key, destination)


def register_asset(
    session: Session,
    *,
    kind: MediaAssetKind,
    storage_key: str,
    video_id: uuid.UUID | None = None,
    clip_id: uuid.UUID | None = None,
    content_type: str | None = None,
    size_bytes: int | None = None,
    checksum_sha256: str | None = None,
    extra: dict | None = None,
) -> MediaAsset:
    asset = MediaAsset(
        video_id=video_id,
        clip_id=clip_id,
        kind=kind,
        storage_key=storage_key,
        content_type=content_type,
        size_bytes=size_bytes,
        checksum_sha256=checksum_sha256,
        extra=extra or {},
    )
    session.add(asset)
    session.flush()
    return asset


def signed_url_for(key: str | None, *, expires_in: int | None = None) -> str | None:
    if not key:
        return None
    try:
        return get_storage_provider().signed_url(key, expires_in=expires_in)
    except Exception as exc:  # noqa: BLE001 - a missing URL must not break a read
        logger.warning("could not sign url", extra={"key": key, "error": str(exc)})
        return None
