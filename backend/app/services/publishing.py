"""Publishing orchestration.

Two gates stand in front of every upload, and both are enforced here rather
than in the UI:

1. the **source** must carry a rights position that permits republishing, and
2. the **clip** must be approved and not blocked by the safety review.

Only official platform APIs are used; credentials come from a connected
account and are never logged.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.orm import Session

from app.core.errors import AuthorizationStatusError, ForbiddenError, NotFoundError
from app.core.logging import get_logger
from app.models.clip import GeneratedClip
from app.models.enums import (
    PUBLISHABLE_AUTHORIZATION,
    ClipStatus,
    MediaAssetKind,
    Platform,
    PublishStatus,
    SafetyVerdict,
)
from app.models.publishing import PlatformAccount, PublishingJob
from app.models.video import Video
from app.providers.registry import get_publishing_provider, get_storage_provider
from app.services import media as media_service
from app.services.storage_paths import job_workdir

logger = get_logger(__name__)


def assert_publishable(session: Session, clip: GeneratedClip) -> Video:
    """Raise unless both the rights and the review gates are satisfied."""
    video = session.get(Video, clip.video_id)
    if video is None:
        raise NotFoundError("The source video for this clip no longer exists.")

    if video.authorization_status not in PUBLISHABLE_AUTHORIZATION:
        raise AuthorizationStatusError(
            "This clip comes from a source with authorization_status="
            f"{video.authorization_status}. Automated publishing is only "
            "available for content you own, licensed, or are otherwise "
            "authorized to republish.",
            details={"authorization_status": str(video.authorization_status)},
        )
    if clip.status is not ClipStatus.APPROVED:
        raise ForbiddenError(
            f"Clip is {clip.status}; only approved clips can be published."
        )
    if clip.safety_verdict == SafetyVerdict.BLOCK.value:
        raise ForbiddenError("This clip was blocked by the content safety review.")
    if not clip.storage_key:
        raise NotFoundError("This clip has no rendered file to publish.")
    return video


def schedule_publish(
    session: Session,
    clip: GeneratedClip,
    *,
    platform: Platform,
    account: PlatformAccount | None,
    scheduled_for: datetime | None = None,
    privacy: str = "private",
    title: str | None = None,
    description: str | None = None,
) -> PublishingJob:
    assert_publishable(session, clip)
    if platform is not Platform.MANUAL_EXPORT and account is None:
        raise ForbiddenError(
            f"No connected {platform.value} account. Connect one before publishing."
        )

    job = PublishingJob(
        clip_id=clip.id,
        account_id=account.id if account else None,
        platform=platform,
        status=PublishStatus.SCHEDULED if scheduled_for else PublishStatus.QUEUED,
        scheduled_for=scheduled_for,
        title=title or clip.title,
        description=description or clip.description,
        hashtags=list(clip.hashtags or []),
        privacy=privacy,
    )
    session.add(job)
    clip.publish_status = job.status
    session.flush()
    return job


def _credentials(account: PlatformAccount | None) -> dict[str, Any]:
    if account is None:
        return {}
    return {
        "access_token": account.access_token_ref,
        "refresh_token": account.refresh_token_ref,
        "external_account_id": account.external_account_id,
        "ig_user_id": account.external_account_id,
    }


def publish_clip(session: Session, publishing_job_id: uuid.UUID) -> dict[str, Any]:
    """Execute one publishing job."""
    job = session.get(PublishingJob, publishing_job_id)
    if job is None:
        raise NotFoundError(f"Publishing job {publishing_job_id} not found.")
    clip = session.get(GeneratedClip, job.clip_id)
    if clip is None:
        raise NotFoundError("The clip for this publishing job no longer exists.")

    assert_publishable(session, clip)
    account = session.get(PlatformAccount, job.account_id) if job.account_id else None
    provider = get_publishing_provider(job.platform)
    storage = get_storage_provider()

    job.status = PublishStatus.UPLOADING
    clip.publish_status = job.status
    session.flush()

    local = storage.local_path(clip.storage_key)
    if local is None:
        local = storage.get_to_path(
            clip.storage_key, job_workdir(job.id) / "publish.mp4"
        )

    credentials = _credentials(account)
    if job.platform is Platform.INSTAGRAM_REELS:
        # The Graph API pulls the file from a URL rather than accepting an
        # upload, so a signed URL is part of the credential bundle.
        credentials["video_url"] = storage.signed_url(clip.storage_key)

    try:
        result = provider.publish(
            video_path=local,
            title=job.title or clip.title or "Clip",
            description=job.description or clip.description or "",
            hashtags=job.hashtags or clip.hashtags or [],
            credentials=credentials,
            privacy=job.privacy,
        )
    except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
        job.status = PublishStatus.FAILED
        job.error_code = type(exc).__name__
        job.error_message = str(exc)[:4000]
        job.retry_count += 1
        clip.publish_status = PublishStatus.FAILED
        if account is not None:
            account.last_error = str(exc)[:2000]
        session.flush()
        logger.error(
            "publish failed",
            extra={"publishing_job_id": str(job.id), "platform": str(job.platform)},
        )
        raise

    job.status = PublishStatus.PUBLISHED
    job.published_at = datetime.now(timezone.utc)
    job.external_post_id = result.external_post_id
    job.external_url = result.external_url
    job.response_payload = result.raw
    job.error_code = None
    job.error_message = None

    clip.publish_status = PublishStatus.PUBLISHED
    clip.status = ClipStatus.PUBLISHED
    session.flush()

    logger.info(
        "clip published",
        extra={
            "clip_id": str(clip.id),
            "platform": str(job.platform),
            "external_post_id": result.external_post_id,
        },
    )
    return {
        "publishing_job_id": str(job.id),
        "external_post_id": result.external_post_id,
        "external_url": result.external_url,
    }


def export_clip_path(session: Session, clip: GeneratedClip):
    """Resolve a local path for download, for the manual-export workflow."""
    if not clip.storage_key:
        raise NotFoundError("This clip has no rendered file.")
    storage = get_storage_provider()
    local = storage.local_path(clip.storage_key)
    if local is not None:
        return local
    # Clip assets hang off the clip, not the video; looking them up by
    # video_id never matched and always fell through to the default name.
    asset = media_service.get_clip_asset(session, clip.id, MediaAssetKind.CLIP_VIDEO)
    name = asset.storage_key.rsplit("/", 1)[-1] if asset else "clip.mp4"
    return storage.get_to_path(clip.storage_key, job_workdir(clip.id) / name)
