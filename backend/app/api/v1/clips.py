"""Clip endpoints: generation, review, editing and export."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Annotated

from fastapi import APIRouter, Query, status
from fastapi.responses import FileResponse
from sqlalchemy import func, select
from sqlalchemy.orm import selectinload

from app.api.deps import CurrentUser, DbSession, RateLimited, Settings
from app.core.errors import ForbiddenError, NotFoundError, ValidationFailure
from app.core.logging import get_logger
from app.models.check import ContentSafetyCheck, QualityCheck
from app.models.clip import CandidateClip, ClipVariant, GeneratedClip
from app.models.enums import PROCESSABLE_AUTHORIZATION, ClipStatus, JobType
from app.models.video import Video
from app.schemas.clip import (
    ApprovalRequest,
    ClipDetailResponse,
    ClipGenerateRequest,
    ClipRegenerateRequest,
    ClipResponse,
    ClipUpdateRequest,
    ClipVariantResponse,
    ManualClipRequest,
    QualityCheckResponse,
    SafetyCheckResponse,
)
from app.schemas.common import MessageResponse, Page
from app.schemas.misc import JobAcceptedResponse
from app.services import jobs as job_service
from app.services.dispatch import dispatch
from app.services.media import signed_url_for

logger = get_logger(__name__)
router = APIRouter(prefix="/clips", tags=["clips"], dependencies=[RateLimited])


async def _get_owned_clip(
    session: DbSession, user: CurrentUser, clip_id: uuid.UUID
) -> GeneratedClip:
    clip = await session.get(GeneratedClip, clip_id)
    if clip is None or clip.user_id != user.id:
        raise NotFoundError(f"Clip {clip_id} not found.")
    return clip


def _with_urls(clip: GeneratedClip, *, video_title: str | None = None) -> ClipResponse:
    response = ClipResponse.model_validate(clip)
    response.media_url = signed_url_for(clip.storage_key)
    response.thumbnail_url = signed_url_for(clip.thumbnail_storage_key)
    response.subtitle_url = signed_url_for(clip.subtitle_storage_key)
    response.video_title = video_title
    return response


@router.post(
    "/generate",
    response_model=JobAcceptedResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def generate_clips(
    payload: ClipGenerateRequest,
    user: CurrentUser,
    session: DbSession,
    user_settings: Settings,
) -> JobAcceptedResponse:
    """Queue rendering for detected candidates."""
    video_id = payload.video_id
    if payload.candidate_ids:
        candidate = await session.get(CandidateClip, payload.candidate_ids[0])
        if candidate is None:
            raise NotFoundError("Candidate not found.")
        video_id = candidate.video_id

    video = await session.get(Video, video_id) if video_id else None
    if video is None or video.user_id != user.id:
        raise NotFoundError("Video not found.")
    if video.authorization_status not in PROCESSABLE_AUTHORIZATION:
        raise ForbiddenError(
            f"This video's authorization_status is {video.authorization_status}; "
            "clips cannot be generated from it."
        )

    job = await job_service.create_job_async(
        session,
        user_id=user.id,
        job_type=JobType.GENERATE_CLIP,
        video_id=video.id,
        payload={
            "candidate_ids": [str(c) for c in payload.candidate_ids],
            "count": payload.count or user_settings.clips_per_video,
            "crop_mode": payload.crop_mode.value if payload.crop_mode else None,
            "caption_style": payload.caption_style.value if payload.caption_style else None,
            "caption_position": (
                payload.caption_position.value if payload.caption_position else None
            ),
            "include_hook_overlay": payload.include_hook_overlay,
            "generate_variants": payload.generate_variants,
        },
    )
    await session.commit()

    from app.workers.tasks import generate_clips_task

    dispatch(generate_clips_task, str(job.id))
    return JobAcceptedResponse(
        job_id=job.id, status=job.status, message="Clip generation queued."
    )


@router.post(
    "/manual",
    response_model=JobAcceptedResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def create_manual_clip(
    payload: ManualClipRequest, user: CurrentUser, session: DbSession
) -> JobAcceptedResponse:
    """Cut a span chosen by hand rather than by the agent."""
    video = await session.get(Video, payload.video_id)
    if video is None or video.user_id != user.id:
        raise NotFoundError("Video not found.")
    if video.authorization_status not in PROCESSABLE_AUTHORIZATION:
        raise ForbiddenError(
            f"This video's authorization_status is {video.authorization_status}."
        )
    if video.duration_seconds and payload.end_time > video.duration_seconds + 0.5:
        raise ValidationFailure(
            f"end_time {payload.end_time:.1f}s is past the end of the video "
            f"({video.duration_seconds:.1f}s)."
        )

    job = await job_service.create_job_async(
        session,
        user_id=user.id,
        job_type=JobType.GENERATE_CLIP,
        video_id=video.id,
        payload={
            "manual": True,
            "start_time": payload.start_time,
            "end_time": payload.end_time,
            "crop_mode": payload.crop_mode.value if payload.crop_mode else None,
            "caption_style": payload.caption_style.value if payload.caption_style else None,
            "caption_position": (
                payload.caption_position.value if payload.caption_position else None
            ),
            "hook_text": payload.hook_text,
            "include_hook_overlay": payload.include_hook_overlay,
        },
    )
    await session.commit()

    from app.workers.tasks import manual_clip_task

    dispatch(manual_clip_task, str(job.id))
    return JobAcceptedResponse(
        job_id=job.id, status=job.status, message="Manual clip queued."
    )


@router.get("", response_model=Page[ClipResponse])
async def list_clips(
    user: CurrentUser,
    session: DbSession,
    status_filter: Annotated[ClipStatus | None, Query(alias="status")] = None,
    video_id: Annotated[uuid.UUID | None, Query()] = None,
    min_score: Annotated[float, Query(ge=0, le=100)] = 0.0,
    order_by: Annotated[
        str, Query(pattern="^(created_at|viral_score|quality_score)$")
    ] = "created_at",
    limit: Annotated[int, Query(ge=1, le=100)] = 25,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> Page[ClipResponse]:
    conditions = [GeneratedClip.user_id == user.id]
    if status_filter is not None:
        conditions.append(GeneratedClip.status == status_filter)
    if video_id is not None:
        conditions.append(GeneratedClip.video_id == video_id)
    if min_score > 0:
        conditions.append(GeneratedClip.viral_score >= min_score)

    total = await session.scalar(
        select(func.count()).select_from(GeneratedClip).where(*conditions)
    )
    column = {
        "created_at": GeneratedClip.created_at,
        "viral_score": GeneratedClip.viral_score,
        "quality_score": GeneratedClip.quality_score,
    }[order_by]
    rows = list(
        (
            await session.execute(
                select(GeneratedClip)
                .where(*conditions)
                .order_by(column.desc().nullslast())
                .limit(limit)
                .offset(offset)
            )
        ).scalars()
    )

    titles: dict[uuid.UUID, str] = {}
    if rows:
        video_rows = await session.execute(
            select(Video.id, Video.title).where(
                Video.id.in_({row.video_id for row in rows})
            )
        )
        titles = {row_id: title for row_id, title in video_rows}

    items = [_with_urls(clip, video_title=titles.get(clip.video_id)) for clip in rows]
    return Page.of(items, total=int(total or 0), limit=limit, offset=offset)


@router.get("/{clip_id}", response_model=ClipDetailResponse)
async def get_clip(
    clip_id: uuid.UUID, user: CurrentUser, session: DbSession
) -> ClipDetailResponse:
    clip = await session.get(
        GeneratedClip,
        clip_id,
        options=[
            selectinload(GeneratedClip.variants),
            selectinload(GeneratedClip.quality_checks),
            selectinload(GeneratedClip.safety_checks),
        ],
    )
    if clip is None or clip.user_id != user.id:
        raise NotFoundError(f"Clip {clip_id} not found.")

    video = await session.get(Video, clip.video_id)
    base = _with_urls(clip, video_title=video.title if video else None)
    detail = ClipDetailResponse(**base.model_dump())
    detail.captions = clip.captions or []
    detail.crop_keyframes = clip.crop_keyframes or []
    detail.render_config = clip.render_config or {}
    detail.variants = [
        ClipVariantResponse(
            **ClipVariantResponse.model_validate(v).model_dump(exclude={"media_url"}),
            media_url=signed_url_for(v.storage_key),
        )
        for v in clip.variants
    ]
    detail.quality_checks = [
        QualityCheckResponse.model_validate(q)
        for q in sorted(clip.quality_checks, key=lambda q: q.attempt)
    ]
    detail.safety_checks = [
        SafetyCheckResponse.model_validate(s) for s in clip.safety_checks
    ]
    return detail


@router.patch("/{clip_id}", response_model=ClipResponse)
async def update_clip(
    clip_id: uuid.UUID,
    payload: ClipUpdateRequest,
    user: CurrentUser,
    session: DbSession,
) -> ClipResponse:
    """Apply editor changes.

    Text-only edits are saved directly. Anything that changes the picture --
    timing, crop, caption style -- needs a re-render; use
    ``POST /clips/{id}/regenerate`` for that, which this endpoint will tell you.
    """
    clip = await _get_owned_clip(session, user, clip_id)
    if payload.requires_rerender:
        raise ValidationFailure(
            "Timing, crop and caption changes require a re-render. "
            "POST to /clips/{id}/regenerate with the same fields."
        )
    for field, value in payload.model_dump(exclude_unset=True).items():
        setattr(clip, field, value)
    await session.commit()
    await session.refresh(clip)
    return _with_urls(clip)


@router.post(
    "/{clip_id}/regenerate",
    response_model=JobAcceptedResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def regenerate_clip(
    clip_id: uuid.UUID,
    payload: ClipRegenerateRequest,
    user: CurrentUser,
    session: DbSession,
) -> JobAcceptedResponse:
    clip = await _get_owned_clip(session, user, clip_id)
    job = await job_service.create_job_async(
        session,
        user_id=user.id,
        job_type=JobType.GENERATE_CLIP,
        video_id=clip.video_id,
        clip_id=clip.id,
        payload={
            "start_time": clip.start_time,
            "end_time": clip.end_time,
            "crop_mode": payload.crop_mode.value if payload.crop_mode else None,
            "caption_style": payload.caption_style.value if payload.caption_style else None,
            "caption_position": (
                payload.caption_position.value if payload.caption_position else None
            ),
            "hook_text": payload.hook_text,
            "regenerate_hook": payload.regenerate_hook,
            "include_hook_overlay": (
                True if payload.include_hook_overlay is None
                else payload.include_hook_overlay
            ),
        },
    )
    await session.commit()

    from app.workers.tasks import regenerate_clip_task

    dispatch(regenerate_clip_task, str(job.id))
    return JobAcceptedResponse(
        job_id=job.id, status=job.status, message="Regeneration queued."
    )


@router.post(
    "/{clip_id}/variants",
    response_model=JobAcceptedResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def create_variants(
    clip_id: uuid.UUID,
    user: CurrentUser,
    session: DbSession,
    labels: Annotated[list[str] | None, Query()] = None,
) -> JobAcceptedResponse:
    clip = await _get_owned_clip(session, user, clip_id)
    job = await job_service.create_job_async(
        session,
        user_id=user.id,
        job_type=JobType.GENERATE_CLIP,
        video_id=clip.video_id,
        clip_id=clip.id,
        payload={"labels": labels or ["B", "C"]},
    )
    await session.commit()

    from app.workers.tasks import generate_variants_task

    dispatch(generate_variants_task, str(job.id))
    return JobAcceptedResponse(
        job_id=job.id, status=job.status, message="Variant rendering queued."
    )


@router.post("/{clip_id}/approve", response_model=ClipResponse)
async def approve_clip(
    clip_id: uuid.UUID,
    payload: ApprovalRequest,
    user: CurrentUser,
    session: DbSession,
) -> ClipResponse:
    clip = await _get_owned_clip(session, user, clip_id)
    if clip.status is ClipStatus.PUBLISHED:
        raise ValidationFailure("This clip has already been published.")
    if clip.safety_verdict == "BLOCK":
        raise ForbiddenError(
            "This clip was blocked by the content safety review and cannot be "
            "approved. Regenerate it from a different moment instead."
        )
    clip.status = ClipStatus.APPROVED
    clip.approved_at = datetime.now(timezone.utc)
    clip.rejected_at = None
    clip.auto_approved = False
    clip.review_note = payload.note
    await session.commit()
    await session.refresh(clip)
    return _with_urls(clip)


@router.post("/{clip_id}/reject", response_model=ClipResponse)
async def reject_clip(
    clip_id: uuid.UUID,
    payload: ApprovalRequest,
    user: CurrentUser,
    session: DbSession,
) -> ClipResponse:
    clip = await _get_owned_clip(session, user, clip_id)
    clip.status = ClipStatus.REJECTED
    clip.rejected_at = datetime.now(timezone.utc)
    clip.approved_at = None
    clip.review_note = payload.note
    await session.commit()
    await session.refresh(clip)
    return _with_urls(clip)


@router.get("/{clip_id}/download")
async def download_clip(
    clip_id: uuid.UUID, user: CurrentUser, session: DbSession
) -> FileResponse:
    """Stream the rendered file for manual upload."""
    clip = await _get_owned_clip(session, user, clip_id)
    if not clip.storage_key:
        raise NotFoundError("This clip has not finished rendering.")

    import anyio

    from app.database.session import get_sync_session_factory
    from app.services.publishing import export_clip_path

    def _resolve():
        factory = get_sync_session_factory()
        with factory() as sync_session:
            record = sync_session.get(GeneratedClip, clip_id)
            return export_clip_path(sync_session, record)

    path = await anyio.to_thread.run_sync(_resolve)
    filename = f"{(clip.title or 'clip').replace('/', '-')[:60]}.mp4"
    return FileResponse(path, media_type="video/mp4", filename=filename)


@router.delete("/{clip_id}", response_model=MessageResponse)
async def delete_clip(
    clip_id: uuid.UUID, user: CurrentUser, session: DbSession
) -> MessageResponse:
    clip = await _get_owned_clip(session, user, clip_id)
    await session.delete(clip)
    await session.commit()
    return MessageResponse(message="Clip deleted.")


@router.get("/{clip_id}/checks", response_model=list[QualityCheckResponse])
async def list_quality_checks(
    clip_id: uuid.UUID, user: CurrentUser, session: DbSession
) -> list[QualityCheck]:
    await _get_owned_clip(session, user, clip_id)
    rows = await session.execute(
        select(QualityCheck)
        .where(QualityCheck.clip_id == clip_id)
        .order_by(QualityCheck.attempt)
    )
    return list(rows.scalars())


@router.get("/{clip_id}/safety", response_model=list[SafetyCheckResponse])
async def list_safety_checks(
    clip_id: uuid.UUID, user: CurrentUser, session: DbSession
) -> list[ContentSafetyCheck]:
    await _get_owned_clip(session, user, clip_id)
    rows = await session.execute(
        select(ContentSafetyCheck)
        .where(ContentSafetyCheck.clip_id == clip_id)
        .order_by(ContentSafetyCheck.created_at.desc())
    )
    return list(rows.scalars())


@router.get("/{clip_id}/variants", response_model=list[ClipVariantResponse])
async def list_variants(
    clip_id: uuid.UUID, user: CurrentUser, session: DbSession
) -> list[ClipVariantResponse]:
    await _get_owned_clip(session, user, clip_id)
    rows = await session.execute(
        select(ClipVariant).where(ClipVariant.clip_id == clip_id).order_by(ClipVariant.label)
    )
    return [
        ClipVariantResponse(
            **ClipVariantResponse.model_validate(v).model_dump(exclude={"media_url"}),
            media_url=signed_url_for(v.storage_key),
        )
        for v in rows.scalars()
    ]
