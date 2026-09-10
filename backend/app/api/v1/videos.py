"""Video endpoints: registration, upload, authorization, analysis, inspection."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any

import anyio
from fastapi import APIRouter, File, Query, UploadFile, status
from sqlalchemy import func, select
from sqlalchemy.orm import selectinload

from app.api.deps import CurrentUser, DbSession, RateLimited, Settings
from app.core.config import settings as app_settings
from app.core.errors import ConflictError, NotFoundError, ValidationFailure
from app.core.logging import get_logger
from app.models.clip import CandidateClip, GeneratedClip
from app.models.enums import (
    PROCESSABLE_AUTHORIZATION,
    AuthorizationStatus,
    JobType,
    MediaAssetKind,
    VideoSource,
    VideoStatus,
)
from app.models.transcript import Scene, Transcript, TranscriptSegment
from app.models.video import MediaAsset, Video
from app.schemas.common import MessageResponse, Page
from app.schemas.misc import JobAcceptedResponse
from app.schemas.video import (
    AnalyzeRequest,
    CandidateResponse,
    SceneResponse,
    TranscriptResponse,
    TranscriptSegmentResponse,
    VideoAuthorizationUpdate,
    VideoCreateRequest,
    VideoResponse,
    VideoUpdateRequest,
)
from app.services import jobs as job_service
from app.services.dispatch import dispatch
from app.services.media import signed_url_for
from app.services.storage_paths import safe_filename, source_video_key

logger = get_logger(__name__)
router = APIRouter(prefix="/videos", tags=["videos"], dependencies=[RateLimited])


async def _get_owned_video(session: DbSession, user: CurrentUser, video_id: uuid.UUID) -> Video:
    video = await session.get(Video, video_id)
    if video is None or video.user_id != user.id:
        raise NotFoundError(f"Video {video_id} not found.")
    return video


async def _decorate(session: DbSession, video: Video) -> VideoResponse:
    """Add the counts and URLs the dashboard needs, which are not columns."""
    candidate_count = await session.scalar(
        select(func.count())
        .select_from(CandidateClip)
        .where(CandidateClip.video_id == video.id)
    )
    clip_count = await session.scalar(
        select(func.count())
        .select_from(GeneratedClip)
        .where(GeneratedClip.video_id == video.id)
    )
    asset = (
        await session.execute(
            select(MediaAsset).where(
                MediaAsset.video_id == video.id,
                MediaAsset.kind == MediaAssetKind.SOURCE_VIDEO,
            )
        )
    ).scalars().first()

    response = VideoResponse.model_validate(video)
    response.candidate_count = int(candidate_count or 0)
    response.clip_count = int(clip_count or 0)
    response.has_media = asset is not None
    response.media_url = signed_url_for(asset.storage_key) if asset else None
    return response


@router.post("", response_model=VideoResponse, status_code=status.HTTP_201_CREATED)
async def create_video(
    payload: VideoCreateRequest, user: CurrentUser, session: DbSession
) -> VideoResponse:
    """Register a video by reference.

    No media is fetched. Discovering a URL does not make a video safe to
    download or republish: the media must be uploaded by the account holder or
    supplied through an authorized integration, and the rights position is
    recorded separately.
    """
    if payload.external_id:
        existing = (
            await session.execute(
                select(Video).where(
                    Video.user_id == user.id,
                    Video.source == payload.source,
                    Video.external_id == payload.external_id,
                )
            )
        ).scalars().first()
        if existing is not None:
            raise ConflictError(
                f"A video with external_id={payload.external_id} is already registered."
            )

    video = Video(
        user_id=user.id,
        source=payload.source,
        external_id=payload.external_id,
        source_url=str(payload.source_url) if payload.source_url else None,
        title=payload.title,
        description=payload.description,
        creator_name=payload.creator_name,
        category=payload.category,
        language=payload.language,
        tags=payload.tags,
        published_at=payload.published_at,
        authorization_status=payload.authorization_status,
        authorization_note=payload.authorization_note,
        authorization_evidence_url=(
            str(payload.authorization_evidence_url)
            if payload.authorization_evidence_url
            else None
        ),
        status=(
            VideoStatus.PENDING_MEDIA
            if payload.authorization_status in PROCESSABLE_AUTHORIZATION
            else VideoStatus.DISCOVERED
        ),
    )
    if payload.authorization_status in PROCESSABLE_AUTHORIZATION:
        video.authorized_at = datetime.now(timezone.utc)
        video.authorized_by_user_id = user.id

    session.add(video)
    await session.commit()
    await session.refresh(video)
    return await _decorate(session, video)


@router.get("", response_model=Page[VideoResponse])
async def list_videos(
    user: CurrentUser,
    session: DbSession,
    status_filter: Annotated[VideoStatus | None, Query(alias="status")] = None,
    authorization: Annotated[AuthorizationStatus | None, Query()] = None,
    source: Annotated[VideoSource | None, Query()] = None,
    search: Annotated[str | None, Query(max_length=200)] = None,
    order_by: Annotated[str, Query(pattern="^(created_at|trend_score|published_at)$")] = "created_at",
    limit: Annotated[int, Query(ge=1, le=100)] = 25,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> Page[VideoResponse]:
    conditions = [Video.user_id == user.id]
    if status_filter is not None:
        conditions.append(Video.status == status_filter)
    if authorization is not None:
        conditions.append(Video.authorization_status == authorization)
    if source is not None:
        conditions.append(Video.source == source)
    if search:
        conditions.append(Video.title.ilike(f"%{search}%"))

    total = await session.scalar(
        select(func.count()).select_from(Video).where(*conditions)
    )
    column = {
        "created_at": Video.created_at,
        "trend_score": Video.trend_score,
        "published_at": Video.published_at,
    }[order_by]
    rows = (
        await session.execute(
            select(Video)
            .where(*conditions)
            .order_by(column.desc().nullslast())
            .limit(limit)
            .offset(offset)
        )
    ).scalars()
    items = [await _decorate(session, video) for video in rows]
    return Page.of(items, total=int(total or 0), limit=limit, offset=offset)


@router.get("/{video_id}", response_model=VideoResponse)
async def get_video(
    video_id: uuid.UUID, user: CurrentUser, session: DbSession
) -> VideoResponse:
    video = await _get_owned_video(session, user, video_id)
    return await _decorate(session, video)


@router.patch("/{video_id}", response_model=VideoResponse)
async def update_video(
    video_id: uuid.UUID,
    payload: VideoUpdateRequest,
    user: CurrentUser,
    session: DbSession,
) -> VideoResponse:
    video = await _get_owned_video(session, user, video_id)
    for field, value in payload.model_dump(exclude_unset=True).items():
        setattr(video, field, value)
    await session.commit()
    await session.refresh(video)
    return await _decorate(session, video)


@router.delete("/{video_id}", response_model=MessageResponse)
async def delete_video(
    video_id: uuid.UUID, user: CurrentUser, session: DbSession
) -> MessageResponse:
    video = await _get_owned_video(session, user, video_id)
    await session.delete(video)
    await session.commit()
    return MessageResponse(message="Video and all derived data deleted.")


@router.put("/{video_id}/authorization", response_model=VideoResponse)
async def set_authorization(
    video_id: uuid.UUID,
    payload: VideoAuthorizationUpdate,
    user: CurrentUser,
    session: DbSession,
) -> VideoResponse:
    """Record the rights position for a video.

    Setting USER_OWNED, LICENSED or AUTHORIZED is an assertion by the account
    holder; the note and evidence URL are stored as the record of it. Only
    videos in one of those states (or USER_UPLOADED) can be processed or
    published.
    """
    video = await _get_owned_video(session, user, video_id)
    video.authorization_status = payload.authorization_status
    video.authorization_note = payload.authorization_note
    video.authorization_evidence_url = (
        str(payload.authorization_evidence_url)
        if payload.authorization_evidence_url
        else None
    )
    if payload.authorization_status in PROCESSABLE_AUTHORIZATION:
        video.authorized_at = datetime.now(timezone.utc)
        video.authorized_by_user_id = user.id
        if video.status is VideoStatus.DISCOVERED:
            video.status = VideoStatus.PENDING_MEDIA
    else:
        video.authorized_at = None
        video.authorized_by_user_id = None

    await session.commit()
    await session.refresh(video)
    logger.info(
        "authorization updated",
        extra={
            "video_id": str(video.id),
            "authorization_status": str(video.authorization_status),
        },
    )
    return await _decorate(session, video)


@router.post(
    "/{video_id}/upload",
    response_model=VideoResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_media(
    video_id: uuid.UUID,
    user: CurrentUser,
    session: DbSession,
    file: Annotated[UploadFile, File(description="The source video file.")],
) -> VideoResponse:
    """Attach the media file for a registered video.

    Uploading is itself an assertion of rights: the video moves to
    USER_UPLOADED unless a stronger status has already been recorded.
    """
    filename = file.filename or "upload.mp4"
    suffix = Path(filename).suffix.lower()
    if suffix not in app_settings.allowed_video_extensions:
        raise ValidationFailure(
            f"{suffix or 'that file type'} is not supported. Allowed: "
            + ", ".join(app_settings.allowed_video_extensions)
        )
    if file.content_type and file.content_type not in app_settings.allowed_video_mimetypes:
        raise ValidationFailure(f"Unsupported content type: {file.content_type}")

    video = await _get_owned_video(session, user, video_id)

    from app.providers.registry import get_storage_provider

    storage = get_storage_provider()
    key = source_video_key(user.id, video.id, safe_filename(filename))

    # Storage writes are blocking; keep them off the event loop.
    stored = await anyio.to_thread.run_sync(
        lambda: storage.put_stream(
            key,
            file.file,
            content_type=file.content_type,
            max_bytes=app_settings.max_upload_bytes,
        )
    )

    existing = (
        await session.execute(
            select(MediaAsset).where(
                MediaAsset.video_id == video.id,
                MediaAsset.kind == MediaAssetKind.SOURCE_VIDEO,
            )
        )
    ).scalars().all()
    for asset in existing:
        await session.delete(asset)

    session.add(
        MediaAsset(
            video_id=video.id,
            kind=MediaAssetKind.SOURCE_VIDEO,
            storage_key=stored.key,
            content_type=file.content_type,
            size_bytes=stored.size_bytes,
            checksum_sha256=stored.checksum_sha256,
            extra={"original_filename": safe_filename(filename)},
        )
    )
    video.file_size_bytes = stored.size_bytes
    video.content_hash = stored.checksum_sha256
    video.status = VideoStatus.UPLOADED
    if video.authorization_status not in PROCESSABLE_AUTHORIZATION:
        video.authorization_status = AuthorizationStatus.USER_UPLOADED
        video.authorized_at = datetime.now(timezone.utc)
        video.authorized_by_user_id = user.id

    await session.commit()
    await session.refresh(video)
    logger.info(
        "media uploaded",
        extra={"video_id": str(video.id), "size_bytes": stored.size_bytes},
    )
    return await _decorate(session, video)


@router.post(
    "/{video_id}/analyze",
    response_model=JobAcceptedResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def analyze_video(
    video_id: uuid.UUID,
    payload: AnalyzeRequest,
    user: CurrentUser,
    session: DbSession,
    user_settings: Settings,
    auto_generate: Annotated[bool, Query()] = False,
) -> JobAcceptedResponse:
    """Queue analysis. Returns immediately with a job id."""
    video = await _get_owned_video(session, user, video_id)
    if video.authorization_status not in PROCESSABLE_AUTHORIZATION:
        raise ValidationFailure(
            "This video's authorization_status is "
            f"{video.authorization_status}. Record that you own it or hold a "
            "licence before analysing it."
        )

    asset = (
        await session.execute(
            select(MediaAsset).where(
                MediaAsset.video_id == video.id,
                MediaAsset.kind == MediaAssetKind.SOURCE_VIDEO,
            )
        )
    ).scalars().first()
    if asset is None:
        raise ValidationFailure(
            "No media is attached to this video. Upload the file first."
        )

    job = await job_service.create_job_async(
        session,
        user_id=user.id,
        job_type=JobType.ANALYZE_VIDEO,
        video_id=video.id,
        payload={
            "force": payload.force,
            "auto_generate": auto_generate,
            "clips_per_video": user_settings.clips_per_video,
        },
    )
    await session.commit()

    from app.workers.tasks import analyze_video_task

    dispatch(analyze_video_task, str(job.id))
    return JobAcceptedResponse(
        job_id=job.id,
        status=job.status,
        message="Analysis queued.",
    )


@router.get("/{video_id}/transcript", response_model=TranscriptResponse)
async def get_transcript(
    video_id: uuid.UUID,
    user: CurrentUser,
    session: DbSession,
    include_words: Annotated[bool, Query()] = False,
) -> TranscriptResponse:
    video = await _get_owned_video(session, user, video_id)
    transcript = (
        await session.execute(
            select(Transcript)
            .where(Transcript.video_id == video.id)
            .options(selectinload(Transcript.segments))
        )
    ).scalars().first()
    if transcript is None:
        raise NotFoundError("This video has not been transcribed yet.")

    segments = [
        TranscriptSegmentResponse(
            id=row.id,
            index=row.index,
            start_time=row.start_time,
            end_time=row.end_time,
            text=row.text,
            speaker=row.speaker,
            prefilter_score=row.prefilter_score,
            words=row.words if include_words else [],
            signals=row.signals or {},
        )
        for row in sorted(transcript.segments, key=lambda s: s.index)
    ]
    response = TranscriptResponse.model_validate(transcript)
    response.segments = segments
    response.is_synthetic = bool((transcript.provider_metadata or {}).get("synthetic"))
    return response


@router.get("/{video_id}/scenes", response_model=list[SceneResponse])
async def list_scenes(
    video_id: uuid.UUID, user: CurrentUser, session: DbSession
) -> list[Scene]:
    video = await _get_owned_video(session, user, video_id)
    rows = await session.execute(
        select(Scene).where(Scene.video_id == video.id).order_by(Scene.start_time)
    )
    return list(rows.scalars())


@router.get("/{video_id}/candidates", response_model=list[CandidateResponse])
async def list_candidates(
    video_id: uuid.UUID,
    user: CurrentUser,
    session: DbSession,
    include_discarded: Annotated[bool, Query()] = False,
    min_score: Annotated[float, Query(ge=0, le=100)] = 0.0,
) -> list[CandidateClip]:
    """Ranked candidate moments, with the reasoning behind each one."""
    video = await _get_owned_video(session, user, video_id)
    conditions: list[Any] = [
        CandidateClip.video_id == video.id,
        CandidateClip.viral_score >= min_score,
    ]
    if not include_discarded:
        from app.models.enums import CandidateStatus

        conditions.append(
            CandidateClip.status.in_(
                [CandidateStatus.PROPOSED, CandidateStatus.SELECTED, CandidateStatus.GENERATED]
            )
        )
    rows = await session.execute(
        select(CandidateClip)
        .where(*conditions)
        .order_by(CandidateClip.viral_score.desc())
    )
    return list(rows.scalars())


@router.get("/{video_id}/segments", response_model=list[TranscriptSegmentResponse])
async def list_segments(
    video_id: uuid.UUID, user: CurrentUser, session: DbSession
) -> list[TranscriptSegment]:
    video = await _get_owned_video(session, user, video_id)
    transcript = (
        await session.execute(select(Transcript).where(Transcript.video_id == video.id))
    ).scalars().first()
    if transcript is None:
        return []
    rows = await session.execute(
        select(TranscriptSegment)
        .where(TranscriptSegment.transcript_id == transcript.id)
        .order_by(TranscriptSegment.index)
    )
    return list(rows.scalars())
