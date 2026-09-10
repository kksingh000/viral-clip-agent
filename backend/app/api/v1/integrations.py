"""Connected platform accounts and publishing.

Credentials are supplied by the deployment's OAuth flow and stored as opaque
references. This router never accepts a raw long-lived secret in a query
string and never logs token values.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Annotated

import anyio
from fastapi import APIRouter, Query, status
from pydantic import Field
from sqlalchemy import select

from app.api.deps import CurrentUser, DbSession, RateLimited
from app.core.errors import ForbiddenError, NotFoundError
from app.core.logging import get_logger
from app.models.clip import GeneratedClip
from app.models.enums import PUBLISHABLE_AUTHORIZATION, JobType, Platform
from app.models.publishing import PlatformAccount, PublishingJob
from app.models.video import Video
from app.schemas.common import APIModel, MessageResponse
from app.schemas.misc import (
    JobAcceptedResponse,
    PlatformAccountResponse,
    PublishingJobResponse,
    PublishRequest,
)
from app.services import jobs as job_service
from app.services.dispatch import dispatch

logger = get_logger(__name__)
router = APIRouter(prefix="/integrations", tags=["integrations"], dependencies=[RateLimited])


class ConnectAccountRequest(APIModel):
    """Register an account whose OAuth flow has already completed.

    The deployment performs the OAuth exchange (the redirect URI and client
    secret live outside the API); this endpoint records the resulting tokens
    against the user.
    """

    platform: Platform
    external_account_id: str = Field(min_length=1, max_length=255)
    display_name: str | None = Field(default=None, max_length=255)
    access_token: str = Field(min_length=1, max_length=4096)
    refresh_token: str | None = Field(default=None, max_length=4096)
    token_expires_at: datetime | None = None
    scopes: list[str] = Field(default_factory=list, max_length=30)


@router.get("/accounts", response_model=list[PlatformAccountResponse])
async def list_accounts(user: CurrentUser, session: DbSession) -> list[PlatformAccount]:
    rows = await session.execute(
        select(PlatformAccount)
        .where(PlatformAccount.user_id == user.id)
        .order_by(PlatformAccount.created_at.desc())
    )
    return list(rows.scalars())


@router.post(
    "/accounts",
    response_model=PlatformAccountResponse,
    status_code=status.HTTP_201_CREATED,
)
async def connect_account(
    payload: ConnectAccountRequest, user: CurrentUser, session: DbSession
) -> PlatformAccount:
    existing = (
        await session.execute(
            select(PlatformAccount).where(
                PlatformAccount.user_id == user.id,
                PlatformAccount.platform == payload.platform,
                PlatformAccount.external_account_id == payload.external_account_id,
            )
        )
    ).scalars().first()

    account = existing or PlatformAccount(
        user_id=user.id,
        platform=payload.platform,
        external_account_id=payload.external_account_id,
    )
    account.display_name = payload.display_name
    account.access_token_ref = payload.access_token
    account.refresh_token_ref = payload.refresh_token
    account.token_expires_at = payload.token_expires_at
    account.scopes = payload.scopes
    account.is_active = True
    account.connected_at = datetime.now(timezone.utc)
    account.last_error = None

    if existing is None:
        session.add(account)
    await session.commit()
    await session.refresh(account)
    logger.info(
        "platform account connected",
        extra={"user_id": str(user.id), "platform": str(payload.platform)},
    )
    return account


@router.delete("/accounts/{account_id}", response_model=MessageResponse)
async def disconnect_account(
    account_id: uuid.UUID, user: CurrentUser, session: DbSession
) -> MessageResponse:
    account = await session.get(PlatformAccount, account_id)
    if account is None or account.user_id != user.id:
        raise NotFoundError("Account not found.")
    await session.delete(account)
    await session.commit()
    return MessageResponse(message="Account disconnected.")


@router.post(
    "/publish/{clip_id}",
    response_model=JobAcceptedResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def publish_clip(
    clip_id: uuid.UUID,
    payload: PublishRequest,
    user: CurrentUser,
    session: DbSession,
) -> JobAcceptedResponse:
    """Queue a publish.

    Refused unless the *source* rights permit republishing and the clip has
    been approved. Both checks are re-run in the worker before upload.
    """
    clip = await session.get(GeneratedClip, clip_id)
    if clip is None or clip.user_id != user.id:
        raise NotFoundError(f"Clip {clip_id} not found.")

    video = await session.get(Video, clip.video_id)
    if video is None or video.authorization_status not in PUBLISHABLE_AUTHORIZATION:
        raise ForbiddenError(
            "This clip comes from a source whose rights do not permit automated "
            "republishing. Only content you own, licensed, or are authorized to "
            "use can be published."
        )

    account = None
    if payload.platform is not Platform.MANUAL_EXPORT:
        query = select(PlatformAccount).where(
            PlatformAccount.user_id == user.id,
            PlatformAccount.platform == payload.platform,
            PlatformAccount.is_active.is_(True),
        )
        if payload.account_id:
            query = query.where(PlatformAccount.id == payload.account_id)
        account = (await session.execute(query.limit(1))).scalars().first()
        if account is None:
            raise ForbiddenError(
                f"No connected {payload.platform.value} account. Connect one first."
            )

    from app.database.session import get_sync_session_factory
    from app.services.publishing import schedule_publish

    def _schedule() -> str:
        factory = get_sync_session_factory()
        with factory() as sync_session:
            record = sync_session.get(GeneratedClip, clip_id)
            connected = (
                sync_session.get(PlatformAccount, account.id) if account else None
            )
            publishing_job = schedule_publish(
                sync_session,
                record,
                platform=payload.platform,
                account=connected,
                scheduled_for=payload.scheduled_for,
                privacy=payload.privacy,
                title=payload.title,
                description=payload.description,
            )
            sync_session.commit()
            return str(publishing_job.id)

    publishing_job_id = await anyio.to_thread.run_sync(_schedule)

    job = await job_service.create_job_async(
        session,
        user_id=user.id,
        job_type=JobType.PUBLISH,
        clip_id=clip.id,
        video_id=clip.video_id,
        payload={
            "publishing_job_id": publishing_job_id,
            "platform": payload.platform.value,
        },
    )
    await session.commit()

    from app.workers.tasks import publish_clip_task

    dispatch(publish_clip_task, str(job.id))
    return JobAcceptedResponse(
        job_id=job.id, status=job.status, message="Publish queued."
    )


@router.get("/publishing-jobs", response_model=list[PublishingJobResponse])
async def list_publishing_jobs(
    user: CurrentUser,
    session: DbSession,
    clip_id: Annotated[uuid.UUID | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
) -> list[PublishingJob]:
    query = (
        select(PublishingJob)
        .join(GeneratedClip, GeneratedClip.id == PublishingJob.clip_id)
        .where(GeneratedClip.user_id == user.id)
        .order_by(PublishingJob.created_at.desc())
        .limit(limit)
    )
    if clip_id is not None:
        query = query.where(PublishingJob.clip_id == clip_id)
    rows = await session.execute(query)
    return list(rows.scalars())


@router.post(
    "/clips/{clip_id}/sync-analytics",
    response_model=JobAcceptedResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def sync_analytics(
    clip_id: uuid.UUID, user: CurrentUser, session: DbSession
) -> JobAcceptedResponse:
    clip = await session.get(GeneratedClip, clip_id)
    if clip is None or clip.user_id != user.id:
        raise NotFoundError(f"Clip {clip_id} not found.")

    job = await job_service.create_job_async(
        session,
        user_id=user.id,
        job_type=JobType.ANALYTICS_SYNC,
        clip_id=clip.id,
        payload={"clip_id": str(clip.id)},
    )
    await session.commit()

    from app.workers.tasks import sync_analytics_task

    dispatch(sync_analytics_task, str(job.id))
    return JobAcceptedResponse(
        job_id=job.id, status=job.status, message="Analytics sync queued."
    )
