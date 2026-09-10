"""Trend discovery and topic intelligence endpoints.

Everything here is metadata. Nothing in this router downloads media or implies
that a discovered video may be republished; that requires a separate,
explicit rights decision on the video itself.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Query, status
from sqlalchemy import func, select

from app.api.deps import CurrentUser, DbSession, RateLimited
from app.core.config import settings as app_settings
from app.core.errors import ValidationFailure
from app.models.enums import JobType, VideoSource
from app.models.trend import Topic
from app.models.video import Video, VideoMetric
from app.schemas.common import Page
from app.schemas.misc import JobAcceptedResponse, TopicResponse, TrendDiscoveryRequest
from app.schemas.video import TrendingVideoResponse
from app.services import jobs as job_service
from app.services.dispatch import dispatch

router = APIRouter(tags=["trending"], dependencies=[RateLimited])


@router.get("/trending", response_model=Page[TrendingVideoResponse])
async def list_trending(
    user: CurrentUser,
    session: DbSession,
    min_score: Annotated[float, Query(ge=0, le=100)] = 0.0,
    category: Annotated[str | None, Query(max_length=128)] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 25,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> Page[TrendingVideoResponse]:
    """Discovered videos ranked by momentum, not by total views."""
    conditions = [
        Video.user_id == user.id,
        Video.source == VideoSource.DISCOVERY,
        Video.trend_score.is_not(None),
        Video.trend_score >= min_score,
    ]
    if category:
        conditions.append(Video.category == category)

    total = await session.scalar(
        select(func.count()).select_from(Video).where(*conditions)
    )
    rows = list(
        (
            await session.execute(
                select(Video)
                .where(*conditions)
                .order_by(Video.trend_score.desc())
                .limit(limit)
                .offset(offset)
            )
        ).scalars()
    )

    items: list[TrendingVideoResponse] = []
    for video in rows:
        latest = (
            await session.execute(
                select(VideoMetric)
                .where(VideoMetric.video_id == video.id)
                .order_by(VideoMetric.captured_at.desc())
                .limit(1)
            )
        ).scalars().first()
        response = TrendingVideoResponse.model_validate(video)
        if latest is not None:
            response.view_velocity = latest.view_velocity
            response.engagement_rate = latest.engagement_rate
            response.growth_acceleration = latest.growth_acceleration
        items.append(response)
    return Page.of(items, total=int(total or 0), limit=limit, offset=offset)


@router.post(
    "/trending/discover",
    response_model=JobAcceptedResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def discover(
    payload: TrendDiscoveryRequest, user: CurrentUser, session: DbSession
) -> JobAcceptedResponse:
    """Queue a metadata-only discovery run."""
    if not app_settings.youtube_api_key:
        raise ValidationFailure(
            "YOUTUBE_API_KEY is not configured, so trend discovery cannot run. "
            "Set it in the environment to enable the discovery engine."
        )
    job = await job_service.create_job_async(
        session,
        user_id=user.id,
        job_type=JobType.TREND_DISCOVERY,
        payload={
            "region": payload.region,
            "category_id": payload.category_id,
            "max_results": payload.max_results,
            "query": payload.query,
        },
    )
    await session.commit()

    from app.workers.tasks import trend_discovery_task

    dispatch(trend_discovery_task, str(job.id))
    return JobAcceptedResponse(
        job_id=job.id,
        status=job.status,
        message=(
            "Discovery queued. Results are metadata only and are recorded with "
            "authorization_status=UNKNOWN."
        ),
    )


@router.get("/topics", response_model=Page[TopicResponse])
async def list_topics(
    user: CurrentUser,
    session: DbSession,
    region: Annotated[str | None, Query(max_length=8)] = None,
    kind: Annotated[str | None, Query(max_length=32)] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 25,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> Page[TopicResponse]:
    conditions = []
    if region:
        conditions.append(Topic.region == region)
    if kind:
        conditions.append(Topic.kind == kind)

    total = await session.scalar(select(func.count()).select_from(Topic).where(*conditions))
    rows = await session.execute(
        select(Topic)
        .where(*conditions)
        .order_by(Topic.trend_score.desc())
        .limit(limit)
        .offset(offset)
    )
    items = [TopicResponse.model_validate(row) for row in rows.scalars()]
    return Page.of(items, total=int(total or 0), limit=limit, offset=offset)
