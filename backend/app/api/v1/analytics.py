"""Dashboard aggregates, performance analytics and score calibration."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Annotated

import anyio
from fastapi import APIRouter, Query
from sqlalchemy import func, select

from app.api.deps import CurrentUser, DbSession, RateLimited
from app.models.analytics import ClipAnalytics
from app.models.clip import GeneratedClip
from app.models.enums import ClipStatus, JobStatus, VideoStatus
from app.models.job import ProcessingJob
from app.models.video import Video
from app.schemas.misc import (
    AnalyticsResponse,
    DashboardResponse,
    ScorePerformancePoint,
)

router = APIRouter(tags=["analytics"], dependencies=[RateLimited])


@router.get("/dashboard", response_model=DashboardResponse)
async def dashboard(user: CurrentUser, session: DbSession) -> DashboardResponse:
    now = datetime.now(timezone.utc)
    day_ago = now - timedelta(hours=24)
    start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0)

    async def count(model, *conditions) -> int:
        value = await session.scalar(
            select(func.count()).select_from(model).where(*conditions)
        )
        return int(value or 0)

    videos_today = await count(
        Video, Video.user_id == user.id, Video.created_at >= start_of_day
    )
    videos_analyzed = await count(
        Video, Video.user_id == user.id, Video.status == VideoStatus.ANALYZED
    )
    videos_pending = await count(
        Video,
        Video.user_id == user.id,
        Video.status.in_([VideoStatus.UPLOADED, VideoStatus.PENDING_MEDIA]),
    )
    clips_generated = await count(GeneratedClip, GeneratedClip.user_id == user.id)
    clips_approved = await count(
        GeneratedClip,
        GeneratedClip.user_id == user.id,
        GeneratedClip.status.in_([ClipStatus.APPROVED, ClipStatus.PUBLISHED]),
    )
    clips_review = await count(
        GeneratedClip,
        GeneratedClip.user_id == user.id,
        GeneratedClip.status == ClipStatus.NEEDS_REVIEW,
    )
    active_jobs = await count(
        ProcessingJob,
        ProcessingJob.user_id == user.id,
        ProcessingJob.status.in_([JobStatus.QUEUED, JobStatus.PROCESSING]),
    )
    failed_jobs = await count(
        ProcessingJob,
        ProcessingJob.user_id == user.id,
        ProcessingJob.status == JobStatus.FAILED,
        ProcessingJob.finished_at >= day_ago,
    )

    average_score = await session.scalar(
        select(func.avg(GeneratedClip.viral_score)).where(
            GeneratedClip.user_id == user.id, GeneratedClip.viral_score.is_not(None)
        )
    )
    cost = await session.scalar(
        select(func.sum(ProcessingJob.estimated_cost_usd)).where(
            ProcessingJob.user_id == user.id, ProcessingJob.created_at >= day_ago
        )
    )

    top = (
        await session.execute(
            select(GeneratedClip)
            .where(
                GeneratedClip.user_id == user.id,
                GeneratedClip.viral_score.is_not(None),
            )
            .order_by(GeneratedClip.viral_score.desc())
            .limit(1)
        )
    ).scalars().first()

    return DashboardResponse(
        videos_discovered_today=videos_today,
        videos_analyzed=videos_analyzed,
        videos_pending=videos_pending,
        clips_generated=clips_generated,
        clips_approved=clips_approved,
        clips_awaiting_review=clips_review,
        average_viral_score=float(average_score) if average_score is not None else None,
        top_clip=(
            {
                "id": str(top.id),
                "title": top.title,
                "viral_score": top.viral_score,
                "duration_seconds": top.duration_seconds,
                "status": str(top.status),
            }
            if top
            else None
        ),
        active_jobs=active_jobs,
        failed_jobs_24h=failed_jobs,
        llm_cost_24h_usd=float(cost or 0.0),
    )


@router.get("/analytics", response_model=AnalyticsResponse)
async def analytics(
    user: CurrentUser,
    session: DbSession,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> AnalyticsResponse:
    """Predicted score against realised performance."""
    from app.database.session import get_sync_session_factory
    from app.services.analytics import account_summary

    def _summary() -> dict:
        factory = get_sync_session_factory()
        with factory() as sync_session:
            return account_summary(sync_session, user.id)

    summary = await anyio.to_thread.run_sync(_summary)

    rows = list(
        (
            await session.execute(
                select(ClipAnalytics, GeneratedClip.title)
                .join(GeneratedClip, GeneratedClip.id == ClipAnalytics.clip_id)
                .where(GeneratedClip.user_id == user.id)
                .order_by(ClipAnalytics.captured_at.desc())
                .limit(limit)
            )
        ).all()
    )
    seen: set = set()
    points: list[ScorePerformancePoint] = []
    for record, title in rows:
        if record.clip_id in seen:
            continue
        seen.add(record.clip_id)
        points.append(
            ScorePerformancePoint(
                clip_id=record.clip_id,
                title=title,
                predicted_score=record.predicted_score,
                views=record.views,
                performance_percentile=record.performance_percentile,
                prediction_error=record.prediction_error,
            )
        )

    return AnalyticsResponse(
        clips_published=summary["clips_published"],
        total_views=summary["total_views"],
        total_likes=summary["total_likes"],
        total_comments=summary["total_comments"],
        average_completion_rate=summary["average_completion_rate"],
        score_correlation=summary["score_correlation"],
        calibration_sample_size=summary["calibration_sample_size"],
        points=points,
    )


@router.post("/analytics/calibrate", response_model=dict)
async def calibrate(user: CurrentUser) -> dict:
    """Refit the scoring weights against realised performance.

    Returns ``status: insufficient_data`` until enough clips have been
    published and measured -- a fit on a handful of clips is noise.
    """
    from app.database.session import get_sync_session_factory
    from app.services.analytics import calibrate_scoring

    def _run() -> dict:
        factory = get_sync_session_factory()
        with factory() as sync_session:
            result = calibrate_scoring(sync_session, user.id)
            sync_session.commit()
            return result

    return await anyio.to_thread.run_sync(_run)
