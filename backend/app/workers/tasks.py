"""Celery tasks.

Every task follows the same shape, provided by :func:`run_job`:

* load the job row, refuse to run a cancelled one,
* mark it PROCESSING,
* run the body inside a transactional scope,
* on success mark COMPLETED with a result payload,
* on failure record ``error_code``/``error_message``/``retry_count`` and
  either schedule a retry with exponential backoff or leave it FAILED.

A task never re-raises past this wrapper, so one bad video cannot take down a
worker or stop a batch.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.agents.base import AgentContext
from app.core.errors import AppError
from app.core.logging import get_logger
from app.database.session import session_scope
from app.models.clip import CandidateClip, GeneratedClip
from app.models.enums import (
    CandidateStatus,
    ClipStatus,
    JobStatus,
    JobType,
    VideoStatus,
)
from app.models.job import ProcessingJob
from app.models.user import UserSettings
from app.models.video import Video
from app.services import clips as clip_service
from app.services import jobs as job_service
from app.services import pipeline as pipeline_service
from app.workers.celery_app import celery_app

logger = get_logger(__name__)

#: A job stuck in PROCESSING for longer than this is presumed dead.
STALE_JOB_HOURS = 6


def _user_settings(session: Session, user_id: uuid.UUID) -> UserSettings | None:
    return session.execute(
        select(UserSettings).where(UserSettings.user_id == user_id)
    ).scalars().first()


def run_job(
    job_id: str,
    body: Callable[[Session, ProcessingJob, AgentContext], dict[str, Any]],
    *,
    task=None,
) -> dict[str, Any]:
    """Execute one job with full lifecycle and error handling."""
    identifier = uuid.UUID(str(job_id))

    with session_scope() as session:
        job = job_service.get_job(session, identifier)
        if job is None:
            logger.error("job row not found", extra={"job_id": job_id})
            return {"status": "missing"}
        if job.status is JobStatus.CANCELLED:
            logger.info("job was cancelled before it started", extra={"job_id": job_id})
            return {"status": JobStatus.CANCELLED.value}
        job_service.mark_started(
            session, job, celery_task_id=getattr(task, "request", None)
            and getattr(task.request, "id", None)
        )

    context = AgentContext(job_id=str(identifier))
    try:
        with session_scope() as session:
            job = job_service.get_job(session, identifier)
            assert job is not None
            result = body(session, job, context)
            job_service.mark_completed(session, job, result=result)
            return {"status": JobStatus.COMPLETED.value, **result}
    except Exception as exc:  # noqa: BLE001 - the wrapper is the safety net
        retryable = isinstance(exc, AppError) and exc.retryable
        with session_scope() as session:
            job = job_service.get_job(session, identifier)
            if job is None:
                return {"status": "missing"}
            job_service.mark_failed(session, job, exc)
            delay = job_service.schedule_retry(session, job) if retryable else None
            job_type = job.job_type
            video_id = job.video_id

            if delay is None and job_type is JobType.ANALYZE_VIDEO and video_id:
                video = session.get(Video, video_id)
                if video is not None:
                    video.status = VideoStatus.FAILED
                    video.status_detail = str(exc)[:2000]

        if delay is not None and task is not None:
            logger.info(
                "retrying job", extra={"job_id": job_id, "countdown": delay}
            )
            raise task.retry(exc=exc, countdown=delay)
        logger.exception("job failed permanently", extra={"job_id": job_id})
        return {"status": JobStatus.FAILED.value, "error": str(exc)}


# ------------------------------------------------------------------- analysis
@celery_app.task(bind=True, name="app.workers.tasks.analyze_video_task", max_retries=3)
def analyze_video_task(self, job_id: str) -> dict[str, Any]:
    def body(session: Session, job: ProcessingJob, context: AgentContext) -> dict[str, Any]:
        video = session.get(Video, job.video_id) if job.video_id else None
        if video is None:
            raise AppError(f"Job {job.id} has no video to analyse.")
        result = pipeline_service.analyze_video(
            session,
            video,
            job=job,
            user_settings=_user_settings(session, job.user_id),
            force=bool(job.payload.get("force")),
            context=context,
        )
        payload = result.to_dict()

        if job.payload.get("auto_generate"):
            count = int(job.payload.get("clips_per_video") or 0) or None
            child = job_service.create_job(
                session,
                user_id=job.user_id,
                job_type=JobType.GENERATE_CLIP,
                video_id=video.id,
                parent_job_id=job.id,
                payload={"count": count} if count else {},
            )
            payload["generation_job_id"] = str(child.id)
            _enqueue_after_commit(generate_clips_task, str(child.id))
        return payload

    return run_job(job_id, body, task=self)


# ------------------------------------------------------------------ rendering
@celery_app.task(bind=True, name="app.workers.tasks.generate_clips_task", max_retries=2)
def generate_clips_task(self, job_id: str) -> dict[str, Any]:
    def body(session: Session, job: ProcessingJob, context: AgentContext) -> dict[str, Any]:
        video = session.get(Video, job.video_id) if job.video_id else None
        if video is None:
            raise AppError(f"Job {job.id} has no video to render from.")
        user_settings = _user_settings(session, job.user_id)

        candidate_ids = [uuid.UUID(str(v)) for v in job.payload.get("candidate_ids", [])]
        if candidate_ids:
            candidates = [
                c
                for c in (session.get(CandidateClip, cid) for cid in candidate_ids)
                if c is not None and c.video_id == video.id
            ]
        else:
            limit = int(
                job.payload.get("count")
                or (user_settings.clips_per_video if user_settings else 3)
            )
            minimum = user_settings.min_viral_score if user_settings else 0.0
            candidates = list(
                session.execute(
                    select(CandidateClip)
                    .where(
                        CandidateClip.video_id == video.id,
                        CandidateClip.status == CandidateStatus.PROPOSED,
                        CandidateClip.viral_score >= minimum,
                    )
                    .order_by(CandidateClip.viral_score.desc())
                    .limit(limit)
                ).scalars()
            )

        if not candidates:
            return {"clips": [], "note": "no candidates met the configured threshold"}

        produced: list[str] = []
        failures: list[dict[str, str]] = []
        for index, candidate in enumerate(candidates):
            job_service.update_progress(
                session,
                job,
                progress=index / max(1, len(candidates)),
                stage=f"clip {index + 1} of {len(candidates)}",
            )
            # Each clip gets its own SAVEPOINT. The whole job body runs inside
            # one transaction, so a plain rollback here would discard every
            # clip already produced in this batch -- the opposite of the
            # isolation this loop exists to provide.
            savepoint = session.begin_nested()
            try:
                result = clip_service.generate_clip(
                    session,
                    clip_service.ClipRequest(
                        video=video,
                        start=candidate.start_time,
                        end=candidate.end_time,
                        candidate=candidate,
                        crop_mode=_enum_or_none(job.payload.get("crop_mode"), "CropMode"),
                        caption_style=_enum_or_none(
                            job.payload.get("caption_style"), "CaptionStyle"
                        ),
                        caption_position=_enum_or_none(
                            job.payload.get("caption_position"), "CaptionPosition"
                        ),
                        include_hook_overlay=bool(
                            job.payload.get("include_hook_overlay", True)
                        ),
                    ),
                    user_settings=user_settings,
                    job=job,
                    context=context,
                )
                clip_id = str(result.clip.id)
                if job.payload.get("generate_variants"):
                    clip_service.generate_variants(
                        session, result.clip, user_settings=user_settings, job=job
                    )
                savepoint.commit()
                produced.append(clip_id)
            except Exception as exc:  # noqa: BLE001 - one clip must not stop the batch
                savepoint.rollback()
                failures.append({"candidate_id": str(candidate.id), "error": str(exc)})
                logger.exception(
                    "clip generation failed",
                    extra={"candidate_id": str(candidate.id)},
                )
        if failures and not produced:
            logger.warning(
                "every clip in the batch failed",
                extra={"video_id": str(video.id), "failures": len(failures)},
            )
        return {"clips": produced, "failures": failures}

    return run_job(job_id, body, task=self)


@celery_app.task(bind=True, name="app.workers.tasks.manual_clip_task", max_retries=2)
def manual_clip_task(self, job_id: str) -> dict[str, Any]:
    """Render a span the user chose by hand, bypassing candidate detection."""

    def body(session: Session, job: ProcessingJob, context: AgentContext) -> dict[str, Any]:
        video = session.get(Video, job.video_id) if job.video_id else None
        if video is None:
            raise AppError(f"Job {job.id} has no video to render from.")
        payload = job.payload
        result = clip_service.generate_clip(
            session,
            clip_service.ClipRequest(
                video=video,
                start=float(payload["start_time"]),
                end=float(payload["end_time"]),
                crop_mode=_enum_or_none(payload.get("crop_mode"), "CropMode"),
                caption_style=_enum_or_none(payload.get("caption_style"), "CaptionStyle"),
                caption_position=_enum_or_none(
                    payload.get("caption_position"), "CaptionPosition"
                ),
                hook_text=payload.get("hook_text"),
                include_hook_overlay=bool(payload.get("include_hook_overlay", True)),
            ),
            user_settings=_user_settings(session, job.user_id),
            job=job,
            context=context,
        )
        return {"clip_id": str(result.clip.id)}

    return run_job(job_id, body, task=self)


@celery_app.task(bind=True, name="app.workers.tasks.regenerate_clip_task", max_retries=2)
def regenerate_clip_task(self, job_id: str) -> dict[str, Any]:
    def body(session: Session, job: ProcessingJob, context: AgentContext) -> dict[str, Any]:
        clip = session.get(GeneratedClip, job.clip_id) if job.clip_id else None
        if clip is None:
            raise AppError(f"Job {job.id} has no clip to regenerate.")
        video = session.get(Video, clip.video_id)
        if video is None:
            raise AppError("The source video no longer exists.")

        payload = job.payload
        result = clip_service.generate_clip(
            session,
            clip_service.ClipRequest(
                video=video,
                start=float(payload.get("start_time", clip.start_time)),
                end=float(payload.get("end_time", clip.end_time)),
                candidate=(
                    session.get(CandidateClip, clip.candidate_id)
                    if clip.candidate_id
                    else None
                ),
                crop_mode=_enum_or_none(payload.get("crop_mode"), "CropMode")
                or clip.crop_mode,
                caption_style=_enum_or_none(payload.get("caption_style"), "CaptionStyle")
                or clip.caption_style,
                caption_position=_enum_or_none(
                    payload.get("caption_position"), "CaptionPosition"
                )
                or clip.caption_position,
                hook_text=None if payload.get("regenerate_hook") else clip.hook_text,
                include_hook_overlay=bool(payload.get("include_hook_overlay", True)),
            ),
            user_settings=_user_settings(session, job.user_id),
            job=job,
            context=context,
        )
        # The old render is superseded; keep the row for history but retire it.
        clip.status = ClipStatus.REJECTED
        clip.review_note = f"Superseded by regenerated clip {result.clip.id}."
        session.flush()
        return {"clip_id": str(result.clip.id), "replaced": str(clip.id)}

    return run_job(job_id, body, task=self)


@celery_app.task(bind=True, name="app.workers.tasks.generate_variants_task", max_retries=1)
def generate_variants_task(self, job_id: str) -> dict[str, Any]:
    def body(session: Session, job: ProcessingJob, context: AgentContext) -> dict[str, Any]:
        clip = session.get(GeneratedClip, job.clip_id) if job.clip_id else None
        if clip is None:
            raise AppError(f"Job {job.id} has no clip to vary.")
        labels = job.payload.get("labels") or ["B", "C"]
        variants = clip_service.generate_variants(
            session,
            clip,
            labels=labels,
            user_settings=_user_settings(session, job.user_id),
            job=job,
        )
        return {"variants": [str(v.id) for v in variants]}

    return run_job(job_id, body, task=self)


# ------------------------------------------------------------------ discovery
@celery_app.task(bind=True, name="app.workers.tasks.trend_discovery_task", max_retries=2)
def trend_discovery_task(self, job_id: str) -> dict[str, Any]:
    def body(session: Session, job: ProcessingJob, context: AgentContext) -> dict[str, Any]:
        from app.services.discovery import run_discovery

        return run_discovery(
            session,
            user_id=job.user_id,
            region=str(job.payload.get("region", "US")),
            category_id=job.payload.get("category_id"),
            max_results=int(job.payload.get("max_results", 50)),
            query=job.payload.get("query"),
            context=context,
        )

    return run_job(job_id, body, task=self)


@celery_app.task(name="app.workers.tasks.refresh_trend_metrics_task")
def refresh_trend_metrics_task() -> dict[str, Any]:
    """Re-snapshot metrics for recently discovered videos.

    Velocity and acceleration need at least two observations, so this is what
    makes the trend score meaningful rather than a static ranking.
    """
    from app.services.discovery import refresh_metrics

    try:
        with session_scope() as session:
            return refresh_metrics(session)
    except Exception as exc:  # noqa: BLE001 - a scheduled job must not crash beat
        logger.exception("trend metric refresh failed")
        return {"status": "failed", "error": str(exc)}


# ----------------------------------------------------------------- publishing
@celery_app.task(bind=True, name="app.workers.tasks.publish_clip_task", max_retries=3)
def publish_clip_task(self, job_id: str) -> dict[str, Any]:
    def body(session: Session, job: ProcessingJob, context: AgentContext) -> dict[str, Any]:
        from app.services.publishing import publish_clip

        publishing_job_id = job.payload.get("publishing_job_id")
        if not publishing_job_id:
            raise AppError("Publish job is missing publishing_job_id.")
        return publish_clip(session, uuid.UUID(str(publishing_job_id)))

    return run_job(job_id, body, task=self)


@celery_app.task(bind=True, name="app.workers.tasks.sync_analytics_task", max_retries=2)
def sync_analytics_task(self, job_id: str) -> dict[str, Any]:
    def body(session: Session, job: ProcessingJob, context: AgentContext) -> dict[str, Any]:
        from app.services.analytics import sync_clip_analytics

        clip_id = job.clip_id or job.payload.get("clip_id")
        if not clip_id:
            raise AppError("Analytics job is missing a clip id.")
        return sync_clip_analytics(session, uuid.UUID(str(clip_id)))

    return run_job(job_id, body, task=self)


@celery_app.task(name="app.workers.tasks.sync_all_analytics_task")
def sync_all_analytics_task() -> dict[str, Any]:
    from app.services.analytics import sync_all_published

    try:
        with session_scope() as session:
            return sync_all_published(session)
    except Exception as exc:  # noqa: BLE001
        logger.exception("analytics sync failed")
        return {"status": "failed", "error": str(exc)}


# ---------------------------------------------------------------- maintenance
@celery_app.task(name="app.workers.tasks.reap_stale_jobs_task")
def reap_stale_jobs_task() -> dict[str, Any]:
    """Fail jobs whose worker died mid-flight.

    Without this a killed worker leaves rows in PROCESSING forever and the
    dashboard shows work that will never finish.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(hours=STALE_JOB_HOURS)
    reaped = 0
    try:
        with session_scope() as session:
            stale = session.execute(
                select(ProcessingJob).where(
                    ProcessingJob.status == JobStatus.PROCESSING,
                    ProcessingJob.started_at < cutoff,
                )
            ).scalars()
            for job in stale:
                job_service.mark_failed(
                    session,
                    job,
                    "The worker processing this job stopped responding.",
                    error_code="worker_lost",
                )
                reaped += 1
    except Exception as exc:  # noqa: BLE001
        logger.exception("stale job reaper failed")
        return {"status": "failed", "error": str(exc)}
    return {"reaped": reaped}


# --------------------------------------------------------------------- helpers
def _enum_or_none(value: Any, enum_name: str):
    if value is None:
        return None
    from app.models import enums

    enum_class = getattr(enums, enum_name)
    try:
        return enum_class(value)
    except ValueError:
        logger.warning(
            "ignoring unknown enum value",
            extra={"enum": enum_name, "value": str(value)},
        )
        return None


def _enqueue_after_commit(task, *args: Any) -> None:
    """Dispatch a follow-up task.

    Called from inside a job body, so the surrounding ``session_scope`` commits
    immediately afterwards; the child task re-reads its own row.
    """
    from app.services.dispatch import dispatch

    dispatch(task, *args)
