"""Processing-job lifecycle.

A job row is the durable record behind every async task: it is created before
the task is dispatched, updated as the task progresses, and always reaches a
terminal state -- including when the worker dies, because the task wrapper
catches everything.

A failure is always local to one job. Nothing here raises into the caller's
loop, so one bad video cannot stop a batch.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from app.core.errors import AppError
from app.core.logging import get_logger
from app.models.enums import JobStatus, JobType
from app.models.job import ProcessingJob

logger = get_logger(__name__)

#: Retry delays in seconds; index by ``retry_count``. Exponential with a cap.
RETRY_BACKOFF_SECONDS = (30, 120, 480, 1800)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def create_job(
    session: Session,
    *,
    user_id: uuid.UUID,
    job_type: JobType,
    video_id: uuid.UUID | None = None,
    clip_id: uuid.UUID | None = None,
    parent_job_id: uuid.UUID | None = None,
    payload: dict[str, Any] | None = None,
    max_retries: int = 3,
) -> ProcessingJob:
    job = ProcessingJob(
        user_id=user_id,
        video_id=video_id,
        clip_id=clip_id,
        parent_job_id=parent_job_id,
        job_type=job_type,
        status=JobStatus.QUEUED,
        payload=payload or {},
        max_retries=max_retries,
        queued_at=_now(),
    )
    session.add(job)
    session.flush()
    logger.info(
        "job created",
        extra={"job_id": str(job.id), "job_type": job_type.value},
    )
    return job


async def create_job_async(
    session: "AsyncSession",
    *,
    user_id: uuid.UUID,
    job_type: JobType,
    video_id: uuid.UUID | None = None,
    clip_id: uuid.UUID | None = None,
    payload: dict[str, Any] | None = None,
    max_retries: int = 3,
) -> ProcessingJob:
    """Async twin of :func:`create_job`, for the API request path.

    The API only ever *creates* jobs; workers own every subsequent transition,
    so this is the single async entry point rather than a parallel module.
    """
    job = ProcessingJob(
        user_id=user_id,
        video_id=video_id,
        clip_id=clip_id,
        job_type=job_type,
        status=JobStatus.QUEUED,
        payload=payload or {},
        max_retries=max_retries,
        queued_at=_now(),
    )
    session.add(job)
    await session.flush()
    logger.info(
        "job created", extra={"job_id": str(job.id), "job_type": job_type.value}
    )
    return job


def get_job(session: Session, job_id: uuid.UUID) -> ProcessingJob | None:
    return session.get(ProcessingJob, job_id)


def mark_started(
    session: Session, job: ProcessingJob, *, celery_task_id: str | None = None
) -> ProcessingJob:
    job.status = JobStatus.PROCESSING
    job.started_at = _now()
    job.progress = 0.0
    if celery_task_id:
        job.celery_task_id = celery_task_id
    session.flush()
    return job


def update_progress(
    session: Session,
    job: ProcessingJob,
    *,
    progress: float,
    stage: str | None = None,
) -> ProcessingJob:
    job.progress = max(0.0, min(1.0, progress))
    if stage:
        job.stage = stage
    session.flush()
    logger.debug(
        "job progress",
        extra={"job_id": str(job.id), "progress": job.progress, "stage": stage},
    )
    return job


def record_usage(
    session: Session,
    job: ProcessingJob,
    *,
    input_tokens: int = 0,
    output_tokens: int = 0,
    calls: int = 0,
    cost_usd: float = 0.0,
) -> ProcessingJob:
    job.llm_input_tokens += input_tokens
    job.llm_output_tokens += output_tokens
    job.llm_calls += calls
    job.estimated_cost_usd += cost_usd
    session.flush()
    return job


def mark_completed(
    session: Session, job: ProcessingJob, *, result: dict[str, Any] | None = None
) -> ProcessingJob:
    job.status = JobStatus.COMPLETED
    job.progress = 1.0
    job.finished_at = _now()
    job.result = result or {}
    if job.started_at:
        job.duration_seconds = (job.finished_at - job.started_at).total_seconds()
    session.flush()
    logger.info(
        "job completed",
        extra={
            "job_id": str(job.id),
            "job_type": job.job_type,
            "duration_s": round(job.duration_seconds or 0.0, 2),
            "llm_calls": job.llm_calls,
            "cost_usd": round(job.estimated_cost_usd, 4),
        },
    )
    return job


def mark_failed(
    session: Session,
    job: ProcessingJob,
    error: BaseException | str,
    *,
    error_code: str | None = None,
    details: dict[str, Any] | None = None,
) -> ProcessingJob:
    """Record a failure. Always sets a code, a message and a timestamp."""
    if isinstance(error, AppError):
        code = error_code or error.error_code
        message = error.message
        payload = {**error.details, **(details or {})}
    elif isinstance(error, BaseException):
        code = error_code or type(error).__name__
        message = str(error) or repr(error)
        payload = details or {}
    else:
        code = error_code or "error"
        message = str(error)
        payload = details or {}

    job.status = JobStatus.FAILED
    job.finished_at = _now()
    job.error_code = code[:64]
    job.error_message = message[:4000]
    job.error_details = payload
    if job.started_at:
        job.duration_seconds = (job.finished_at - job.started_at).total_seconds()
    session.flush()
    logger.error(
        "job failed",
        extra={
            "job_id": str(job.id),
            "job_type": job.job_type,
            "error_code": job.error_code,
            "retry_count": job.retry_count,
        },
    )
    return job


def mark_cancelled(session: Session, job: ProcessingJob, reason: str = "") -> ProcessingJob:
    job.status = JobStatus.CANCELLED
    job.finished_at = _now()
    job.error_code = "cancelled"
    job.error_message = reason or "Cancelled by request."
    session.flush()
    return job


def schedule_retry(session: Session, job: ProcessingJob) -> int | None:
    """Increment the retry counter and return the backoff delay in seconds.

    Returns ``None`` when the job has exhausted its retries.
    """
    if job.retry_count >= job.max_retries:
        return None
    delay = RETRY_BACKOFF_SECONDS[min(job.retry_count, len(RETRY_BACKOFF_SECONDS) - 1)]
    job.retry_count += 1
    job.status = JobStatus.QUEUED
    job.error_details = {**job.error_details, "retry_scheduled_in": delay}
    session.flush()
    logger.info(
        "job retry scheduled",
        extra={
            "job_id": str(job.id),
            "retry_count": job.retry_count,
            "delay_s": delay,
        },
    )
    return delay


def active_jobs(
    session: Session, *, user_id: uuid.UUID | None = None, limit: int = 100
) -> list[ProcessingJob]:
    statement = select(ProcessingJob).where(
        ProcessingJob.status.in_([JobStatus.QUEUED, JobStatus.PROCESSING])
    )
    if user_id is not None:
        statement = statement.where(ProcessingJob.user_id == user_id)
    statement = statement.order_by(ProcessingJob.created_at.desc()).limit(limit)
    return list(session.execute(statement).scalars())


def queue_depth(session: Session) -> dict[str, int]:
    counts: dict[str, int] = {}
    for status in (JobStatus.QUEUED, JobStatus.PROCESSING, JobStatus.FAILED):
        statement = select(ProcessingJob).where(ProcessingJob.status == status)
        counts[status.value] = len(list(session.execute(statement).scalars()))
    return counts
