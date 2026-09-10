"""Job inspection endpoints."""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Query
from sqlalchemy import func, select

from app.api.deps import CurrentUser, DbSession, RateLimited
from app.core.errors import NotFoundError, ValidationFailure
from app.models.enums import TERMINAL_JOB_STATUSES, JobStatus, JobType
from app.models.job import ProcessingJob
from app.schemas.common import MessageResponse, Page
from app.schemas.misc import JobResponse

router = APIRouter(prefix="/jobs", tags=["jobs"], dependencies=[RateLimited])


@router.get("", response_model=Page[JobResponse])
async def list_jobs(
    user: CurrentUser,
    session: DbSession,
    status_filter: Annotated[JobStatus | None, Query(alias="status")] = None,
    job_type: Annotated[JobType | None, Query()] = None,
    video_id: Annotated[uuid.UUID | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 25,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> Page[JobResponse]:
    conditions = [ProcessingJob.user_id == user.id]
    if status_filter is not None:
        conditions.append(ProcessingJob.status == status_filter)
    if job_type is not None:
        conditions.append(ProcessingJob.job_type == job_type)
    if video_id is not None:
        conditions.append(ProcessingJob.video_id == video_id)

    total = await session.scalar(
        select(func.count()).select_from(ProcessingJob).where(*conditions)
    )
    rows = await session.execute(
        select(ProcessingJob)
        .where(*conditions)
        .order_by(ProcessingJob.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    items = [JobResponse.model_validate(row) for row in rows.scalars()]
    return Page.of(items, total=int(total or 0), limit=limit, offset=offset)


@router.get("/{job_id}", response_model=JobResponse)
async def get_job(
    job_id: uuid.UUID, user: CurrentUser, session: DbSession
) -> ProcessingJob:
    job = await session.get(ProcessingJob, job_id)
    if job is None or job.user_id != user.id:
        raise NotFoundError(f"Job {job_id} not found.")
    return job


@router.post("/{job_id}/cancel", response_model=MessageResponse)
async def cancel_job(
    job_id: uuid.UUID, user: CurrentUser, session: DbSession
) -> MessageResponse:
    """Request cancellation.

    A QUEUED job will not start. A job already running is marked cancelled and
    stops at its next checkpoint; ffmpeg work already in flight still completes.
    """
    job = await session.get(ProcessingJob, job_id)
    if job is None or job.user_id != user.id:
        raise NotFoundError(f"Job {job_id} not found.")
    if job.status in TERMINAL_JOB_STATUSES:
        raise ValidationFailure(f"Job is already {job.status}.")

    job.status = JobStatus.CANCELLED
    job.error_code = "cancelled"
    job.error_message = "Cancelled by the user."
    await session.commit()
    return MessageResponse(message="Cancellation requested.")
