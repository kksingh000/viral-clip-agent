"""Processing job records -- the durable state behind every async task."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import (
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import Base, TimestampMixin, UUIDPrimaryKeyMixin
from app.database.types import EnumString, GUID, JSONB, UTCDateTime
from app.models.enums import JobStatus, JobType

if TYPE_CHECKING:
    from app.models.video import Video


class ProcessingJob(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """One unit of asynchronous work.

    A failure here is always local: the row records ``error_code``,
    ``error_message``, ``retry_count`` and timing, and the pipeline continues
    with other videos.
    """

    __tablename__ = "processing_jobs"
    __table_args__ = (
        Index("ix_jobs_user_status", "user_id", "status"),
        Index("ix_jobs_type_status", "job_type", "status"),
        Index("ix_jobs_video", "video_id", "created_at"),
    )

    user_id: Mapped[uuid.UUID] = mapped_column(
        GUID(), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    video_id: Mapped[uuid.UUID | None] = mapped_column(
        GUID(), ForeignKey("videos.id", ondelete="CASCADE")
    )
    clip_id: Mapped[uuid.UUID | None] = mapped_column(
        GUID(), ForeignKey("generated_clips.id", ondelete="CASCADE")
    )
    parent_job_id: Mapped[uuid.UUID | None] = mapped_column(
        GUID(), ForeignKey("processing_jobs.id", ondelete="SET NULL")
    )

    job_type: Mapped[JobType] = mapped_column(EnumString(JobType, 32), nullable=False)
    status: Mapped[JobStatus] = mapped_column(
        EnumString(JobStatus, 32), default=JobStatus.QUEUED, nullable=False, index=True
    )
    celery_task_id: Mapped[str | None] = mapped_column(String(128), index=True)

    progress: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    stage: Mapped[str | None] = mapped_column(String(64))

    payload: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    result: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)

    error_code: Mapped[str | None] = mapped_column(String(64))
    error_message: Mapped[str | None] = mapped_column(Text)
    error_details: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    retry_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    max_retries: Mapped[int] = mapped_column(Integer, default=3, nullable=False)

    queued_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
    started_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
    finished_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
    duration_seconds: Mapped[float | None] = mapped_column(Float)

    # Token/cost accounting rolled up from every agent call inside the job.
    llm_input_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    llm_output_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    llm_calls: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    estimated_cost_usd: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)

    video: Mapped["Video | None"] = relationship(back_populates="jobs")

    @property
    def is_terminal(self) -> bool:
        from app.models.enums import TERMINAL_JOB_STATUSES

        return self.status in TERMINAL_JOB_STATUSES
