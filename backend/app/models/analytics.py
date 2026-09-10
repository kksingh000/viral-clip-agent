"""Post-publication performance and the score-calibration feedback loop."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import (
    BigInteger,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import Base, TimestampMixin, UUIDPrimaryKeyMixin
from app.database.types import EnumString, GUID, JSONB, UTCDateTime
from app.models.enums import Platform

if TYPE_CHECKING:
    from app.models.clip import GeneratedClip


class ClipAnalytics(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """A snapshot of a published clip's performance on one platform."""

    __tablename__ = "clip_analytics"
    __table_args__ = (
        UniqueConstraint(
            "clip_id", "platform", "captured_at", name="uq_analytics_clip_platform_time"
        ),
        Index("ix_analytics_clip_time", "clip_id", "captured_at"),
    )

    clip_id: Mapped[uuid.UUID] = mapped_column(
        GUID(), ForeignKey("generated_clips.id", ondelete="CASCADE"), nullable=False
    )
    platform: Mapped[Platform] = mapped_column(EnumString(Platform, 32), nullable=False)
    external_post_id: Mapped[str | None] = mapped_column(String(255))
    captured_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)

    views: Mapped[int | None] = mapped_column(BigInteger)
    likes: Mapped[int | None] = mapped_column(BigInteger)
    comments: Mapped[int | None] = mapped_column(BigInteger)
    shares: Mapped[int | None] = mapped_column(BigInteger)
    saves: Mapped[int | None] = mapped_column(BigInteger)

    average_view_duration: Mapped[float | None] = mapped_column(Float)
    completion_rate: Mapped[float | None] = mapped_column(Float)
    retention_curve: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB, default=list, nullable=False
    )

    # Populated by the feedback loop: how this clip performed relative to the
    # user's other clips in the same window (0-100).
    performance_percentile: Mapped[float | None] = mapped_column(Float)
    predicted_score: Mapped[float | None] = mapped_column(Float)
    prediction_error: Mapped[float | None] = mapped_column(Float)

    clip: Mapped["GeneratedClip"] = relationship(back_populates="analytics")


class ScoreCalibration(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """A learned adjustment to the viral-scoring weights for one user.

    The feedback loop fits weights against realised performance and writes a
    new row; scoring reads the most recent active row. Keeping every fit makes
    the change auditable and reversible.
    """

    __tablename__ = "score_calibrations"
    __table_args__ = (Index("ix_calibration_user_active", "user_id", "is_active"),)

    user_id: Mapped[uuid.UUID] = mapped_column(
        GUID(), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    weights: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    sample_size: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    baseline_mae: Mapped[float | None] = mapped_column(Float)
    calibrated_mae: Mapped[float | None] = mapped_column(Float)
    correlation: Mapped[float | None] = mapped_column(Float)
    method: Mapped[str] = mapped_column(String(64), default="ridge", nullable=False)
    notes: Mapped[str | None] = mapped_column(String(1024))
    is_active: Mapped[bool] = mapped_column(default=False, nullable=False)
