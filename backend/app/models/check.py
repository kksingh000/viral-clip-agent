"""Automated quality-control and content-safety records."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

from sqlalchemy import Boolean, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import Base, TimestampMixin, UUIDPrimaryKeyMixin
from app.database.types import EnumString, GUID, JSONB
from app.models.enums import QualityStatus, SafetyVerdict

if TYPE_CHECKING:
    from app.models.clip import GeneratedClip


class QualityCheck(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Result of one VideoQualityAgent inspection of a rendered clip."""

    __tablename__ = "quality_checks"
    __table_args__ = (Index("ix_quality_clip_attempt", "clip_id", "attempt"),)

    clip_id: Mapped[uuid.UUID] = mapped_column(
        GUID(), ForeignKey("generated_clips.id", ondelete="CASCADE"), nullable=False
    )
    attempt: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    status: Mapped[QualityStatus] = mapped_column(EnumString(QualityStatus, 16), nullable=False)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    threshold: Mapped[float] = mapped_column(Float, nullable=False)

    # Each issue: {check, severity, message, detail, auto_fixable, fix_action}
    issues: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, default=list, nullable=False)
    recommendations: Mapped[list[str]] = mapped_column(JSONB, default=list, nullable=False)
    # Deterministic probe output (duration/resolution/loudness/black frames...)
    technical: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    auto_corrected: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    correction_applied: Mapped[str | None] = mapped_column(Text)

    clip: Mapped["GeneratedClip"] = relationship(back_populates="quality_checks")


class ContentSafetyCheck(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Result of one ContentSafetyAgent review."""

    __tablename__ = "content_safety_checks"
    __table_args__ = (Index("ix_safety_clip", "clip_id", "created_at"),)

    clip_id: Mapped[uuid.UUID] = mapped_column(
        GUID(), ForeignKey("generated_clips.id", ondelete="CASCADE"), nullable=False
    )
    verdict: Mapped[SafetyVerdict] = mapped_column(EnumString(SafetyVerdict, 16), nullable=False)
    # {hate: 0-10, harassment: ..., sexual: ..., violence: ..., ...}
    category_scores: Mapped[dict[str, Any]] = mapped_column(
        JSONB, default=dict, nullable=False
    )
    flagged_categories: Mapped[list[str]] = mapped_column(
        JSONB, default=list, nullable=False
    )
    rationale: Mapped[str | None] = mapped_column(Text)
    evidence: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB, default=list, nullable=False
    )
    model_name: Mapped[str | None] = mapped_column(String(128))

    clip: Mapped["GeneratedClip"] = relationship(back_populates="safety_checks")
