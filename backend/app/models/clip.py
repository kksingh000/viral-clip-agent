"""Candidate moments, rendered clips and their variants."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import (
    Boolean,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import Base, TimestampMixin, UUIDPrimaryKeyMixin
from app.database.types import EnumString, GUID, JSONB, UTCDateTime, Vector
from app.models.enums import (
    CandidateStatus,
    CaptionPosition,
    CaptionStyle,
    ClipStatus,
    CropMode,
    PublishStatus,
)
from app.models.transcript import EMBEDDING_DIM

if TYPE_CHECKING:
    from app.models.analytics import ClipAnalytics
    from app.models.check import ContentSafetyCheck, QualityCheck
    from app.models.publishing import PublishingJob
    from app.models.video import MediaAsset, Video


class CandidateClip(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """A moment the ViralMomentAgent proposed, before any rendering."""

    __tablename__ = "candidate_clips"
    __table_args__ = (
        Index("ix_candidates_video_score", "video_id", "viral_score"),
        Index("ix_candidates_status", "status"),
    )

    video_id: Mapped[uuid.UUID] = mapped_column(
        GUID(), ForeignKey("videos.id", ondelete="CASCADE"), nullable=False, index=True
    )

    start_time: Mapped[float] = mapped_column(Float, nullable=False)
    end_time: Mapped[float] = mapped_column(Float, nullable=False)
    duration_seconds: Mapped[float] = mapped_column(Float, nullable=False)

    # The agent may request extra lead-in for context; both are recorded so a
    # human can see how much runway the model asked for and why.
    context_lead_in: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    raw_moment_start: Mapped[float | None] = mapped_column(Float)

    hook: Mapped[str | None] = mapped_column(Text)
    summary: Mapped[str | None] = mapped_column(Text)
    reason: Mapped[str | None] = mapped_column(Text)
    transcript_excerpt: Mapped[str | None] = mapped_column(Text)

    viral_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    score_breakdown: Mapped[dict[str, Any]] = mapped_column(
        JSONB, default=dict, nullable=False
    )
    status: Mapped[CandidateStatus] = mapped_column(
        EnumString(CandidateStatus, 32), default=CandidateStatus.PROPOSED, nullable=False
    )
    duplicate_of_id: Mapped[uuid.UUID | None] = mapped_column(
        GUID(), ForeignKey("candidate_clips.id", ondelete="SET NULL")
    )
    rank: Mapped[int | None] = mapped_column(Integer)

    model_name: Mapped[str | None] = mapped_column(String(128))
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIM))

    video: Mapped["Video"] = relationship(back_populates="candidates")
    clips: Mapped[list["GeneratedClip"]] = relationship(back_populates="candidate")


class GeneratedClip(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """A rendered vertical short."""

    __tablename__ = "generated_clips"
    __table_args__ = (
        Index("ix_clips_user_status", "user_id", "status"),
        Index("ix_clips_video", "video_id", "created_at"),
    )

    user_id: Mapped[uuid.UUID] = mapped_column(
        GUID(), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    video_id: Mapped[uuid.UUID] = mapped_column(
        GUID(), ForeignKey("videos.id", ondelete="CASCADE"), nullable=False
    )
    candidate_id: Mapped[uuid.UUID | None] = mapped_column(
        GUID(), ForeignKey("candidate_clips.id", ondelete="SET NULL"), index=True
    )

    # -- cut
    start_time: Mapped[float] = mapped_column(Float, nullable=False)
    end_time: Mapped[float] = mapped_column(Float, nullable=False)
    duration_seconds: Mapped[float] = mapped_column(Float, nullable=False)

    # -- render settings actually used (a full record, so a regeneration is
    #    reproducible and a variant diff is meaningful)
    crop_mode: Mapped[CropMode] = mapped_column(EnumString(CropMode, 32), nullable=False)
    caption_style: Mapped[CaptionStyle] = mapped_column(EnumString(CaptionStyle, 32), nullable=False)
    caption_position: Mapped[CaptionPosition] = mapped_column(EnumString(CaptionPosition, 32), nullable=False)
    render_config: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    crop_keyframes: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB, default=list, nullable=False
    )

    # -- generated copy
    title: Mapped[str | None] = mapped_column(String(300))
    hook_text: Mapped[str | None] = mapped_column(Text)
    hook_alternatives: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB, default=list, nullable=False
    )
    description: Mapped[str | None] = mapped_column(Text)
    instagram_caption: Mapped[str | None] = mapped_column(Text)
    hashtags: Mapped[list[str]] = mapped_column(JSONB, default=list, nullable=False)
    keywords: Mapped[list[str]] = mapped_column(JSONB, default=list, nullable=False)
    transcript_text: Mapped[str | None] = mapped_column(Text)
    captions: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, default=list, nullable=False)

    # -- output
    status: Mapped[ClipStatus] = mapped_column(
        EnumString(ClipStatus, 32), default=ClipStatus.DRAFT, nullable=False, index=True
    )
    status_detail: Mapped[str | None] = mapped_column(Text)
    width: Mapped[int | None] = mapped_column(Integer)
    height: Mapped[int | None] = mapped_column(Integer)
    fps: Mapped[float | None] = mapped_column(Float)
    file_size_bytes: Mapped[int | None] = mapped_column(Integer)
    storage_key: Mapped[str | None] = mapped_column(String(1024))
    thumbnail_storage_key: Mapped[str | None] = mapped_column(String(1024))
    subtitle_storage_key: Mapped[str | None] = mapped_column(String(1024))
    perceptual_hash: Mapped[str | None] = mapped_column(String(64), index=True)

    # -- scores
    viral_score: Mapped[float | None] = mapped_column(Float, index=True)
    quality_score: Mapped[float | None] = mapped_column(Float)
    safety_verdict: Mapped[str | None] = mapped_column(String(16))
    regeneration_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    # -- review
    approved_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
    rejected_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
    review_note: Mapped[str | None] = mapped_column(Text)
    auto_approved: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    publish_status: Mapped[PublishStatus] = mapped_column(
        EnumString(PublishStatus, 32), default=PublishStatus.NOT_SCHEDULED, nullable=False
    )
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIM))

    video: Mapped["Video"] = relationship(back_populates="clips")
    candidate: Mapped["CandidateClip | None"] = relationship(back_populates="clips")
    variants: Mapped[list["ClipVariant"]] = relationship(
        back_populates="clip", cascade="all, delete-orphan"
    )
    assets: Mapped[list["MediaAsset"]] = relationship(back_populates="clip")
    quality_checks: Mapped[list["QualityCheck"]] = relationship(
        back_populates="clip", cascade="all, delete-orphan"
    )
    safety_checks: Mapped[list["ContentSafetyCheck"]] = relationship(
        back_populates="clip", cascade="all, delete-orphan"
    )
    publishing_jobs: Mapped[list["PublishingJob"]] = relationship(
        back_populates="clip", cascade="all, delete-orphan"
    )
    analytics: Mapped[list["ClipAnalytics"]] = relationship(
        back_populates="clip", cascade="all, delete-orphan"
    )


class ClipVariant(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """An alternative render of the same moment (different opening/crop/style).

    Variants never alter what the speaker said; they change framing, captions
    and the optional overlay only.
    """

    __tablename__ = "clip_variants"
    __table_args__ = (Index("ix_variants_clip_label", "clip_id", "label"),)

    clip_id: Mapped[uuid.UUID] = mapped_column(
        GUID(), ForeignKey("generated_clips.id", ondelete="CASCADE"), nullable=False
    )
    label: Mapped[str] = mapped_column(String(32), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    start_time: Mapped[float] = mapped_column(Float, nullable=False)
    end_time: Mapped[float] = mapped_column(Float, nullable=False)
    crop_mode: Mapped[CropMode] = mapped_column(EnumString(CropMode, 32), nullable=False)
    caption_style: Mapped[CaptionStyle] = mapped_column(EnumString(CaptionStyle, 32), nullable=False)
    hook_text: Mapped[str | None] = mapped_column(Text)
    render_config: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    storage_key: Mapped[str | None] = mapped_column(String(1024))
    thumbnail_storage_key: Mapped[str | None] = mapped_column(String(1024))
    status: Mapped[ClipStatus] = mapped_column(
        EnumString(ClipStatus, 32), default=ClipStatus.DRAFT, nullable=False
    )
    quality_score: Mapped[float | None] = mapped_column(Float)
    is_primary: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    clip: Mapped["GeneratedClip"] = relationship(back_populates="variants")
