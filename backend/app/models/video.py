"""Source content: channels, videos, metric snapshots and stored media."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import Base, TimestampMixin, UUIDPrimaryKeyMixin
from app.database.types import EnumString, GUID, JSONB, UTCDateTime
from app.models.enums import (
    AuthorizationStatus,
    MediaAssetKind,
    VideoSource,
    VideoStatus,
)

if TYPE_CHECKING:
    from app.models.clip import CandidateClip, GeneratedClip
    from app.models.job import ProcessingJob
    from app.models.transcript import Scene, Transcript
    from app.models.trend import TopicMention
    from app.models.user import User


class Channel(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """A creator/channel observed by trend discovery."""

    __tablename__ = "channels"
    __table_args__ = (
        UniqueConstraint("platform", "external_id", name="uq_channel_platform_external"),
    )

    platform: Mapped[str] = mapped_column(String(32), nullable=False, default="youtube")
    external_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    handle: Mapped[str | None] = mapped_column(String(128))
    description: Mapped[str | None] = mapped_column(Text)
    thumbnail_url: Mapped[str | None] = mapped_column(String(1024))
    country: Mapped[str | None] = mapped_column(String(8))
    subscriber_count: Mapped[int | None] = mapped_column(BigInteger)
    video_count: Mapped[int | None] = mapped_column(BigInteger)
    view_count: Mapped[int | None] = mapped_column(BigInteger)
    metrics_fetched_at: Mapped[datetime | None] = mapped_column(UTCDateTime())

    videos: Mapped[list["Video"]] = relationship(back_populates="channel")


class Video(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """A source video.

    A row may exist purely as *discovery metadata* (``status=DISCOVERED``,
    ``authorization_status=UNKNOWN``, no media asset). Media is only ever
    attached when :attr:`authorization_status` permits processing.
    """

    __tablename__ = "videos"
    __table_args__ = (
        UniqueConstraint(
            "user_id", "source", "external_id", name="uq_video_user_source_external"
        ),
        Index("ix_videos_user_status", "user_id", "status"),
        Index("ix_videos_trend_score", "trend_score"),
    )

    user_id: Mapped[uuid.UUID] = mapped_column(
        GUID(), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    channel_id: Mapped[uuid.UUID | None] = mapped_column(
        GUID(), ForeignKey("channels.id", ondelete="SET NULL"), index=True
    )

    # -- identity
    source: Mapped[VideoSource] = mapped_column(EnumString(VideoSource, 32), nullable=False)
    external_id: Mapped[str | None] = mapped_column(String(255), index=True)
    source_url: Mapped[str | None] = mapped_column(String(2048))
    title: Mapped[str] = mapped_column(String(1024), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    creator_name: Mapped[str | None] = mapped_column(String(512))
    category: Mapped[str | None] = mapped_column(String(128), index=True)
    tags: Mapped[list[str]] = mapped_column(JSONB, default=list, nullable=False)
    language: Mapped[str | None] = mapped_column(String(16))
    thumbnail_url: Mapped[str | None] = mapped_column(String(2048))
    published_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), index=True)

    # -- rights (see AuthorizationStatus docstring)
    authorization_status: Mapped[AuthorizationStatus] = mapped_column(
        EnumString(AuthorizationStatus, 32), nullable=False, default=AuthorizationStatus.UNKNOWN, index=True
    )
    authorization_note: Mapped[str | None] = mapped_column(Text)
    authorization_evidence_url: Mapped[str | None] = mapped_column(String(2048))
    authorized_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
    authorized_by_user_id: Mapped[uuid.UUID | None] = mapped_column(
        GUID(), ForeignKey("users.id", ondelete="SET NULL")
    )

    # -- pipeline state
    status: Mapped[VideoStatus] = mapped_column(
        EnumString(VideoStatus, 32), nullable=False, default=VideoStatus.DISCOVERED, index=True
    )
    status_detail: Mapped[str | None] = mapped_column(Text)
    analyzed_at: Mapped[datetime | None] = mapped_column(UTCDateTime())

    # -- technical metadata (populated by ffprobe once media exists)
    duration_seconds: Mapped[float | None] = mapped_column(Float)
    width: Mapped[int | None] = mapped_column(Integer)
    height: Mapped[int | None] = mapped_column(Integer)
    fps: Mapped[float | None] = mapped_column(Float)
    video_codec: Mapped[str | None] = mapped_column(String(64))
    audio_codec: Mapped[str | None] = mapped_column(String(64))
    audio_available: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    file_size_bytes: Mapped[int | None] = mapped_column(BigInteger)
    content_hash: Mapped[str | None] = mapped_column(String(64), index=True)

    # -- discovery metrics snapshot (latest)
    view_count: Mapped[int | None] = mapped_column(BigInteger)
    like_count: Mapped[int | None] = mapped_column(BigInteger)
    comment_count: Mapped[int | None] = mapped_column(BigInteger)
    trend_score: Mapped[float | None] = mapped_column(Float)
    trend_breakdown: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    extra_metadata: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)

    user: Mapped["User"] = relationship(back_populates="videos", foreign_keys=[user_id])
    channel: Mapped["Channel | None"] = relationship(back_populates="videos")
    assets: Mapped[list["MediaAsset"]] = relationship(
        back_populates="video", cascade="all, delete-orphan"
    )
    metrics: Mapped[list["VideoMetric"]] = relationship(
        back_populates="video", cascade="all, delete-orphan", order_by="VideoMetric.captured_at"
    )
    transcript: Mapped["Transcript | None"] = relationship(
        back_populates="video", uselist=False, cascade="all, delete-orphan"
    )
    scenes: Mapped[list["Scene"]] = relationship(
        back_populates="video", cascade="all, delete-orphan", order_by="Scene.start_time"
    )
    candidates: Mapped[list["CandidateClip"]] = relationship(
        back_populates="video", cascade="all, delete-orphan"
    )
    clips: Mapped[list["GeneratedClip"]] = relationship(
        back_populates="video", cascade="all, delete-orphan"
    )
    jobs: Mapped[list["ProcessingJob"]] = relationship(
        back_populates="video", cascade="all, delete-orphan"
    )
    topic_mentions: Mapped[list["TopicMention"]] = relationship(
        back_populates="video", cascade="all, delete-orphan"
    )

    @property
    def is_processable(self) -> bool:
        from app.models.enums import PROCESSABLE_AUTHORIZATION

        return self.authorization_status in PROCESSABLE_AUTHORIZATION

    @property
    def is_publishable(self) -> bool:
        from app.models.enums import PUBLISHABLE_AUTHORIZATION

        return self.authorization_status in PUBLISHABLE_AUTHORIZATION

    def primary_asset(self) -> "MediaAsset | None":
        from app.models.enums import MediaAssetKind as _Kind

        for asset in self.assets:
            if asset.kind == _Kind.SOURCE_VIDEO:
                return asset
        return None


class VideoMetric(UUIDPrimaryKeyMixin, Base):
    """A time-series snapshot used to compute velocity and acceleration."""

    __tablename__ = "video_metrics"
    __table_args__ = (
        Index("ix_video_metrics_video_time", "video_id", "captured_at"),
    )

    video_id: Mapped[uuid.UUID] = mapped_column(
        GUID(), ForeignKey("videos.id", ondelete="CASCADE"), nullable=False
    )
    captured_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    view_count: Mapped[int | None] = mapped_column(BigInteger)
    like_count: Mapped[int | None] = mapped_column(BigInteger)
    comment_count: Mapped[int | None] = mapped_column(BigInteger)

    # derived, stored so the dashboard does not recompute on every read
    view_velocity: Mapped[float | None] = mapped_column(Float)
    engagement_rate: Mapped[float | None] = mapped_column(Float)
    comment_rate: Mapped[float | None] = mapped_column(Float)
    growth_acceleration: Mapped[float | None] = mapped_column(Float)
    trend_score: Mapped[float | None] = mapped_column(Float)

    video: Mapped["Video"] = relationship(back_populates="metrics")


class MediaAsset(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """A file held in object storage."""

    __tablename__ = "media_assets"
    __table_args__ = (Index("ix_media_assets_video_kind", "video_id", "kind"),)

    video_id: Mapped[uuid.UUID | None] = mapped_column(
        GUID(), ForeignKey("videos.id", ondelete="CASCADE"), index=True
    )
    clip_id: Mapped[uuid.UUID | None] = mapped_column(
        GUID(), ForeignKey("generated_clips.id", ondelete="CASCADE"), index=True
    )
    kind: Mapped[MediaAssetKind] = mapped_column(EnumString(MediaAssetKind, 32), nullable=False)
    storage_key: Mapped[str] = mapped_column(String(1024), nullable=False)
    content_type: Mapped[str | None] = mapped_column(String(128))
    size_bytes: Mapped[int | None] = mapped_column(BigInteger)
    checksum_sha256: Mapped[str | None] = mapped_column(String(64))
    extra: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)

    video: Mapped["Video | None"] = relationship(back_populates="assets")
    clip: Mapped["GeneratedClip | None"] = relationship(back_populates="assets")
