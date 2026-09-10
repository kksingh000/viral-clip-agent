"""User, credential and per-user settings models."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import (
    Boolean,
    Float,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import Base, TimestampMixin, UUIDPrimaryKeyMixin
from app.database.types import EnumString, GUID, JSONB, UTCDateTime
from app.models.enums import CaptionPosition, CaptionStyle, CropMode

if TYPE_CHECKING:
    from app.models.publishing import PlatformAccount
    from app.models.video import Video


class User(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "users"

    email: Mapped[str] = mapped_column(String(320), unique=True, index=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    full_name: Mapped[str | None] = mapped_column(String(255))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    is_superuser: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    last_login_at: Mapped[datetime | None] = mapped_column(UTCDateTime())

    settings: Mapped["UserSettings"] = relationship(
        back_populates="user", uselist=False, cascade="all, delete-orphan"
    )
    # Video has two foreign keys to users (owner and the person who recorded
    # the authorization decision), so the join must be stated explicitly.
    videos: Mapped[list["Video"]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan",
        foreign_keys="Video.user_id",
    )
    platform_accounts: Mapped[list["PlatformAccount"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    api_keys: Mapped[list["ApiKey"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )


class ApiKey(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Machine credential. Only the SHA-256 digest of the key is stored."""

    __tablename__ = "api_keys"

    user_id: Mapped[uuid.UUID] = mapped_column(
        GUID(), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    key_prefix: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    key_hash: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    last_used_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
    revoked_at: Mapped[datetime | None] = mapped_column(UTCDateTime())

    user: Mapped["User"] = relationship(back_populates="api_keys")


class UserSettings(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Tunables that drive selection, rendering and approval for one account."""

    __tablename__ = "user_settings"
    __table_args__ = (UniqueConstraint("user_id", name="uq_user_settings_user"),)

    user_id: Mapped[uuid.UUID] = mapped_column(
        GUID(), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )

    # -- selection thresholds
    min_viral_score: Mapped[float] = mapped_column(Float, default=70.0, nullable=False)
    auto_approve_score: Mapped[float] = mapped_column(Float, default=90.0, nullable=False)
    reject_below_score: Mapped[float] = mapped_column(Float, default=70.0, nullable=False)
    clips_per_video: Mapped[int] = mapped_column(Integer, default=3, nullable=False)

    # -- clip shape
    min_clip_seconds: Mapped[float] = mapped_column(Float, default=15.0, nullable=False)
    max_clip_seconds: Mapped[float] = mapped_column(Float, default=60.0, nullable=False)

    # -- render defaults
    crop_mode: Mapped[CropMode] = mapped_column(
        EnumString(CropMode, 32), default=CropMode.SMART, nullable=False
    )
    caption_style: Mapped[CaptionStyle] = mapped_column(
        EnumString(CaptionStyle, 32), default=CaptionStyle.WORD_HIGHLIGHT, nullable=False
    )
    caption_position: Mapped[CaptionPosition] = mapped_column(
        EnumString(CaptionPosition, 32), default=CaptionPosition.LOWER_THIRD, nullable=False
    )
    caption_overrides: Mapped[dict[str, Any]] = mapped_column(
        JSONB, default=dict, nullable=False
    )
    output_width: Mapped[int] = mapped_column(Integer, default=1080, nullable=False)
    output_height: Mapped[int] = mapped_column(Integer, default=1920, nullable=False)
    output_fps: Mapped[int | None] = mapped_column(Integer)  # None == follow source

    # -- scoring / discovery
    scoring_weights: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    trend_weights: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    preferred_categories: Mapped[list[str]] = mapped_column(JSONB, default=list, nullable=False)
    preferred_languages: Mapped[list[str]] = mapped_column(
        JSONB, default=lambda: ["en"], nullable=False
    )
    trend_regions: Mapped[list[str]] = mapped_column(
        JSONB, default=lambda: ["US"], nullable=False
    )

    # -- quality / safety gates
    quality_threshold: Mapped[float] = mapped_column(Float, default=75.0, nullable=False)
    max_regeneration_attempts: Mapped[int] = mapped_column(Integer, default=2, nullable=False)
    block_on_safety_review: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    user: Mapped["User"] = relationship(back_populates="settings")
