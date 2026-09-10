"""Connected platform accounts and publishing jobs."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import (
    Boolean,
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
from app.models.enums import Platform, PublishStatus

if TYPE_CHECKING:
    from app.models.clip import GeneratedClip
    from app.models.user import User


class PlatformAccount(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """An OAuth connection to an official platform API.

    Tokens are stored encrypted-at-rest by the deployment (column-level
    encryption or a KMS-backed secret store); this table holds the ciphertext
    reference, never a plaintext long-lived secret in application logs.
    """

    __tablename__ = "platform_accounts"
    __table_args__ = (
        UniqueConstraint(
            "user_id", "platform", "external_account_id", name="uq_platform_account"
        ),
    )

    user_id: Mapped[uuid.UUID] = mapped_column(
        GUID(), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    platform: Mapped[Platform] = mapped_column(EnumString(Platform, 32), nullable=False)
    external_account_id: Mapped[str] = mapped_column(String(255), nullable=False)
    display_name: Mapped[str | None] = mapped_column(String(255))
    scopes: Mapped[list[str]] = mapped_column(JSONB, default=list, nullable=False)

    access_token_ref: Mapped[str | None] = mapped_column(Text)
    refresh_token_ref: Mapped[str | None] = mapped_column(Text)
    token_expires_at: Mapped[datetime | None] = mapped_column(UTCDateTime())

    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    connected_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
    last_error: Mapped[str | None] = mapped_column(Text)

    user: Mapped["User"] = relationship(back_populates="platform_accounts")
    publishing_jobs: Mapped[list["PublishingJob"]] = relationship(
        back_populates="account"
    )


class PublishingJob(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "publishing_jobs"
    __table_args__ = (
        Index("ix_publishing_status_scheduled", "status", "scheduled_for"),
    )

    clip_id: Mapped[uuid.UUID] = mapped_column(
        GUID(), ForeignKey("generated_clips.id", ondelete="CASCADE"), nullable=False, index=True
    )
    account_id: Mapped[uuid.UUID | None] = mapped_column(
        GUID(), ForeignKey("platform_accounts.id", ondelete="SET NULL"), index=True
    )
    platform: Mapped[Platform] = mapped_column(EnumString(Platform, 32), nullable=False)
    status: Mapped[PublishStatus] = mapped_column(
        EnumString(PublishStatus, 32), default=PublishStatus.SCHEDULED, nullable=False
    )
    scheduled_for: Mapped[datetime | None] = mapped_column(UTCDateTime())
    published_at: Mapped[datetime | None] = mapped_column(UTCDateTime())

    title: Mapped[str | None] = mapped_column(String(300))
    description: Mapped[str | None] = mapped_column(Text)
    hashtags: Mapped[list[str]] = mapped_column(JSONB, default=list, nullable=False)
    privacy: Mapped[str] = mapped_column(String(32), default="private", nullable=False)

    external_post_id: Mapped[str | None] = mapped_column(String(255), index=True)
    external_url: Mapped[str | None] = mapped_column(String(2048))

    error_code: Mapped[str | None] = mapped_column(String(64))
    error_message: Mapped[str | None] = mapped_column(Text)
    retry_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    response_payload: Mapped[dict[str, Any]] = mapped_column(
        JSONB, default=dict, nullable=False
    )

    clip: Mapped["GeneratedClip"] = relationship(back_populates="publishing_jobs")
    account: Mapped["PlatformAccount | None"] = relationship(
        back_populates="publishing_jobs"
    )
