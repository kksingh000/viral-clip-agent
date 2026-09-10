"""Topic intelligence: the topic database and video-to-topic mentions."""

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
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import Base, TimestampMixin, UUIDPrimaryKeyMixin
from app.database.types import GUID, JSONB, UTCDateTime, Vector
from app.models.transcript import EMBEDDING_DIM

if TYPE_CHECKING:
    from app.models.video import Video


class Topic(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """An emerging subject tracked across discovered videos."""

    __tablename__ = "topics"
    __table_args__ = (
        UniqueConstraint("slug", name="uq_topic_slug"),
        Index("ix_topics_trend_score", "trend_score"),
    )

    slug: Mapped[str] = mapped_column(String(160), nullable=False, index=True)
    label: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    kind: Mapped[str] = mapped_column(String(32), default="topic", nullable=False)
    keywords: Mapped[list[str]] = mapped_column(JSONB, default=list, nullable=False)

    trend_score: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    growth_rate: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    video_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    total_views: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    first_seen_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
    last_seen_at: Mapped[datetime | None] = mapped_column(
        UTCDateTime(), index=True
    )
    region: Mapped[str | None] = mapped_column(String(8), index=True)
    history: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB, default=list, nullable=False
    )
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIM))

    mentions: Mapped[list["TopicMention"]] = relationship(
        back_populates="topic", cascade="all, delete-orphan"
    )


class TopicMention(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "topic_mentions"
    __table_args__ = (
        UniqueConstraint("topic_id", "video_id", name="uq_mention_topic_video"),
    )

    topic_id: Mapped[uuid.UUID] = mapped_column(
        GUID(), ForeignKey("topics.id", ondelete="CASCADE"), nullable=False, index=True
    )
    video_id: Mapped[uuid.UUID] = mapped_column(
        GUID(), ForeignKey("videos.id", ondelete="CASCADE"), nullable=False, index=True
    )
    relevance: Mapped[float] = mapped_column(Float, default=1.0, nullable=False)
    matched_terms: Mapped[list[str]] = mapped_column(JSONB, default=list, nullable=False)

    topic: Mapped["Topic"] = relationship(back_populates="mentions")
    video: Mapped["Video"] = relationship(back_populates="topic_mentions")
