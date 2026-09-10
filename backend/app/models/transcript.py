"""Transcripts, semantic segments and detected scenes."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

from sqlalchemy import Float, ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import Base, TimestampMixin, UUIDPrimaryKeyMixin
from app.database.types import GUID, JSONB, Vector

if TYPE_CHECKING:
    from app.models.video import Video

EMBEDDING_DIM = 384


class Transcript(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Full transcript for one video.

    Word-level timings live on :class:`TranscriptSegment.words` as a JSON array
    of ``{"word", "start", "end", "probability"}`` objects. Word rows in their
    own table would multiply row counts by ~150x per hour of audio for no
    query benefit -- every consumer reads words per segment.
    """

    __tablename__ = "transcripts"
    __table_args__ = (UniqueConstraint("video_id", name="uq_transcript_video"),)

    video_id: Mapped[uuid.UUID] = mapped_column(
        GUID(), ForeignKey("videos.id", ondelete="CASCADE"), nullable=False, index=True
    )
    provider: Mapped[str] = mapped_column(String(64), nullable=False)
    model: Mapped[str | None] = mapped_column(String(128))
    language: Mapped[str | None] = mapped_column(String(16), index=True)
    language_confidence: Mapped[float | None] = mapped_column(Float)
    duration_seconds: Mapped[float | None] = mapped_column(Float)
    word_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    text: Mapped[str] = mapped_column(Text, default="", nullable=False)
    provider_metadata: Mapped[dict[str, Any]] = mapped_column(
        JSONB, default=dict, nullable=False
    )

    video: Mapped["Video"] = relationship(back_populates="transcript")
    segments: Mapped[list["TranscriptSegment"]] = relationship(
        back_populates="transcript",
        cascade="all, delete-orphan",
        order_by="TranscriptSegment.index",
    )


class TranscriptSegment(UUIDPrimaryKeyMixin, Base):
    """A semantic segment: a coherent unit bounded by topic/pause/speaker."""

    __tablename__ = "transcript_segments"
    __table_args__ = (
        Index("ix_segments_transcript_index", "transcript_id", "index"),
        Index("ix_segments_time", "transcript_id", "start_time"),
    )

    transcript_id: Mapped[uuid.UUID] = mapped_column(
        GUID(), ForeignKey("transcripts.id", ondelete="CASCADE"), nullable=False
    )
    index: Mapped[int] = mapped_column(Integer, nullable=False)
    start_time: Mapped[float] = mapped_column(Float, nullable=False)
    end_time: Mapped[float] = mapped_column(Float, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    speaker: Mapped[str | None] = mapped_column(String(64))
    words: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, default=list, nullable=False)

    # Cheap deterministic signals computed without an LLM; used to pre-filter
    # which parts of a long transcript are worth spending model tokens on.
    words_per_second: Mapped[float | None] = mapped_column(Float)
    leading_pause: Mapped[float | None] = mapped_column(Float)
    trailing_pause: Mapped[float | None] = mapped_column(Float)
    prefilter_score: Mapped[float | None] = mapped_column(Float, index=True)
    signals: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIM))

    transcript: Mapped["Transcript"] = relationship(back_populates="segments")

    @property
    def duration(self) -> float:
        return max(0.0, self.end_time - self.start_time)


class Scene(UUIDPrimaryKeyMixin, Base):
    """A shot/scene boundary detected from the video track."""

    __tablename__ = "scenes"
    __table_args__ = (Index("ix_scenes_video_time", "video_id", "start_time"),)

    video_id: Mapped[uuid.UUID] = mapped_column(
        GUID(), ForeignKey("videos.id", ondelete="CASCADE"), nullable=False
    )
    index: Mapped[int] = mapped_column(Integer, nullable=False)
    start_time: Mapped[float] = mapped_column(Float, nullable=False)
    end_time: Mapped[float] = mapped_column(Float, nullable=False)
    change_score: Mapped[float | None] = mapped_column(Float)
    # Aggregated face-tracking output for the scene, used by the smart crop.
    face_summary: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    visual_metrics: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)

    video: Mapped["Video"] = relationship(back_populates="scenes")

    @property
    def duration(self) -> float:
        return max(0.0, self.end_time - self.start_time)
