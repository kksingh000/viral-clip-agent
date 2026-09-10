"""Video, transcript, scene and candidate schemas."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import Field, HttpUrl, field_validator, model_validator

from app.models.enums import (
    AuthorizationStatus,
    CandidateStatus,
    VideoSource,
    VideoStatus,
)
from app.schemas.common import APIModel

#: Statuses a client may assert directly. USER_UPLOADED is set by the upload
#: endpoint itself and is not a client-assertable value.
CLIENT_SETTABLE_AUTHORIZATION = {
    AuthorizationStatus.UNKNOWN,
    AuthorizationStatus.USER_OWNED,
    AuthorizationStatus.LICENSED,
    AuthorizationStatus.AUTHORIZED,
    AuthorizationStatus.NOT_AUTHORIZED,
}

#: Rights that rest on a third party's permission rather than the account
#: holder's own authorship. Asserting one requires recording its basis, so the
#: claim is auditable later.
REQUIRES_EVIDENCE = {
    AuthorizationStatus.LICENSED,
    AuthorizationStatus.AUTHORIZED,
}


def _check_evidence(
    status: AuthorizationStatus, note: str | None, evidence_url: object | None
) -> None:
    """Shared rule: the same assertion carries the same burden of proof on
    whichever endpoint makes it."""
    if status in REQUIRES_EVIDENCE and not (note or evidence_url):
        raise ValueError(
            f"{status.value} requires a note or an evidence URL recording the "
            "basis of the rights."
        )


class VideoCreateRequest(APIModel):
    """Register a video by reference.

    This does not fetch media. It records metadata and a rights position; the
    media arrives via the upload endpoint or an authorized integration.
    """

    title: str = Field(min_length=1, max_length=1024)
    source: VideoSource = VideoSource.UPLOAD
    source_url: HttpUrl | None = None
    external_id: str | None = Field(default=None, max_length=255)
    description: str | None = None
    creator_name: str | None = Field(default=None, max_length=512)
    category: str | None = Field(default=None, max_length=128)
    language: str | None = Field(default=None, max_length=16)
    tags: list[str] = Field(default_factory=list, max_length=50)
    published_at: datetime | None = None
    authorization_status: AuthorizationStatus = AuthorizationStatus.UNKNOWN
    authorization_note: str | None = None
    authorization_evidence_url: HttpUrl | None = None

    @field_validator("authorization_status")
    @classmethod
    def _client_settable(cls, value: AuthorizationStatus) -> AuthorizationStatus:
        if value not in CLIENT_SETTABLE_AUTHORIZATION:
            raise ValueError(
                f"{value} cannot be asserted directly; it is set by the platform."
            )
        return value

    @model_validator(mode="after")
    def _require_evidence(self) -> "VideoCreateRequest":
        _check_evidence(
            self.authorization_status,
            self.authorization_note,
            self.authorization_evidence_url,
        )
        return self


class VideoAuthorizationUpdate(APIModel):
    """Change the rights position of a video.

    Asserting anything other than UNKNOWN or NOT_AUTHORIZED is a statement that
    the account holder owns the content or holds a licence for it; the note is
    stored as the record of that assertion.
    """

    authorization_status: AuthorizationStatus
    authorization_note: str | None = Field(default=None, max_length=4000)
    authorization_evidence_url: HttpUrl | None = None

    @model_validator(mode="after")
    def _require_evidence(self) -> "VideoAuthorizationUpdate":
        _check_evidence(
            self.authorization_status,
            self.authorization_note,
            self.authorization_evidence_url,
        )
        return self


class VideoUpdateRequest(APIModel):
    title: str | None = Field(default=None, min_length=1, max_length=1024)
    description: str | None = None
    category: str | None = Field(default=None, max_length=128)
    language: str | None = Field(default=None, max_length=16)
    tags: list[str] | None = Field(default=None, max_length=50)


class ChannelResponse(APIModel):
    id: uuid.UUID
    platform: str
    external_id: str
    title: str
    handle: str | None
    thumbnail_url: str | None
    subscriber_count: int | None


class VideoResponse(APIModel):
    id: uuid.UUID
    title: str
    description: str | None
    source: VideoSource
    source_url: str | None
    external_id: str | None
    creator_name: str | None
    category: str | None
    language: str | None
    tags: list[str]
    thumbnail_url: str | None
    published_at: datetime | None

    authorization_status: AuthorizationStatus
    authorization_note: str | None
    authorized_at: datetime | None

    status: VideoStatus
    status_detail: str | None
    analyzed_at: datetime | None

    duration_seconds: float | None
    width: int | None
    height: int | None
    fps: float | None
    audio_available: bool
    file_size_bytes: int | None

    view_count: int | None
    like_count: int | None
    comment_count: int | None
    trend_score: float | None
    trend_breakdown: dict[str, Any]

    created_at: datetime
    updated_at: datetime

    #: Populated by the API layer, not stored.
    has_media: bool = False
    candidate_count: int = 0
    clip_count: int = 0
    media_url: str | None = None


class TranscriptWordSchema(APIModel):
    word: str
    start: float
    end: float
    probability: float | None = None


class TranscriptSegmentResponse(APIModel):
    id: uuid.UUID
    index: int
    start_time: float
    end_time: float
    text: str
    speaker: str | None
    prefilter_score: float | None
    words: list[dict[str, Any]] = Field(default_factory=list)
    signals: dict[str, Any] = Field(default_factory=dict)


class TranscriptResponse(APIModel):
    id: uuid.UUID
    video_id: uuid.UUID
    provider: str
    model: str | None
    language: str | None
    language_confidence: float | None
    duration_seconds: float | None
    word_count: int
    text: str
    #: True when no transcription provider was configured: the timings are
    #: real but the words are placeholders.
    is_synthetic: bool = False
    segments: list[TranscriptSegmentResponse] = Field(default_factory=list)


class SceneResponse(APIModel):
    id: uuid.UUID
    index: int
    start_time: float
    end_time: float
    change_score: float | None


class CandidateResponse(APIModel):
    id: uuid.UUID
    video_id: uuid.UUID
    start_time: float
    end_time: float
    duration_seconds: float
    context_lead_in: float
    hook: str | None
    summary: str | None
    reason: str | None
    transcript_excerpt: str | None
    viral_score: float
    score_breakdown: dict[str, Any]
    status: CandidateStatus
    rank: int | None
    model_name: str | None
    created_at: datetime


class AnalyzeRequest(APIModel):
    force: bool = Field(
        default=False,
        description="Re-transcribe and re-detect scenes instead of reusing cached results.",
    )


class TrendingVideoResponse(VideoResponse):
    view_velocity: float | None = None
    engagement_rate: float | None = None
    growth_acceleration: float | None = None
