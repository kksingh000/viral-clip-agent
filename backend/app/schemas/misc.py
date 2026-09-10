"""Job, settings, topic, analytics and publishing schemas."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import Field, model_validator

from app.models.enums import (
    CaptionPosition,
    CaptionStyle,
    CropMode,
    JobStatus,
    JobType,
    Platform,
    PublishStatus,
)
from app.schemas.common import APIModel


# ----------------------------------------------------------------------- jobs
class JobResponse(APIModel):
    id: uuid.UUID
    job_type: JobType
    status: JobStatus
    progress: float
    stage: str | None
    video_id: uuid.UUID | None
    clip_id: uuid.UUID | None
    payload: dict[str, Any]
    result: dict[str, Any]
    error_code: str | None
    error_message: str | None
    retry_count: int
    max_retries: int
    queued_at: datetime | None
    started_at: datetime | None
    finished_at: datetime | None
    duration_seconds: float | None
    llm_calls: int
    llm_input_tokens: int
    llm_output_tokens: int
    estimated_cost_usd: float
    created_at: datetime


class JobAcceptedResponse(APIModel):
    job_id: uuid.UUID
    status: JobStatus
    message: str


# ------------------------------------------------------------------- settings
class SettingsResponse(APIModel):
    min_viral_score: float
    auto_approve_score: float
    reject_below_score: float
    clips_per_video: int
    min_clip_seconds: float
    max_clip_seconds: float
    crop_mode: CropMode
    caption_style: CaptionStyle
    caption_position: CaptionPosition
    caption_overrides: dict[str, Any]
    output_width: int
    output_height: int
    output_fps: int | None
    scoring_weights: dict[str, Any]
    trend_weights: dict[str, Any]
    preferred_categories: list[str]
    preferred_languages: list[str]
    trend_regions: list[str]
    quality_threshold: float
    max_regeneration_attempts: int
    block_on_safety_review: bool


class SettingsUpdateRequest(APIModel):
    min_viral_score: float | None = Field(default=None, ge=0, le=100)
    auto_approve_score: float | None = Field(default=None, ge=0, le=100)
    reject_below_score: float | None = Field(default=None, ge=0, le=100)
    clips_per_video: int | None = Field(default=None, ge=1, le=20)
    min_clip_seconds: float | None = Field(default=None, ge=5, le=180)
    max_clip_seconds: float | None = Field(default=None, ge=5, le=180)
    crop_mode: CropMode | None = None
    caption_style: CaptionStyle | None = None
    caption_position: CaptionPosition | None = None
    caption_overrides: dict[str, Any] | None = None
    output_width: int | None = Field(default=None, ge=360, le=2160)
    output_height: int | None = Field(default=None, ge=640, le=3840)
    output_fps: int | None = Field(default=None, ge=24, le=60)
    scoring_weights: dict[str, float] | None = None
    trend_weights: dict[str, float] | None = None
    preferred_categories: list[str] | None = Field(default=None, max_length=40)
    preferred_languages: list[str] | None = Field(default=None, max_length=20)
    trend_regions: list[str] | None = Field(default=None, max_length=20)
    quality_threshold: float | None = Field(default=None, ge=0, le=100)
    max_regeneration_attempts: int | None = Field(default=None, ge=0, le=5)
    block_on_safety_review: bool | None = None

    @model_validator(mode="after")
    def _coherent(self) -> "SettingsUpdateRequest":
        if (
            self.min_clip_seconds is not None
            and self.max_clip_seconds is not None
            and self.min_clip_seconds >= self.max_clip_seconds
        ):
            raise ValueError("min_clip_seconds must be less than max_clip_seconds.")
        if (
            self.auto_approve_score is not None
            and self.reject_below_score is not None
            and self.auto_approve_score < self.reject_below_score
        ):
            raise ValueError(
                "auto_approve_score must be greater than or equal to reject_below_score."
            )
        return self


# --------------------------------------------------------------------- topics
class TopicResponse(APIModel):
    id: uuid.UUID
    slug: str
    label: str
    description: str | None
    kind: str
    keywords: list[str]
    trend_score: float
    growth_rate: float
    video_count: int
    total_views: int
    region: str | None
    first_seen_at: datetime | None
    last_seen_at: datetime | None


class TrendDiscoveryRequest(APIModel):
    """Kick off a metadata-only discovery run.

    Discovery never downloads media. Videos it finds are recorded with
    ``authorization_status = UNKNOWN`` and cannot be processed until rights are
    established.
    """

    region: str = Field(default="US", max_length=8)
    category_id: str | None = Field(default=None, max_length=16)
    max_results: int = Field(default=50, ge=1, le=200)
    query: str | None = Field(default=None, max_length=200)


# ------------------------------------------------------------------ analytics
class DashboardResponse(APIModel):
    videos_discovered_today: int
    videos_analyzed: int
    videos_pending: int
    clips_generated: int
    clips_approved: int
    clips_awaiting_review: int
    average_viral_score: float | None
    top_clip: dict[str, Any] | None
    active_jobs: int
    failed_jobs_24h: int
    llm_cost_24h_usd: float


class ScorePerformancePoint(APIModel):
    clip_id: uuid.UUID
    title: str | None
    predicted_score: float | None
    views: int | None
    performance_percentile: float | None
    prediction_error: float | None


class AnalyticsResponse(APIModel):
    clips_published: int
    total_views: int
    total_likes: int
    total_comments: int
    average_completion_rate: float | None
    score_correlation: float | None
    calibration_sample_size: int
    points: list[ScorePerformancePoint] = Field(default_factory=list)


# ----------------------------------------------------------------- publishing
class PlatformAccountResponse(APIModel):
    id: uuid.UUID
    platform: Platform
    external_account_id: str
    display_name: str | None
    scopes: list[str]
    is_active: bool
    connected_at: datetime | None
    token_expires_at: datetime | None
    last_error: str | None


class PublishRequest(APIModel):
    platform: Platform
    account_id: uuid.UUID | None = None
    scheduled_for: datetime | None = None
    privacy: str = Field(default="private", pattern="^(private|unlisted|public)$")
    title: str | None = Field(default=None, max_length=300)
    description: str | None = None


class PublishingJobResponse(APIModel):
    id: uuid.UUID
    clip_id: uuid.UUID
    platform: Platform
    status: PublishStatus
    scheduled_for: datetime | None
    published_at: datetime | None
    external_post_id: str | None
    external_url: str | None
    error_code: str | None
    error_message: str | None
    created_at: datetime
