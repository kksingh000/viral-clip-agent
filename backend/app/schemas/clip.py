"""Clip, variant, quality and safety schemas."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import Field, model_validator

from app.models.enums import (
    CaptionPosition,
    CaptionStyle,
    ClipStatus,
    CropMode,
    PublishStatus,
    QualityStatus,
    SafetyVerdict,
)
from app.schemas.common import APIModel


class ClipGenerateRequest(APIModel):
    """Generate one or more clips.

    Supply ``candidate_ids`` to render specific detected moments, or
    ``video_id`` alone to render the top ``count`` candidates.
    """

    video_id: uuid.UUID | None = None
    candidate_ids: list[uuid.UUID] = Field(default_factory=list, max_length=20)
    count: int = Field(default=3, ge=1, le=20)
    crop_mode: CropMode | None = None
    caption_style: CaptionStyle | None = None
    caption_position: CaptionPosition | None = None
    caption_overrides: dict[str, Any] = Field(default_factory=dict)
    include_hook_overlay: bool = True
    generate_variants: bool = False

    @model_validator(mode="after")
    def _need_a_target(self) -> "ClipGenerateRequest":
        if not self.video_id and not self.candidate_ids:
            raise ValueError("Provide either video_id or candidate_ids.")
        return self


class ManualClipRequest(APIModel):
    """Cut an arbitrary span, bypassing candidate detection."""

    video_id: uuid.UUID
    start_time: float = Field(ge=0)
    end_time: float = Field(gt=0)
    crop_mode: CropMode | None = None
    caption_style: CaptionStyle | None = None
    caption_position: CaptionPosition | None = None
    hook_text: str | None = Field(default=None, max_length=280)
    include_hook_overlay: bool = True

    @model_validator(mode="after")
    def _ordered(self) -> "ManualClipRequest":
        if self.end_time <= self.start_time:
            raise ValueError("end_time must be greater than start_time.")
        if self.end_time - self.start_time > 180:
            raise ValueError("Manual clips are limited to 180 seconds.")
        return self


class ClipUpdateRequest(APIModel):
    """Edits applied from the browser editor.

    Changing timing, crop or captions requires a re-render; the API returns a
    job when that is the case.
    """

    start_time: float | None = Field(default=None, ge=0)
    end_time: float | None = Field(default=None, gt=0)
    title: str | None = Field(default=None, max_length=300)
    hook_text: str | None = Field(default=None, max_length=280)
    description: str | None = None
    instagram_caption: str | None = None
    hashtags: list[str] | None = Field(default=None, max_length=15)
    caption_style: CaptionStyle | None = None
    caption_position: CaptionPosition | None = None
    crop_mode: CropMode | None = None
    captions: list[dict[str, Any]] | None = None

    @property
    def requires_rerender(self) -> bool:
        return any(
            value is not None
            for value in (
                self.start_time,
                self.end_time,
                self.caption_style,
                self.caption_position,
                self.crop_mode,
                self.captions,
            )
        )


class ClipRegenerateRequest(APIModel):
    crop_mode: CropMode | None = None
    caption_style: CaptionStyle | None = None
    caption_position: CaptionPosition | None = None
    hook_text: str | None = Field(default=None, max_length=280)
    regenerate_hook: bool = False
    include_hook_overlay: bool | None = None


class ApprovalRequest(APIModel):
    note: str | None = Field(default=None, max_length=2000)


class QualityCheckResponse(APIModel):
    id: uuid.UUID
    attempt: int
    status: QualityStatus
    score: float
    threshold: float
    issues: list[dict[str, Any]]
    recommendations: list[str]
    technical: dict[str, Any]
    created_at: datetime


class SafetyCheckResponse(APIModel):
    id: uuid.UUID
    verdict: SafetyVerdict
    category_scores: dict[str, Any]
    flagged_categories: list[str]
    rationale: str | None
    created_at: datetime


class ClipVariantResponse(APIModel):
    id: uuid.UUID
    label: str
    description: str | None
    start_time: float
    end_time: float
    crop_mode: CropMode
    caption_style: CaptionStyle
    status: ClipStatus
    is_primary: bool
    media_url: str | None = None


class ClipResponse(APIModel):
    id: uuid.UUID
    video_id: uuid.UUID
    candidate_id: uuid.UUID | None

    start_time: float
    end_time: float
    duration_seconds: float

    crop_mode: CropMode
    caption_style: CaptionStyle
    caption_position: CaptionPosition

    title: str | None
    hook_text: str | None
    hook_alternatives: list[dict[str, Any]]
    description: str | None
    instagram_caption: str | None
    hashtags: list[str]
    keywords: list[str]
    transcript_text: str | None

    status: ClipStatus
    status_detail: str | None
    width: int | None
    height: int | None
    fps: float | None
    file_size_bytes: int | None

    viral_score: float | None
    quality_score: float | None
    safety_verdict: str | None
    regeneration_count: int
    auto_approved: bool
    approved_at: datetime | None
    rejected_at: datetime | None
    review_note: str | None
    publish_status: PublishStatus

    created_at: datetime
    updated_at: datetime

    #: Populated by the API layer.
    media_url: str | None = None
    thumbnail_url: str | None = None
    subtitle_url: str | None = None
    video_title: str | None = None


class ClipDetailResponse(ClipResponse):
    captions: list[dict[str, Any]] = Field(default_factory=list)
    crop_keyframes: list[dict[str, Any]] = Field(default_factory=list)
    render_config: dict[str, Any] = Field(default_factory=dict)
    variants: list[ClipVariantResponse] = Field(default_factory=list)
    quality_checks: list[QualityCheckResponse] = Field(default_factory=list)
    safety_checks: list[SafetyCheckResponse] = Field(default_factory=list)
