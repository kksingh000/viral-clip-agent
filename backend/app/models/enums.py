"""Enumerations shared by the ORM models, schemas and agents.

Every enum is stored in the database by *value* (a lowercase or uppercase
string), never by ordinal, so that adding members is backwards compatible.
"""

from __future__ import annotations

from enum import Enum


class StrEnum(str, Enum):
    """String-valued enum that serialises to its value."""

    def __str__(self) -> str:  # pragma: no cover - trivial
        return str(self.value)


# --------------------------------------------------------------------- rights
class AuthorizationStatus(StrEnum):
    """Rights position for a piece of source content.

    Only :attr:`USER_OWNED`, :attr:`LICENSED`, :attr:`AUTHORIZED` and
    :attr:`USER_UPLOADED` may enter the processing pipeline; only those four
    may reach an automatic publishing workflow. Discovery-only records stay
    :attr:`UNKNOWN` and hold metadata alone -- never media.
    """

    UNKNOWN = "UNKNOWN"
    USER_OWNED = "USER_OWNED"
    LICENSED = "LICENSED"
    AUTHORIZED = "AUTHORIZED"
    USER_UPLOADED = "USER_UPLOADED"
    NOT_AUTHORIZED = "NOT_AUTHORIZED"


#: Statuses that permit downloading/processing the media itself.
PROCESSABLE_AUTHORIZATION = frozenset(
    {
        AuthorizationStatus.USER_OWNED,
        AuthorizationStatus.LICENSED,
        AuthorizationStatus.AUTHORIZED,
        AuthorizationStatus.USER_UPLOADED,
    }
)

#: Statuses that permit automated republishing to a connected platform.
PUBLISHABLE_AUTHORIZATION = frozenset(
    {
        AuthorizationStatus.USER_OWNED,
        AuthorizationStatus.LICENSED,
        AuthorizationStatus.AUTHORIZED,
        AuthorizationStatus.USER_UPLOADED,
    }
)


class VideoSource(StrEnum):
    UPLOAD = "UPLOAD"
    LOCAL_PATH = "LOCAL_PATH"
    CLOUD_STORAGE = "CLOUD_STORAGE"
    YOUTUBE_OWNED = "YOUTUBE_OWNED"
    DISCOVERY = "DISCOVERY"
    API_INTEGRATION = "API_INTEGRATION"


class VideoStatus(StrEnum):
    DISCOVERED = "DISCOVERED"        # metadata only, no media
    PENDING_MEDIA = "PENDING_MEDIA"  # authorized, awaiting the file
    UPLOADED = "UPLOADED"
    ANALYZING = "ANALYZING"
    ANALYZED = "ANALYZED"
    FAILED = "FAILED"
    ARCHIVED = "ARCHIVED"


# ------------------------------------------------------------------- pipeline
class JobType(StrEnum):
    INGEST = "INGEST"
    ANALYZE_VIDEO = "ANALYZE_VIDEO"
    TRANSCRIBE = "TRANSCRIBE"
    DETECT_SCENES = "DETECT_SCENES"
    FIND_MOMENTS = "FIND_MOMENTS"
    GENERATE_CLIP = "GENERATE_CLIP"
    QUALITY_CHECK = "QUALITY_CHECK"
    SAFETY_CHECK = "SAFETY_CHECK"
    PUBLISH = "PUBLISH"
    TREND_DISCOVERY = "TREND_DISCOVERY"
    ANALYTICS_SYNC = "ANALYTICS_SYNC"


class JobStatus(StrEnum):
    QUEUED = "QUEUED"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


TERMINAL_JOB_STATUSES = frozenset(
    {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED}
)


class ClipStatus(StrEnum):
    DRAFT = "DRAFT"
    QUEUED = "QUEUED"
    RENDERING = "RENDERING"
    RENDERED = "RENDERED"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    PUBLISHED = "PUBLISHED"
    FAILED = "FAILED"


class CandidateStatus(StrEnum):
    PROPOSED = "PROPOSED"
    SELECTED = "SELECTED"
    DISCARDED_DUPLICATE = "DISCARDED_DUPLICATE"
    DISCARDED_LOW_SCORE = "DISCARDED_LOW_SCORE"
    GENERATED = "GENERATED"


# --------------------------------------------------------------------- render
class CropMode(StrEnum):
    SMART = "SMART"                # face/speaker tracking, dynamic crop
    CENTER = "CENTER"              # static centre crop
    BLUR_PAD = "BLUR_PAD"          # full frame over a blurred fill
    SPLIT_SPEAKERS = "SPLIT_SPEAKERS"  # two stacked speaker crops


class CaptionStyle(StrEnum):
    NONE = "NONE"
    NORMAL = "NORMAL"
    BOLD = "BOLD"
    KARAOKE = "KARAOKE"
    WORD_HIGHLIGHT = "WORD_HIGHLIGHT"
    MINIMAL = "MINIMAL"
    PODCAST = "PODCAST"


class CaptionPosition(StrEnum):
    TOP = "TOP"
    UPPER_THIRD = "UPPER_THIRD"
    CENTER = "CENTER"
    LOWER_THIRD = "LOWER_THIRD"
    BOTTOM = "BOTTOM"


# --------------------------------------------------------------------- checks
class QualityStatus(StrEnum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


class QualitySeverity(StrEnum):
    INFO = "INFO"
    MINOR = "MINOR"
    MAJOR = "MAJOR"
    CRITICAL = "CRITICAL"


class SafetyVerdict(StrEnum):
    SAFE = "SAFE"
    REVIEW = "REVIEW"
    BLOCK = "BLOCK"


# ----------------------------------------------------------------- publishing
class Platform(StrEnum):
    YOUTUBE_SHORTS = "YOUTUBE_SHORTS"
    INSTAGRAM_REELS = "INSTAGRAM_REELS"
    TIKTOK = "TIKTOK"
    MANUAL_EXPORT = "MANUAL_EXPORT"


class PublishStatus(StrEnum):
    NOT_SCHEDULED = "NOT_SCHEDULED"
    SCHEDULED = "SCHEDULED"
    QUEUED = "QUEUED"
    UPLOADING = "UPLOADING"
    PUBLISHED = "PUBLISHED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class MediaAssetKind(StrEnum):
    SOURCE_VIDEO = "SOURCE_VIDEO"
    SOURCE_AUDIO = "SOURCE_AUDIO"
    CLIP_VIDEO = "CLIP_VIDEO"
    THUMBNAIL = "THUMBNAIL"
    SUBTITLE = "SUBTITLE"
    OVERLAY = "OVERLAY"
