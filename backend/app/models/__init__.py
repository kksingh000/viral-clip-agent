"""ORM models.

Importing this package registers every mapper on :class:`app.database.base.Base`,
which Alembic's autogenerate and ``Base.metadata.create_all`` both rely on.
"""

from app.database.base import Base
from app.models.analytics import ClipAnalytics, ScoreCalibration
from app.models.check import ContentSafetyCheck, QualityCheck
from app.models.clip import CandidateClip, ClipVariant, GeneratedClip
from app.models.enums import (
    PROCESSABLE_AUTHORIZATION,
    PUBLISHABLE_AUTHORIZATION,
    TERMINAL_JOB_STATUSES,
    AuthorizationStatus,
    CandidateStatus,
    CaptionPosition,
    CaptionStyle,
    ClipStatus,
    CropMode,
    JobStatus,
    JobType,
    MediaAssetKind,
    Platform,
    PublishStatus,
    QualitySeverity,
    QualityStatus,
    SafetyVerdict,
    VideoSource,
    VideoStatus,
)
from app.models.job import ProcessingJob
from app.models.publishing import PlatformAccount, PublishingJob
from app.models.transcript import EMBEDDING_DIM, Scene, Transcript, TranscriptSegment
from app.models.trend import Topic, TopicMention
from app.models.user import ApiKey, User, UserSettings
from app.models.video import Channel, MediaAsset, Video, VideoMetric

__all__ = [
    "Base",
    "EMBEDDING_DIM",
    "PROCESSABLE_AUTHORIZATION",
    "PUBLISHABLE_AUTHORIZATION",
    "TERMINAL_JOB_STATUSES",
    "ApiKey",
    "AuthorizationStatus",
    "CandidateClip",
    "CandidateStatus",
    "CaptionPosition",
    "CaptionStyle",
    "Channel",
    "ClipAnalytics",
    "ClipStatus",
    "ClipVariant",
    "ContentSafetyCheck",
    "CropMode",
    "GeneratedClip",
    "JobStatus",
    "JobType",
    "MediaAsset",
    "MediaAssetKind",
    "Platform",
    "PlatformAccount",
    "ProcessingJob",
    "PublishStatus",
    "PublishingJob",
    "QualityCheck",
    "QualitySeverity",
    "QualityStatus",
    "SafetyVerdict",
    "Scene",
    "ScoreCalibration",
    "Topic",
    "TopicMention",
    "Transcript",
    "TranscriptSegment",
    "User",
    "UserSettings",
    "Video",
    "VideoMetric",
    "VideoSource",
    "VideoStatus",
]
