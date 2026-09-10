"""Provider protocols.

Every external capability the application depends on sits behind one of these
interfaces so a deployment can swap implementations without touching agent or
pipeline code (see docs/architecture.md, "Model abstraction").
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Literal, Sequence

Role = Literal["user", "assistant"]


# ------------------------------------------------------------------------ LLM
@dataclass(slots=True)
class LLMMessage:
    role: Role
    content: str


@dataclass(slots=True)
class LLMUsage:
    input_tokens: int = 0
    output_tokens: int = 0

    def __add__(self, other: "LLMUsage") -> "LLMUsage":
        return LLMUsage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
        )


@dataclass(slots=True)
class LLMResponse:
    text: str
    model: str
    usage: LLMUsage = field(default_factory=LLMUsage)
    stop_reason: str | None = None
    parsed: dict[str, Any] | None = None


class LLMProvider(abc.ABC):
    """Text/structured completion."""

    name: str = "base"
    #: ``True`` when this provider does not call a real model. Agents check
    #: this to route straight to their deterministic fallback instead of
    #: fabricating model output.
    is_mock: bool = False

    @abc.abstractmethod
    def complete(
        self,
        *,
        system: str,
        messages: Sequence[LLMMessage],
        max_tokens: int | None = None,
        temperature: float | None = None,
        model: str | None = None,
    ) -> LLMResponse:
        """Free-form completion. Prefer :meth:`complete_json`."""

    @abc.abstractmethod
    def complete_json(
        self,
        *,
        system: str,
        messages: Sequence[LLMMessage],
        schema: dict[str, Any],
        schema_name: str = "response",
        max_tokens: int | None = None,
        temperature: float | None = None,
        model: str | None = None,
    ) -> LLMResponse:
        """Completion constrained to ``schema``.

        Implementations must set :attr:`LLMResponse.parsed` to the decoded
        object, or raise :class:`app.core.errors.LLMResponseError`.
        """

    def estimate_cost_usd(self, model: str, usage: LLMUsage) -> float:
        return 0.0


# --------------------------------------------------------------- transcription
@dataclass(slots=True)
class TranscriptWord:
    word: str
    start: float
    end: float
    probability: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "word": self.word,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "probability": self.probability,
        }


@dataclass(slots=True)
class TranscriptChunk:
    start: float
    end: float
    text: str
    words: list[TranscriptWord] = field(default_factory=list)
    speaker: str | None = None


@dataclass(slots=True)
class TranscriptionResult:
    language: str | None
    language_probability: float | None
    duration: float
    chunks: list[TranscriptChunk]
    provider: str
    model: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return " ".join(c.text.strip() for c in self.chunks if c.text.strip())

    @property
    def words(self) -> list[TranscriptWord]:
        out: list[TranscriptWord] = []
        for chunk in self.chunks:
            out.extend(chunk.words)
        return out


class TranscriptionProvider(abc.ABC):
    name: str = "base"
    is_mock: bool = False

    @abc.abstractmethod
    def transcribe(
        self,
        audio_path: Path,
        *,
        language: str | None = None,
        word_timestamps: bool = True,
    ) -> TranscriptionResult:
        ...


# -------------------------------------------------------------------- storage
@dataclass(slots=True)
class StoredObject:
    key: str
    size_bytes: int
    content_type: str | None
    checksum_sha256: str | None = None


class StorageProvider(abc.ABC):
    name: str = "base"

    @abc.abstractmethod
    def put_file(
        self, key: str, path: Path, *, content_type: str | None = None
    ) -> StoredObject:
        ...

    @abc.abstractmethod
    def put_stream(
        self,
        key: str,
        stream: BinaryIO,
        *,
        content_type: str | None = None,
        max_bytes: int | None = None,
    ) -> StoredObject:
        ...

    @abc.abstractmethod
    def get_to_path(self, key: str, destination: Path) -> Path:
        ...

    @abc.abstractmethod
    def exists(self, key: str) -> bool:
        ...

    @abc.abstractmethod
    def delete(self, key: str) -> None:
        ...

    @abc.abstractmethod
    def signed_url(self, key: str, *, expires_in: int | None = None) -> str:
        ...

    @abc.abstractmethod
    def local_path(self, key: str) -> Path | None:
        """Return a directly readable path when the backend is local, else ``None``."""


# --------------------------------------------------------------------- vision
@dataclass(slots=True)
class FaceBox:
    x: float
    y: float
    width: float
    height: float
    confidence: float
    #: Mean absolute pixel change in the mouth region between two *adjacent*
    #: frames. A talking face moves its mouth; a listening one does not. This
    #: is the active-speaker signal the crop tracker uses.
    mouth_activity: float = 0.0

    @property
    def center_x(self) -> float:
        return self.x + self.width / 2

    @property
    def center_y(self) -> float:
        return self.y + self.height / 2

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    def to_dict(self) -> dict[str, Any]:
        return {
            "x": round(self.x, 2),
            "y": round(self.y, 2),
            "w": round(self.width, 2),
            "h": round(self.height, 2),
            "confidence": round(self.confidence, 4),
            "mouth_activity": round(self.mouth_activity, 4),
        }


@dataclass(slots=True)
class FrameAnalysis:
    timestamp: float
    width: int
    height: int
    faces: list[FaceBox] = field(default_factory=list)
    brightness: float = 0.0
    sharpness: float = 0.0
    motion: float = 0.0
    is_black: bool = False


class VisionProvider(abc.ABC):
    name: str = "base"
    is_mock: bool = False

    @abc.abstractmethod
    def analyze_frames(
        self,
        video_path: Path,
        *,
        start: float = 0.0,
        end: float | None = None,
        sample_fps: float = 2.0,
        detect_faces: bool = True,
    ) -> list[FrameAnalysis]:
        ...


# ----------------------------------------------------------------- publishing
@dataclass(slots=True)
class PublishResult:
    external_post_id: str
    external_url: str | None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PlatformMetrics:
    views: int | None = None
    likes: int | None = None
    comments: int | None = None
    shares: int | None = None
    saves: int | None = None
    average_view_duration: float | None = None
    completion_rate: float | None = None
    retention_curve: list[dict[str, Any]] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)


class PublishingProvider(abc.ABC):
    name: str = "base"
    #: ``True`` when the provider only records intent (no network calls).
    is_manual: bool = False

    @abc.abstractmethod
    def publish(
        self,
        *,
        video_path: Path,
        title: str,
        description: str,
        hashtags: Sequence[str],
        credentials: dict[str, Any],
        privacy: str = "private",
    ) -> PublishResult:
        ...

    @abc.abstractmethod
    def fetch_metrics(
        self, *, external_post_id: str, credentials: dict[str, Any]
    ) -> PlatformMetrics:
        ...
