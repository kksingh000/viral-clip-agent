"""Provider registry.

Single place where configuration is turned into a concrete implementation.
Instances are cached per process; tests override them with
:func:`override_providers`.
"""

from __future__ import annotations

from contextlib import contextmanager
from threading import Lock
from typing import Iterator

from app.core.config import settings
from app.core.errors import ProviderError
from app.core.logging import get_logger
from app.models.enums import Platform
from app.providers.base import (
    LLMProvider,
    PublishingProvider,
    StorageProvider,
    TranscriptionProvider,
    VisionProvider,
)

logger = get_logger(__name__)

_lock = Lock()
_cache: dict[str, object] = {}
_overrides: dict[str, object] = {}


def _cached(key: str, factory):
    if key in _overrides:
        return _overrides[key]
    with _lock:
        if key not in _cache:
            _cache[key] = factory()
        return _cache[key]


# --------------------------------------------------------------------- getters
def get_llm_provider() -> LLMProvider:
    def build() -> LLMProvider:
        choice = settings.llm_provider
        if choice == "anthropic":
            from app.providers.llm.anthropic_provider import AnthropicLLMProvider

            return AnthropicLLMProvider()
        if choice == "openai":
            from app.providers.llm.openai_provider import OpenAILLMProvider

            return OpenAILLMProvider()
        from app.providers.llm.mock_provider import MockLLMProvider

        logger.warning(
            "LLM_PROVIDER=mock -- agents will use deterministic fallbacks, "
            "not model output."
        )
        return MockLLMProvider()

    return _cached("llm", build)  # type: ignore[return-value]


def get_transcription_provider() -> TranscriptionProvider:
    def build() -> TranscriptionProvider:
        choice = settings.transcription_provider
        if choice == "faster_whisper":
            from app.providers.transcription.faster_whisper_provider import (
                FasterWhisperProvider,
            )

            return FasterWhisperProvider()
        if choice == "openai":
            from app.providers.transcription.openai_provider import (
                OpenAITranscriptionProvider,
            )

            return OpenAITranscriptionProvider()
        from app.providers.transcription.mock_provider import MockTranscriptionProvider

        logger.warning(
            "TRANSCRIPTION_PROVIDER=mock -- transcripts will be structural only."
        )
        return MockTranscriptionProvider()

    return _cached("transcription", build)  # type: ignore[return-value]


def get_storage_provider() -> StorageProvider:
    def build() -> StorageProvider:
        if settings.storage_provider == "s3":
            from app.providers.storage.s3_storage import S3StorageProvider

            return S3StorageProvider()
        from app.providers.storage.local_storage import LocalStorageProvider

        return LocalStorageProvider()

    return _cached("storage", build)  # type: ignore[return-value]


def get_vision_provider() -> VisionProvider:
    def build() -> VisionProvider:
        if settings.vision_provider == "opencv":
            try:
                import cv2  # noqa: F401

                from app.providers.vision.opencv_provider import OpenCVVisionProvider

                return OpenCVVisionProvider()
            except ImportError:  # pragma: no cover - dependency guard
                logger.warning("opencv not installed; falling back to geometry-only vision")
        from app.providers.vision.opencv_provider import MockVisionProvider

        return MockVisionProvider()

    return _cached("vision", build)  # type: ignore[return-value]


def get_publishing_provider(platform: Platform | str) -> PublishingProvider:
    platform = Platform(platform)

    def build() -> PublishingProvider:
        if platform is Platform.YOUTUBE_SHORTS:
            from app.providers.publishing.youtube import YouTubeShortsProvider

            return YouTubeShortsProvider()
        if platform is Platform.INSTAGRAM_REELS:
            from app.providers.publishing.instagram import InstagramReelsProvider

            return InstagramReelsProvider()
        if platform is Platform.MANUAL_EXPORT:
            from app.providers.publishing.manual import ManualExportProvider

            return ManualExportProvider()
        raise ProviderError(f"No publishing provider is implemented for {platform}.")

    return _cached(f"publishing:{platform.value}", build)  # type: ignore[return-value]


# ----------------------------------------------------------------- test hooks
@contextmanager
def override_providers(**providers: object) -> Iterator[None]:
    """Temporarily replace providers.

    ``override_providers(llm=FakeLLM(), storage=FakeStorage())``
    """
    previous = dict(_overrides)
    _overrides.update(providers)
    try:
        yield
    finally:
        _overrides.clear()
        _overrides.update(previous)


def reset_providers() -> None:
    with _lock:
        _cache.clear()
    _overrides.clear()
