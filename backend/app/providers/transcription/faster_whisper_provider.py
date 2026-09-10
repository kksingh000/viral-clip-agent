"""On-box speech recognition via faster-whisper (CTranslate2).

Produces word-level timestamps, which the whole downstream pipeline depends on
for frame-accurate cuts and caption timing.
"""

from __future__ import annotations

import threading
from pathlib import Path

from app.core.config import settings
from app.core.errors import TranscriptionError
from app.core.logging import get_logger
from app.providers.base import (
    TranscriptChunk,
    TranscriptionProvider,
    TranscriptionResult,
    TranscriptWord,
)

logger = get_logger(__name__)

_model_lock = threading.Lock()
_model_cache: dict[tuple[str, str, str], object] = {}


def _load_model(size: str, device: str, compute_type: str):
    """Whisper models are expensive to load; cache one per configuration.

    Guarded by a lock because a Celery prefork worker may run several tasks in
    threads within one process.
    """
    key = (size, device, compute_type)
    with _model_lock:
        if key not in _model_cache:
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise TranscriptionError(
                    "faster-whisper is not installed. Install requirements-ml.txt "
                    "or set TRANSCRIPTION_PROVIDER to 'openai' or 'mock'."
                ) from exc
            resolved_device = device
            if device == "auto":
                resolved_device = "cpu"
                try:
                    import ctranslate2

                    if ctranslate2.get_cuda_device_count() > 0:
                        resolved_device = "cuda"
                except Exception:  # pragma: no cover - environment dependent
                    pass
            logger.info(
                "loading whisper model",
                extra={"size": size, "device": resolved_device, "compute": compute_type},
            )
            _model_cache[key] = WhisperModel(
                size, device=resolved_device, compute_type=compute_type
            )
        return _model_cache[key]


class FasterWhisperProvider(TranscriptionProvider):
    name = "faster_whisper"

    def __init__(
        self,
        model_size: str | None = None,
        device: str | None = None,
        compute_type: str | None = None,
    ) -> None:
        self.model_size = model_size or settings.whisper_model_size
        self.device = device or settings.whisper_device
        self.compute_type = compute_type or settings.whisper_compute_type

    def transcribe(
        self,
        audio_path: Path,
        *,
        language: str | None = None,
        word_timestamps: bool = True,
    ) -> TranscriptionResult:
        if not audio_path.exists():
            raise TranscriptionError(f"Audio file not found: {audio_path}")
        model = _load_model(self.model_size, self.device, self.compute_type)
        try:
            segments, info = model.transcribe(  # type: ignore[attr-defined]
                str(audio_path),
                language=language or settings.transcription_default_language,
                word_timestamps=word_timestamps,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 400},
                beam_size=5,
                condition_on_previous_text=False,
            )
        except Exception as exc:  # noqa: BLE001
            raise TranscriptionError(f"faster-whisper failed: {exc}") from exc

        chunks: list[TranscriptChunk] = []
        for segment in segments:  # generator: consuming it runs the decode
            words = [
                TranscriptWord(
                    word=w.word.strip(),
                    start=float(w.start),
                    end=float(w.end),
                    probability=float(getattr(w, "probability", 0.0) or 0.0),
                )
                for w in (getattr(segment, "words", None) or [])
                if w.word and w.word.strip() and w.start is not None and w.end is not None
            ]
            chunks.append(
                TranscriptChunk(
                    start=float(segment.start),
                    end=float(segment.end),
                    text=segment.text.strip(),
                    words=words,
                )
            )

        if not chunks:
            raise TranscriptionError(
                "Transcription produced no segments; the audio may be silent."
            )

        return TranscriptionResult(
            language=getattr(info, "language", None),
            language_probability=getattr(info, "language_probability", None),
            duration=float(getattr(info, "duration", chunks[-1].end)),
            chunks=chunks,
            provider=self.name,
            model=self.model_size,
            metadata={"vad_filter": True, "device": self.device},
        )
