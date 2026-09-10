"""Hosted speech recognition through the OpenAI audio transcription API."""

from __future__ import annotations

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

#: Hard API limit on upload size for the transcription endpoint.
MAX_UPLOAD_BYTES = 25 * 1024 * 1024


class OpenAITranscriptionProvider(TranscriptionProvider):
    name = "openai"

    def __init__(self, api_key: str | None = None, model: str = "whisper-1") -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise TranscriptionError("The 'openai' package is required.") from exc
        key = api_key or settings.openai_api_key
        if not key:
            raise TranscriptionError("OPENAI_API_KEY is not configured.")
        self._client = OpenAI(api_key=key, base_url=settings.openai_base_url or None)
        self.model = model

    def transcribe(
        self,
        audio_path: Path,
        *,
        language: str | None = None,
        word_timestamps: bool = True,
    ) -> TranscriptionResult:
        if not audio_path.exists():
            raise TranscriptionError(f"Audio file not found: {audio_path}")
        size = audio_path.stat().st_size
        if size > MAX_UPLOAD_BYTES:
            raise TranscriptionError(
                f"{audio_path.name} is {size / 1e6:.1f} MB, over the "
                f"{MAX_UPLOAD_BYTES / 1e6:.0f} MB API limit. Split the audio, "
                "lower the sample rate, or use the faster_whisper provider."
            )
        granularities = ["segment"]
        if word_timestamps:
            granularities.append("word")
        try:
            with audio_path.open("rb") as handle:
                response = self._client.audio.transcriptions.create(
                    model=self.model,
                    file=handle,
                    response_format="verbose_json",
                    timestamp_granularities=granularities,
                    language=language or settings.transcription_default_language or None,
                )
        except Exception as exc:  # noqa: BLE001
            raise TranscriptionError(f"OpenAI transcription failed: {exc}") from exc

        payload = response.model_dump() if hasattr(response, "model_dump") else dict(response)
        raw_words = payload.get("words") or []
        words = [
            TranscriptWord(
                word=str(w.get("word", "")).strip(),
                start=float(w.get("start", 0.0)),
                end=float(w.get("end", 0.0)),
                probability=None,
            )
            for w in raw_words
            if str(w.get("word", "")).strip()
        ]

        chunks: list[TranscriptChunk] = []
        for segment in payload.get("segments") or []:
            start = float(segment.get("start", 0.0))
            end = float(segment.get("end", 0.0))
            chunks.append(
                TranscriptChunk(
                    start=start,
                    end=end,
                    text=str(segment.get("text", "")).strip(),
                    words=[w for w in words if start <= w.start < end],
                )
            )
        if not chunks and payload.get("text"):
            duration = float(payload.get("duration") or (words[-1].end if words else 0.0))
            chunks = [
                TranscriptChunk(
                    start=0.0, end=duration, text=str(payload["text"]).strip(), words=words
                )
            ]
        if not chunks:
            raise TranscriptionError("Transcription returned no segments.")

        return TranscriptionResult(
            language=payload.get("language"),
            language_probability=None,
            duration=float(payload.get("duration") or chunks[-1].end),
            chunks=chunks,
            provider=self.name,
            model=self.model,
            metadata={"granularities": granularities},
        )
