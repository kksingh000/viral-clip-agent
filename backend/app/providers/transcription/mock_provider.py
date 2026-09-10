"""Offline transcription provider.

Two behaviours, in priority order:

1. **Sidecar transcript.** If ``<media>.transcript.json`` sits beside the media
   file, it is loaded verbatim. This is the supported way to run the pipeline
   on real content whose transcript you already have (an existing caption file,
   a previous run, or a fixture in the test suite).
2. **Synthetic structure.** Otherwise the provider derives *real* segment
   timings from the audio (silence boundaries) but cannot invent words: every
   segment carries the placeholder ``[no transcription provider configured]``
   and the result is flagged ``synthetic=True``.

The synthetic mode exists so the media pipeline is exercisable without a model;
it is never presented to the user as a real transcript -- the API surfaces the
``synthetic`` flag, and agents downgrade to structural-only scoring.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.core.errors import TranscriptionError
from app.core.logging import get_logger
from app.providers.base import (
    TranscriptChunk,
    TranscriptionProvider,
    TranscriptionResult,
    TranscriptWord,
)
from app.video.audio import detect_silence
from app.video.ffmpeg import probe

logger = get_logger(__name__)

PLACEHOLDER_TEXT = "[no transcription provider configured]"
_MAX_SEGMENT = 20.0
_MIN_SEGMENT = 3.0


def _sidecar_for(audio_path: Path) -> Path | None:
    candidates = [
        audio_path.with_suffix(".transcript.json"),
        audio_path.parent / f"{audio_path.stem}.transcript.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def load_transcript_json(path: Path, *, provider: str = "sidecar") -> TranscriptionResult:
    """Load a transcript sidecar.

    Accepted shape::

        {"language": "en", "duration": 90.0,
         "segments": [{"start": 0.0, "end": 4.2, "text": "...",
                       "words": [{"word": "hi", "start": 0.0, "end": 0.2}]}]}
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TranscriptionError(f"Could not read transcript sidecar {path}: {exc}") from exc

    raw_segments = payload.get("segments")
    if not isinstance(raw_segments, list) or not raw_segments:
        raise TranscriptionError(f"Transcript sidecar {path} has no segments.")

    chunks: list[TranscriptChunk] = []
    for segment in raw_segments:
        words = [
            TranscriptWord(
                word=str(w["word"]).strip(),
                start=float(w["start"]),
                end=float(w["end"]),
                probability=w.get("probability"),
            )
            for w in segment.get("words", [])
            if str(w.get("word", "")).strip()
        ]
        text = str(segment.get("text") or " ".join(w.word for w in words)).strip()
        if not text:
            continue
        chunks.append(
            TranscriptChunk(
                start=float(segment["start"]),
                end=float(segment["end"]),
                text=text,
                words=words,
                speaker=segment.get("speaker"),
            )
        )
    if not chunks:
        raise TranscriptionError(f"Transcript sidecar {path} contained no usable text.")

    return TranscriptionResult(
        language=payload.get("language"),
        language_probability=payload.get("language_probability"),
        duration=float(payload.get("duration") or chunks[-1].end),
        chunks=chunks,
        provider=provider,
        model=payload.get("model"),
        metadata={"source": str(path), "synthetic": False},
    )


class MockTranscriptionProvider(TranscriptionProvider):
    name = "mock"
    is_mock = True

    def transcribe(
        self,
        audio_path: Path,
        *,
        language: str | None = None,
        word_timestamps: bool = True,
    ) -> TranscriptionResult:
        if not audio_path.exists():
            raise TranscriptionError(f"Audio file not found: {audio_path}")

        if (sidecar := _sidecar_for(audio_path)) is not None:
            logger.info("using transcript sidecar", extra={"path": str(sidecar)})
            return load_transcript_json(sidecar, provider="sidecar")

        duration = probe(audio_path).duration
        boundaries = self._segment_boundaries(audio_path, duration)
        chunks = [
            TranscriptChunk(
                start=start,
                end=end,
                text=PLACEHOLDER_TEXT,
                words=[TranscriptWord(word=PLACEHOLDER_TEXT, start=start, end=end)],
            )
            for start, end in boundaries
        ]
        logger.warning(
            "no transcription provider configured; emitting synthetic segments",
            extra={"segments": len(chunks), "duration": round(duration, 2)},
        )
        return TranscriptionResult(
            language=language,
            language_probability=None,
            duration=duration,
            chunks=chunks,
            provider=self.name,
            model=None,
            metadata={"synthetic": True, "placeholder_text": PLACEHOLDER_TEXT},
        )

    @staticmethod
    def _segment_boundaries(audio_path: Path, duration: float) -> list[tuple[float, float]]:
        """Split on real silences, then enforce sane segment lengths."""
        try:
            silences = detect_silence(audio_path)
        except Exception as exc:  # noqa: BLE001 - analysis must not be fatal here
            logger.debug("silence detection failed", extra={"error": str(exc)})
            silences = []

        cuts = [0.0]
        for silence in silences:
            midpoint = (silence.start + silence.end) / 2
            if midpoint - cuts[-1] >= _MIN_SEGMENT:
                cuts.append(midpoint)
        cuts.append(duration)

        segments: list[tuple[float, float]] = []
        for start, end in zip(cuts, cuts[1:]):
            if end - start <= 0:
                continue
            span = end - start
            if span <= _MAX_SEGMENT:
                segments.append((start, end))
                continue
            pieces = int(span // _MAX_SEGMENT) + 1
            step = span / pieces
            for index in range(pieces):
                segments.append((start + index * step, start + (index + 1) * step))
        return segments or [(0.0, duration)]
