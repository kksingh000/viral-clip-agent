"""Semantic transcript segmentation.

Fixed 30-second slices are the wrong unit: they cut through sentences, jokes
and answers. This module places boundaries where the *content* breaks --
sentence endings, pauses, speaker changes and shot changes -- then merges the
result into segments long enough to reason about but short enough to score.

No model is involved; the output is fully reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from app.core.logging import get_logger
from app.providers.base import TranscriptionResult, TranscriptWord
from app.services.signals import SegmentSignals, analyze_text, is_sentence_end

logger = get_logger(__name__)

#: A gap at least this long is treated as a natural break.
PAUSE_BREAK = 0.50
#: A gap at least this long always breaks, regardless of grammar.
HARD_PAUSE_BREAK = 1.10
#: Target bounds for a segment. Short enough to score, long enough to mean
#: something on its own.
MIN_SEGMENT_SECONDS = 4.0
MAX_SEGMENT_SECONDS = 45.0
IDEAL_SEGMENT_SECONDS = 20.0
#: A shot change within this distance of a word gap snaps the boundary to it.
SCENE_SNAP_WINDOW = 0.6

#: Break strengths. A *strong* boundary (a speaker change, or a sentence that
#: ends into a long pause) closes a segment even when it is shorter than
#: ``MIN_SEGMENT_SECONDS`` -- forcing a minimum length across such a boundary
#: is what glues a channel intro onto the actual hook.
STRENGTH_SPEAKER_CHANGE = 1.0
STRENGTH_HARD_PAUSE = 0.95
STRENGTH_SENTENCE_INTO_PAUSE = 0.9
STRENGTH_SENTENCE = 0.7
STRENGTH_PAUSE = 0.6
STRENGTH_SCENE_CHANGE = 0.5

#: A boundary at least this strong may close a short segment.
STRONG_BREAK = 0.85
#: ...but never below this duration, or captions become unreadable churn.
ABSOLUTE_MIN_SECONDS = 2.0


@dataclass(slots=True)
class Segment:
    index: int
    start: float
    end: float
    text: str
    words: list[TranscriptWord] = field(default_factory=list)
    speaker: str | None = None
    signals: SegmentSignals | None = None

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "duration": round(self.duration, 3),
            "text": self.text,
            "speaker": self.speaker,
            "words": [w.to_dict() for w in self.words],
            "signals": self.signals.to_dict() if self.signals else {},
        }


def _speaker_at(chunks, timestamp: float) -> str | None:
    for chunk in chunks:
        if chunk.start <= timestamp < chunk.end:
            return chunk.speaker
    return None


def segment_transcript(
    transcript: TranscriptionResult,
    *,
    scene_boundaries: Sequence[float] = (),
    min_seconds: float = MIN_SEGMENT_SECONDS,
    max_seconds: float = MAX_SEGMENT_SECONDS,
) -> list[Segment]:
    """Split a transcript into semantic segments."""
    words = transcript.words
    if not words:
        # No word timings (some providers, or the synthetic path): fall back to
        # the provider's own chunking, which is still time-accurate.
        return _segments_from_chunks(transcript)

    boundaries = _boundary_indices(words, transcript, scene_boundaries)
    segments = _assemble(words, boundaries, transcript, min_seconds, max_seconds)

    scene_set = sorted(scene_boundaries)
    for segment in segments:
        text_signals = analyze_text(segment.text)
        leading, trailing = _pauses_around(words, segment)
        segment.signals = SegmentSignals(
            text=text_signals,
            words_per_second=(
                len(segment.words) / segment.duration if segment.duration > 0 else 0.0
            ),
            leading_pause=leading,
            trailing_pause=trailing,
            scene_aligned_start=_near_scene(segment.start, scene_set),
            scene_aligned_end=_near_scene(segment.end, scene_set),
        )
    logger.info(
        "transcript segmented",
        extra={
            "segments": len(segments),
            "words": len(words),
            "mean_seconds": round(
                sum(s.duration for s in segments) / max(1, len(segments)), 2
            ),
        },
    )
    return segments


def _boundary_indices(
    words: Sequence[TranscriptWord],
    transcript: TranscriptionResult,
    scene_boundaries: Sequence[float],
) -> dict[int, float]:
    """Word indices *after which* a break is allowed, mapped to a strength."""
    breaks: dict[int, float] = {}
    scenes = sorted(scene_boundaries)

    def offer(index: int, strength: float) -> None:
        if strength > breaks.get(index, 0.0):
            breaks[index] = strength

    for index in range(len(words) - 1):
        current, following = words[index], words[index + 1]
        gap = following.start - current.end
        sentence_end = is_sentence_end(current.word)

        if _speaker_at(transcript.chunks, current.start) != _speaker_at(
            transcript.chunks, following.start
        ):
            offer(index, STRENGTH_SPEAKER_CHANGE)
        if gap >= HARD_PAUSE_BREAK:
            offer(index, STRENGTH_HARD_PAUSE)
        if sentence_end and gap >= PAUSE_BREAK:
            offer(index, STRENGTH_SENTENCE_INTO_PAUSE)
        elif sentence_end:
            offer(index, STRENGTH_SENTENCE)
        elif gap >= PAUSE_BREAK:
            offer(index, STRENGTH_PAUSE)
        # A shot change landing inside the gap.
        if any(
            current.end - SCENE_SNAP_WINDOW <= s <= following.start + SCENE_SNAP_WINDOW
            for s in scenes
        ):
            offer(index, STRENGTH_SCENE_CHANGE)

    breaks[len(words) - 1] = 1.0
    return breaks


def _assemble(
    words: Sequence[TranscriptWord],
    breaks: dict[int, float],
    transcript: TranscriptionResult,
    min_seconds: float,
    max_seconds: float,
) -> list[Segment]:
    segments: list[Segment] = []
    current: list[TranscriptWord] = []
    start_index = 0

    for index, word in enumerate(words):
        current.append(word)
        span = current[-1].end - current[0].start
        strength = breaks.get(index, 0.0)

        should_close = False
        if strength > 0 and span >= min_seconds:
            should_close = True
        elif strength >= STRONG_BREAK and span >= ABSOLUTE_MIN_SECONDS:
            # A speaker change or a sentence into a long pause is a real
            # content boundary; honour it rather than gluing two topics
            # together to satisfy a length target.
            should_close = True
        elif span >= max_seconds:
            # Overlong without a natural break: close at the most recent
            # allowed break inside the window, else force a cut here.
            recent = [b for b in breaks if start_index <= b < index]
            if recent:
                cut = max(recent)
                head = words[start_index : cut + 1]
                segments.append(_make_segment(len(segments), head, transcript))
                current = list(words[cut + 1 : index + 1])
                start_index = cut + 1
                continue
            should_close = True

        if should_close:
            segments.append(_make_segment(len(segments), current, transcript))
            current = []
            start_index = index + 1

    if current:
        if segments and (current[-1].end - current[0].start) < min_seconds / 2:
            # A stub tail belongs to the previous segment.
            previous = segments[-1]
            merged = previous.words + current
            segments[-1] = _make_segment(previous.index, merged, transcript)
        else:
            segments.append(_make_segment(len(segments), current, transcript))
    return segments


def _make_segment(
    index: int, words: Sequence[TranscriptWord], transcript: TranscriptionResult
) -> Segment:
    text = " ".join(w.word for w in words).strip()
    return Segment(
        index=index,
        start=words[0].start,
        end=words[-1].end,
        text=text,
        words=list(words),
        speaker=_speaker_at(transcript.chunks, words[0].start),
    )


def _pauses_around(
    words: Sequence[TranscriptWord], segment: Segment
) -> tuple[float, float]:
    leading = trailing = 0.0
    for index, word in enumerate(words):
        if abs(word.start - segment.start) < 1e-6 and index > 0:
            leading = max(0.0, word.start - words[index - 1].end)
        if abs(word.end - segment.end) < 1e-6 and index + 1 < len(words):
            trailing = max(0.0, words[index + 1].start - word.end)
    return leading, trailing


def _near_scene(timestamp: float, scenes: Sequence[float]) -> bool:
    return any(abs(timestamp - s) <= SCENE_SNAP_WINDOW for s in scenes)


def _segments_from_chunks(transcript: TranscriptionResult) -> list[Segment]:
    segments: list[Segment] = []
    for index, chunk in enumerate(transcript.chunks):
        segment = Segment(
            index=index,
            start=chunk.start,
            end=chunk.end,
            text=chunk.text,
            words=list(chunk.words),
            speaker=chunk.speaker,
        )
        segment.signals = SegmentSignals(
            text=analyze_text(chunk.text),
            words_per_second=(
                len(chunk.text.split()) / segment.duration if segment.duration else 0.0
            ),
        )
        segments.append(segment)
    return segments
