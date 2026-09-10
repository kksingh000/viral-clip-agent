"""Turning raw candidate spans into final, cuttable clips.

Everything here is deterministic post-processing of the agent's proposals:

* **Boundary snapping** -- never cut mid-word; prefer sentence ends; snap onto
  a shot change when one is within a frame or two.
* **Context repair** -- if the opening line depends on a pronoun with no
  antecedent inside the clip, extend backwards to the previous sentence when
  the duration budget allows (spec: "start slightly before the viral moment").
* **Duration fitting** -- trim or extend to the configured window without
  breaking a sentence.
* **Scoring** -- combine the agent's dimensions with structural penalties.
* **De-duplication** -- drop near-identical clips, keeping the better one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from app.core.logging import get_logger
from app.providers.base import TranscriptWord
from app.services.scoring import (
    ScoreResult,
    ScoringWeights,
    StructuralContext,
    ViralDimensions,
    compute_score,
)
from app.services.segmentation import Segment
from app.services.signals import analyze_text, is_sentence_end, jaccard, keyword_set

logger = get_logger(__name__)

#: Padding kept around speech so a clip does not start on a clipped consonant.
LEAD_PADDING = 0.12
TAIL_PADDING = 0.28
#: How far a boundary may move to reach a sentence edge.
MAX_SENTENCE_SEEK = 3.5
#: How close a shot change must be for the cut to snap onto it.
SCENE_SNAP = 0.35
#: Two clips overlapping by more than this fraction are duplicates.
OVERLAP_DUPLICATE = 0.55
#: Two clips whose transcripts are this similar are duplicates.
TEXT_DUPLICATE = 0.72


@dataclass(slots=True)
class SnappedSpan:
    start: float
    end: float
    text: str
    context_lead_in: float
    raw_start: float
    starts_mid_sentence: bool
    ends_mid_sentence: bool
    adjustments: list[str] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass(slots=True)
class ScoredCandidate:
    start: float
    end: float
    duration: float
    hook: str
    summary: str
    reason: str
    text: str
    dimensions: ViralDimensions
    score: ScoreResult
    context_lead_in: float
    raw_start: float
    adjustments: list[str] = field(default_factory=list)
    keywords: set[str] = field(default_factory=set)

    @property
    def viral_score(self) -> float:
        return self.score.final_score

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_time": round(self.start, 3),
            "end_time": round(self.end, 3),
            "duration": round(self.duration, 3),
            "hook": self.hook,
            "summary": self.summary,
            "reason": self.reason,
            "viral_score": round(self.viral_score, 2),
            "context_lead_in": round(self.context_lead_in, 3),
            "score_breakdown": self.score.to_dict(),
            "adjustments": self.adjustments,
        }


# ------------------------------------------------------------------- snapping
#: How much filler makes a neighbouring segment not worth extending into.
UNUSABLE_FILLER_RATIO = 0.35


@dataclass(slots=True)
class ExtensionLimits:
    """How far a span may grow before it runs into unusable material.

    Reaching the minimum duration is not worth swallowing a channel intro or a
    rambling sign-off, so the limits mark where growth must stop in each
    direction. ``None`` means unbounded.
    """

    earliest_start: float | None = None
    latest_end: float | None = None


def extension_limits(
    segments: Sequence[Segment],
    start: float,
    end: float,
    *,
    filler_ratio: float = UNUSABLE_FILLER_RATIO,
) -> ExtensionLimits:
    """Find the nearest unusable segment on each side of a span.

    "Unusable" is channel boilerplate ("please subscribe", "thanks for
    watching") or speech that is mostly filler -- material that would make the
    clip worse even though it would make it longer.
    """

    def unusable(segment: Segment) -> bool:
        if segment.signals is None:
            return False
        text = segment.signals.text
        return text.boilerplate_hits > 0 or text.filler_ratio > filler_ratio

    earliest: float | None = None
    latest: float | None = None
    for segment in segments:
        if not unusable(segment):
            continue
        if segment.end <= start + 1e-6:
            # The latest bad segment before the span sets the backward floor.
            earliest = max(earliest or 0.0, segment.end)
        elif segment.start >= end - 1e-6:
            # The earliest bad segment after the span sets the forward ceiling.
            latest = segment.start if latest is None else min(latest, segment.start)
    return ExtensionLimits(earliest_start=earliest, latest_end=latest)


def snap_boundaries(
    start: float,
    end: float,
    words: Sequence[TranscriptWord],
    *,
    scene_boundaries: Sequence[float] = (),
    min_duration: float = 15.0,
    max_duration: float = 60.0,
    total_duration: float | None = None,
    limits: ExtensionLimits | None = None,
) -> SnappedSpan:
    """Move a proposed span onto clean speech boundaries."""
    adjustments: list[str] = []
    if not words:
        return SnappedSpan(
            start=max(0.0, start),
            end=max(start + 0.5, end),
            text="",
            context_lead_in=0.0,
            raw_start=start,
            starts_mid_sentence=False,
            ends_mid_sentence=False,
            adjustments=["no word timings available; span used verbatim"],
        )

    raw_start = start
    first_index = _first_word_at_or_after(words, start)
    last_index = _last_word_at_or_before(words, end)

    if first_index is None:
        first_index = 0
    if last_index is None or last_index < first_index:
        last_index = min(len(words) - 1, first_index)

    # 1. Never open mid-word.
    if words[first_index].start < start - 1e-6:
        adjustments.append("moved start to the next word boundary")

    # 2. Prefer opening at a sentence start.
    sentence_start = _sentence_start_index(words, first_index)
    if sentence_start != first_index:
        candidate_start = words[sentence_start].start
        if start - candidate_start <= MAX_SENTENCE_SEEK and (
            words[last_index].end - candidate_start
        ) <= max_duration:
            first_index = sentence_start
            adjustments.append("extended start back to the sentence beginning")

    # 3. Close on a sentence end when one is near.
    sentence_end = _sentence_end_index(words, last_index)
    if sentence_end != last_index:
        candidate_end = words[sentence_end].end
        if candidate_end - end <= MAX_SENTENCE_SEEK and (
            candidate_end - words[first_index].start
        ) <= max_duration:
            last_index = sentence_end
            adjustments.append("extended end to complete the sentence")

    snapped_start = max(0.0, words[first_index].start - LEAD_PADDING)
    if first_index > 0:
        snapped_start = max(snapped_start, words[first_index - 1].end + 0.01)
    snapped_end = words[last_index].end + TAIL_PADDING
    if last_index + 1 < len(words):
        snapped_end = min(snapped_end, words[last_index + 1].start - 0.01)
    if total_duration is not None:
        snapped_end = min(snapped_end, total_duration)

    # 4. Snap onto a shot change when one sits within a couple of frames.
    for boundary in scene_boundaries:
        if abs(boundary - snapped_start) <= SCENE_SNAP and boundary <= snapped_start:
            snapped_start = boundary
            adjustments.append("snapped start to a shot change")
        if abs(boundary - snapped_end) <= SCENE_SNAP and boundary >= snapped_end:
            snapped_end = boundary
            adjustments.append("snapped end to a shot change")

    # 5. Fit the duration window without breaking sentences.
    snapped_start, snapped_end, first_index, last_index, fit_notes = _fit_duration(
        words, first_index, last_index, snapped_start, snapped_end,
        min_duration, max_duration, total_duration, limits or ExtensionLimits(),
    )
    adjustments.extend(fit_notes)

    text = " ".join(w.word for w in words[first_index : last_index + 1]).strip()
    return SnappedSpan(
        start=snapped_start,
        end=snapped_end,
        text=text,
        context_lead_in=max(0.0, raw_start - snapped_start),
        raw_start=raw_start,
        starts_mid_sentence=not _is_sentence_boundary(words, first_index),
        ends_mid_sentence=not is_sentence_end(words[last_index].word),
        adjustments=adjustments,
    )


def _fit_duration(
    words: Sequence[TranscriptWord],
    first: int,
    last: int,
    start: float,
    end: float,
    min_duration: float,
    max_duration: float,
    total_duration: float | None,
    limits: ExtensionLimits,
) -> tuple[float, float, int, int, list[str]]:
    notes: list[str] = []
    ceiling = limits.latest_end
    floor = limits.earliest_start

    # Too long: pull the end back to the last sentence end that fits.
    while end - start > max_duration and last > first:
        for index in range(last, first, -1):
            if is_sentence_end(words[index].word) and (
                words[index].end + TAIL_PADDING - start
            ) <= max_duration:
                last = index
                end = words[index].end + TAIL_PADDING
                notes.append("trimmed to the last sentence that fits the maximum")
                break
        else:
            last -= 1
            end = words[last].end + TAIL_PADDING
            notes.append("hard-trimmed to the maximum duration")
    # Too short: extend forwards, then backwards, on sentence boundaries.
    while end - start < min_duration and last + 1 < len(words):
        if ceiling is not None and words[last + 1].end > ceiling:
            notes.append(
                "stopped short of the minimum rather than extending into "
                "unusable material"
            )
            break
        last += 1
        end = words[last].end + TAIL_PADDING
        if total_duration is not None:
            end = min(end, total_duration)
        if is_sentence_end(words[last].word) and end - start >= min_duration:
            notes.append("extended forwards to reach the minimum duration")
            break
    else:
        # Reaching the minimum is not enough: stopping there would leave the
        # clip ending mid-sentence. Run on to the next sentence end, provided
        # it still fits inside the maximum.
        if last + 1 < len(words) and not is_sentence_end(words[last].word):
            probe = last
            while probe + 1 < len(words):
                probe += 1
                candidate_end = words[probe].end + TAIL_PADDING
                if total_duration is not None:
                    candidate_end = min(candidate_end, total_duration)
                if candidate_end - start > max_duration:
                    break
                if ceiling is not None and candidate_end > ceiling:
                    break
                if is_sentence_end(words[probe].word):
                    last = probe
                    end = candidate_end
                    notes.append("ran on to the end of the sentence")
                    break

    while end - start < min_duration and first > 0:
        if floor is not None and words[first - 1].start < floor:
            notes.append(
                "stopped short of the minimum rather than extending back into "
                "unusable material"
            )
            break
        first -= 1
        start = max(0.0, words[first].start - LEAD_PADDING)
        if _is_sentence_boundary(words, first) and end - start >= min_duration:
            notes.append("extended backwards to reach the minimum duration")
            break
    return start, end, first, last, notes


def ensure_standalone_opening(
    span: SnappedSpan,
    words: Sequence[TranscriptWord],
    *,
    max_duration: float,
    limits: ExtensionLimits | None = None,
) -> SnappedSpan:
    """Extend backwards when the clip opens on an unresolved reference.

    "...and that is why it works" is a bad opening; the previous sentence is
    usually the missing antecedent.
    """
    if not span.text or not words:
        return span
    signals = analyze_text(span.text)
    if not signals.starts_with_dangling_reference:
        return span

    first_index = _first_word_at_or_after(words, span.start)
    if first_index is None or first_index == 0:
        return span
    previous_sentence = _sentence_start_index(words, max(0, first_index - 1))
    candidate_start = max(0.0, words[previous_sentence].start - LEAD_PADDING)
    floor = (limits or ExtensionLimits()).earliest_start
    if floor is not None and candidate_start < floor:
        span.adjustments.append(
            "opening needs context but the preceding material is unusable"
        )
        return span
    if span.end - candidate_start > max_duration:
        span.adjustments.append(
            "opening needs context but extending would exceed the maximum duration"
        )
        return span

    new_text = " ".join(
        w.word for w in words if candidate_start <= w.start and w.end <= span.end
    ).strip()
    span.context_lead_in += span.start - candidate_start
    span.start = candidate_start
    span.text = new_text
    span.starts_mid_sentence = False
    span.adjustments.append("extended backwards to supply missing context")
    return span


# --------------------------------------------------------------------- scoring
def score_candidate(
    span: SnappedSpan,
    dimensions: ViralDimensions,
    *,
    weights: ScoringWeights | None = None,
    min_duration: float = 15.0,
    max_duration: float = 60.0,
    silence_ratio: float = 0.0,
    has_audio: bool = True,
) -> ScoreResult:
    structure = StructuralContext(
        duration=span.duration,
        min_duration=min_duration,
        max_duration=max_duration,
        starts_mid_sentence=span.starts_mid_sentence,
        ends_mid_sentence=span.ends_mid_sentence,
        starts_with_dangling_reference=analyze_text(
            span.text
        ).starts_with_dangling_reference,
        silence_ratio=silence_ratio,
        has_audio=has_audio,
    )
    return compute_score(dimensions, weights=weights, structure=structure)


# ------------------------------------------------------------- de-duplication
def deduplicate(
    candidates: Sequence[ScoredCandidate],
    *,
    overlap_threshold: float = OVERLAP_DUPLICATE,
    text_threshold: float = TEXT_DUPLICATE,
) -> tuple[list[ScoredCandidate], list[ScoredCandidate]]:
    """Keep the better of any two near-identical clips.

    Returns ``(kept, discarded)``. Overlap is measured against the shorter
    clip, so a 20s clip fully inside a 45s one counts as a duplicate.
    """
    ordered = sorted(candidates, key=lambda c: c.viral_score, reverse=True)
    kept: list[ScoredCandidate] = []
    discarded: list[ScoredCandidate] = []

    for candidate in ordered:
        duplicate = False
        for existing in kept:
            overlap = _overlap_ratio(
                (candidate.start, candidate.end), (existing.start, existing.end)
            )
            similarity = jaccard(candidate.keywords, existing.keywords)
            if overlap >= overlap_threshold or similarity >= text_threshold:
                duplicate = True
                logger.debug(
                    "dropped duplicate candidate",
                    extra={
                        "overlap": round(overlap, 3),
                        "similarity": round(similarity, 3),
                        "kept_score": round(existing.viral_score, 1),
                        "dropped_score": round(candidate.viral_score, 1),
                    },
                )
                break
        (discarded if duplicate else kept).append(candidate)
    return kept, discarded


def _overlap_ratio(a: tuple[float, float], b: tuple[float, float]) -> float:
    overlap = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    shorter = min(a[1] - a[0], b[1] - b[0])
    return overlap / shorter if shorter > 0 else 0.0


# --------------------------------------------------------------------- helpers
def _first_word_at_or_after(
    words: Sequence[TranscriptWord], timestamp: float
) -> int | None:
    for index, word in enumerate(words):
        if word.end > timestamp:
            return index
    return len(words) - 1 if words else None


def _last_word_at_or_before(
    words: Sequence[TranscriptWord], timestamp: float
) -> int | None:
    result: int | None = None
    for index, word in enumerate(words):
        if word.start <= timestamp:
            result = index
        else:
            break
    return result


def _sentence_start_index(words: Sequence[TranscriptWord], index: int) -> int:
    cursor = index
    while cursor > 0 and not is_sentence_end(words[cursor - 1].word):
        cursor -= 1
    return cursor


def _sentence_end_index(words: Sequence[TranscriptWord], index: int) -> int:
    cursor = index
    while cursor + 1 < len(words) and not is_sentence_end(words[cursor].word):
        cursor += 1
    return cursor


def _is_sentence_boundary(words: Sequence[TranscriptWord], index: int) -> bool:
    return index == 0 or is_sentence_end(words[index - 1].word)


def words_between(
    words: Sequence[TranscriptWord], start: float, end: float
) -> list[TranscriptWord]:
    return [w for w in words if w.start >= start - 1e-6 and w.end <= end + 1e-6]


def build_scored_candidate(
    span: SnappedSpan,
    dimensions: ViralDimensions,
    *,
    hook: str,
    summary: str,
    reason: str,
    weights: ScoringWeights | None = None,
    min_duration: float = 15.0,
    max_duration: float = 60.0,
    silence_ratio: float = 0.0,
    has_audio: bool = True,
) -> ScoredCandidate:
    score = score_candidate(
        span,
        dimensions,
        weights=weights,
        min_duration=min_duration,
        max_duration=max_duration,
        silence_ratio=silence_ratio,
        has_audio=has_audio,
    )
    return ScoredCandidate(
        start=span.start,
        end=span.end,
        duration=span.duration,
        hook=hook,
        summary=summary,
        reason=reason,
        text=span.text,
        dimensions=dimensions,
        score=score,
        context_lead_in=span.context_lead_in,
        raw_start=span.raw_start,
        adjustments=list(span.adjustments),
        keywords=keyword_set(span.text),
    )


def segments_between(
    segments: Sequence[Segment], start: float, end: float
) -> list[Segment]:
    return [s for s in segments if s.end > start and s.start < end]
