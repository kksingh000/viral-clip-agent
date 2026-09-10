"""ViralMomentAgent -- finds candidate short-form moments in a transcript.

Cost control (see docs/architecture.md): the whole transcript is never sent to
an expensive model in one piece. Segments are pre-filtered by the deterministic
signal score, then packed into windows sized to
``settings.transcript_chunk_chars``. Each window is one model call; results are
merged and de-duplicated afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from pydantic import Field, field_validator

from app.agents.base import (
    AgentContext,
    AgentOutput,
    AgentResult,
    BaseAgent,
    compact_json,
)
from app.ai.prompts.viral_moment import EXAMPLE, SYSTEM
from app.core.config import settings
from app.core.logging import get_logger
from app.services.scoring import ViralDimensions, estimate_dimensions
from app.services.segmentation import Segment

logger = get_logger(__name__)


# ---------------------------------------------------------------- I/O contract
class MomentCandidate(AgentOutput):
    start_time: float = Field(ge=0, description="Seconds from the start of the source.")
    end_time: float = Field(gt=0, description="Seconds from the start of the source.")
    hook: str = Field(
        min_length=3,
        max_length=280,
        description="The opening line, faithful to what is actually said.",
    )
    summary: str = Field(min_length=3, max_length=400)
    reason: str = Field(min_length=3, max_length=600)
    context_lead_in_seconds: float = Field(
        default=0.0,
        ge=0,
        le=30,
        description="How much of the span is setup included purely for context.",
    )
    dimensions: ViralDimensions

    @field_validator("hook", "summary", "reason")
    @classmethod
    def _strip(cls, value: str) -> str:
        return value.strip()


class ViralMomentOutput(AgentOutput):
    candidates: list[MomentCandidate] = Field(default_factory=list, max_length=25)


@dataclass(slots=True)
class MomentSearchPayload:
    video_title: str
    video_duration: float
    segments: list[Segment]
    min_clip_seconds: float = 15.0
    max_clip_seconds: float = 60.0
    max_candidates: int = 8
    creator: str | None = None
    category: str | None = None
    language: str | None = None
    trending_topics: list[str] = field(default_factory=list)
    #: Set when the transcript is structural only (no real words).
    transcript_is_synthetic: bool = False


# ------------------------------------------------------------------- the agent
class ViralMomentAgent(BaseAgent[MomentSearchPayload, ViralMomentOutput]):
    name = "viral_moment_agent"
    output_model = ViralMomentOutput
    temperature = 0.4

    #: Segments below this deterministic prefilter score are omitted from the
    #: prompt unless doing so would leave too little material. Calibrated
    #: against ``signals.PREFILTER_BASELINE``: the floor sits just above a
    #: neutral span, so only actively-bad material is dropped.
    prefilter_floor = 0.30
    #: Always keep at least this many segments, regardless of the floor.
    min_segments_in_prompt = 12

    # ---------------------------------------------------------------- prompting
    def build_system_prompt(self, payload: MomentSearchPayload) -> str:
        return SYSTEM + "\n" + EXAMPLE

    def build_user_content(self, payload: MomentSearchPayload) -> str:
        return compact_json(
            {
                "video": {
                    "title": payload.video_title,
                    "creator": payload.creator,
                    "category": payload.category,
                    "duration_seconds": round(payload.video_duration, 2),
                    "language": payload.language,
                },
                "constraints": {
                    "min_clip_seconds": payload.min_clip_seconds,
                    "max_clip_seconds": payload.max_clip_seconds,
                    "max_candidates": payload.max_candidates,
                },
                "trending_topics": payload.trending_topics,
                "segments": [_segment_payload(s) for s in payload.segments],
            }
        )

    # --------------------------------------------------------------- execution
    def run_windowed(
        self, payload: MomentSearchPayload, *, context: AgentContext | None = None
    ) -> AgentResult[ViralMomentOutput]:
        """Run over the whole transcript, one model call per window."""
        context = context or AgentContext()

        if payload.transcript_is_synthetic:
            message = (
                "transcript is structural only (no transcription provider); "
                "moment detection cannot read what is said"
            )
            context.warn(message)
            logger.warning("skipping model call", extra={"agent": self.name})
            result = AgentResult(
                output=self.fallback(payload),
                used_fallback=True,
                attempts=0,
                model="deterministic",
                warnings=[message],
            )
            return result

        windows = self._windows(payload)
        if len(windows) <= 1:
            return self.run(payload, context=context)

        logger.info(
            "running moment detection in windows",
            extra={"agent": self.name, "windows": len(windows)},
        )
        merged: list[MomentCandidate] = []
        warnings: list[str] = []
        used_fallback = True
        attempts = 0
        model = "deterministic"
        for index, window in enumerate(windows):
            sub = MomentSearchPayload(
                video_title=payload.video_title,
                video_duration=payload.video_duration,
                segments=window,
                min_clip_seconds=payload.min_clip_seconds,
                max_clip_seconds=payload.max_clip_seconds,
                max_candidates=max(2, payload.max_candidates // len(windows) + 1),
                creator=payload.creator,
                category=payload.category,
                language=payload.language,
                trending_topics=payload.trending_topics,
            )
            result = self.run(sub, context=context)
            merged.extend(result.output.candidates)
            warnings.extend(result.warnings)
            attempts += result.attempts
            used_fallback = used_fallback and result.used_fallback
            if not result.used_fallback:
                model = result.model
            logger.debug(
                "window complete",
                extra={"window": index, "candidates": len(result.output.candidates)},
            )

        merged.sort(key=lambda c: _mean_dimension(c), reverse=True)
        return AgentResult(
            output=ViralMomentOutput(candidates=merged[: payload.max_candidates * 2]),
            used_fallback=used_fallback,
            attempts=attempts,
            model=model,
            usage=context.usage,
            cost_usd=context.cost_usd,
            warnings=warnings,
        )

    def _windows(self, payload: MomentSearchPayload) -> list[list[Segment]]:
        """Pack pre-filtered segments into character-bounded windows.

        Windows overlap by one segment so a moment straddling a boundary is
        still visible to one of the calls in full.
        """
        segments = self._prefilter(payload.segments)
        budget = max(2000, settings.transcript_chunk_chars)
        windows: list[list[Segment]] = []
        current: list[Segment] = []
        size = 0
        for segment in segments:
            cost = len(segment.text) + 220  # rough per-segment JSON overhead
            if current and size + cost > budget:
                windows.append(current)
                current = [current[-1]]
                size = len(current[0].text) + 220
            current.append(segment)
            size += cost
        if current:
            windows.append(current)
        return windows or [[]]

    def _prefilter(self, segments: Sequence[Segment]) -> list[Segment]:
        """Drop segments the deterministic signals say are hopeless.

        Neighbours of a strong segment are kept even when weak on their own,
        because setup often scores low while being necessary.
        """
        if len(segments) <= self.min_segments_in_prompt:
            return list(segments)

        scores = [
            (s.signals.text.prefilter_score if s.signals else 0.0) for s in segments
        ]
        keep = {
            index
            for index, score in enumerate(scores)
            if score >= self.prefilter_floor
        }
        for index in list(keep):
            keep.add(max(0, index - 1))
            keep.add(min(len(segments) - 1, index + 1))

        if len(keep) < self.min_segments_in_prompt:
            ranked = sorted(range(len(segments)), key=lambda i: scores[i], reverse=True)
            keep.update(ranked[: self.min_segments_in_prompt])

        kept = [segments[i] for i in sorted(keep)]
        if len(kept) < len(segments):
            logger.info(
                "prefiltered segments before model call",
                extra={"kept": len(kept), "total": len(segments)},
            )
        return kept

    # ---------------------------------------------------------------- fallback
    def fallback(self, payload: MomentSearchPayload) -> ViralMomentOutput:
        """Deterministic candidate generation.

        Greedily grows a window around each high-signal segment until it falls
        inside the duration constraints and ends on a sentence boundary. The
        result is genuinely usable -- it is the same machinery that pre-filters
        for the model -- but it cannot judge meaning, and every candidate is
        marked as such in its ``reason``.
        """
        segments = payload.segments
        if not segments:
            return ViralMomentOutput(candidates=[])

        # Unusable material must not seed a candidate: growing outwards from
        # "please subscribe" produces a clip that opens on the ask, and growing
        # outwards from a rambling sign-off produces one that ends on it.
        seeds = [s for s in segments if not _is_unusable(s)]
        ranked = sorted(seeds or list(segments), key=_prior, reverse=True)

        candidates: list[MomentCandidate] = []
        used: list[tuple[float, float]] = []
        for seed in ranked:
            if len(candidates) >= payload.max_candidates:
                break
            span = self._grow(seed, segments, payload)
            if span is None:
                continue
            start, end, text = span
            if any(_overlap_ratio((start, end), other) > 0.3 for other in used):
                continue

            duration = end - start
            dimensions = estimate_dimensions(
                seed.signals if seed.signals else _empty_signals(text),
                duration=duration,
            )
            hook = _first_sentence(text) or text[:160]
            candidates.append(
                MomentCandidate(
                    start_time=round(start, 3),
                    end_time=round(end, 3),
                    hook=hook[:280],
                    summary=(text[:240] + ("..." if len(text) > 240 else "")),
                    reason=(
                        "Selected by the deterministic signal model (no LLM "
                        "available): high hook/payoff marker density and a "
                        "clean sentence boundary at both ends."
                    ),
                    context_lead_in_seconds=max(0.0, seed.start - start),
                    dimensions=dimensions,
                )
            )
            used.append((start, end))
        return ViralMomentOutput(candidates=candidates)

    def _grow(
        self,
        seed: Segment,
        segments: Sequence[Segment],
        payload: MomentSearchPayload,
    ) -> tuple[float, float, str] | None:
        """Expand around ``seed`` until the duration constraints are met."""
        order = {s.index: position for position, s in enumerate(segments)}
        position = order.get(seed.index)
        if position is None:
            return None

        left = right = position
        start, end = seed.start, seed.end
        text_parts = [seed.text]
        seed_prior = _prior(seed)

        def acceptable(neighbour: Segment, *, backwards: bool) -> bool:
            """Refuse to absorb material that would poison the clip.

            Forward and backward extension are not symmetric. Extending
            *forward* completes the thought, and a payoff legitimately carries
            fewer attention markers than the hook that set it up -- so only
            boilerplate is refused. Extending *backward* adds material the
            viewer did not ask for, so it must also be of comparable quality,
            or a rambling intro gets glued onto the front of the clip.
            """
            if _is_unusable(neighbour):
                return False
            if backwards:
                return _prior(neighbour) >= seed_prior - MAX_PRIOR_DROP
            return True

        while end - start < payload.min_clip_seconds:
            grew = False
            # Prefer extending forwards: a payoff usually follows the hook.
            if right + 1 < len(segments):
                nxt = segments[right + 1]
                if nxt.end - start <= payload.max_clip_seconds and acceptable(
                    nxt, backwards=False
                ):
                    right += 1
                    end = nxt.end
                    text_parts.append(nxt.text)
                    grew = True
            if end - start < payload.min_clip_seconds and left - 1 >= 0:
                prev = segments[left - 1]
                if end - prev.start <= payload.max_clip_seconds and acceptable(
                    prev, backwards=True
                ):
                    left -= 1
                    start = prev.start
                    text_parts.insert(0, prev.text)
                    grew = True
            if not grew:
                break

        duration = end - start
        if duration < min(payload.min_clip_seconds, 8.0):
            return None
        if duration > payload.max_clip_seconds:
            end = start + payload.max_clip_seconds
        return start, end, " ".join(text_parts).strip()


# --------------------------------------------------------------------- helpers
def _segment_payload(segment: Segment) -> dict[str, Any]:
    signals = segment.signals
    compact: dict[str, Any] = {}
    if signals is not None:
        text = signals.text
        compact = {
            "hook_opener": text.hook_opener,
            "curiosity": text.curiosity_hits,
            "payoff": text.payoff_hits,
            "emotion": text.emotion_hits,
            "advice": text.advice_hits,
            "controversy": text.controversy_hits,
            "questions": text.question_count,
            "filler_ratio": round(text.filler_ratio, 3),
            "opens_with_pronoun": text.starts_with_dangling_reference,
            "leading_pause": round(signals.leading_pause, 2),
            "trailing_pause": round(signals.trailing_pause, 2),
            "prior": round(text.prefilter_score, 3),
        }
    return {
        "index": segment.index,
        "start": round(segment.start, 2),
        "end": round(segment.end, 2),
        "text": segment.text,
        "signals": compact,
    }


#: A neighbouring segment this much weaker than the seed is not worth
#: absorbing just to reach the minimum duration.
MAX_PRIOR_DROP = 0.15

#: A neighbour this full of filler drags the clip down in either direction.
MAX_NEIGHBOUR_FILLER = 0.22


def _prior(segment: Segment) -> float:
    return segment.signals.text.prefilter_score if segment.signals else 0.0


def _is_unusable(segment: Segment) -> bool:
    """Channel boilerplate, or speech that is mostly filler.

    Shares its definition with :func:`app.services.clip_selection.extension_limits`
    so seeding, growth and boundary snapping all agree on what is unusable.
    """
    if segment.signals is None:
        return False
    text = segment.signals.text
    return text.boilerplate_hits > 0 or text.filler_ratio > MAX_NEIGHBOUR_FILLER


def _mean_dimension(candidate: MomentCandidate) -> float:
    values = candidate.dimensions.as_dict().values()
    return sum(values) / len(values) if values else 0.0


def _overlap_ratio(a: tuple[float, float], b: tuple[float, float]) -> float:
    overlap = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    shorter = min(a[1] - a[0], b[1] - b[0])
    return overlap / shorter if shorter > 0 else 0.0


def _first_sentence(text: str) -> str:
    for terminator in (". ", "! ", "? ", "। "):
        index = text.find(terminator)
        if 12 <= index <= 200:
            return text[: index + 1].strip()
    return text[:160].strip()


def _empty_signals(text: str):
    from app.services.signals import analyze_text

    return analyze_text(text)
