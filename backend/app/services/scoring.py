"""The viral scoring model.

Scoring is deliberately split in two:

* **Dimensions** (0-10 each) are judgements -- "is this hook strong?", "does
  this stand alone?" -- and come from the ViralMomentAgent, or from the
  deterministic estimator below when no model is available.
* **Combination** is arithmetic: a configurable weighted sum plus explicit
  structural penalties, producing 0-100.

Keeping the arithmetic out of the model means the final number is auditable,
reproducible, and adjustable by the analytics feedback loop without touching a
prompt.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from pydantic import BaseModel, Field

from app.services.signals import SegmentSignals, TextSignals

DIMENSION_NAMES = (
    "hook_strength",
    "emotional_intensity",
    "curiosity",
    "standalone_context",
    "payoff",
    "information_density",
    "shareability",
    "comment_potential",
    "visual_interest",
    "speech_quality",
    "trend_relevance",
    "originality",
)

DEFAULT_WEIGHTS: dict[str, float] = {
    "hook_strength": 0.16,
    "curiosity": 0.13,
    "payoff": 0.12,
    "standalone_context": 0.12,
    "emotional_intensity": 0.10,
    "shareability": 0.09,
    "information_density": 0.07,
    "comment_potential": 0.06,
    "visual_interest": 0.05,
    "speech_quality": 0.04,
    "trend_relevance": 0.04,
    "originality": 0.02,
}


class ViralDimensions(BaseModel):
    """The twelve judgement axes, each 0-10."""

    hook_strength: float = Field(
        ge=0, le=10, description="Would the first 2 seconds stop a scroll?"
    )
    emotional_intensity: float = Field(
        ge=0, le=10, description="Strength of the emotion the moment carries."
    )
    curiosity: float = Field(
        ge=0, le=10, description="Does it open a gap the viewer needs closed?"
    )
    standalone_context: float = Field(
        ge=0,
        le=10,
        description="Can someone who never saw the source follow it? 10 = no "
        "outside context needed at all.",
    )
    payoff: float = Field(
        ge=0, le=10, description="Does the ending resolve what the opening promised?"
    )
    information_density: float = Field(
        ge=0, le=10, description="Substance per second; penalise padding."
    )
    shareability: float = Field(
        ge=0, le=10, description="Would a viewer send this to a specific person?"
    )
    comment_potential: float = Field(
        ge=0, le=10, description="Does it invite a reply, disagreement or story?"
    )
    visual_interest: float = Field(
        ge=0, le=10, description="Is there anything to look at beyond a static head?"
    )
    speech_quality: float = Field(
        ge=0, le=10, description="Clarity, pace and absence of filler."
    )
    trend_relevance: float = Field(
        ge=0, le=10, description="Alignment with a currently rising topic."
    )
    originality: float = Field(
        ge=0, le=10, description="Non-obviousness relative to common takes."
    )

    def as_dict(self) -> dict[str, float]:
        return {name: float(getattr(self, name)) for name in DIMENSION_NAMES}


@dataclass(slots=True)
class ScoringWeights:
    weights: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_WEIGHTS)
    )

    @classmethod
    def from_mapping(cls, overrides: Mapping[str, Any] | None) -> "ScoringWeights":
        """Merge user overrides over the defaults, ignoring unknown keys.

        Weights are renormalised so a partial override cannot silently change
        the effective scale of the score.
        """
        merged = dict(DEFAULT_WEIGHTS)
        for key, value in (overrides or {}).items():
            if key in merged:
                try:
                    parsed = float(value)
                except (TypeError, ValueError):
                    continue
                if parsed >= 0:
                    merged[key] = parsed
        total = sum(merged.values())
        if total <= 0:
            merged = dict(DEFAULT_WEIGHTS)
            total = sum(merged.values())
        return cls(weights={k: v / total for k, v in merged.items()})


@dataclass(slots=True)
class ScoreResult:
    final_score: float
    base_score: float
    contributions: dict[str, float]
    penalties: dict[str, float]
    dimensions: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "final_score": round(self.final_score, 2),
            "base_score": round(self.base_score, 2),
            "dimensions": {k: round(v, 2) for k, v in self.dimensions.items()},
            "contributions": {k: round(v, 3) for k, v in self.contributions.items()},
            "penalties": {k: round(v, 2) for k, v in self.penalties.items()},
        }


@dataclass(slots=True)
class StructuralContext:
    """Facts about the cut itself, used for penalties the model cannot see."""

    duration: float
    min_duration: float = 15.0
    max_duration: float = 60.0
    starts_mid_sentence: bool = False
    ends_mid_sentence: bool = False
    starts_with_dangling_reference: bool = False
    silence_ratio: float = 0.0
    has_audio: bool = True


def compute_score(
    dimensions: ViralDimensions,
    *,
    weights: ScoringWeights | None = None,
    structure: StructuralContext | None = None,
) -> ScoreResult:
    """Combine dimensions and structural penalties into a 0-100 score."""
    weights = weights or ScoringWeights()
    values = dimensions.as_dict()
    contributions = {
        name: values[name] * weights.weights.get(name, 0.0) for name in DIMENSION_NAMES
    }
    base = sum(contributions.values()) * 10.0  # 0-10 weighted mean -> 0-100

    penalties: dict[str, float] = {}
    if structure is not None:
        penalties.update(_structural_penalties(structure))

    final = base - sum(penalties.values())
    return ScoreResult(
        final_score=max(0.0, min(100.0, final)),
        base_score=base,
        contributions=contributions,
        penalties=penalties,
        dimensions=values,
    )


def _structural_penalties(structure: StructuralContext) -> dict[str, float]:
    penalties: dict[str, float] = {}

    if structure.duration < structure.min_duration:
        shortfall = structure.min_duration - structure.duration
        penalties["too_short"] = min(20.0, shortfall * 1.5)
    elif structure.duration > structure.max_duration:
        excess = structure.duration - structure.max_duration
        penalties["too_long"] = min(20.0, excess * 0.8)

    if structure.starts_mid_sentence:
        penalties["starts_mid_sentence"] = 6.0
    if structure.ends_mid_sentence:
        penalties["ends_mid_sentence"] = 8.0
    if structure.starts_with_dangling_reference:
        penalties["needs_outside_context"] = 7.0
    if structure.silence_ratio > 0.35:
        penalties["excessive_silence"] = min(12.0, (structure.silence_ratio - 0.35) * 40)
    if not structure.has_audio:
        penalties["no_audio"] = 25.0
    return penalties


# --------------------------------------------------- deterministic estimation
def estimate_dimensions(
    signals: SegmentSignals | TextSignals,
    *,
    duration: float,
    visual_interest: float | None = None,
    trend_relevance: float | None = None,
) -> ViralDimensions:
    """Estimate the dimensions without a model.

    Used for the cheap pre-filter, for the fallback path, and as a sanity
    reference the model's own scores are logged against. It is a prior, not a
    replacement: it cannot judge whether a joke lands.
    """
    text = signals.text if isinstance(signals, SegmentSignals) else signals
    timing = signals if isinstance(signals, SegmentSignals) else None

    def clamp(value: float) -> float:
        return max(0.0, min(10.0, value))

    # Channel boilerplate ("welcome back", "subscribe") is fluent speech that
    # scores well on every positive lexicon, so it is subtracted explicitly.
    boilerplate = min(4.0, 2.0 * text.boilerplate_hits)

    hook = 3.0
    hook += 3.5 if text.hook_opener else 0.0
    hook += min(1.5, text.curiosity_hits * 0.5)
    hook += min(1.0, text.question_count * 0.5)
    hook -= 2.0 if text.starts_with_dangling_reference else 0.0
    hook -= boilerplate

    emotion = 3.0 + min(4.0, text.emotion_hits * 1.1) + min(1.5, text.story_hits * 0.7)
    curiosity = 3.0 + min(4.0, text.curiosity_hits * 1.0) + min(2.0, text.question_count * 0.8)

    standalone = 7.0
    standalone -= 3.0 if text.starts_with_dangling_reference else 0.0
    standalone += 1.0 if text.hook_opener else 0.0
    standalone -= 1.5 if text.word_count < 25 else 0.0
    standalone -= boilerplate * 0.5

    payoff = 3.5 + min(4.0, text.payoff_hits * 1.3)
    payoff -= 2.0 if text.ends_mid_sentence else 0.0

    density = 4.0 + min(3.0, text.number_count * 0.6) + min(2.0, text.advice_hits * 0.8)
    density -= min(3.0, text.filler_ratio * 12)

    share = 3.0 + min(3.0, text.superlative_hits * 0.8) + min(2.0, text.advice_hits * 0.7)
    share += min(1.5, text.second_person_ratio * 10)
    share -= boilerplate

    comment = 3.0 + min(4.0, text.controversy_hits * 1.6) + min(2.0, text.question_count * 0.9)
    comment -= boilerplate * 0.5

    speech = 6.5 - min(4.0, text.filler_ratio * 14)
    if timing is not None and timing.words_per_second:
        # 2.2-3.4 words/second is a comfortable delivery band.
        rate = timing.words_per_second
        if rate < 1.6 or rate > 4.2:
            speech -= 1.5

    # Duration fit: 20-45s is the sweet spot for a self-contained short.
    fit = 1.0 if 20.0 <= duration <= 45.0 else 0.0

    return ViralDimensions(
        hook_strength=clamp(hook),
        emotional_intensity=clamp(emotion),
        curiosity=clamp(curiosity),
        standalone_context=clamp(standalone),
        payoff=clamp(payoff + fit),
        information_density=clamp(density),
        shareability=clamp(share),
        comment_potential=clamp(comment),
        visual_interest=clamp(visual_interest if visual_interest is not None else 5.0),
        speech_quality=clamp(speech),
        trend_relevance=clamp(trend_relevance if trend_relevance is not None else 5.0),
        originality=clamp(4.5 + min(2.0, text.controversy_hits * 0.9)),
    )
