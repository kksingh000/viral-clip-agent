"""Boundary snapping, context repair, scoring and de-duplication."""

from __future__ import annotations

import pytest

from app.services.clip_selection import (
    build_scored_candidate,
    deduplicate,
    ensure_standalone_opening,
    snap_boundaries,
)
from app.services.scoring import (
    DEFAULT_WEIGHTS,
    DIMENSION_NAMES,
    ScoringWeights,
    StructuralContext,
    ViralDimensions,
    compute_score,
    estimate_dimensions,
)
from app.services.signals import analyze_text
from tests.fixtures.transcripts import build_transcript

TEXT = """\
Most people completely misunderstand how compounding works. They think it is \
about patience. It is not. It is about survival.
The reason is simple. Compounding only works if you never interrupt it, and \
almost everyone interrupts it.
And that is why it works so well for people who never look at their accounts.
"""


@pytest.fixture(scope="module")
def transcript():
    return build_transcript(TEXT)


def flat_dimensions(value: float = 5.0) -> ViralDimensions:
    return ViralDimensions(**{name: value for name in DIMENSION_NAMES})


class TestSnapBoundaries:
    def test_never_starts_or_ends_mid_word(self, transcript):
        words = transcript.words
        # Deliberately land both ends inside a word.
        target = words[10]
        span = snap_boundaries(
            target.start + (target.end - target.start) / 2,
            words[30].start + 0.05,
            words,
            min_duration=5,
            max_duration=60,
            total_duration=transcript.duration,
        )
        for word in words:
            overlaps_start = word.start < span.start < word.end
            overlaps_end = word.start < span.end < word.end
            assert not overlaps_start, f"start cuts through {word.word!r}"
            assert not overlaps_end, f"end cuts through {word.word!r}"

    def test_extends_to_finish_the_sentence(self, transcript):
        words = transcript.words
        span = snap_boundaries(
            words[0].start,
            words[6].end,  # mid-sentence
            words,
            min_duration=2,
            max_duration=60,
            total_duration=transcript.duration,
        )
        assert span.ends_mid_sentence is False
        assert span.text.rstrip().endswith((".", "!", "?"))

    def test_respects_the_maximum_duration(self, transcript):
        span = snap_boundaries(
            0.0,
            transcript.duration,
            transcript.words,
            min_duration=5,
            max_duration=12,
            total_duration=transcript.duration,
        )
        assert span.duration <= 12.5

    def test_reaches_the_minimum_duration(self, transcript):
        words = transcript.words
        span = snap_boundaries(
            words[0].start,
            words[3].end,
            words,
            min_duration=10,
            max_duration=60,
            total_duration=transcript.duration,
        )
        assert span.duration >= 9.5

    def test_snaps_onto_a_nearby_shot_change(self, transcript):
        words = transcript.words
        rough_start = words[5].start
        span = snap_boundaries(
            rough_start,
            words[25].end,
            words,
            scene_boundaries=[rough_start - 0.15],
            min_duration=5,
            max_duration=60,
            total_duration=transcript.duration,
        )
        assert any("shot change" in note for note in span.adjustments)

    def test_survives_an_empty_word_list(self):
        span = snap_boundaries(1.0, 5.0, [], min_duration=5, max_duration=60)
        assert span.start == 1.0 and span.end == 5.0
        assert span.adjustments


class TestContextRepair:
    def test_extends_back_when_the_opening_dangles(self, transcript):
        words = transcript.words
        dangling = next(
            i for i, w in enumerate(words) if w.word.lower().startswith("and")
        )
        span = snap_boundaries(
            words[dangling].start,
            words[-1].end,
            words,
            min_duration=3,
            max_duration=60,
            total_duration=transcript.duration,
        )
        assert analyze_text(span.text).starts_with_dangling_reference

        repaired = ensure_standalone_opening(span, words, max_duration=60)
        assert repaired.start < span.raw_start
        assert repaired.context_lead_in > 0
        assert any("context" in note for note in repaired.adjustments)

    def test_refuses_to_exceed_the_maximum_duration(self, transcript):
        words = transcript.words
        dangling = next(
            i for i, w in enumerate(words) if w.word.lower().startswith("and")
        )
        span = snap_boundaries(
            words[dangling].start,
            words[-1].end,
            words,
            min_duration=3,
            max_duration=60,
            total_duration=transcript.duration,
        )
        original_start = span.start
        # A maximum barely larger than the current span leaves no room.
        repaired = ensure_standalone_opening(
            span, words, max_duration=span.duration + 0.1
        )
        assert repaired.start == original_start
        assert any("exceed the maximum" in note for note in repaired.adjustments)


class TestScoring:
    def test_default_weights_sum_to_one(self):
        assert sum(DEFAULT_WEIGHTS.values()) == pytest.approx(1.0, abs=1e-9)

    def test_perfect_dimensions_score_100_without_penalties(self):
        result = compute_score(flat_dimensions(10.0))
        assert result.final_score == pytest.approx(100.0, abs=0.01)

    def test_zero_dimensions_score_zero(self):
        assert compute_score(flat_dimensions(0.0)).final_score == 0.0

    def test_structural_penalties_reduce_the_score(self):
        base = compute_score(flat_dimensions(8.0))
        penalised = compute_score(
            flat_dimensions(8.0),
            structure=StructuralContext(
                duration=4.0,
                min_duration=15.0,
                ends_mid_sentence=True,
                starts_with_dangling_reference=True,
            ),
        )
        assert penalised.final_score < base.final_score
        assert "too_short" in penalised.penalties
        assert "ends_mid_sentence" in penalised.penalties

    def test_score_is_always_clamped(self):
        result = compute_score(
            flat_dimensions(0.5),
            structure=StructuralContext(duration=1.0, has_audio=False),
        )
        assert 0.0 <= result.final_score <= 100.0

    def test_partial_weight_overrides_are_renormalised(self):
        weights = ScoringWeights.from_mapping({"hook_strength": 5.0})
        assert sum(weights.weights.values()) == pytest.approx(1.0, abs=1e-9)
        assert weights.weights["hook_strength"] > DEFAULT_WEIGHTS["hook_strength"]

    def test_unknown_and_invalid_weights_are_ignored(self):
        weights = ScoringWeights.from_mapping(
            {"not_a_dimension": 1.0, "curiosity": "abc", "payoff": -3}
        )
        assert "not_a_dimension" not in weights.weights
        assert weights.weights["payoff"] == pytest.approx(
            DEFAULT_WEIGHTS["payoff"], abs=1e-9
        )

    def test_all_zero_weights_fall_back_to_defaults(self):
        weights = ScoringWeights.from_mapping({name: 0 for name in DIMENSION_NAMES})
        assert weights.weights == pytest.approx(DEFAULT_WEIGHTS)

    def test_estimator_ranks_a_strong_hook_above_boilerplate(self):
        strong = estimate_dimensions(
            analyze_text(
                "Most people completely misunderstand this, and the reason is simple."
            ),
            duration=30.0,
        )
        weak = estimate_dimensions(
            analyze_text("Welcome back to the channel, please subscribe."),
            duration=30.0,
        )
        assert strong.hook_strength > weak.hook_strength
        assert strong.shareability > weak.shareability

    def test_estimator_output_is_within_range(self):
        for text in ("", "Subscribe " * 40, "Wow! " * 30):
            dimensions = estimate_dimensions(analyze_text(text), duration=20.0)
            assert all(0 <= v <= 10 for v in dimensions.as_dict().values())


class TestDeduplicate:
    def _candidate(self, start, end, score, text):
        """Build a candidate with an explicit final score.

        Dimensions are 0-10; the 0-100 ``score`` under test is set directly so
        the ranking being asserted does not depend on the weighting maths.
        """
        span = snap_boundaries(start, end, [], min_duration=1, max_duration=120)
        span.text = text
        candidate = build_scored_candidate(
            span, flat_dimensions(5.0), hook="h", summary="s", reason="r"
        )
        candidate.score.final_score = float(score)
        return candidate

    def test_heavily_overlapping_clips_collapse_to_the_better_one(self):
        kept, dropped = deduplicate(
            [
                self._candidate(0, 30, 90, "alpha beta gamma"),
                self._candidate(2, 29, 60, "different words entirely here"),
            ]
        )
        assert len(kept) == 1
        assert kept[0].viral_score == 90
        assert len(dropped) == 1

    def test_similar_text_collapses_even_without_overlap(self):
        shared = "compounding interruption panic selling returns decades"
        kept, _ = deduplicate(
            [
                self._candidate(0, 30, 80, shared),
                self._candidate(200, 230, 70, shared),
            ]
        )
        assert len(kept) == 1

    def test_distinct_clips_are_all_kept(self):
        kept, dropped = deduplicate(
            [
                self._candidate(0, 20, 80, "compounding survival interruption"),
                self._candidate(60, 80, 75, "sourdough starter hydration baking"),
            ]
        )
        assert len(kept) == 2
        assert dropped == []

    def test_empty_input(self):
        kept, dropped = deduplicate([])
        assert kept == [] and dropped == []
