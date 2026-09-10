"""Transcript segmentation and the deterministic signal model."""

from __future__ import annotations

import pytest

from app.services.segmentation import (
    ABSOLUTE_MIN_SECONDS,
    MAX_SEGMENT_SECONDS,
    segment_transcript,
)
from app.services.signals import (
    PREFILTER_BASELINE,
    analyze_text,
    is_sentence_end,
    jaccard,
    keyword_set,
)
from tests.fixtures.transcripts import build_transcript, sample_transcript


class TestSignals:
    def test_hook_opener_is_detected_at_the_start(self):
        signals = analyze_text("Most people completely misunderstand compounding.")
        assert signals.hook_opener is True
        assert signals.prefilter_score > PREFILTER_BASELINE

    def test_boilerplate_scores_below_a_neutral_span(self):
        boilerplate = analyze_text(
            "Welcome back to the channel, please subscribe before we start."
        )
        neutral = analyze_text(
            "The temperature outside was about fifteen degrees that morning."
        )
        assert boilerplate.boilerplate_hits >= 1
        assert boilerplate.prefilter_score < neutral.prefilter_score

    def test_multi_word_fillers_are_counted(self):
        """Regression: phrases like "you know" were in the single-token list
        and could never match, so rambling speech scored as clean."""
        signals = analyze_text(
            "So you know I mean it is kind of like sort of the same thing really"
        )
        assert signals.filler_ratio > 0.3

    def test_dangling_opener_is_flagged(self):
        assert analyze_text("And that is why it works.").starts_with_dangling_reference
        assert analyze_text("It changed everything.").starts_with_dangling_reference
        assert not analyze_text(
            "Compounding changed everything."
        ).starts_with_dangling_reference

    def test_empty_text_is_safe(self):
        signals = analyze_text("   ")
        assert signals.word_count == 0
        assert signals.prefilter_score == 0.0

    def test_scores_stay_in_range(self):
        for text in ("", "a", "Most people " * 200, "Subscribe! " * 50):
            assert 0.0 <= analyze_text(text).prefilter_score <= 1.0

    def test_devanagari_and_hinglish_markers(self):
        signals = analyze_text("Sabse bada mistake jo log karte hain wo ye hai")
        assert signals.hook_opener is True

    def test_sentence_end_supports_the_danda(self):
        assert is_sentence_end("बात।")
        assert is_sentence_end("done.")
        assert not is_sentence_end("and then")

    def test_keyword_overlap(self):
        a = keyword_set("compounding returns investing patience")
        b = keyword_set("compounding returns investing survival")
        assert 0.4 < jaccard(a, b) < 1.0
        assert jaccard(a, set()) == 0.0


class TestSegmentation:
    def test_segments_cover_the_transcript_without_overlap(self):
        transcript = sample_transcript()
        segments = segment_transcript(transcript)

        assert len(segments) > 3
        for previous, current in zip(segments, segments[1:]):
            assert previous.end <= current.start + 1e-6
        assert all(s.duration > 0 for s in segments)
        assert all(s.duration <= MAX_SEGMENT_SECONDS + 1 for s in segments)

    def test_a_strong_boundary_splits_below_the_minimum_length(self):
        """Regression: enforcing a minimum length across a sentence-into-pause
        boundary glued the channel intro onto the actual hook."""
        transcript = sample_transcript()
        segments = segment_transcript(transcript)

        subscribe = [s for s in segments if "subscribe" in s.text.lower()]
        assert subscribe, "expected a segment containing the subscribe ask"
        assert all(
            "misunderstand" not in s.text.lower() for s in subscribe
        ), "the boilerplate and the hook must not share a segment"

    def test_no_segment_is_absurdly_short(self):
        segments = segment_transcript(sample_transcript())
        assert all(s.duration >= ABSOLUTE_MIN_SECONDS - 0.5 for s in segments)

    def test_every_segment_carries_signals(self):
        segments = segment_transcript(sample_transcript())
        assert all(s.signals is not None for s in segments)
        assert all(s.signals.text.word_count > 0 for s in segments)

    def test_scene_boundaries_are_recorded(self):
        transcript = sample_transcript()
        boundary = transcript.chunks[1].start
        segments = segment_transcript(transcript, scene_boundaries=[boundary])
        assert any(s.signals and s.signals.scene_aligned_start for s in segments)

    def test_handles_a_single_short_utterance(self):
        transcript = build_transcript("Hello there.")
        segments = segment_transcript(transcript)
        assert len(segments) == 1
        assert segments[0].text.startswith("Hello")

    def test_words_are_preserved_in_order(self):
        transcript = sample_transcript()
        segments = segment_transcript(transcript)
        flattened = [w.word for s in segments for w in s.words]
        assert flattened == [w.word for w in transcript.words]

    @pytest.mark.parametrize("min_seconds,max_seconds", [(3.0, 20.0), (8.0, 60.0)])
    def test_respects_configured_bounds(self, min_seconds, max_seconds):
        segments = segment_transcript(
            sample_transcript(), min_seconds=min_seconds, max_seconds=max_seconds
        )
        assert all(s.duration <= max_seconds + 1 for s in segments)
