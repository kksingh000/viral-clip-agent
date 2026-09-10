"""Two-speaker split framing, and the unusable-material extension limits."""

from __future__ import annotations

import pytest

from app.models.enums import CropMode
from app.services.clip_selection import (
    ExtensionLimits,
    extension_limits,
    snap_boundaries,
)
from app.services.segmentation import segment_transcript
from app.video.tracking import plan_split_crop
from tests.fixtures.transcripts import build_transcript
from tests.unit.test_video import frames

TRANSCRIPT_WITH_OUTRO = """\
Okay so welcome back to the channel, please subscribe before we start.
Most people completely misunderstand how compounding works. They think it is \
about patience. It is not. It is about survival.
The reason is simple. Compounding only works if you never interrupt it, and \
almost everyone interrupts it.
Anyway that is basically all I wanted to say about that today.
"""


class TestSplitCropPlanning:
    def test_two_speakers_produce_two_windows(self):
        positions = [[(400.0, 500.0), (1500.0, 500.0)]] * 16
        plan = plan_split_crop(
            frames(positions), source_width=1920, source_height=1080
        )
        assert plan is not None
        assert len(plan.windows) == 2

    def test_windows_match_the_half_frame_aspect(self):
        """Each window fills half of a 1080x1920 output, so it must already be
        1080:960 -- otherwise the scale step distorts faces."""
        positions = [[(400.0, 500.0), (1500.0, 500.0)]] * 16
        plan = plan_split_crop(
            frames(positions), source_width=1920, source_height=1080
        )
        assert plan is not None
        expected = 1080 / 960
        for window in plan.windows:
            assert window.width % 2 == 0 and window.height % 2 == 0
            assert abs(window.width / window.height - expected) < 0.02

    def test_windows_stay_inside_the_source_frame(self):
        # Speakers pushed hard against both edges.
        positions = [[(60.0, 500.0), (1860.0, 500.0)]] * 16
        plan = plan_split_crop(
            frames(positions), source_width=1920, source_height=1080
        )
        assert plan is not None
        for window in plan.windows:
            assert 0 <= window.x <= 1920 - window.width
            assert 0 <= window.y <= 1080 - window.height

    def test_left_speaker_is_placed_on_top(self):
        positions = [[(1500.0, 500.0), (400.0, 500.0)]] * 16
        plan = plan_split_crop(
            frames(positions), source_width=1920, source_height=1080
        )
        assert plan is not None
        assert plan.windows[0].x < plan.windows[1].x

    def test_a_single_speaker_is_not_a_split_screen(self):
        plan = plan_split_crop(
            frames([[(900.0, 500.0)]] * 16), source_width=1920, source_height=1080
        )
        assert plan is None

    def test_no_faces_is_not_a_split_screen(self):
        assert plan_split_crop([], source_width=1920, source_height=1080) is None

    def test_two_tracks_of_the_same_person_are_rejected(self):
        """A detection that breaks and restarts must not render one face twice."""
        positions = [[(900.0, 500.0)]] * 8 + [[(930.0, 505.0)]] * 8
        plan = plan_split_crop(
            frames(positions),
            source_width=1920,
            source_height=1080,
            # Force two tracks by cutting between the runs.
            scene_boundaries=[8 / 3],
        )
        assert plan is None

    def test_plan_serialises_for_storage(self):
        positions = [[(400.0, 500.0), (1500.0, 500.0)]] * 16
        plan = plan_split_crop(
            frames(positions), source_width=1920, source_height=1080
        )
        assert plan is not None
        payload = plan.to_dict()
        assert payload["layout"] == "split_speakers"
        assert len(payload["windows"]) == 2


class TestFramingService:
    """``build_framing`` must never silently produce something other than the
    mode that was asked for."""

    def _framing(self, monkeypatch, mode: CropMode, positions):
        from app.services import clips as clip_service

        monkeypatch.setattr(
            clip_service, "_sample_frames", lambda *a, **k: frames(positions)
        )
        return clip_service.build_framing(
            __import__("pathlib").Path("unused.mp4"),
            start=0.0,
            end=10.0,
            crop_mode=mode,
            width=1920,
            height=1080,
            target_width=1080,
            target_height=1920,
        )

    def test_split_mode_with_two_speakers_yields_a_split_plan(self, monkeypatch):
        framing = self._framing(
            monkeypatch,
            CropMode.SPLIT_SPEAKERS,
            [[(400.0, 500.0), (1500.0, 500.0)]] * 16,
        )
        assert framing.split is not None
        assert framing.crop is None
        assert framing.face_driven is True

    def test_split_mode_with_one_speaker_falls_back_and_says_so(self, monkeypatch):
        framing = self._framing(
            monkeypatch, CropMode.SPLIT_SPEAKERS, [[(900.0, 500.0)]] * 16
        )
        assert framing.split is None
        assert framing.crop is not None
        assert framing.mode is CropMode.SMART
        assert any("one speaker" in note for note in framing.notes)

    def test_blur_pad_needs_no_window(self, monkeypatch):
        framing = self._framing(monkeypatch, CropMode.BLUR_PAD, [])
        assert framing.crop is None and framing.split is None
        assert framing.to_dict()["layout"] == "blur_pad"

    def test_centre_mode_is_static_and_not_face_driven(self, monkeypatch):
        framing = self._framing(
            monkeypatch, CropMode.CENTER, [[(400.0, 500.0)]] * 16
        )
        assert framing.crop is not None
        assert framing.crop.is_static
        assert framing.face_driven is False


class TestExtensionLimits:
    @pytest.fixture(scope="class")
    def parts(self):
        transcript = build_transcript(TRANSCRIPT_WITH_OUTRO)
        return transcript, segment_transcript(transcript)

    def test_boilerplate_and_outro_are_both_detected(self, parts):
        _, segments = parts
        intro = segments[0]
        outro = segments[-1]
        assert intro.signals.text.boilerplate_hits > 0
        assert outro.signals.text.filler_ratio > 0.35

    def test_limits_bracket_a_span(self, parts):
        _, segments = parts
        middle = segments[len(segments) // 2]
        limits = extension_limits(segments, middle.start, middle.end)
        assert limits.earliest_start is not None
        assert limits.latest_end is not None
        assert limits.earliest_start <= middle.start
        assert limits.latest_end >= middle.end

    def test_forward_growth_stops_before_the_outro(self, parts):
        """Regression: reaching the minimum duration used to justify swallowing
        a rambling sign-off."""
        transcript, segments = parts
        outro = segments[-1]
        target = segments[-2]

        limits = extension_limits(segments, target.start, target.end)
        span = snap_boundaries(
            target.start,
            target.end,
            transcript.words,
            min_duration=40.0,  # unreachable without eating the outro
            max_duration=60.0,
            total_duration=transcript.duration,
            limits=limits,
        )
        assert span.end <= outro.start + 0.5
        assert "wanted to say" not in span.text
        assert any("unusable" in note for note in span.adjustments)

    def test_backward_growth_stops_after_the_intro(self, parts):
        transcript, segments = parts
        intro = segments[0]
        target = segments[1]

        limits = extension_limits(segments, target.start, target.end)
        span = snap_boundaries(
            target.start,
            target.end,
            transcript.words,
            min_duration=40.0,
            max_duration=60.0,
            total_duration=transcript.duration,
            limits=limits,
        )
        assert span.start >= intro.end - 0.5
        assert "subscribe" not in span.text.lower()

    def test_without_limits_the_span_still_grows(self, parts):
        """The limits are what stop growth -- not some other guard."""
        transcript, segments = parts
        target = segments[-2]
        span = snap_boundaries(
            target.start,
            target.end,
            transcript.words,
            min_duration=40.0,
            max_duration=60.0,
            total_duration=transcript.duration,
            limits=ExtensionLimits(),
        )
        assert span.duration > (target.end - target.start)

    def test_no_unusable_segments_means_no_limits(self):
        transcript = build_transcript(
            "Compounding only works if you never interrupt it. "
            "One bad year erases nine good ones."
        )
        segments = segment_transcript(transcript)
        limits = extension_limits(segments, 0.0, transcript.duration)
        assert limits.earliest_start is None
        assert limits.latest_end is None
