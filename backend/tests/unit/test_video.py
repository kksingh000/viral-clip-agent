"""Framing, captions and media probing."""

from __future__ import annotations

import pytest

from app.providers.base import FaceBox, FrameAnalysis
from app.video.captions import (
    CaptionStyleConfig,
    CaptionWord,
    band_geometry,
    build_cues,
    build_states,
    words_from_segments,
)
from app.models.enums import CaptionPosition, CaptionStyle
from app.video.tracking import (
    TrackingConfig,
    build_tracks,
    choose_targets,
    plan_crop,
    smooth_path,
)


def frames(
    positions: list[list[tuple[float, float]]],
    *,
    width: int = 1920,
    height: int = 1080,
    fps: float = 3.0,
    face_size: float = 200.0,
    activity: list[list[float]] | None = None,
) -> list[FrameAnalysis]:
    """Build a frame series from per-frame face centre positions."""
    out: list[FrameAnalysis] = []
    for index, centres in enumerate(positions):
        faces = [
            FaceBox(
                x=cx - face_size / 2,
                y=cy - face_size / 2,
                width=face_size,
                height=face_size,
                confidence=0.9,
                mouth_activity=(
                    activity[index][face_index] if activity else 0.0
                ),
            )
            for face_index, (cx, cy) in enumerate(centres)
        ]
        out.append(
            FrameAnalysis(
                timestamp=index / fps,
                width=width,
                height=height,
                faces=faces,
            )
        )
    return out


class TestCropGeometry:
    def test_landscape_source_yields_an_exact_9_16_window(self):
        plan = plan_crop([], source_width=1920, source_height=1080)
        assert plan.crop_height == 1080
        assert abs(plan.crop_width / plan.crop_height - 9 / 16) < 0.01
        assert plan.crop_width % 2 == 0 and plan.crop_height % 2 == 0

    def test_already_vertical_source_is_left_alone(self):
        plan = plan_crop([], source_width=1080, source_height=1920)
        assert (plan.crop_width, plan.crop_height) == (1080, 1920)

    def test_taller_than_9_16_source_crops_vertically(self):
        plan = plan_crop([], source_width=1000, source_height=2400)
        assert plan.crop_width == 1000
        assert plan.crop_height <= 2400
        assert abs(plan.crop_width / plan.crop_height - 9 / 16) < 0.02

    def test_no_faces_produces_a_static_centred_crop(self):
        plan = plan_crop([], source_width=1920, source_height=1080)
        assert plan.face_driven is False
        assert plan.is_static
        assert plan.keyframes[0].x == pytest.approx((1920 - plan.crop_width) / 2)
        assert any("no faces" in note for note in plan.notes)

    def test_crop_window_never_leaves_the_frame(self):
        positions = [[(50.0, 500.0)]] * 5 + [[(1870.0, 500.0)]] * 15
        plan = plan_crop(frames(positions), source_width=1920, source_height=1080)
        for keyframe in plan.keyframes:
            assert 0 <= keyframe.x <= 1920 - plan.crop_width
            assert 0 <= keyframe.y <= 1080 - plan.crop_height


class TestTracking:
    def test_a_steady_face_forms_one_track(self):
        tracks = build_tracks(frames([[(900.0, 500.0)]] * 12))
        assert len(tracks) == 1
        assert len(tracks[0].samples) == 12

    def test_two_faces_form_two_tracks(self):
        tracks = build_tracks(frames([[(500.0, 500.0), (1400.0, 500.0)]] * 12))
        assert len(tracks) == 2

    def test_a_scene_cut_terminates_tracks(self):
        """The same screen position across an edit is a different person."""
        positions = [[(900.0, 500.0)]] * 12
        tracks = build_tracks(
            frames(positions), scene_boundaries=[2.0]
        )
        assert len(tracks) >= 2

    def test_the_talking_face_is_framed(self):
        """Mouth activity, not size or position, decides the subject."""
        positions = [[(400.0, 500.0), (1500.0, 500.0)]] * 20
        # The right-hand face is the one moving its mouth.
        activity = [[0.2, 9.0]] * 20
        analysed = frames(positions, activity=activity)
        targets, _ = choose_targets(analysed, build_tracks(analysed))
        chosen = [x for _, x in targets if x is not None]
        assert chosen, "expected a subject to be chosen"
        assert all(x > 1000 for x in chosen)

    def test_hysteresis_prevents_ping_ponging(self):
        """Alternating marginal advantage must not flip the crop every frame."""
        positions = [[(400.0, 500.0), (1500.0, 500.0)]] * 24
        activity = [[9.0, 8.6] if i % 2 == 0 else [8.6, 9.0] for i in range(24)]
        analysed = frames(positions, activity=activity)
        _, switches = choose_targets(analysed, build_tracks(analysed))
        assert switches <= 1


class TestSmoothing:
    def test_pan_speed_is_limited(self):
        config = TrackingConfig()
        targets = [(0.0, 100.0)] + [(i / 3, 1800.0) for i in range(1, 20)]
        path = smooth_path(
            targets, frame_width=1920, crop_width=608, config=config
        )
        for (t0, x0), (t1, x1) in zip(path, path[1:]):
            allowed = config.max_velocity * 1920 * (t1 - t0) + 1e-6
            assert abs(x1 - x0) <= allowed

    def test_small_jitter_is_ignored(self):
        targets = [(i / 3, 960.0 + (5 if i % 2 else -5)) for i in range(20)]
        path = smooth_path(targets, frame_width=1920, crop_width=608)
        movement = max(x for _, x in path) - min(x for _, x in path)
        assert movement < 10

    def test_outliers_are_filtered(self):
        targets = [(i / 3, 960.0) for i in range(20)]
        targets[10] = (10 / 3, 100.0)  # single-frame detection glitch
        path = smooth_path(targets, frame_width=1920, crop_width=608)
        movement = max(x for _, x in path) - min(x for _, x in path)
        assert movement < 60

    def test_a_scene_change_cuts_rather_than_pans(self):
        targets = [(i / 3, 300.0) for i in range(6)] + [
            (i / 3, 1600.0) for i in range(6, 12)
        ]
        path = smooth_path(
            targets,
            frame_width=1920,
            crop_width=608,
            scene_boundaries=[6 / 3],
        )
        jumps = [abs(b - a) for (_, a), (_, b) in zip(path, path[1:])]
        assert max(jumps) > 500, "expected a hard cut at the scene boundary"

    def test_missing_detections_hold_the_last_position(self):
        targets = [(0.0, 400.0), (0.33, None), (0.66, None), (1.0, 400.0)]
        path = smooth_path(targets, frame_width=1920, crop_width=608)
        assert len({round(x) for _, x in path}) <= 2


class TestCaptions:
    def _words(self, text: str, per: float = 0.35) -> list[CaptionWord]:
        return [
            CaptionWord(word, index * per, (index + 1) * per - 0.03)
            for index, word in enumerate(text.split())
        ]

    def test_cues_respect_the_word_limit(self):
        config = CaptionStyleConfig.for_style(CaptionStyle.WORD_HIGHLIGHT)
        cues = build_cues(
            self._words("one two three four five six seven eight nine ten"),
            config,
            canvas_width=1080,
        )
        assert cues
        for cue in cues:
            for line in cue.lines:
                assert len(line.words) <= config.max_words_per_line

    def test_sentence_punctuation_breaks_a_line(self):
        config = CaptionStyleConfig.for_style(CaptionStyle.NORMAL)
        cues = build_cues(
            self._words("this ends here. and this begins"), config, canvas_width=1080
        )
        first_line = cues[0].lines[0]
        assert first_line.text.endswith(".")

    def test_a_long_pause_starts_a_new_cue(self):
        words = [
            CaptionWord("before", 0.0, 0.4),
            CaptionWord("after", 3.0, 3.4),  # 2.6s gap
        ]
        cues = build_cues(
            words, CaptionStyleConfig.for_style(CaptionStyle.NORMAL), canvas_width=1080
        )
        assert len(cues) == 2

    def test_word_highlight_emits_one_state_per_word(self):
        config = CaptionStyleConfig.for_style(CaptionStyle.WORD_HIGHLIGHT)
        words = self._words("alpha beta gamma delta")
        cues = build_cues(words, config, canvas_width=1080)
        states = build_states(cues, config, clip_duration=2.0)
        with_cue = [s for s in states if s.cue is not None]
        assert len(with_cue) == len(words)
        assert [s.active_word for s in with_cue] == list(range(len(words)))

    def test_static_styles_emit_one_state_per_cue(self):
        config = CaptionStyleConfig.for_style(CaptionStyle.NORMAL)
        words = self._words("alpha beta gamma delta")
        cues = build_cues(words, config, canvas_width=1080)
        states = build_states(cues, config, clip_duration=2.0)
        assert len([s for s in states if s.cue is not None]) == len(cues)

    def test_states_cover_the_whole_clip_without_gaps(self):
        config = CaptionStyleConfig.for_style(CaptionStyle.WORD_HIGHLIGHT)
        words = self._words("alpha beta gamma")
        cues = build_cues(words, config, canvas_width=1080)
        states = build_states(cues, config, clip_duration=5.0)
        assert states[0].start == pytest.approx(0.0, abs=0.05)
        assert states[-1].end == pytest.approx(5.0, abs=0.05)
        for previous, current in zip(states, states[1:]):
            assert previous.end == pytest.approx(current.start, abs=0.01)

    def test_states_never_run_past_the_clip(self):
        config = CaptionStyleConfig.for_style(CaptionStyle.WORD_HIGHLIGHT)
        words = self._words("alpha beta gamma delta epsilon zeta", per=1.0)
        cues = build_cues(words, config, canvas_width=1080)
        states = build_states(cues, config, clip_duration=2.0)
        assert all(state.end <= 2.0001 for state in states)

    @pytest.mark.parametrize("position", list(CaptionPosition))
    def test_band_stays_inside_the_safe_area(self, position):
        config = CaptionStyleConfig.for_style(CaptionStyle.NORMAL, position)
        band = band_geometry(config, 1080, 1920, config.max_lines)
        assert band.y_offset >= 0
        assert band.y_offset + band.height <= 1920

    def test_words_are_rebased_onto_the_clip(self):
        segments = [
            {
                "words": [
                    {"word": "hello", "start": 30.0, "end": 30.4},
                    {"word": "there", "start": 30.5, "end": 30.9},
                ]
            }
        ]
        words = words_from_segments(segments, offset=30.0)
        assert words[0].start == pytest.approx(0.0)
        assert words[1].end == pytest.approx(0.9)

    def test_zero_length_words_are_given_a_duration(self):
        segments = [{"words": [{"word": "x", "start": 5.0, "end": 5.0}]}]
        words = words_from_segments(segments, offset=0.0)
        assert words[0].end > words[0].start

    def test_empty_input_produces_no_cues(self):
        assert build_cues([], CaptionStyleConfig(), canvas_width=1080) == []
