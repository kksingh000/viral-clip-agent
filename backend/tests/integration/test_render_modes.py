"""Real ffmpeg renders for each framing branch of the filter graph.

The renderer has three distinct video chains (dynamic crop, blurred padding,
split speakers) plus an optional caption overlay. Each is a different
filter graph, so each needs an actual render to prove it composes.
"""

from __future__ import annotations

import pytest

from app.models.enums import CaptionPosition, CaptionStyle, CropMode
from app.video.captions import CaptionStyleConfig, CaptionWord
from app.video.ffmpeg import probe
from app.video.renderer import HookOverlay, RenderRequest, render_clip
from app.video.tracking import CropWindow, SplitCropPlan, plan_crop

pytestmark = [pytest.mark.integration, pytest.mark.slow]

START, END = 12.0, 22.0


def caption_words() -> list[CaptionWord]:
    text = "most people completely misunderstand this and it costs them years"
    words = text.split()
    per = (END - START) / len(words)
    return [
        CaptionWord(word, index * per, (index + 1) * per - 0.03)
        for index, word in enumerate(words)
    ]


@pytest.fixture(scope="module")
def source_info(sample_video_path):
    return probe(sample_video_path)


def render(tmp_path, sample_video_path, **overrides):
    request = RenderRequest(
        source=sample_video_path,
        output=tmp_path / "out.mp4",
        start=START,
        end=END,
        workdir=tmp_path,
        **overrides,
    )
    return render_clip(request)


def assert_vertical(result) -> None:
    assert result.width == 1080 and result.height == 1920
    assert result.duration == pytest.approx(END - START, abs=0.35)
    assert result.size_bytes > 50_000


class TestFramingBranches:
    def test_smart_crop_renders(self, tmp_path, sample_video_path, source_info):
        plan = plan_crop(
            [], source_width=source_info.width, source_height=source_info.height
        )
        result = render(
            tmp_path,
            sample_video_path,
            crop_mode=CropMode.SMART,
            crop_plan=plan,
        )
        assert_vertical(result)
        assert "crop@dyn" in result.filter_graph

    def test_blur_pad_renders(self, tmp_path, sample_video_path):
        result = render(tmp_path, sample_video_path, crop_mode=CropMode.BLUR_PAD)
        assert_vertical(result)
        assert "gblur" in result.filter_graph
        assert "overlay=(W-w)/2" in result.filter_graph

    def test_split_speakers_renders_two_stacked_windows(
        self, tmp_path, sample_video_path, source_info
    ):
        half_aspect = 1080 / 960
        height = min(source_info.height, int(round(source_info.width / half_aspect)))
        width = min(source_info.width, int(round(height * half_aspect)))
        width -= width % 2
        height -= height % 2

        plan = SplitCropPlan(
            source_width=source_info.width,
            source_height=source_info.height,
            windows=[
                CropWindow(x=0, y=0, width=width, height=height),
                CropWindow(
                    x=source_info.width - width, y=0, width=width, height=height
                ),
            ],
            tracks_detected=2,
        )
        result = render(
            tmp_path,
            sample_video_path,
            crop_mode=CropMode.SPLIT_SPEAKERS,
            split_plan=plan,
        )
        assert_vertical(result)
        assert "vstack=inputs=2" in result.filter_graph

    def test_split_without_a_plan_does_not_silently_centre_crop(
        self, tmp_path, sample_video_path, source_info
    ):
        """Regression: SPLIT_SPEAKERS was a valid enum value with no
        implementation, so it rendered a static centre crop and said nothing."""
        plan = plan_crop(
            [], source_width=source_info.width, source_height=source_info.height
        )
        result = render(
            tmp_path,
            sample_video_path,
            crop_mode=CropMode.SPLIT_SPEAKERS,
            crop_plan=plan,
            split_plan=None,
        )
        assert_vertical(result)
        assert "vstack" not in result.filter_graph


class TestCaptionsAndHook:
    def test_captions_and_hook_compose_into_one_pass(
        self, tmp_path, sample_video_path, source_info
    ):
        plan = plan_crop(
            [], source_width=source_info.width, source_height=source_info.height
        )
        result = render(
            tmp_path,
            sample_video_path,
            crop_mode=CropMode.SMART,
            crop_plan=plan,
            caption_words=caption_words(),
            caption_config=CaptionStyleConfig.for_style(
                CaptionStyle.WORD_HIGHLIGHT, CaptionPosition.LOWER_THIRD
            ),
            hook=HookOverlay(text="Most people get this backwards"),
        )
        assert_vertical(result)
        assert result.captions is not None and result.captions.has_content
        # Two overlays chained onto the base in a single invocation.
        assert result.filter_graph.count("overlay=") >= 2
        assert result.subtitle_files, "an .srt/.ass sidecar should be written"
        assert result.thumbnail is not None and result.thumbnail.exists()

    def test_caption_cues_stay_inside_the_clip(
        self, tmp_path, sample_video_path, source_info
    ):
        plan = plan_crop(
            [], source_width=source_info.width, source_height=source_info.height
        )
        result = render(
            tmp_path,
            sample_video_path,
            crop_mode=CropMode.SMART,
            crop_plan=plan,
            caption_words=caption_words(),
            caption_config=CaptionStyleConfig.for_style(CaptionStyle.KARAOKE),
        )
        assert result.captions is not None
        for cue in result.captions.cues:
            assert cue.start >= -0.01
            assert cue.end <= (END - START) + 0.35

    def test_caption_style_none_renders_without_an_overlay(
        self, tmp_path, sample_video_path, source_info
    ):
        plan = plan_crop(
            [], source_width=source_info.width, source_height=source_info.height
        )
        result = render(
            tmp_path,
            sample_video_path,
            crop_mode=CropMode.SMART,
            crop_plan=plan,
            caption_words=caption_words(),
            caption_config=CaptionStyleConfig.for_style(CaptionStyle.NONE),
        )
        assert_vertical(result)
        assert result.captions is None
