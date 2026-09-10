"""Short-form clip renderer.

One ffmpeg invocation performs the cut, the 9:16 conversion (with a dynamic,
speaker-following crop), the caption overlay and the optional hook card. Doing
it in a single pass avoids intermediate generation loss and keeps render time
roughly proportional to clip length.

Filter graph (smart crop, captions on):

    [0:v] setpts=PTS-STARTPTS,
          sendcmd=f=crop.cmd,          <- drives the crop below over time
          crop@dyn=CW:CH:X:Y,
          scale=1080:1920,setsar=1,fps=OUT [base]
    [1:v] fps=OUT,format=rgba          [cap]
    [base][cap] overlay=0:BANDY:eof_action=pass [v1]
    [v1][2:v]   overlay=0:HOOKY:enable='between(t,0,3)' [vout]
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from app.core.config import settings
from app.core.errors import FFmpegError, ValidationFailure
from app.core.logging import get_logger
from app.models.enums import CaptionStyle, CropMode
from app.video.captions import (
    CaptionStyleConfig,
    CaptionWord,
    RenderedCaptions,
    render_captions,
)
from app.video.ffmpeg import MediaInfo, ffmpeg_capabilities, probe, run_ffmpeg
from app.video.tracking import CropPlan, SplitCropPlan

logger = get_logger(__name__)

DEFAULT_WIDTH = 1080
DEFAULT_HEIGHT = 1920


# ------------------------------------------------------------------- requests
@dataclass(slots=True)
class HookOverlay:
    """Opening text card. The copy must describe the clip truthfully; the
    HookAgent is responsible for that, the renderer only draws it."""

    text: str
    duration: float = 2.6
    font_size: int = 76
    text_color: tuple[int, int, int, int] = (255, 255, 255, 255)
    background_color: tuple[int, int, int, int] = (0, 0, 0, 165)
    position_ratio: float = 0.16
    max_width_ratio: float = 0.84


@dataclass(slots=True)
class RenderRequest:
    source: Path
    output: Path
    start: float
    end: float
    workdir: Path
    crop_mode: CropMode = CropMode.SMART
    crop_plan: CropPlan | None = None
    #: Required when ``crop_mode`` is SPLIT_SPEAKERS; without it the renderer
    #: falls back to single-subject framing rather than silently centre-cropping.
    split_plan: SplitCropPlan | None = None
    caption_words: Sequence[CaptionWord] = ()
    caption_config: CaptionStyleConfig | None = None
    hook: HookOverlay | None = None
    width: int = DEFAULT_WIDTH
    height: int = DEFAULT_HEIGHT
    fps: float | None = None
    crf: int | None = None
    preset: str | None = None
    audio_bitrate: str = "192k"
    normalize_loudness: bool = True

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass(slots=True)
class RenderResult:
    output: Path
    duration: float
    width: int
    height: int
    fps: float | None
    size_bytes: int
    captions: RenderedCaptions | None
    thumbnail: Path | None = None
    subtitle_files: list[Path] = field(default_factory=list)
    filter_graph: str = ""
    warnings: list[str] = field(default_factory=list)
    info: MediaInfo | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "output": str(self.output),
            "duration": round(self.duration, 3),
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "size_bytes": self.size_bytes,
            "warnings": self.warnings,
        }


# --------------------------------------------------------------------- helpers
def _even(value: float) -> int:
    number = int(round(value))
    return number - (number % 2)


def write_sendcmd_script(plan: CropPlan, destination: Path) -> Path:
    """Emit a ``sendcmd`` script that pans the crop window over time."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for keyframe in plan.keyframes:
        # Time 0 is set by the filter's own initialisation, so skip it.
        if keyframe.t <= 0.0005:
            continue
        lines.append(f"{keyframe.t:.3f} crop@dyn x {int(round(keyframe.x))};")
    destination.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return destination


def render_hook_card(
    hook: HookOverlay, width: int, height: int, destination: Path
) -> tuple[Path, int]:
    """Draw the hook card. Returns ``(path, y_offset)``."""
    from PIL import Image, ImageDraw

    from app.video.fonts import detect_script, load_font

    font = load_font(hook.font_size, script=detect_script(hook.text))
    max_width = width * hook.max_width_ratio

    words = hook.text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        try:
            candidate_width = font.getlength(candidate)
        except AttributeError:  # pragma: no cover
            candidate_width = font.getsize(candidate)[0]
        if current and candidate_width > max_width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    lines = lines[:4]

    line_height = int(hook.font_size * 1.25)
    padding = 34
    card_height = len(lines) * line_height + padding * 2
    card_height += card_height % 2

    image = Image.new("RGBA", (width, card_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    try:
        draw.rounded_rectangle(
            (width * 0.05, 0, width * 0.95, card_height),
            radius=26,
            fill=hook.background_color,
        )
    except AttributeError:  # pragma: no cover
        draw.rectangle((width * 0.05, 0, width * 0.95, card_height), fill=hook.background_color)

    for index, line in enumerate(lines):
        try:
            text_width = font.getlength(line)
        except AttributeError:  # pragma: no cover
            text_width = font.getsize(line)[0]
        draw.text(
            ((width - text_width) / 2, padding + index * line_height),
            line,
            font=font,
            fill=hook.text_color,
            stroke_width=3,
            stroke_fill=(0, 0, 0, 220),
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination, "PNG", optimize=True)
    y_offset = int(height * hook.position_ratio)
    return destination, max(0, min(y_offset, height - card_height))


# ------------------------------------------------------------------- rendering
def render_clip(request: RenderRequest) -> RenderResult:
    """Render one vertical short."""
    source = Path(request.source)
    if not source.exists():
        raise ValidationFailure(f"Source video not found: {source}")
    if request.duration <= 0.2:
        raise ValidationFailure(
            f"Clip duration must be positive, got {request.duration:.2f}s."
        )

    workdir = Path(request.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    source_info = probe(source)
    warnings: list[str] = []

    out_width = _even(request.width)
    out_height = _even(request.height)
    out_fps = request.fps or source_info.fps or 30.0
    out_fps = min(60.0, max(24.0, float(out_fps)))

    caption_config = request.caption_config or CaptionStyleConfig()
    captions: RenderedCaptions | None = None
    if request.caption_words and caption_config.style is not CaptionStyle.NONE:
        captions = render_captions(
            request.caption_words,
            config=caption_config,
            canvas_width=out_width,
            canvas_height=out_height,
            clip_duration=request.duration,
            workdir=workdir,
        )
        warnings.extend(captions.warnings)
        if not captions.has_content:
            captions = None

    inputs: list[str] = ["-ss", f"{request.start:.3f}", "-i", str(source)]
    filters: list[str] = []
    caption_index: int | None = None
    hook_index: int | None = None
    next_input = 1

    if captions is not None and captions.concat_file is not None:
        inputs += ["-f", "concat", "-safe", "0", "-i", str(captions.concat_file)]
        caption_index = next_input
        next_input += 1

    hook_y = 0
    if request.hook and request.hook.text.strip():
        hook_path, hook_y = render_hook_card(
            request.hook, out_width, out_height, workdir / "hook.png"
        )
        inputs += ["-i", str(hook_path)]
        hook_index = next_input
        next_input += 1

    # ---------------------------------------------------------- video chain
    chain = ["setpts=PTS-STARTPTS"]
    if request.crop_mode is CropMode.SPLIT_SPEAKERS and request.split_plan is not None:
        # Two static speaker windows scaled into stacked halves. Static by
        # design: a split screen that also pans is disorienting.
        plan = request.split_plan
        half_height = out_height // 2
        half_height -= half_height % 2
        top, bottom = plan.windows[0], plan.windows[1]
        filters.append(
            f"[0:v]{','.join(chain)},split=2[sp0][sp1];"
            f"[sp0]crop={top.width}:{top.height}:{int(top.x)}:{int(top.y)},"
            f"scale={out_width}:{half_height},setsar=1[sptop];"
            f"[sp1]crop={bottom.width}:{bottom.height}:{int(bottom.x)}:{int(bottom.y)},"
            f"scale={out_width}:{half_height},setsar=1[spbot];"
            f"[sptop][spbot]vstack=inputs=2,fps={out_fps:.4f}[base]"
        )
    elif request.crop_mode is CropMode.BLUR_PAD or request.crop_plan is None:
        if request.crop_mode is CropMode.SMART and request.crop_plan is None:
            warnings.append("no crop plan supplied; falling back to blurred padding")
        filters.append(
            f"[0:v]{','.join(chain)},split=2[bgsrc][fgsrc];"
            f"[bgsrc]scale={out_width}:{out_height}:force_original_aspect_ratio=increase,"
            f"crop={out_width}:{out_height},gblur=sigma=28[bg];"
            f"[fgsrc]scale={out_width}:-2:force_original_aspect_ratio=decrease[fg];"
            f"[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1,fps={out_fps:.4f}[base]"
        )
    else:
        plan = request.crop_plan
        crop_width = _even(plan.crop_width)
        crop_height = _even(plan.crop_height)
        first = plan.keyframes[0] if plan.keyframes else None
        start_x = int(round(first.x)) if first else int((plan.source_width - crop_width) / 2)
        crop_y = int(round(first.y)) if first else int((plan.source_height - crop_height) / 2)

        use_sendcmd = (
            request.crop_mode is CropMode.SMART
            and not plan.is_static
            and ffmpeg_capabilities().get("sendcmd", False)
        )
        if use_sendcmd:
            script = write_sendcmd_script(plan, workdir / "crop.cmd")
            escaped = _escape_filter_path(script)
            chain.append(f"sendcmd=f={escaped}")
        elif request.crop_mode is CropMode.SMART and not plan.is_static:
            warnings.append(
                "this ffmpeg build has no sendcmd filter; using a static crop"
            )
        chain.append(f"crop@dyn={crop_width}:{crop_height}:{start_x}:{crop_y}")
        chain.append(f"scale={out_width}:{out_height}")
        chain.append("setsar=1")
        chain.append(f"fps={out_fps:.4f}")
        filters.append(f"[0:v]{','.join(chain)}[base]")

    current_label = "base"
    if caption_index is not None and captions is not None:
        filters.append(
            f"[{caption_index}:v]fps={out_fps:.4f},format=rgba[cap]"
        )
        filters.append(
            f"[{current_label}][cap]overlay=0:{captions.band.y_offset}:"
            f"eof_action=pass:format=auto[vcap]"
        )
        current_label = "vcap"
    if hook_index is not None and request.hook is not None:
        end = max(0.4, min(request.hook.duration, request.duration))
        filters.append(
            f"[{current_label}][{hook_index}:v]overlay=0:{hook_y}:"
            f"enable='between(t,0,{end:.2f})':format=auto[vhook]"
        )
        current_label = "vhook"

    filter_complex = ";".join(filters)

    # ---------------------------------------------------------- audio chain
    audio_args: list[str] = []
    if source_info.has_audio:
        audio_filters = ["asetpts=PTS-STARTPTS"]
        if request.normalize_loudness:
            audio_filters.append("loudnorm=I=-14:TP=-1.5:LRA=11")
        filter_complex += f";[0:a]{','.join(audio_filters)}[aout]"
        audio_args = [
            "-map", "[aout]",
            "-c:a", "aac",
            "-b:a", request.audio_bitrate,
            "-ar", "48000",
            "-ac", "2",
        ]
    else:
        warnings.append("source has no audio track")
        audio_args = ["-an"]

    output = Path(request.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    args = [
        "-y",
        *inputs,
        "-t", f"{request.duration:.3f}",
        "-filter_complex", filter_complex,
        "-map", f"[{current_label}]",
        *audio_args,
        "-c:v", "libx264",
        "-preset", request.preset or settings.video_encode_preset,
        "-crf", str(request.crf if request.crf is not None else settings.video_encode_crf),
        "-profile:v", "high",
        "-level", "4.2",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-max_muxing_queue_size", "1024",
        str(output),
    ]

    try:
        run_ffmpeg(args)
    except FFmpegError as exc:
        # A caption overlay is the most fragile input (many PNGs through the
        # concat demuxer). Retry once without it rather than losing the clip.
        if caption_index is not None:
            logger.warning(
                "render failed with captions; retrying without them",
                extra={"error": exc.message},
            )
            fallback = RenderRequest(
                source=request.source,
                output=request.output,
                start=request.start,
                end=request.end,
                workdir=workdir,
                crop_mode=request.crop_mode,
                crop_plan=request.crop_plan,
                split_plan=request.split_plan,
                caption_words=(),
                caption_config=request.caption_config,
                hook=request.hook,
                width=request.width,
                height=request.height,
                fps=request.fps,
                crf=request.crf,
                preset=request.preset,
                audio_bitrate=request.audio_bitrate,
                normalize_loudness=request.normalize_loudness,
            )
            result = render_clip(fallback)
            result.warnings.append("captions were dropped: " + exc.message)
            return result
        raise

    info = probe(output)
    thumbnail = extract_thumbnail(output, workdir / "thumbnail.jpg")
    subtitle_files = [
        path
        for path in (
            captions.srt_path if captions else None,
            captions.ass_path if captions else None,
        )
        if path is not None
    ]
    return RenderResult(
        output=output,
        duration=info.duration,
        width=info.width or out_width,
        height=info.height or out_height,
        fps=info.fps,
        size_bytes=info.size_bytes,
        captions=captions,
        thumbnail=thumbnail,
        subtitle_files=subtitle_files,
        filter_graph=filter_complex,
        warnings=warnings,
        info=info,
    )


def _escape_filter_path(path: Path) -> str:
    """Escape a path for use inside an ffmpeg filter argument.

    Windows drive letters and backslashes both need escaping; forward slashes
    are accepted by ffmpeg on every platform, so normalise to those first.
    """
    text = path.as_posix()
    return text.replace(":", "\\:").replace(",", "\\,").replace("'", "\\'")


def extract_thumbnail(
    video: Path, destination: Path, *, timestamp: float | None = None
) -> Path | None:
    """Grab a representative frame.

    Defaults to 25% in, which avoids the fade-in that often opens a clip.
    """
    try:
        info = probe(video)
        at = timestamp if timestamp is not None else max(0.1, info.duration * 0.25)
        destination.parent.mkdir(parents=True, exist_ok=True)
        run_ffmpeg(
            [
                "-y",
                "-ss", f"{at:.3f}",
                "-i", str(video),
                "-frames:v", "1",
                "-q:v", "3",
                str(destination),
            ],
            timeout=120,
        )
        return destination if destination.exists() else None
    except Exception as exc:  # noqa: BLE001 - a missing thumbnail is not fatal
        logger.warning("thumbnail extraction failed", extra={"error": str(exc)})
        return None


def cut_segment(
    source: Path, destination: Path, start: float, end: float, *, copy: bool = True
) -> Path:
    """Extract a segment without re-framing.

    Stream copy is fast but snaps to keyframes; ``copy=False`` re-encodes for a
    frame-accurate cut. Used for analysis inputs and for the editor preview.
    """
    duration = max(0.0, end - start)
    if duration <= 0:
        raise ValidationFailure("Segment duration must be positive.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    args = ["-y", "-ss", f"{start:.3f}", "-i", str(source), "-t", f"{duration:.3f}"]
    if copy:
        args += ["-c", "copy", "-avoid_negative_ts", "make_zero"]
    else:
        args += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-c:a", "aac"]
    args.append(str(destination))
    run_ffmpeg(args)
    return destination


def cleanup_workdir(workdir: Path) -> None:
    shutil.rmtree(workdir, ignore_errors=True)
