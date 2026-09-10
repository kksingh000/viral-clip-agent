"""Caption layout, styling and rendering.

Captions are rendered as a sequence of RGBA PNG *states* which ffmpeg
concatenates into a single alpha overlay. That approach was chosen over
burning ``.ass`` subtitles because it does not depend on the ffmpeg build
carrying libass, and because word-level highlighting, per-word boxes and
karaoke fills are all just drawing -- no subtitle-format gymnastics.

A ``.srt`` and an ``.ass`` file are still written alongside every clip so the
captions are editable and portable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from app.core.logging import get_logger
from app.models.enums import CaptionPosition, CaptionStyle
from app.video.fonts import detect_script, load_font

logger = get_logger(__name__)

RGBA = tuple[int, int, int, int]


# ------------------------------------------------------------------- structure
@dataclass(slots=True)
class CaptionWord:
    text: str
    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass(slots=True)
class CaptionLine:
    words: list[CaptionWord]

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)

    @property
    def start(self) -> float:
        return self.words[0].start

    @property
    def end(self) -> float:
        return self.words[-1].end


@dataclass(slots=True)
class CaptionCue:
    """What is on screen at one time: one or two lines."""

    lines: list[CaptionLine]

    @property
    def words(self) -> list[CaptionWord]:
        return [w for line in self.lines for w in line.words]

    @property
    def start(self) -> float:
        return self.lines[0].start

    @property
    def end(self) -> float:
        return self.lines[-1].end

    @property
    def text(self) -> str:
        return " ".join(line.text for line in self.lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "text": self.text,
            "lines": [
                {
                    "text": line.text,
                    "start": round(line.start, 3),
                    "end": round(line.end, 3),
                    "words": [
                        {
                            "word": w.text,
                            "start": round(w.start, 3),
                            "end": round(w.end, 3),
                        }
                        for w in line.words
                    ],
                }
                for line in self.lines
            ],
        }


# ----------------------------------------------------------------------- style
@dataclass(slots=True)
class CaptionStyleConfig:
    style: CaptionStyle = CaptionStyle.WORD_HIGHLIGHT
    position: CaptionPosition = CaptionPosition.LOWER_THIRD

    font_path: str | None = None
    font_size: int = 68
    line_spacing: float = 1.22
    max_words_per_line: int = 4
    max_lines: int = 2
    max_line_width_ratio: float = 0.86

    text_color: RGBA = (255, 255, 255, 255)
    highlight_color: RGBA = (255, 214, 10, 255)
    highlight_text_color: RGBA = (17, 17, 17, 255)
    outline_color: RGBA = (0, 0, 0, 235)
    outline_width: int = 6
    shadow_offset: int = 4
    shadow_color: RGBA = (0, 0, 0, 140)

    background_color: RGBA | None = None
    background_padding: int = 22
    background_radius: int = 18

    uppercase: bool = False
    #: Fraction of the frame height kept clear of platform UI at top/bottom.
    safe_area_ratio: float = 0.14

    @classmethod
    def for_style(
        cls,
        style: CaptionStyle,
        position: CaptionPosition = CaptionPosition.LOWER_THIRD,
        **overrides: Any,
    ) -> "CaptionStyleConfig":
        """Presets. Every field remains overridable per user/clip."""
        presets: dict[CaptionStyle, dict[str, Any]] = {
            CaptionStyle.NORMAL: {
                "font_size": 62,
                "max_words_per_line": 5,
                "outline_width": 5,
            },
            CaptionStyle.BOLD: {
                "font_size": 78,
                "max_words_per_line": 3,
                "outline_width": 8,
                "uppercase": True,
            },
            CaptionStyle.KARAOKE: {
                "font_size": 70,
                "max_words_per_line": 4,
                "text_color": (235, 235, 235, 230),
                "highlight_color": (255, 214, 10, 255),
            },
            CaptionStyle.WORD_HIGHLIGHT: {
                "font_size": 72,
                "max_words_per_line": 4,
                "outline_width": 7,
            },
            CaptionStyle.MINIMAL: {
                "font_size": 56,
                "max_words_per_line": 3,
                "max_lines": 1,
                "outline_width": 4,
                "shadow_offset": 2,
            },
            CaptionStyle.PODCAST: {
                "font_size": 58,
                "max_words_per_line": 6,
                "max_lines": 2,
                "outline_width": 0,
                "background_color": (0, 0, 0, 170),
            },
        }
        config = cls(style=style, position=position)
        for key, value in presets.get(style, {}).items():
            setattr(config, key, value)
        for key, value in overrides.items():
            if value is not None and hasattr(config, key):
                setattr(config, key, value)
        return config


# ---------------------------------------------------------------------- layout
#: A pause longer than this ends a caption line even mid-sentence.
LINE_BREAK_PAUSE = 0.45
#: A gap longer than this ends the cue entirely (captions disappear).
CUE_BREAK_PAUSE = 0.9
_SENTENCE_END = (".", "!", "?", "।")  # includes the Devanagari danda


def build_cues(
    words: Sequence[CaptionWord], config: CaptionStyleConfig, *, canvas_width: int
) -> list[CaptionCue]:
    """Group words into lines and cues.

    Breaks happen at, in priority order: long pauses, sentence punctuation,
    the word-count cap, and the measured pixel width of the line.
    """
    if not words:
        return []

    max_pixels = canvas_width * config.max_line_width_ratio
    font = load_font(
        config.font_size,
        script=detect_script(" ".join(w.text for w in words[:40])),
        path=config.font_path,
    )

    lines: list[CaptionLine] = []
    current: list[CaptionWord] = []
    force_cue_break_after: set[int] = set()

    def flush_line() -> None:
        if current:
            lines.append(CaptionLine(words=list(current)))
            current.clear()

    previous_end: float | None = None
    for word in words:
        text = word.text.strip()
        if not text:
            continue
        if previous_end is not None and word.start - previous_end >= CUE_BREAK_PAUSE:
            flush_line()
            force_cue_break_after.add(len(lines) - 1)
        elif previous_end is not None and word.start - previous_end >= LINE_BREAK_PAUSE:
            flush_line()
        previous_end = word.end

        candidate = current + [CaptionWord(text, word.start, word.end)]
        candidate_text = " ".join(w.text for w in candidate)
        if config.uppercase:
            candidate_text = candidate_text.upper()
        too_wide = _text_width(candidate_text, font) > max_pixels
        too_many = len(candidate) > config.max_words_per_line
        if current and (too_wide or too_many):
            flush_line()
            current.append(CaptionWord(text, word.start, word.end))
        else:
            current.append(CaptionWord(text, word.start, word.end))

        if text.endswith(_SENTENCE_END):
            flush_line()
    flush_line()

    cues: list[CaptionCue] = []
    buffer: list[CaptionLine] = []
    for index, line in enumerate(lines):
        buffer.append(line)
        if len(buffer) >= config.max_lines or index in force_cue_break_after:
            cues.append(CaptionCue(lines=list(buffer)))
            buffer.clear()
    if buffer:
        cues.append(CaptionCue(lines=list(buffer)))
    return cues


def _text_width(text: str, font) -> float:
    try:
        return float(font.getlength(text))
    except AttributeError:  # pragma: no cover - very old Pillow
        return float(font.getsize(text)[0])


# ---------------------------------------------------------------------- states
@dataclass(slots=True)
class CaptionState:
    """One distinct on-screen image and the interval it is shown for."""

    start: float
    end: float
    cue: CaptionCue | None
    active_word: int | None = None
    path: Path | None = None

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


def build_states(
    cues: Sequence[CaptionCue], config: CaptionStyleConfig, clip_duration: float
) -> list[CaptionState]:
    """Expand cues into per-frame-change states covering ``0..clip_duration``.

    Styles that highlight the spoken word produce one state per word; static
    styles produce one state per cue. Gaps become transparent states so the
    overlay stream is continuous.
    """
    per_word = config.style in (CaptionStyle.KARAOKE, CaptionStyle.WORD_HIGHLIGHT)
    states: list[CaptionState] = []
    cursor = 0.0

    for cue in cues:
        cue_start = max(0.0, min(cue.start, clip_duration))
        cue_end = max(cue_start, min(cue.end, clip_duration))
        if cue_end <= cue_start:
            continue
        if cue_start > cursor + 1e-3:
            states.append(CaptionState(start=cursor, end=cue_start, cue=None))
        if per_word:
            words = cue.words
            for index, word in enumerate(words):
                start = max(cue_start, min(word.start, clip_duration))
                end = (
                    max(start, min(words[index + 1].start, clip_duration))
                    if index + 1 < len(words)
                    else cue_end
                )
                if end <= start:
                    continue
                states.append(
                    CaptionState(start=start, end=end, cue=cue, active_word=index)
                )
        else:
            states.append(CaptionState(start=cue_start, end=cue_end, cue=cue))
        cursor = cue_end

    if cursor < clip_duration - 1e-3:
        states.append(CaptionState(start=cursor, end=clip_duration, cue=None))
    return [s for s in states if s.duration > 1e-3]


# --------------------------------------------------------------------- drawing
@dataclass(slots=True)
class CaptionBand:
    """Geometry of the caption overlay within the output frame."""

    width: int
    height: int
    y_offset: int


def band_geometry(
    config: CaptionStyleConfig, canvas_width: int, canvas_height: int, max_lines: int
) -> CaptionBand:
    line_height = int(config.font_size * config.line_spacing)
    padding = config.background_padding + config.outline_width + config.shadow_offset
    height = max_lines * line_height + padding * 2
    height = min(height, canvas_height)
    height += height % 2

    safe = int(canvas_height * config.safe_area_ratio)
    positions = {
        CaptionPosition.TOP: safe,
        CaptionPosition.UPPER_THIRD: int(canvas_height * 0.22),
        CaptionPosition.CENTER: (canvas_height - height) // 2,
        CaptionPosition.LOWER_THIRD: int(canvas_height * 0.66),
        CaptionPosition.BOTTOM: canvas_height - safe - height,
    }
    y = positions.get(config.position, int(canvas_height * 0.66))
    y = max(safe, min(y, canvas_height - safe - height))
    return CaptionBand(width=canvas_width, height=height, y_offset=max(0, y))


def render_state(
    state: CaptionState,
    config: CaptionStyleConfig,
    band: CaptionBand,
    destination: Path,
) -> Path:
    """Draw one caption state to a transparent PNG."""
    from PIL import Image, ImageDraw

    image = Image.new("RGBA", (band.width, band.height), (0, 0, 0, 0))
    if state.cue is None:
        image.save(destination, "PNG", optimize=True)
        return destination

    draw = ImageDraw.Draw(image)
    lines = state.cue.lines
    script = detect_script(state.cue.text)
    font = load_font(config.font_size, script=script, path=config.font_path)
    line_height = int(config.font_size * config.line_spacing)
    total_height = len(lines) * line_height
    top = max(0, (band.height - total_height) // 2)

    # Optional rounded background behind the whole block.
    if config.background_color:
        widths = [
            _text_width(_display(line.text, config), font) for line in lines
        ]
        block_width = max(widths) + config.background_padding * 2
        left = (band.width - block_width) / 2
        box = (
            left,
            top - config.background_padding,
            left + block_width,
            top + total_height + config.background_padding,
        )
        try:
            draw.rounded_rectangle(
                box, radius=config.background_radius, fill=config.background_color
            )
        except AttributeError:  # pragma: no cover - old Pillow
            draw.rectangle(box, fill=config.background_color)

    word_cursor = 0
    for line_index, line in enumerate(lines):
        y = top + line_index * line_height
        rendered = [_display(w.text, config) for w in line.words]
        space_width = _text_width(" ", font)
        widths = [_text_width(t, font) for t in rendered]
        line_width = sum(widths) + space_width * max(0, len(rendered) - 1)
        x = (band.width - line_width) / 2

        for word_index, text in enumerate(rendered):
            absolute = word_cursor + word_index
            is_active = state.active_word is not None and absolute == state.active_word
            is_spoken = state.active_word is not None and absolute <= state.active_word
            _draw_word(
                draw,
                text,
                (x, y),
                font,
                config,
                width=widths[word_index],
                line_height=line_height,
                is_active=is_active,
                is_spoken=is_spoken,
            )
            x += widths[word_index] + space_width
        word_cursor += len(line.words)

    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination, "PNG", optimize=True)
    return destination


def _display(text: str, config: CaptionStyleConfig) -> str:
    return text.upper() if config.uppercase else text


def _draw_word(
    draw,
    text: str,
    origin: tuple[float, float],
    font,
    config: CaptionStyleConfig,
    *,
    width: float,
    line_height: int,
    is_active: bool,
    is_spoken: bool,
) -> None:
    x, y = origin

    if config.style is CaptionStyle.KARAOKE:
        fill = config.highlight_color if is_spoken else config.text_color
    elif config.style is CaptionStyle.WORD_HIGHLIGHT and is_active:
        pad = max(6, config.font_size // 8)
        box = (x - pad, y - pad // 2, x + width + pad, y + line_height - pad // 2)
        try:
            draw.rounded_rectangle(box, radius=pad, fill=config.highlight_color)
        except AttributeError:  # pragma: no cover
            draw.rectangle(box, fill=config.highlight_color)
        fill = config.highlight_text_color
    else:
        fill = config.text_color

    if config.shadow_offset:
        draw.text(
            (x + config.shadow_offset, y + config.shadow_offset),
            text,
            font=font,
            fill=config.shadow_color,
        )
    if config.outline_width and not (
        config.style is CaptionStyle.WORD_HIGHLIGHT and is_active
    ):
        draw.text(
            (x, y),
            text,
            font=font,
            fill=fill,
            stroke_width=config.outline_width,
            stroke_fill=config.outline_color,
        )
    else:
        draw.text((x, y), text, font=font, fill=fill)


# ---------------------------------------------------------------- subtitle IO
def _srt_timestamp(seconds: float) -> str:
    seconds = max(0.0, seconds)
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    millis = int(round((seconds - math.floor(seconds)) * 1000))
    if millis == 1000:
        millis = 999
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def write_srt(cues: Sequence[CaptionCue], destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    blocks: list[str] = []
    for index, cue in enumerate(cues, start=1):
        blocks.append(
            f"{index}\n"
            f"{_srt_timestamp(cue.start)} --> {_srt_timestamp(cue.end)}\n"
            + "\n".join(line.text for line in cue.lines)
            + "\n"
        )
    destination.write_text("\n".join(blocks), encoding="utf-8")
    return destination


def _ass_timestamp(seconds: float) -> str:
    seconds = max(0.0, seconds)
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    centis = int(round((seconds - math.floor(seconds)) * 100))
    if centis == 100:
        centis = 99
    return f"{hours:d}:{minutes:02d}:{secs:02d}.{centis:02d}"


def write_ass(
    cues: Sequence[CaptionCue],
    destination: Path,
    *,
    config: CaptionStyleConfig,
    width: int,
    height: int,
) -> Path:
    """Editable ASS sidecar (also usable with the ffmpeg ``subtitles`` filter)."""
    destination.parent.mkdir(parents=True, exist_ok=True)

    def bgr(color: RGBA) -> str:
        r, g, b, a = color
        return f"&H{255 - a:02X}{b:02X}{g:02X}{r:02X}"

    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {width}\nPlayResY: {height}\n"
        "WrapStyle: 2\nScaledBorderAndShadow: yes\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, "
        "ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, "
        "MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Default,Sans,{config.font_size},{bgr(config.text_color)},"
        f"{bgr(config.highlight_color)},{bgr(config.outline_color)},&H80000000,"
        f"-1,0,0,0,100,100,0,0,1,{config.outline_width},{config.shadow_offset},2,"
        f"60,60,{int(height * 0.16)},1\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )
    events = [
        "Dialogue: 0,{start},{end},Default,,0,0,0,,{text}".format(
            start=_ass_timestamp(cue.start),
            end=_ass_timestamp(cue.end),
            text="\\N".join(line.text for line in cue.lines),
        )
        for cue in cues
    ]
    destination.write_text(header + "\n".join(events) + "\n", encoding="utf-8")
    return destination


# ------------------------------------------------------------------ public API
@dataclass(slots=True)
class RenderedCaptions:
    band: CaptionBand
    states: list[CaptionState]
    cues: list[CaptionCue]
    concat_file: Path | None
    srt_path: Path | None = None
    ass_path: Path | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def has_content(self) -> bool:
        return any(state.cue is not None for state in self.states)


def render_captions(
    words: Sequence[CaptionWord],
    *,
    config: CaptionStyleConfig,
    canvas_width: int,
    canvas_height: int,
    clip_duration: float,
    workdir: Path,
) -> RenderedCaptions:
    """Render every caption state and write the ffmpeg concat manifest."""
    workdir.mkdir(parents=True, exist_ok=True)
    cues = build_cues(words, config, canvas_width=canvas_width)
    band = band_geometry(config, canvas_width, canvas_height, config.max_lines)
    states = build_states(cues, config, clip_duration)

    warnings: list[str] = []
    if not cues:
        warnings.append("no words available for captions")
        return RenderedCaptions(
            band=band, states=[], cues=[], concat_file=None, warnings=warnings
        )

    frames_dir = workdir / "caption_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    # Identical images are common (e.g. blank gaps); cache by signature so we
    # neither redraw nor rewrite them.
    cache: dict[tuple, Path] = {}
    for index, state in enumerate(states):
        signature = (
            id(state.cue) if state.cue is not None else None,
            state.active_word,
        )
        if signature in cache:
            state.path = cache[signature]
            continue
        path = frames_dir / f"state_{index:05d}.png"
        render_state(state, config, band, path)
        cache[signature] = path
        state.path = path

    concat_file = workdir / "captions_concat.txt"
    lines: list[str] = []
    for state in states:
        assert state.path is not None
        lines.append(f"file '{state.path.as_posix()}'")
        lines.append(f"duration {state.duration:.4f}")
    # The concat demuxer needs the final entry repeated to give it a duration.
    lines.append(f"file '{states[-1].path.as_posix()}'")
    concat_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    srt_path = write_srt(cues, workdir / "captions.srt")
    ass_path = write_ass(
        cues, workdir / "captions.ass", config=config, width=canvas_width, height=canvas_height
    )

    logger.info(
        "captions rendered",
        extra={
            "cues": len(cues),
            "states": len(states),
            "images": len(cache),
            "band_height": band.height,
        },
    )
    return RenderedCaptions(
        band=band,
        states=states,
        cues=cues,
        concat_file=concat_file,
        srt_path=srt_path,
        ass_path=ass_path,
        warnings=warnings,
    )


def words_from_segments(
    segments: Iterable[dict[str, Any]], *, offset: float = 0.0
) -> list[CaptionWord]:
    """Flatten stored transcript segments into clip-relative caption words."""
    out: list[CaptionWord] = []
    for segment in segments:
        for word in segment.get("words") or []:
            text = str(word.get("word", "")).strip()
            if not text:
                continue
            start = float(word.get("start", 0.0)) - offset
            end = float(word.get("end", start)) - offset
            if end <= start:
                end = start + 0.08
            out.append(CaptionWord(text=text, start=start, end=end))
    return out
