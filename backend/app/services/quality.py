"""Deterministic technical inspection of a rendered clip.

Everything measurable is measured here, in code -- duration, geometry, audio
presence and loudness, black frames, silence, caption timing and readability.
The QualityAgent adds only the judgements a model is actually needed for.

Each issue carries ``auto_fixable`` and ``fix_action`` so the pipeline can try
to correct the clip before asking a human (spec: "attempt automatic correction
before asking the user").
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from app.core.logging import get_logger
from app.models.enums import QualitySeverity
from app.video.audio import detect_silence, measure_loudness
from app.video.ffmpeg import probe, run_ffmpeg

logger = get_logger(__name__)

# ------------------------------------------------------------------ thresholds
TARGET_ASPECT = 9 / 16
ASPECT_TOLERANCE = 0.02
MIN_LOUDNESS_LUFS = -30.0
MAX_LOUDNESS_LUFS = -8.0
MAX_TRUE_PEAK_DB = -0.5
MAX_SILENCE_RATIO = 0.35
MAX_LEADING_SILENCE = 1.2
MAX_BLACK_RATIO = 0.10
MAX_LEADING_BLACK = 0.5
#: Captions must not run past the end of the video, nor stop long before it.
MAX_CAPTION_OVERHANG = 0.35
MAX_CAPTION_SHORTFALL_RATIO = 0.25
#: Caption text smaller than this fraction of frame height is unreadable on a
#: phone held at arm's length.
MIN_CAPTION_FONT_RATIO = 0.026
#: Platform chrome sits in roughly the top and bottom 12% of the screen.
UI_SAFE_RATIO = 0.12

_BLACK_RE = re.compile(
    r"black_start:([\d.]+)\s+black_end:([\d.]+)\s+black_duration:([\d.]+)"
)


@dataclass(slots=True)
class QualityIssue:
    check: str
    severity: QualitySeverity
    message: str
    detail: dict[str, Any] = field(default_factory=dict)
    auto_fixable: bool = False
    fix_action: str | None = None

    #: Points deducted from the 0-100 quality score.
    @property
    def weight(self) -> float:
        return {
            QualitySeverity.INFO: 0.0,
            QualitySeverity.MINOR: 4.0,
            QualitySeverity.MAJOR: 14.0,
            QualitySeverity.CRITICAL: 40.0,
        }[self.severity]

    def to_dict(self) -> dict[str, Any]:
        return {
            "check": self.check,
            "severity": self.severity.value,
            "message": self.message,
            "detail": self.detail,
            "auto_fixable": self.auto_fixable,
            "fix_action": self.fix_action,
        }


@dataclass(slots=True)
class TechnicalReport:
    duration: float
    width: int | None
    height: int | None
    fps: float | None
    has_audio: bool
    size_bytes: int
    aspect_ratio: float | None
    integrated_lufs: float | None
    true_peak_db: float | None
    silence_ratio: float
    leading_silence: float
    black_ratio: float
    leading_black: float
    caption_cue_count: int
    caption_coverage: float
    issues: list[QualityIssue] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "duration": round(self.duration, 3),
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "has_audio": self.has_audio,
            "size_bytes": self.size_bytes,
            "aspect_ratio": round(self.aspect_ratio, 4) if self.aspect_ratio else None,
            "integrated_lufs": self.integrated_lufs,
            "true_peak_db": self.true_peak_db,
            "silence_ratio": round(self.silence_ratio, 4),
            "leading_silence": round(self.leading_silence, 3),
            "black_ratio": round(self.black_ratio, 4),
            "leading_black": round(self.leading_black, 3),
            "caption_cue_count": self.caption_cue_count,
            "caption_coverage": round(self.caption_coverage, 4),
        }


def detect_black_frames(
    video: Path, *, min_duration: float = 0.12, threshold: float = 0.10
) -> list[tuple[float, float]]:
    result = run_ffmpeg(
        [
            "-i", str(video),
            "-vf", f"blackdetect=d={min_duration}:pic_th=0.98:pix_th={threshold}",
            "-an", "-f", "null", "-",
        ],
        check=False,
    )
    return [
        (float(start), float(end)) for start, end, _ in _BLACK_RE.findall(result.stderr)
    ]


def inspect_clip(
    video: Path,
    *,
    expected_duration: float | None = None,
    expected_width: int = 1080,
    expected_height: int = 1920,
    min_duration: float = 15.0,
    max_duration: float = 60.0,
    caption_cues: Sequence[dict[str, Any]] = (),
    caption_font_size: int | None = None,
    caption_y_offset: int | None = None,
    caption_band_height: int | None = None,
) -> TechnicalReport:
    """Measure a rendered clip and list everything wrong with it."""
    info = probe(video)
    issues: list[QualityIssue] = []

    # ---------------------------------------------------------- geometry
    aspect = info.aspect_ratio
    if info.width != expected_width or info.height != expected_height:
        issues.append(
            QualityIssue(
                check="resolution",
                severity=QualitySeverity.MAJOR,
                message=(
                    f"Output is {info.width}x{info.height}, expected "
                    f"{expected_width}x{expected_height}."
                ),
                detail={"actual": [info.width, info.height]},
                auto_fixable=True,
                fix_action="rescale",
            )
        )
    if aspect is not None and abs(aspect - TARGET_ASPECT) > ASPECT_TOLERANCE:
        issues.append(
            QualityIssue(
                check="aspect_ratio",
                severity=QualitySeverity.CRITICAL,
                message=f"Aspect ratio is {aspect:.3f}, expected {TARGET_ASPECT:.3f} (9:16).",
                detail={"aspect_ratio": aspect},
                auto_fixable=True,
                fix_action="rescale",
            )
        )
    if not info.fps or info.fps < 23:
        issues.append(
            QualityIssue(
                check="framerate",
                severity=QualitySeverity.MINOR,
                message=f"Frame rate is {info.fps}; short-form platforms expect 24-60 fps.",
                detail={"fps": info.fps},
            )
        )

    # ---------------------------------------------------------- duration
    if info.duration < min_duration - 0.5:
        issues.append(
            QualityIssue(
                check="duration_too_short",
                severity=QualitySeverity.MAJOR,
                message=f"Clip is {info.duration:.1f}s, below the {min_duration:.0f}s minimum.",
                detail={"duration": info.duration},
            )
        )
    if info.duration > max_duration + 0.5:
        issues.append(
            QualityIssue(
                check="duration_too_long",
                severity=QualitySeverity.MAJOR,
                message=f"Clip is {info.duration:.1f}s, above the {max_duration:.0f}s maximum.",
                detail={"duration": info.duration},
                auto_fixable=True,
                fix_action="trim",
            )
        )
    if expected_duration is not None and abs(info.duration - expected_duration) > 1.0:
        issues.append(
            QualityIssue(
                check="duration_mismatch",
                severity=QualitySeverity.MINOR,
                message=(
                    f"Rendered {info.duration:.2f}s but {expected_duration:.2f}s was "
                    "requested."
                ),
                detail={"requested": expected_duration, "actual": info.duration},
            )
        )

    # ------------------------------------------------------------- audio
    integrated = true_peak = None
    silence_ratio = leading_silence = 0.0
    if not info.has_audio:
        issues.append(
            QualityIssue(
                check="audio_missing",
                severity=QualitySeverity.CRITICAL,
                message="The clip has no audio track.",
            )
        )
    else:
        loudness = measure_loudness(video)
        integrated = loudness.get("input_i")
        true_peak = loudness.get("input_tp")
        if integrated is not None:
            if integrated < MIN_LOUDNESS_LUFS:
                issues.append(
                    QualityIssue(
                        check="audio_too_quiet",
                        severity=QualitySeverity.MAJOR,
                        message=f"Integrated loudness is {integrated:.1f} LUFS (very quiet).",
                        detail={"lufs": integrated},
                        auto_fixable=True,
                        fix_action="normalize_audio",
                    )
                )
            elif integrated > MAX_LOUDNESS_LUFS:
                issues.append(
                    QualityIssue(
                        check="audio_too_loud",
                        severity=QualitySeverity.MINOR,
                        message=f"Integrated loudness is {integrated:.1f} LUFS (hot).",
                        detail={"lufs": integrated},
                        auto_fixable=True,
                        fix_action="normalize_audio",
                    )
                )
        if true_peak is not None and true_peak > MAX_TRUE_PEAK_DB:
            issues.append(
                QualityIssue(
                    check="audio_clipping",
                    severity=QualitySeverity.MINOR,
                    message=f"True peak is {true_peak:.1f} dBTP; clipping is likely.",
                    detail={"true_peak": true_peak},
                    auto_fixable=True,
                    fix_action="normalize_audio",
                )
            )

        silences = detect_silence(video)
        silent_seconds = sum(s.duration for s in silences)
        silence_ratio = silent_seconds / info.duration if info.duration else 0.0
        leading_silence = next(
            (s.duration for s in silences if s.start <= 0.05), 0.0
        )
        if silence_ratio > MAX_SILENCE_RATIO:
            issues.append(
                QualityIssue(
                    check="excessive_silence",
                    severity=QualitySeverity.MAJOR,
                    message=f"{silence_ratio * 100:.0f}% of the clip is silent.",
                    detail={"silence_ratio": silence_ratio},
                )
            )
        if leading_silence > MAX_LEADING_SILENCE:
            issues.append(
                QualityIssue(
                    check="leading_silence",
                    severity=QualitySeverity.MINOR,
                    message=f"The clip opens with {leading_silence:.1f}s of silence.",
                    detail={"leading_silence": leading_silence},
                    auto_fixable=True,
                    fix_action="trim_start",
                )
            )

    # -------------------------------------------------------- black frames
    black_spans = detect_black_frames(video)
    black_seconds = sum(end - start for start, end in black_spans)
    black_ratio = black_seconds / info.duration if info.duration else 0.0
    leading_black = next((end for start, end in black_spans if start <= 0.05), 0.0)
    if black_ratio > MAX_BLACK_RATIO:
        issues.append(
            QualityIssue(
                check="black_frames",
                severity=QualitySeverity.MAJOR,
                message=f"{black_ratio * 100:.0f}% of the clip is black or near-black.",
                detail={"black_ratio": black_ratio, "spans": black_spans[:10]},
            )
        )
    if leading_black > MAX_LEADING_BLACK:
        issues.append(
            QualityIssue(
                check="leading_black",
                severity=QualitySeverity.MAJOR,
                message=f"The clip opens on {leading_black:.1f}s of black.",
                detail={"leading_black": leading_black},
                auto_fixable=True,
                fix_action="trim_start",
            )
        )

    # ----------------------------------------------------------- captions
    cue_count = len(caption_cues)
    coverage = 0.0
    if cue_count:
        covered = sum(
            max(0.0, float(c.get("end", 0)) - float(c.get("start", 0)))
            for c in caption_cues
        )
        coverage = covered / info.duration if info.duration else 0.0
        last_end = max(float(c.get("end", 0)) for c in caption_cues)
        first_start = min(float(c.get("start", 0)) for c in caption_cues)

        if last_end > info.duration + MAX_CAPTION_OVERHANG:
            issues.append(
                QualityIssue(
                    check="caption_overhang",
                    severity=QualitySeverity.MAJOR,
                    message=(
                        f"Captions run to {last_end:.1f}s but the clip ends at "
                        f"{info.duration:.1f}s."
                    ),
                    detail={"caption_end": last_end, "clip_end": info.duration},
                    auto_fixable=True,
                    fix_action="retime_captions",
                )
            )
        if info.duration - last_end > info.duration * MAX_CAPTION_SHORTFALL_RATIO:
            issues.append(
                QualityIssue(
                    check="caption_shortfall",
                    severity=QualitySeverity.MINOR,
                    message=(
                        f"Captions stop {info.duration - last_end:.1f}s before the "
                        "clip ends."
                    ),
                    detail={"gap": info.duration - last_end},
                )
            )
        if first_start > 2.5:
            issues.append(
                QualityIssue(
                    check="caption_late_start",
                    severity=QualitySeverity.MINOR,
                    message=f"First caption appears at {first_start:.1f}s.",
                    detail={"first_start": first_start},
                )
            )
    elif caption_font_size is not None:
        issues.append(
            QualityIssue(
                check="captions_missing",
                severity=QualitySeverity.MAJOR,
                message="Captions were requested but none were rendered.",
                auto_fixable=True,
                fix_action="regenerate_captions",
            )
        )

    height = info.height or expected_height
    if caption_font_size is not None and height:
        ratio = caption_font_size / height
        if ratio < MIN_CAPTION_FONT_RATIO:
            issues.append(
                QualityIssue(
                    check="caption_too_small",
                    severity=QualitySeverity.MINOR,
                    message=(
                        f"Caption text is {ratio * 100:.1f}% of frame height; "
                        f"{MIN_CAPTION_FONT_RATIO * 100:.1f}% is the readable minimum."
                    ),
                    detail={"font_ratio": ratio},
                    auto_fixable=True,
                    fix_action="enlarge_captions",
                )
            )
    if caption_y_offset is not None and caption_band_height is not None and height:
        safe_top = height * UI_SAFE_RATIO
        safe_bottom = height * (1 - UI_SAFE_RATIO)
        if caption_y_offset < safe_top or (
            caption_y_offset + caption_band_height
        ) > safe_bottom:
            issues.append(
                QualityIssue(
                    check="caption_unsafe_area",
                    severity=QualitySeverity.MINOR,
                    message="Captions overlap the area where platform UI is drawn.",
                    detail={
                        "y": caption_y_offset,
                        "band_height": caption_band_height,
                        "safe": [round(safe_top), round(safe_bottom)],
                    },
                    auto_fixable=True,
                    fix_action="reposition_captions",
                )
            )

    # ---------------------------------------------------------- integrity
    if info.size_bytes < 20_000:
        issues.append(
            QualityIssue(
                check="file_too_small",
                severity=QualitySeverity.CRITICAL,
                message=f"Output is only {info.size_bytes} bytes; the render likely failed.",
                detail={"size_bytes": info.size_bytes},
            )
        )

    return TechnicalReport(
        duration=info.duration,
        width=info.width,
        height=info.height,
        fps=info.fps,
        has_audio=info.has_audio,
        size_bytes=info.size_bytes,
        aspect_ratio=aspect,
        integrated_lufs=integrated,
        true_peak_db=true_peak,
        silence_ratio=silence_ratio,
        leading_silence=leading_silence,
        black_ratio=black_ratio,
        leading_black=leading_black,
        caption_cue_count=cue_count,
        caption_coverage=coverage,
        issues=issues,
    )


def technical_score(issues: Sequence[QualityIssue]) -> float:
    """0-100 from the deducted weight of every issue."""
    return max(0.0, 100.0 - sum(issue.weight for issue in issues))


def perceptual_hash(video: Path, *, timestamp: float | None = None) -> str | None:
    """A 64-bit dHash of one frame, used for duplicate detection.

    Implemented directly rather than pulling in an image-hashing dependency:
    downscale to 9x8 greyscale, compare horizontally adjacent pixels.
    """
    try:
        import cv2

        capture = cv2.VideoCapture(str(video))
        if not capture.isOpened():
            return None
        try:
            fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
            frames = capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0
            target = timestamp if timestamp is not None else (frames / fps) * 0.4
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(max(0, target * fps)))
            ok, frame = capture.read()
            if not ok or frame is None:
                return None
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
            bits = (small[:, 1:] > small[:, :-1]).flatten()
            value = 0
            for bit in bits:
                value = (value << 1) | int(bit)
            return f"{value:016x}"
        finally:
            capture.release()
    except Exception as exc:  # noqa: BLE001 - hashing is best-effort
        logger.debug("perceptual hash failed", extra={"error": str(exc)})
        return None


def hamming_distance(a: str, b: str) -> int:
    try:
        return bin(int(a, 16) ^ int(b, 16)).count("1")
    except (TypeError, ValueError):
        return 64
