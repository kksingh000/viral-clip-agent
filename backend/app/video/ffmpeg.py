"""Low-level FFmpeg/FFprobe access.

Everything that shells out to ffmpeg goes through :func:`run_ffmpeg` so that
failures are normalised into :class:`FFmpegError` (with the command and the
tail of stderr attached) and every invocation is logged with its duration.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

from app.core.config import settings
from app.core.errors import FFmpegError, MediaAnalysisError
from app.core.logging import get_logger

logger = get_logger(__name__)


# --------------------------------------------------------------- binary paths
@lru_cache(maxsize=1)
def ffmpeg_path() -> str:
    """Resolve the ffmpeg binary: explicit setting, then PATH, then the
    ``imageio-ffmpeg`` wheel (which ships a static build)."""
    if settings.ffmpeg_path:
        return settings.ffmpeg_path
    if found := shutil.which("ffmpeg"):
        return found
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:  # pragma: no cover - only when nothing is installed
        raise FFmpegError(
            "ffmpeg was not found. Install it, set FFMPEG_PATH, or "
            "'pip install imageio-ffmpeg'."
        ) from exc


@lru_cache(maxsize=1)
def ffprobe_path() -> str | None:
    """Resolve ffprobe, or ``None`` when it is unavailable.

    The ``imageio-ffmpeg`` wheel ships ffmpeg but not ffprobe, so probing has a
    documented fallback path (see :func:`probe`).
    """
    if settings.ffprobe_path:
        return settings.ffprobe_path
    if found := shutil.which("ffprobe"):
        return found
    sibling = Path(ffmpeg_path()).with_name(
        "ffprobe.exe" if os.name == "nt" else "ffprobe"
    )
    return str(sibling) if sibling.exists() else None


@lru_cache(maxsize=1)
def ffmpeg_capabilities() -> dict[str, bool]:
    """Which optional filters/encoders this build provides."""
    caps = {
        "sendcmd": False,
        "qtrle": False,
        "libx264": False,
        "aac": False,
        "subtitles": False,
        "gblur": False,
        "blackdetect": False,
        "silencedetect": False,
    }
    try:
        filters = subprocess.run(
            [ffmpeg_path(), "-hide_banner", "-filters"],
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout
        encoders = subprocess.run(
            [ffmpeg_path(), "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout
    except Exception as exc:  # pragma: no cover - environment dependent
        logger.warning("could not query ffmpeg capabilities", extra={"error": str(exc)})
        return caps
    for key in ("sendcmd", "subtitles", "gblur", "blackdetect", "silencedetect"):
        caps[key] = bool(re.search(rf"\s{key}\s", filters))
    for key in ("qtrle", "libx264", "aac"):
        caps[key] = bool(re.search(rf"\s{key}\s", encoders))
    return caps


# ------------------------------------------------------------------ execution
@dataclass(slots=True)
class FFmpegResult:
    args: list[str]
    returncode: int
    stdout: str
    stderr: str
    duration_seconds: float


def run_ffmpeg(
    args: Sequence[str],
    *,
    timeout: int | None = None,
    check: bool = True,
    binary: str | None = None,
    capture_stdout: bool = False,
) -> FFmpegResult:
    """Run ffmpeg with ``args`` (the binary itself is prepended)."""
    command = [binary or ffmpeg_path(), "-hide_banner", "-nostdin", *map(str, args)]
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE if capture_stdout else subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=timeout or settings.ffmpeg_timeout_seconds,
            text=True,
            errors="replace",
        )
    except subprocess.TimeoutExpired as exc:
        raise FFmpegError(
            f"ffmpeg timed out after {timeout or settings.ffmpeg_timeout_seconds}s",
            command=command,
            stderr=str(exc.stderr or ""),
        ) from exc
    except OSError as exc:
        raise FFmpegError(f"could not launch ffmpeg: {exc}", command=command) from exc

    elapsed = time.perf_counter() - started
    result = FFmpegResult(
        args=command,
        returncode=completed.returncode,
        stdout=completed.stdout or "",
        stderr=completed.stderr or "",
        duration_seconds=elapsed,
    )
    logger.debug(
        "ffmpeg finished",
        extra={
            "returncode": result.returncode,
            "elapsed_s": round(elapsed, 2),
            "op": _describe(command),
        },
    )
    if check and completed.returncode != 0:
        raise FFmpegError(
            f"ffmpeg exited with code {completed.returncode}",
            command=command,
            stderr=result.stderr,
            returncode=completed.returncode,
        )
    return result


def _describe(command: Sequence[str]) -> str:
    outputs = [a for a in command[1:] if not a.startswith("-")]
    return Path(outputs[-1]).name if outputs else "ffmpeg"


# -------------------------------------------------------------------- probing
@dataclass(slots=True)
class StreamInfo:
    index: int
    codec_type: str
    codec_name: str | None = None
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    sample_rate: int | None = None
    channels: int | None = None
    bit_rate: int | None = None


@dataclass(slots=True)
class MediaInfo:
    path: Path
    duration: float
    size_bytes: int
    format_name: str | None = None
    bit_rate: int | None = None
    streams: list[StreamInfo] = field(default_factory=list)
    probe_source: str = "ffprobe"

    @property
    def video(self) -> StreamInfo | None:
        return next((s for s in self.streams if s.codec_type == "video"), None)

    @property
    def audio(self) -> StreamInfo | None:
        return next((s for s in self.streams if s.codec_type == "audio"), None)

    @property
    def width(self) -> int | None:
        return self.video.width if self.video else None

    @property
    def height(self) -> int | None:
        return self.video.height if self.video else None

    @property
    def fps(self) -> float | None:
        return self.video.fps if self.video else None

    @property
    def has_audio(self) -> bool:
        return self.audio is not None

    @property
    def aspect_ratio(self) -> float | None:
        if self.width and self.height:
            return self.width / self.height
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "duration": self.duration,
            "size_bytes": self.size_bytes,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "video_codec": self.video.codec_name if self.video else None,
            "audio_codec": self.audio.codec_name if self.audio else None,
            "has_audio": self.has_audio,
            "format": self.format_name,
            "bit_rate": self.bit_rate,
            "probe_source": self.probe_source,
        }


def _parse_fraction(value: str | None) -> float | None:
    if not value or value in ("0/0", "N/A"):
        return None
    try:
        if "/" in value:
            num, den = value.split("/", 1)
            den_f = float(den)
            return float(num) / den_f if den_f else None
        return float(value)
    except (TypeError, ValueError):
        return None


def probe(path: Path | str) -> MediaInfo:
    """Return technical metadata for a media file.

    Uses ffprobe when available (authoritative JSON). Falls back to parsing
    ``ffmpeg -i`` output, which every build supports, so a deployment without
    ffprobe still works -- just with slightly coarser numbers.
    """
    path = Path(path)
    if not path.exists():
        raise MediaAnalysisError(f"Media file not found: {path}")
    size = path.stat().st_size
    if size == 0:
        raise MediaAnalysisError(f"Media file is empty: {path}")

    if (probe_bin := ffprobe_path()) is not None:
        return _probe_with_ffprobe(path, size, probe_bin)
    return _probe_with_ffmpeg(path, size)


def _probe_with_ffprobe(path: Path, size: int, probe_bin: str) -> MediaInfo:
    result = run_ffmpeg(
        [
            "-v", "error",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            str(path),
        ],
        binary=probe_bin,
        capture_stdout=True,
        timeout=120,
    )
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise MediaAnalysisError(f"ffprobe returned invalid JSON for {path.name}") from exc

    streams: list[StreamInfo] = []
    for raw in payload.get("streams", []):
        codec_type = raw.get("codec_type", "")
        streams.append(
            StreamInfo(
                index=int(raw.get("index", 0)),
                codec_type=codec_type,
                codec_name=raw.get("codec_name"),
                width=int(raw["width"]) if raw.get("width") else None,
                height=int(raw["height"]) if raw.get("height") else None,
                fps=_parse_fraction(raw.get("avg_frame_rate"))
                or _parse_fraction(raw.get("r_frame_rate")),
                sample_rate=int(raw["sample_rate"]) if raw.get("sample_rate") else None,
                channels=raw.get("channels"),
                bit_rate=int(raw["bit_rate"]) if raw.get("bit_rate") else None,
            )
        )
    fmt = payload.get("format", {})
    duration = float(fmt.get("duration") or 0.0)
    if duration <= 0:
        for stream in payload.get("streams", []):
            if stream.get("duration"):
                duration = max(duration, float(stream["duration"]))
    return MediaInfo(
        path=path,
        duration=duration,
        size_bytes=size,
        format_name=fmt.get("format_name"),
        bit_rate=int(fmt["bit_rate"]) if fmt.get("bit_rate") else None,
        streams=streams,
        probe_source="ffprobe",
    )


_DURATION_RE = re.compile(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)")
_VIDEO_RE = re.compile(
    r"Stream #\d+:(\d+).*?: Video: ([\w0-9]+).*?, (\d+)x(\d+)[^,]*(?:.*?([\d.]+) fps)?",
)
_AUDIO_RE = re.compile(
    r"Stream #\d+:(\d+).*?: Audio: ([\w0-9]+).*?, (\d+) Hz, ([^,]+)"
)


def _probe_with_ffmpeg(path: Path, size: int) -> MediaInfo:
    """Fallback probe by parsing ``ffmpeg -i`` diagnostics."""
    result = run_ffmpeg(["-i", str(path)], check=False, timeout=120)
    text = result.stderr

    duration = 0.0
    if match := _DURATION_RE.search(text):
        hours, minutes, seconds = match.groups()
        duration = int(hours) * 3600 + int(minutes) * 60 + float(seconds)

    streams: list[StreamInfo] = []
    if match := _VIDEO_RE.search(text):
        index, codec, width, height, fps = match.groups()
        streams.append(
            StreamInfo(
                index=int(index),
                codec_type="video",
                codec_name=codec,
                width=int(width),
                height=int(height),
                fps=float(fps) if fps else None,
            )
        )
    if match := _AUDIO_RE.search(text):
        index, codec, rate, channels = match.groups()
        streams.append(
            StreamInfo(
                index=int(index),
                codec_type="audio",
                codec_name=codec,
                sample_rate=int(rate),
                channels=2 if "stereo" in channels else 1,
            )
        )

    video = next((s for s in streams if s.codec_type == "video"), None)
    if video is not None and (video.fps is None or duration <= 0):
        # OpenCV fills the gaps the text output leaves.
        try:
            import cv2

            capture = cv2.VideoCapture(str(path))
            if capture.isOpened():
                fps = capture.get(cv2.CAP_PROP_FPS)
                frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
                if fps and not math.isnan(fps) and fps > 0:
                    video.fps = video.fps or float(fps)
                    if duration <= 0 and frames > 0:
                        duration = float(frames) / float(fps)
            capture.release()
        except Exception as exc:  # pragma: no cover - opencv optional at runtime
            logger.debug("opencv probe fallback failed", extra={"error": str(exc)})

    if duration <= 0:
        raise MediaAnalysisError(
            f"Could not determine the duration of {path.name}. "
            "Install ffprobe for reliable probing."
        )
    return MediaInfo(
        path=path,
        duration=duration,
        size_bytes=size,
        format_name=None,
        streams=streams,
        probe_source="ffmpeg-stderr",
    )
