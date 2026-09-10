"""Audio extraction and audio-domain analysis."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from app.core.logging import get_logger
from app.video.ffmpeg import probe, run_ffmpeg

logger = get_logger(__name__)


@dataclass(slots=True)
class SilenceInterval:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass(slots=True)
class AudioProfile:
    """Deterministic audio measurements used by scoring and quality control."""

    duration: float
    integrated_lufs: float | None = None
    true_peak_db: float | None = None
    loudness_range: float | None = None
    silences: list[SilenceInterval] = field(default_factory=list)
    #: Fraction of the timeline that is silent.
    silence_ratio: float = 0.0
    #: Mean RMS energy per second, indexed by whole second.
    energy_by_second: list[float] = field(default_factory=list)

    def speech_ratio(self) -> float:
        return max(0.0, 1.0 - self.silence_ratio)


def extract_audio(
    source: Path,
    destination: Path,
    *,
    sample_rate: int = 16000,
    channels: int = 1,
    start: float | None = None,
    duration: float | None = None,
) -> Path:
    """Extract a mono PCM WAV suitable for speech recognition."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    args: list[str] = ["-y"]
    if start is not None:
        args += ["-ss", f"{start:.3f}"]
    args += ["-i", str(source)]
    if duration is not None:
        args += ["-t", f"{duration:.3f}"]
    args += [
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", str(sample_rate),
        "-ac", str(channels),
        str(destination),
    ]
    run_ffmpeg(args)
    return destination


_SILENCE_START = re.compile(r"silence_start:\s*(-?[\d.]+)")
_SILENCE_END = re.compile(r"silence_end:\s*(-?[\d.]+)")


def detect_silence(
    source: Path,
    *,
    noise_db: float = -32.0,
    min_duration: float = 0.35,
) -> list[SilenceInterval]:
    """Silent intervals, used for segment boundaries and quality checks."""
    result = run_ffmpeg(
        [
            "-i", str(source),
            "-af", f"silencedetect=noise={noise_db}dB:d={min_duration}",
            "-f", "null", "-",
        ],
        check=False,
    )
    starts = [float(m) for m in _SILENCE_START.findall(result.stderr)]
    ends = [float(m) for m in _SILENCE_END.findall(result.stderr)]
    intervals: list[SilenceInterval] = []
    for index, start in enumerate(starts):
        end = ends[index] if index < len(ends) else None
        if end is None:
            continue
        if end > start:
            intervals.append(SilenceInterval(max(0.0, start), end))
    return intervals


_LOUDNORM_JSON = re.compile(r"\{[^{}]*\"input_i\"[^{}]*\}", re.DOTALL)


def measure_loudness(source: Path) -> dict[str, float]:
    """EBU R128 measurement via ``loudnorm`` in analysis mode."""
    result = run_ffmpeg(
        [
            "-i", str(source),
            "-af", "loudnorm=I=-14:TP=-1.5:LRA=11:print_format=json",
            "-f", "null", "-",
        ],
        check=False,
    )
    match = _LOUDNORM_JSON.search(result.stderr)
    if not match:
        return {}
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    out: dict[str, float] = {}
    for key in ("input_i", "input_tp", "input_lra", "input_thresh"):
        try:
            out[key] = float(payload[key])
        except (KeyError, TypeError, ValueError):
            continue
    return out


def energy_envelope(source: Path, *, resolution: float = 1.0) -> list[float]:
    """Mean RMS energy per ``resolution``-second bucket, normalised to 0..1.

    Implemented with ``astats`` over a resampled stream so no numpy decode of
    the whole file is needed.
    """
    frame_rate = max(1, int(round(1.0 / resolution)))
    result = run_ffmpeg(
        [
            "-i", str(source),
            "-af",
            f"aresample=8000,asetnsamples=n={8000 // frame_rate},"
            "astats=metadata=1:reset=1,ametadata=print:key=lavfi.astats.Overall.RMS_level",
            "-f", "null", "-",
        ],
        check=False,
    )
    values: list[float] = []
    for line in result.stderr.splitlines():
        if "RMS_level=" in line:
            try:
                values.append(float(line.split("RMS_level=")[-1].strip()))
            except ValueError:
                continue
    if not values:
        return []
    # RMS level is in dBFS (negative). Map -60..0 dB onto 0..1.
    return [max(0.0, min(1.0, (v + 60.0) / 60.0)) for v in values]


def profile_audio(source: Path) -> AudioProfile:
    """Full deterministic audio profile for one media file."""
    info = probe(source)
    duration = info.duration
    silences = detect_silence(source)
    silent_seconds = sum(s.duration for s in silences)
    loudness = measure_loudness(source)
    return AudioProfile(
        duration=duration,
        integrated_lufs=loudness.get("input_i"),
        true_peak_db=loudness.get("input_tp"),
        loudness_range=loudness.get("input_lra"),
        silences=silences,
        silence_ratio=(silent_seconds / duration) if duration > 0 else 0.0,
        energy_by_second=energy_envelope(source),
    )
