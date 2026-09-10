"""Shot/scene boundary detection.

Uses ffmpeg's ``select='gt(scene,...)'`` predicate, which compares consecutive
frames in the decoder -- accurate, no extra dependency, and one pass over the
file. Boundaries feed two consumers: clip cutting (never cut mid-shot when a
shot edge is nearby) and the crop tracker (reset the camera path on a cut
instead of panning across it).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.core.logging import get_logger
from app.video.ffmpeg import probe, run_ffmpeg

logger = get_logger(__name__)

_PTS_RE = re.compile(r"pts_time:([\d.]+)")

#: Below this a shot is a flash frame, not a scene worth reasoning about.
MIN_SCENE_SECONDS = 0.8


@dataclass(slots=True)
class SceneSpan:
    index: int
    start: float
    end: float
    change_score: float = 0.0
    visual_metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def contains(self, timestamp: float) -> bool:
        return self.start <= timestamp < self.end

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "duration": round(self.duration, 3),
            "change_score": round(self.change_score, 4),
        }


def detect_scene_changes(
    source: Path, *, threshold: float = 0.30, max_scenes: int = 4000
) -> list[float]:
    """Timestamps (seconds) at which a shot change was detected."""
    result = run_ffmpeg(
        [
            "-i", str(source),
            "-filter:v", f"select='gt(scene,{threshold})',showinfo",
            "-an",
            "-f", "null", "-",
        ],
        check=False,
    )
    timestamps = sorted({float(m) for m in _PTS_RE.findall(result.stderr)})
    if len(timestamps) > max_scenes:
        logger.warning(
            "scene detector produced an implausible number of cuts; raising threshold",
            extra={"count": len(timestamps), "threshold": threshold},
        )
        if threshold < 0.6:
            return detect_scene_changes(source, threshold=threshold + 0.15)
        timestamps = timestamps[:max_scenes]
    return timestamps


def detect_scenes(
    source: Path,
    *,
    threshold: float = 0.30,
    duration: float | None = None,
    min_scene_seconds: float = MIN_SCENE_SECONDS,
) -> list[SceneSpan]:
    """Contiguous, non-overlapping scenes covering the whole timeline."""
    total = duration if duration is not None else probe(source).duration
    changes = detect_scene_changes(source, threshold=threshold)

    boundaries = [0.0]
    for timestamp in changes:
        if timestamp - boundaries[-1] >= min_scene_seconds and timestamp < total:
            boundaries.append(timestamp)
    boundaries.append(total)

    scenes = [
        SceneSpan(index=index, start=start, end=end, change_score=threshold)
        for index, (start, end) in enumerate(zip(boundaries, boundaries[1:]))
        if end > start
    ]
    logger.info(
        "scene detection complete",
        extra={"scenes": len(scenes), "duration": round(total, 2)},
    )
    return scenes or [SceneSpan(index=0, start=0.0, end=total)]


def nearest_boundary(
    scenes: list[SceneSpan], timestamp: float, *, max_distance: float = 1.0
) -> float | None:
    """The closest shot edge within ``max_distance`` seconds, or ``None``.

    Used to snap a cut onto a shot change so a clip does not open or close a
    fraction of a second into the wrong shot.
    """
    best: float | None = None
    best_distance = max_distance
    for scene in scenes:
        for edge in (scene.start, scene.end):
            distance = abs(edge - timestamp)
            if distance < best_distance:
                best_distance = distance
                best = edge
    return best


def scenes_overlapping(
    scenes: list[SceneSpan], start: float, end: float
) -> list[SceneSpan]:
    return [s for s in scenes if s.end > start and s.start < end]
