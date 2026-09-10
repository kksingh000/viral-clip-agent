"""Smart vertical framing.

Turns sampled face detections into a smooth camera path for a 9:16 crop.

The pipeline is:

    detections -> face tracks -> active-speaker score -> raw target x(t)
        -> median filter -> deadband -> velocity/acceleration limit
        -> scene-boundary cuts -> minimal keyframe list

Every stage is deterministic and unit-testable; no model is involved. The
output keyframes are consumed by :mod:`app.video.renderer`, which drives
ffmpeg's ``crop`` filter through ``sendcmd``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from app.core.logging import get_logger
from app.providers.base import FaceBox, FrameAnalysis

logger = get_logger(__name__)


# ------------------------------------------------------------------ parameters
@dataclass(slots=True)
class TrackingConfig:
    """Tunables for the crop path. Defaults are chosen for talking-head video."""

    #: Two detections belong to the same track when their centres are closer
    #: than this fraction of the frame width.
    association_distance: float = 0.12
    #: A track must survive this many samples to be trusted.
    min_track_samples: int = 2
    #: Weights for the active-speaker score.
    weight_mouth_activity: float = 0.50
    weight_size: float = 0.25
    weight_centrality: float = 0.10
    weight_persistence: float = 0.15
    #: Hysteresis: a challenger must beat the incumbent by this margin to take
    #: over the frame, which stops the crop ping-ponging between speakers.
    switch_margin: float = 0.18
    #: The incumbent keeps the frame for at least this long after a switch.
    min_hold_seconds: float = 1.2
    #: Median filter window (samples, odd).
    median_window: int = 5
    #: Ignore target movements smaller than this fraction of frame width.
    deadband: float = 0.015
    #: Maximum pan speed as a fraction of frame width per second.
    max_velocity: float = 0.25
    #: Exponential smoothing factor per second (higher == more responsive).
    smoothing_rate: float = 3.5
    #: Cut (rather than pan) when a scene boundary is crossed.
    cut_on_scene_change: bool = True
    #: Emit a keyframe only when x moves at least this many pixels.
    keyframe_epsilon_px: float = 1.5
    #: Vertical bias for the crop window: 0 == top, 0.5 == centre. Faces sit
    #: high in the frame, so a slightly high window keeps heads out of the
    #: caption band.
    vertical_bias: float = 0.42


@dataclass(slots=True)
class FaceTrack:
    track_id: int
    samples: list[tuple[float, FaceBox]] = field(default_factory=list)

    @property
    def first_time(self) -> float:
        return self.samples[0][0]

    @property
    def last_time(self) -> float:
        return self.samples[-1][0]

    @property
    def duration(self) -> float:
        return self.last_time - self.first_time

    def box_at(self, timestamp: float) -> FaceBox | None:
        for time_value, box in self.samples:
            if abs(time_value - timestamp) < 1e-6:
                return box
        return None


@dataclass(slots=True)
class CropKeyframe:
    t: float
    x: float
    y: float

    def to_dict(self) -> dict[str, Any]:
        return {"t": round(self.t, 3), "x": round(self.x, 1), "y": round(self.y, 1)}


@dataclass(slots=True)
class CropWindow:
    """A single static crop rectangle in source coordinates."""

    x: float
    y: float
    width: int
    height: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "x": round(self.x, 1),
            "y": round(self.y, 1),
            "w": self.width,
            "h": self.height,
        }


@dataclass(slots=True)
class SplitCropPlan:
    """Two stacked speaker crops, for interviews and two-host podcasts.

    Each window is framed on one speaker and scaled into half of the output
    frame. Windows are static: a split screen that also pans is disorienting,
    and the point of the layout is that both speakers stay visible.
    """

    source_width: int
    source_height: int
    #: Exactly two windows, in render order (top, then bottom).
    windows: list[CropWindow]
    tracks_detected: int
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "layout": "split_speakers",
            "source": [self.source_width, self.source_height],
            "windows": [w.to_dict() for w in self.windows],
            "tracks_detected": self.tracks_detected,
            "notes": self.notes,
        }


@dataclass(slots=True)
class CropPlan:
    """A complete framing decision for one clip."""

    source_width: int
    source_height: int
    crop_width: int
    crop_height: int
    keyframes: list[CropKeyframe]
    #: ``True`` when faces were found and actually drove the path.
    face_driven: bool
    tracks_detected: int
    speaker_switches: int
    notes: list[str] = field(default_factory=list)

    @property
    def is_static(self) -> bool:
        return len(self.keyframes) <= 1

    def x_at(self, timestamp: float) -> float:
        if not self.keyframes:
            return max(0.0, (self.source_width - self.crop_width) / 2)
        previous = self.keyframes[0]
        for keyframe in self.keyframes:
            if keyframe.t > timestamp:
                break
            previous = keyframe
        return previous.x

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": [self.source_width, self.source_height],
            "crop": [self.crop_width, self.crop_height],
            "face_driven": self.face_driven,
            "tracks_detected": self.tracks_detected,
            "speaker_switches": self.speaker_switches,
            "keyframes": [k.to_dict() for k in self.keyframes],
            "notes": self.notes,
        }


# -------------------------------------------------------------------- tracking
def build_tracks(
    frames: Sequence[FrameAnalysis],
    *,
    config: TrackingConfig | None = None,
    scene_boundaries: Sequence[float] = (),
) -> list[FaceTrack]:
    """Associate per-frame detections into tracks by nearest centre.

    Tracks are terminated at scene boundaries: the same screen position across
    a cut is a different person.
    """
    config = config or TrackingConfig()
    tracks: list[FaceTrack] = []
    active: list[FaceTrack] = []
    next_id = 0
    boundaries = sorted(scene_boundaries)
    previous_time: float | None = None

    for frame in frames:
        width = max(1, frame.width)
        threshold = config.association_distance * width

        if previous_time is not None and _crosses_boundary(
            previous_time, frame.timestamp, boundaries
        ):
            active = []
        previous_time = frame.timestamp

        unmatched = list(frame.faces)
        still_active: list[FaceTrack] = []
        for track in active:
            last_box = track.samples[-1][1]
            best: FaceBox | None = None
            best_distance = threshold
            for candidate in unmatched:
                distance = math.dist(
                    (last_box.center_x, last_box.center_y),
                    (candidate.center_x, candidate.center_y),
                )
                if distance < best_distance:
                    best_distance = distance
                    best = candidate
            if best is not None:
                track.samples.append((frame.timestamp, best))
                unmatched.remove(best)
                still_active.append(track)
        for leftover in unmatched:
            track = FaceTrack(track_id=next_id, samples=[(frame.timestamp, leftover)])
            next_id += 1
            tracks.append(track)
            still_active.append(track)
        active = still_active

    return [t for t in tracks if len(t.samples) >= config.min_track_samples]


def _crosses_boundary(start: float, end: float, boundaries: Sequence[float]) -> bool:
    return any(start < b <= end for b in boundaries)


# ------------------------------------------------------------- speaker scoring
def _speaker_score(
    box: FaceBox,
    frame_width: int,
    track_duration: float,
    clip_duration: float,
    max_activity: float,
    config: TrackingConfig,
) -> float:
    size = min(1.0, box.width / (frame_width * 0.45))
    centrality = 1.0 - min(
        1.0, abs(box.center_x - frame_width / 2) / (frame_width / 2)
    )
    activity = (box.mouth_activity / max_activity) if max_activity > 1e-6 else 0.0
    persistence = min(1.0, track_duration / clip_duration) if clip_duration > 0 else 0.0
    return (
        config.weight_mouth_activity * min(1.0, activity)
        + config.weight_size * size
        + config.weight_centrality * centrality
        + config.weight_persistence * persistence
    )


def choose_targets(
    frames: Sequence[FrameAnalysis],
    tracks: Sequence[FaceTrack],
    *,
    config: TrackingConfig | None = None,
) -> tuple[list[tuple[float, float | None]], int]:
    """Pick the framed subject at each sample.

    Returns ``([(timestamp, target_centre_x or None)], switch_count)``.
    ``None`` means "no confident subject" -- the caller holds the last position.
    """
    config = config or TrackingConfig()
    if not frames:
        return [], 0

    duration = max(1e-6, frames[-1].timestamp - frames[0].timestamp)
    activities = [
        box.mouth_activity
        for track in tracks
        for _, box in track.samples
        if box.mouth_activity > 0
    ]
    max_activity = max(activities) if activities else 0.0

    by_time: dict[float, list[tuple[FaceTrack, FaceBox]]] = {}
    for track in tracks:
        for timestamp, box in track.samples:
            by_time.setdefault(timestamp, []).append((track, box))

    targets: list[tuple[float, float | None]] = []
    incumbent: int | None = None
    incumbent_since = -1e9
    switches = 0

    for frame in frames:
        entries = by_time.get(frame.timestamp, [])
        if not entries:
            targets.append((frame.timestamp, None))
            continue

        scored = [
            (
                _speaker_score(
                    box, frame.width, track.duration, duration, max_activity, config
                ),
                track,
                box,
            )
            for track, box in entries
        ]
        scored.sort(key=lambda item: item[0], reverse=True)
        best_score, best_track, best_box = scored[0]

        chosen_box = best_box
        if incumbent is not None and incumbent != best_track.track_id:
            held = next(
                (item for item in scored if item[1].track_id == incumbent), None
            )
            if held is not None:
                incumbent_score = held[0]
                too_soon = (frame.timestamp - incumbent_since) < config.min_hold_seconds
                not_decisive = best_score - incumbent_score < config.switch_margin
                if too_soon or not_decisive:
                    chosen_box = held[2]
                    targets.append((frame.timestamp, chosen_box.center_x))
                    continue
            switches += 1
            incumbent = best_track.track_id
            incumbent_since = frame.timestamp
        elif incumbent is None:
            incumbent = best_track.track_id
            incumbent_since = frame.timestamp

        targets.append((frame.timestamp, chosen_box.center_x))

    return targets, switches


# ------------------------------------------------------------------ smoothing
def _median(values: Iterable[float]) -> float:
    ordered = sorted(values)
    count = len(ordered)
    if count == 0:
        return 0.0
    middle = count // 2
    if count % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def smooth_path(
    targets: Sequence[tuple[float, float | None]],
    *,
    frame_width: int,
    crop_width: int,
    config: TrackingConfig | None = None,
    scene_boundaries: Sequence[float] = (),
) -> list[tuple[float, float]]:
    """Median filter, deadband and rate-limit the raw target series.

    Returns ``[(timestamp, crop_left_x)]`` clamped to the frame.
    """
    config = config or TrackingConfig()
    max_left = max(0.0, float(frame_width - crop_width))
    default_centre = frame_width / 2

    filled: list[tuple[float, float]] = []
    last_known = default_centre
    for timestamp, value in targets:
        if value is None:
            filled.append((timestamp, last_known))
        else:
            last_known = value
            filled.append((timestamp, value))
    if not filled:
        return []

    # Median filter over a centred window kills single-frame detection noise.
    window = max(1, config.median_window | 1)
    half = window // 2
    centres = [value for _, value in filled]
    smoothed_centres = [
        _median(centres[max(0, i - half) : min(len(centres), i + half + 1)])
        for i in range(len(centres))
    ]

    boundaries = sorted(scene_boundaries)
    path: list[tuple[float, float]] = []
    current = _clamp(smoothed_centres[0] - crop_width / 2, 0.0, max_left)
    previous_time = filled[0][0]
    deadband_px = config.deadband * frame_width
    max_step_per_second = config.max_velocity * frame_width

    for index, (timestamp, _) in enumerate(filled):
        desired = _clamp(smoothed_centres[index] - crop_width / 2, 0.0, max_left)
        dt = max(1e-3, timestamp - previous_time)
        previous_time = timestamp

        if config.cut_on_scene_change and _crosses_boundary(
            timestamp - dt, timestamp, boundaries
        ):
            current = desired  # hard cut: no pan across an edit
        else:
            delta = desired - current
            if abs(delta) > deadband_px:
                # Exponential approach, then clamp the per-step displacement.
                alpha = 1.0 - math.exp(-config.smoothing_rate * dt)
                step = delta * alpha
                limit = max_step_per_second * dt
                step = _clamp(step, -limit, limit)
                current = _clamp(current + step, 0.0, max_left)
        path.append((timestamp, current))
    return path


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def to_keyframes(
    path: Sequence[tuple[float, float]],
    *,
    y: float,
    epsilon: float = 1.5,
) -> list[CropKeyframe]:
    """Drop points that do not move the crop, so the sendcmd script stays small."""
    keyframes: list[CropKeyframe] = []
    for timestamp, x in path:
        if not keyframes or abs(x - keyframes[-1].x) >= epsilon:
            keyframes.append(CropKeyframe(t=timestamp, x=x, y=y))
    return keyframes


# ------------------------------------------------------------ split framing
def _track_prominence(track: FaceTrack, frame_width: int) -> float:
    """How much of the clip this face owns: size, airtime and mouth activity."""
    boxes = [box for _, box in track.samples]
    if not boxes:
        return 0.0
    mean_width = sum(b.width for b in boxes) / len(boxes)
    mean_activity = sum(b.mouth_activity for b in boxes) / len(boxes)
    size = min(1.0, mean_width / (frame_width * 0.35))
    airtime = min(1.0, len(boxes) / 30.0)
    return 0.45 * size + 0.35 * airtime + 0.20 * min(1.0, mean_activity / 6.0)


def _median_centre(track: FaceTrack) -> tuple[float, float]:
    xs = [box.center_x for _, box in track.samples]
    ys = [box.center_y for _, box in track.samples]
    return _median(xs), _median(ys)


def plan_split_crop(
    frames: Sequence[FrameAnalysis],
    *,
    source_width: int,
    source_height: int,
    target_width: int = 1080,
    target_height: int = 1920,
    config: TrackingConfig | None = None,
    scene_boundaries: Sequence[float] = (),
) -> SplitCropPlan | None:
    """Frame the two most prominent speakers into stacked halves.

    Returns ``None`` when fewer than two distinct faces are present, so the
    caller can fall back to single-subject framing rather than rendering a
    split screen with the same person twice.
    """
    config = config or TrackingConfig()
    tracks = build_tracks(frames, config=config, scene_boundaries=scene_boundaries)
    if len(tracks) < 2:
        return None

    ranked = sorted(
        tracks, key=lambda t: _track_prominence(t, source_width), reverse=True
    )
    chosen = ranked[:2]

    # Two tracks of the same person (a detection that broke and restarted)
    # would produce a split screen showing one face twice.
    centres = [_median_centre(track) for track in chosen]
    if abs(centres[0][0] - centres[1][0]) < source_width * 0.12:
        return None

    # Each half of the output frame; the source window must match its aspect
    # so the scale step does not distort.
    half_height = target_height // 2
    half_aspect = target_width / half_height

    window_height = min(source_height, int(round(source_width / half_aspect)))
    window_width = min(source_width, int(round(window_height * half_aspect)))
    window_width -= window_width % 2
    window_height -= window_height % 2

    # Render left-hand speaker on top: it matches reading order and stays
    # stable when the speakers swap who is talking.
    ordered = sorted(zip(chosen, centres), key=lambda pair: pair[1][0])

    windows: list[CropWindow] = []
    for _, (centre_x, centre_y) in ordered:
        x = _clamp(centre_x - window_width / 2, 0.0, max(0.0, source_width - window_width))
        y = _clamp(
            centre_y - window_height * 0.45,
            0.0,
            max(0.0, source_height - window_height),
        )
        windows.append(
            CropWindow(x=x, y=y, width=window_width, height=window_height)
        )

    logger.info(
        "split crop planned",
        extra={
            "tracks": len(tracks),
            "window": f"{window_width}x{window_height}",
        },
    )
    return SplitCropPlan(
        source_width=source_width,
        source_height=source_height,
        windows=windows,
        tracks_detected=len(tracks),
        notes=[f"two speakers framed from {len(tracks)} detected face track(s)"],
    )


# ------------------------------------------------------------------ public API
def plan_crop(
    frames: Sequence[FrameAnalysis],
    *,
    source_width: int,
    source_height: int,
    target_aspect: float = 9 / 16,
    config: TrackingConfig | None = None,
    scene_boundaries: Sequence[float] = (),
) -> CropPlan:
    """Produce a complete :class:`CropPlan` for a clip."""
    config = config or TrackingConfig()
    notes: list[str] = []

    crop_height = source_height
    crop_width = int(round(crop_height * target_aspect))
    if crop_width > source_width:
        # Source is already narrower than 9:16 -- crop vertically instead.
        crop_width = source_width
        crop_height = int(round(crop_width / target_aspect))
        crop_height = min(crop_height, source_height)
        notes.append("source is narrower than 9:16; cropped vertically")
    crop_width -= crop_width % 2
    crop_height -= crop_height % 2

    y = _clamp(
        (source_height - crop_height) * config.vertical_bias,
        0.0,
        max(0.0, source_height - crop_height),
    )

    tracks = build_tracks(frames, config=config, scene_boundaries=scene_boundaries)
    if not tracks:
        notes.append("no faces detected; using a centred static crop")
        centre_x = max(0.0, (source_width - crop_width) / 2)
        return CropPlan(
            source_width=source_width,
            source_height=source_height,
            crop_width=crop_width,
            crop_height=crop_height,
            keyframes=[CropKeyframe(t=0.0, x=centre_x, y=y)],
            face_driven=False,
            tracks_detected=0,
            speaker_switches=0,
            notes=notes,
        )

    targets, switches = choose_targets(frames, tracks, config=config)
    path = smooth_path(
        targets,
        frame_width=source_width,
        crop_width=crop_width,
        config=config,
        scene_boundaries=scene_boundaries,
    )
    keyframes = to_keyframes(path, y=y, epsilon=config.keyframe_epsilon_px)
    if not keyframes:
        keyframes = [CropKeyframe(t=0.0, x=max(0.0, (source_width - crop_width) / 2), y=y)]

    if switches:
        notes.append(f"active speaker changed {switches} time(s)")
    logger.info(
        "crop plan built",
        extra={
            "tracks": len(tracks),
            "keyframes": len(keyframes),
            "switches": switches,
            "crop": f"{crop_width}x{crop_height}",
        },
    )
    return CropPlan(
        source_width=source_width,
        source_height=source_height,
        crop_width=crop_width,
        crop_height=crop_height,
        keyframes=keyframes,
        face_driven=True,
        tracks_detected=len(tracks),
        speaker_switches=switches,
        notes=notes,
    )
