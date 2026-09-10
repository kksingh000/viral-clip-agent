"""Trend metrics and the configurable TrendScore.

Discovery ranks by *momentum*, not by totals: a 100k-view video that gained
50k in the last hour is more interesting than a 10M-view video that has
stopped moving. All of that is arithmetic over metric snapshots -- no model is
involved, and every weight is configurable.

Nothing here downloads or stores media. It works on publicly available
metadata only, and the records it produces stay at
``authorization_status = UNKNOWN`` until a human establishes rights.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping, Sequence

from app.core.logging import get_logger

logger = get_logger(__name__)

DEFAULT_TREND_WEIGHTS: dict[str, float] = {
    "view_velocity": 0.30,
    "engagement": 0.18,
    "comment_velocity": 0.12,
    "recency": 0.15,
    "growth_acceleration": 0.15,
    "topic_trend": 0.10,
}

#: Views per hour that maps to a full-marks velocity score. Log-scaled, so the
#: exact value only sets where the curve saturates.
VELOCITY_REFERENCE = 50_000.0
#: Engagement rate ((likes + comments) / views) that maps to full marks.
ENGAGEMENT_REFERENCE = 0.09
COMMENT_RATE_REFERENCE = 0.012
#: Half-life used for the recency term.
RECENCY_HALF_LIFE_HOURS = 36.0


@dataclass(slots=True)
class MetricSnapshot:
    captured_at: datetime
    view_count: int | None = None
    like_count: int | None = None
    comment_count: int | None = None


@dataclass(slots=True)
class TrendMetrics:
    view_velocity: float = 0.0            # views per hour
    engagement_rate: float = 0.0          # (likes + comments) / views
    comment_rate: float = 0.0             # comments / views
    growth_acceleration: float = 0.0      # change in velocity per hour
    hours_since_publication: float = 0.0
    recent_view_velocity: float | None = None  # from the two latest snapshots

    def to_dict(self) -> dict[str, Any]:
        return {
            "view_velocity": round(self.view_velocity, 3),
            "engagement_rate": round(self.engagement_rate, 6),
            "comment_rate": round(self.comment_rate, 6),
            "growth_acceleration": round(self.growth_acceleration, 4),
            "hours_since_publication": round(self.hours_since_publication, 2),
            "recent_view_velocity": (
                round(self.recent_view_velocity, 3)
                if self.recent_view_velocity is not None
                else None
            ),
        }


@dataclass(slots=True)
class TrendScoreResult:
    score: float
    components: dict[str, float]
    weights: dict[str, float]
    metrics: TrendMetrics

    def to_dict(self) -> dict[str, Any]:
        return {
            "trend_score": round(self.score, 2),
            "components": {k: round(v, 3) for k, v in self.components.items()},
            "weights": self.weights,
            "metrics": self.metrics.to_dict(),
        }


def _hours_between(later: datetime, earlier: datetime) -> float:
    return max(0.0, (later - earlier).total_seconds() / 3600.0)


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def compute_metrics(
    *,
    published_at: datetime | None,
    snapshots: Sequence[MetricSnapshot],
    now: datetime | None = None,
) -> TrendMetrics:
    """Derive velocity, engagement and acceleration from metric snapshots."""
    now = _aware(now or datetime.now(timezone.utc))
    if not snapshots:
        return TrendMetrics()

    ordered = sorted(snapshots, key=lambda s: _aware(s.captured_at))
    latest = ordered[-1]
    views = latest.view_count or 0
    likes = latest.like_count or 0
    comments = latest.comment_count or 0

    hours_live = (
        _hours_between(_aware(latest.captured_at), _aware(published_at))
        if published_at
        else 0.0
    )
    # Guard against a just-published video producing an infinite rate.
    effective_hours = max(0.5, hours_live)
    view_velocity = views / effective_hours if views else 0.0

    engagement_rate = ((likes + comments) / views) if views else 0.0
    comment_rate = (comments / views) if views else 0.0

    recent_velocity: float | None = None
    acceleration = 0.0
    if len(ordered) >= 2:
        previous = ordered[-2]
        window = _hours_between(_aware(latest.captured_at), _aware(previous.captured_at))
        if window >= 1 / 60:  # at least a minute apart
            delta_views = (latest.view_count or 0) - (previous.view_count or 0)
            recent_velocity = max(0.0, delta_views / window)
            # Acceleration as a ratio: how much faster than the lifetime
            # average is it moving right now?
            if view_velocity > 0:
                acceleration = (recent_velocity - view_velocity) / view_velocity

    return TrendMetrics(
        view_velocity=view_velocity,
        engagement_rate=engagement_rate,
        comment_rate=comment_rate,
        growth_acceleration=acceleration,
        hours_since_publication=(
            _hours_between(now, _aware(published_at)) if published_at else 0.0
        ),
        recent_view_velocity=recent_velocity,
    )


def _log_scale(value: float, reference: float) -> float:
    """Map 0..inf onto 0..1 with diminishing returns above ``reference``."""
    if value <= 0:
        return 0.0
    return min(1.0, math.log1p(value) / math.log1p(reference))


def resolve_weights(overrides: Mapping[str, Any] | None) -> dict[str, float]:
    merged = dict(DEFAULT_TREND_WEIGHTS)
    for key, value in (overrides or {}).items():
        if key in merged:
            try:
                parsed = float(value)
            except (TypeError, ValueError):
                continue
            if parsed >= 0:
                merged[key] = parsed
    total = sum(merged.values())
    if total <= 0:
        return dict(DEFAULT_TREND_WEIGHTS)
    return {k: v / total for k, v in merged.items()}


def compute_trend_score(
    metrics: TrendMetrics,
    *,
    weights: Mapping[str, Any] | None = None,
    topic_trend: float = 0.0,
) -> TrendScoreResult:
    """Combine the metrics into a 0-100 TrendScore.

    ``topic_trend`` is a 0-1 signal supplied by the topic database: how hot the
    subject of the video is right now, independent of this video's own numbers.
    """
    resolved = resolve_weights(weights)

    components = {
        "view_velocity": _log_scale(
            metrics.recent_view_velocity
            if metrics.recent_view_velocity is not None
            else metrics.view_velocity,
            VELOCITY_REFERENCE,
        ),
        "engagement": min(1.0, metrics.engagement_rate / ENGAGEMENT_REFERENCE),
        "comment_velocity": min(1.0, metrics.comment_rate / COMMENT_RATE_REFERENCE),
        "recency": 0.5 ** (metrics.hours_since_publication / RECENCY_HALF_LIFE_HOURS),
        # Acceleration is a ratio centred on 0; map -1..+2 onto 0..1.
        "growth_acceleration": max(
            0.0, min(1.0, (metrics.growth_acceleration + 1.0) / 3.0)
        ),
        "topic_trend": max(0.0, min(1.0, topic_trend)),
    }
    score = sum(components[key] * resolved.get(key, 0.0) for key in components) * 100.0
    return TrendScoreResult(
        score=max(0.0, min(100.0, score)),
        components=components,
        weights=resolved,
        metrics=metrics,
    )


# ------------------------------------------------------------------- topics
@dataclass(slots=True)
class TopicObservation:
    slug: str
    label: str
    keywords: list[str] = field(default_factory=list)
    video_ids: list[str] = field(default_factory=list)
    total_views: int = 0
    first_seen: datetime | None = None
    last_seen: datetime | None = None


def topic_trend_score(
    *,
    video_count: int,
    previous_video_count: int,
    total_views: int,
    window_hours: float = 24.0,
) -> tuple[float, float]:
    """Return ``(trend_score_0_100, growth_rate)`` for a topic.

    Growth is measured against the previous window, so a topic that appeared
    from nothing scores highly even at a modest absolute volume -- which is the
    point of emerging-topic detection.
    """
    if previous_video_count <= 0:
        growth = 1.0 if video_count > 0 else 0.0
    else:
        growth = (video_count - previous_video_count) / previous_video_count

    volume = _log_scale(float(video_count), 250.0)
    reach = _log_scale(float(total_views), 5_000_000.0)
    velocity = max(0.0, min(1.0, (growth + 0.5) / 2.0))
    freshness = max(0.0, min(1.0, 24.0 / max(1.0, window_hours)))

    score = (
        0.35 * velocity + 0.30 * volume + 0.25 * reach + 0.10 * freshness
    ) * 100.0
    return max(0.0, min(100.0, score)), growth


def window_bounds(hours: float, *, now: datetime | None = None) -> tuple[datetime, datetime]:
    end = _aware(now or datetime.now(timezone.utc))
    return end - timedelta(hours=hours), end
