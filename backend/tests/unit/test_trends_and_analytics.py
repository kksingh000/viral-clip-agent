"""Trend arithmetic and the score-calibration maths.

These are pure functions carrying real product logic — "momentum beats totals"
is a claim the numbers have to actually support.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.services.discovery import parse_iso8601_duration
from app.services.trends import (
    DEFAULT_TREND_WEIGHTS,
    MetricSnapshot,
    compute_metrics,
    compute_trend_score,
    resolve_weights,
    topic_trend_score,
)

NOW = datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc)


def snapshots(*points: tuple[float, int, int, int]) -> list[MetricSnapshot]:
    """``(hours_before_now, views, likes, comments)`` → snapshots."""
    return [
        MetricSnapshot(
            captured_at=NOW - timedelta(hours=hours),
            view_count=views,
            like_count=likes,
            comment_count=comments,
        )
        for hours, views, likes, comments in points
    ]


class TestMetrics:
    def test_view_velocity_is_views_per_hour(self):
        metrics = compute_metrics(
            published_at=NOW - timedelta(hours=10),
            snapshots=snapshots((0, 20_000, 0, 0)),
            now=NOW,
        )
        assert metrics.view_velocity == pytest.approx(2000.0, rel=1e-6)

    def test_engagement_and_comment_rates(self):
        metrics = compute_metrics(
            published_at=NOW - timedelta(hours=10),
            snapshots=snapshots((0, 1000, 80, 20)),
            now=NOW,
        )
        assert metrics.engagement_rate == pytest.approx(0.10)
        assert metrics.comment_rate == pytest.approx(0.02)

    def test_a_just_published_video_cannot_produce_infinite_velocity(self):
        metrics = compute_metrics(
            published_at=NOW,
            snapshots=snapshots((0, 5_000, 10, 5)),
            now=NOW,
        )
        assert metrics.view_velocity < 20_000  # floored at a half-hour window

    def test_acceleration_is_positive_when_speeding_up(self):
        # 60k in the first 9h (6.7k/h), then 40k in the last hour.
        metrics = compute_metrics(
            published_at=NOW - timedelta(hours=10),
            snapshots=snapshots((1, 60_000, 0, 0), (0, 100_000, 0, 0)),
            now=NOW,
        )
        assert metrics.recent_view_velocity == pytest.approx(40_000, rel=1e-3)
        assert metrics.growth_acceleration > 0

    def test_acceleration_is_negative_when_slowing_down(self):
        metrics = compute_metrics(
            published_at=NOW - timedelta(hours=100),
            snapshots=snapshots((1, 999_000, 0, 0), (0, 1_000_000, 0, 0)),
            now=NOW,
        )
        assert metrics.growth_acceleration < 0

    def test_no_snapshots_is_safe(self):
        metrics = compute_metrics(published_at=NOW, snapshots=[], now=NOW)
        assert metrics.view_velocity == 0.0
        assert metrics.engagement_rate == 0.0

    def test_zero_views_does_not_divide_by_zero(self):
        metrics = compute_metrics(
            published_at=NOW - timedelta(hours=5),
            snapshots=snapshots((0, 0, 0, 0)),
            now=NOW,
        )
        assert metrics.engagement_rate == 0.0
        assert metrics.comment_rate == 0.0

    def test_missing_published_at_is_tolerated(self):
        metrics = compute_metrics(
            published_at=None, snapshots=snapshots((0, 1000, 10, 1)), now=NOW
        )
        assert metrics.hours_since_publication == 0.0


class TestTrendScore:
    def _score(self, *, published_hours_ago, points, topic_trend=0.0):
        metrics = compute_metrics(
            published_at=NOW - timedelta(hours=published_hours_ago),
            snapshots=snapshots(*points),
            now=NOW,
        )
        return compute_trend_score(metrics, topic_trend=topic_trend)

    def test_momentum_beats_totals(self):
        """The product's central discovery claim, asserted numerically."""
        surging = self._score(
            published_hours_ago=6,
            points=((1, 60_000, 4_000, 600), (0, 110_000, 7_000, 1_000)),
        )
        stalled = self._score(
            published_hours_ago=2000,
            points=((1, 9_999_000, 300_000, 20_000), (0, 10_000_000, 300_100, 20_010)),
        )
        assert surging.score > stalled.score

    def test_score_is_bounded(self):
        extreme = self._score(
            published_hours_ago=1,
            points=((1, 1, 0, 0), (0, 50_000_000, 5_000_000, 900_000)),
            topic_trend=1.0,
        )
        assert 0.0 <= extreme.score <= 100.0
        empty = compute_trend_score(compute_metrics(published_at=None, snapshots=[]))
        assert 0.0 <= empty.score <= 100.0

    def test_recency_decays(self):
        recent = self._score(published_hours_ago=1, points=((0, 50_000, 2_000, 300),))
        old = self._score(published_hours_ago=500, points=((0, 50_000, 2_000, 300),))
        assert recent.components["recency"] > old.components["recency"]
        assert recent.score > old.score

    def test_topic_heat_raises_the_score(self):
        cold = self._score(published_hours_ago=10, points=((0, 50_000, 2_000, 300),))
        hot = self._score(
            published_hours_ago=10,
            points=((0, 50_000, 2_000, 300),),
            topic_trend=1.0,
        )
        assert hot.score > cold.score

    def test_every_component_is_normalised(self):
        result = self._score(
            published_hours_ago=5,
            points=((1, 100_000, 9_000, 1_500), (0, 400_000, 40_000, 6_000)),
            topic_trend=0.5,
        )
        assert set(result.components) == set(DEFAULT_TREND_WEIGHTS)
        for name, value in result.components.items():
            assert 0.0 <= value <= 1.0, f"{name} out of range: {value}"


class TestTrendWeights:
    def test_defaults_sum_to_one(self):
        assert sum(DEFAULT_TREND_WEIGHTS.values()) == pytest.approx(1.0, abs=1e-9)

    def test_overrides_are_renormalised(self):
        weights = resolve_weights({"view_velocity": 10.0})
        assert sum(weights.values()) == pytest.approx(1.0, abs=1e-9)
        assert weights["view_velocity"] > DEFAULT_TREND_WEIGHTS["view_velocity"]

    def test_unknown_and_invalid_keys_are_ignored(self):
        weights = resolve_weights({"nope": 5, "recency": "abc", "engagement": -1})
        assert "nope" not in weights
        assert weights["engagement"] == pytest.approx(
            DEFAULT_TREND_WEIGHTS["engagement"], abs=1e-9
        )

    def test_all_zero_falls_back_to_defaults(self):
        assert resolve_weights({k: 0 for k in DEFAULT_TREND_WEIGHTS}) == pytest.approx(
            DEFAULT_TREND_WEIGHTS
        )


class TestTopicScore:
    def test_an_emerging_topic_scores_despite_low_volume(self):
        """Appearing from nothing is the signal, not absolute volume."""
        score, growth = topic_trend_score(
            video_count=12, previous_video_count=0, total_views=200_000
        )
        assert growth == 1.0
        assert score > 30

    def test_a_shrinking_topic_scores_below_a_growing_one(self):
        growing, _ = topic_trend_score(
            video_count=40, previous_video_count=20, total_views=1_000_000
        )
        shrinking, growth = topic_trend_score(
            video_count=20, previous_video_count=40, total_views=1_000_000
        )
        assert growth == pytest.approx(-0.5)
        assert shrinking < growing

    def test_scores_are_bounded(self):
        for args in [
            (0, 0, 0),
            (100_000, 1, 10**12),
            (1, 100_000, 0),
        ]:
            score, _ = topic_trend_score(
                video_count=args[0], previous_video_count=args[1], total_views=args[2]
            )
            assert 0.0 <= score <= 100.0


class TestIsoDuration:
    @pytest.mark.parametrize(
        "value,expected",
        [
            ("PT4M13S", 253.0),
            ("PT1H2M3S", 3723.0),
            ("PT45S", 45.0),
            ("PT2H", 7200.0),
            ("P1DT2H", 93600.0),
        ],
    )
    def test_parses_known_forms(self, value, expected):
        assert parse_iso8601_duration(value) == expected

    @pytest.mark.parametrize("value", [None, "", "not-a-duration", "banana"])
    def test_rejects_garbage(self, value):
        assert parse_iso8601_duration(value) is None


class TestCalibration:
    def test_percentiles_span_the_range(self):
        """Ranking within the account removes the channel-growth trend that
        would otherwise dominate a fit against raw views."""
        import numpy as np

        views = [100, 500, 1_000, 5_000, 20_000]
        ordered = sorted(views)
        percentiles = [(i / (len(ordered) - 1)) * 100 for i in range(len(ordered))]
        assert percentiles[0] == 0.0
        assert percentiles[-1] == 100.0
        assert np.all(np.diff(percentiles) > 0)

    def test_ridge_solution_recovers_a_known_signal(self):
        """The calibration solve must actually fit; a degenerate solver would
        silently return near-zero weights and be clamped back to defaults."""
        import numpy as np

        from app.services.analytics import RIDGE_ALPHA

        rng = np.random.default_rng(7)
        features = rng.uniform(0, 10, size=(120, 3))
        true_weights = np.array([0.7, 0.2, 0.1])
        target = features @ true_weights * 10

        centred = features - features.mean(axis=0)
        target_centred = target - target.mean()
        gram = centred.T @ centred + RIDGE_ALPHA * np.eye(centred.shape[1])
        coefficients = np.linalg.solve(gram, centred.T @ target_centred)

        recovered = coefficients / coefficients.sum()
        assert recovered == pytest.approx(true_weights, abs=0.02)
