"""Performance analytics and the score-calibration feedback loop.

The loop is: predicted viral score -> published clip -> realised performance ->
error -> adjusted scoring weights -> better predictions.

Calibration uses ridge-regularised least squares over the stored dimension
breakdowns, implemented with numpy rather than a modelling framework: the
feature count is twelve and the sample count is in the hundreds, so anything
heavier would be ceremony. Every fit is stored, so a bad one can be rolled
back by reactivating the previous row.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.errors import NotFoundError
from app.core.logging import get_logger
from app.models.analytics import ClipAnalytics, ScoreCalibration
from app.models.clip import GeneratedClip
from app.models.enums import Platform, PublishStatus
from app.models.publishing import PlatformAccount, PublishingJob
from app.providers.registry import get_publishing_provider
from app.services.scoring import DEFAULT_WEIGHTS, DIMENSION_NAMES

logger = get_logger(__name__)

#: Below this many published clips a fit is noise, not signal.
MIN_CALIBRATION_SAMPLES = 25
#: Ridge penalty. High enough that a small sample cannot swing a weight wildly.
RIDGE_ALPHA = 1.0
#: A calibrated weight may not move more than this fraction from the default,
#: so one unusual fortnight cannot rewrite the model.
MAX_WEIGHT_DRIFT = 0.5


def _now() -> datetime:
    return datetime.now(timezone.utc)


def sync_clip_analytics(session: Session, clip_id: uuid.UUID) -> dict[str, Any]:
    """Fetch fresh platform metrics for one published clip."""
    clip = session.get(GeneratedClip, clip_id)
    if clip is None:
        raise NotFoundError(f"Clip {clip_id} not found.")

    jobs = list(
        session.execute(
            select(PublishingJob).where(
                PublishingJob.clip_id == clip.id,
                PublishingJob.status == PublishStatus.PUBLISHED,
                PublishingJob.external_post_id.is_not(None),
            )
        ).scalars()
    )
    if not jobs:
        return {"clip_id": str(clip_id), "synced": 0, "note": "clip is not published"}

    synced = 0
    for job in jobs:
        account = session.get(PlatformAccount, job.account_id) if job.account_id else None
        provider = get_publishing_provider(job.platform)
        credentials = {
            "access_token": account.access_token_ref if account else None,
            "external_account_id": account.external_account_id if account else None,
            "ig_user_id": account.external_account_id if account else None,
        }
        try:
            metrics = provider.fetch_metrics(
                external_post_id=job.external_post_id, credentials=credentials
            )
        except Exception as exc:  # noqa: BLE001 - one platform must not stop the rest
            logger.warning(
                "metric fetch failed",
                extra={"clip_id": str(clip_id), "platform": str(job.platform),
                       "error": str(exc)},
            )
            continue

        session.add(
            ClipAnalytics(
                clip_id=clip.id,
                platform=job.platform,
                external_post_id=job.external_post_id,
                captured_at=_now(),
                views=metrics.views,
                likes=metrics.likes,
                comments=metrics.comments,
                shares=metrics.shares,
                saves=metrics.saves,
                average_view_duration=metrics.average_view_duration,
                completion_rate=metrics.completion_rate,
                retention_curve=metrics.retention_curve,
                predicted_score=clip.viral_score,
            )
        )
        synced += 1
    session.flush()
    return {"clip_id": str(clip_id), "synced": synced}


def sync_all_published(session: Session, *, max_clips: int = 200) -> dict[str, Any]:
    clips = list(
        session.execute(
            select(GeneratedClip)
            .where(GeneratedClip.publish_status == PublishStatus.PUBLISHED)
            .order_by(GeneratedClip.updated_at.desc())
            .limit(max_clips)
        ).scalars()
    )
    total = 0
    for clip in clips:
        try:
            total += sync_clip_analytics(session, clip.id).get("synced", 0)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "analytics sync failed for clip",
                extra={"clip_id": str(clip.id), "error": str(exc)},
            )
    return {"clips": len(clips), "snapshots": total}


# ------------------------------------------------------------------ ranking
def latest_analytics(session: Session, user_id: uuid.UUID) -> list[ClipAnalytics]:
    """Most recent snapshot per clip for one account."""
    rows = list(
        session.execute(
            select(ClipAnalytics)
            .join(GeneratedClip, GeneratedClip.id == ClipAnalytics.clip_id)
            .where(GeneratedClip.user_id == user_id)
            .order_by(ClipAnalytics.clip_id, ClipAnalytics.captured_at.desc())
        ).scalars()
    )
    seen: set[uuid.UUID] = set()
    latest: list[ClipAnalytics] = []
    for row in rows:
        if row.clip_id in seen:
            continue
        seen.add(row.clip_id)
        latest.append(row)
    return latest


def assign_percentiles(session: Session, user_id: uuid.UUID) -> int:
    """Rank each clip's views against the account's other clips.

    Percentile, not absolute views, is the training target: it removes the
    channel-growth trend that would otherwise dominate the fit.
    """
    rows = [r for r in latest_analytics(session, user_id) if r.views is not None]
    if len(rows) < 2:
        return 0
    ordered = sorted(rows, key=lambda r: r.views or 0)
    count = len(ordered)
    for index, row in enumerate(ordered):
        row.performance_percentile = (index / (count - 1)) * 100.0
        if row.predicted_score is not None:
            row.prediction_error = row.predicted_score - row.performance_percentile
    session.flush()
    return count


def _feature_matrix(
    session: Session, user_id: uuid.UUID
) -> tuple[list[list[float]], list[float]]:
    features: list[list[float]] = []
    targets: list[float] = []
    for row in latest_analytics(session, user_id):
        if row.performance_percentile is None:
            continue
        clip = session.get(GeneratedClip, row.clip_id)
        if clip is None or not clip.candidate_id:
            continue
        from app.models.clip import CandidateClip

        candidate = session.get(CandidateClip, clip.candidate_id)
        if candidate is None:
            continue
        dimensions = (candidate.score_breakdown or {}).get("dimensions") or {}
        if not dimensions:
            continue
        features.append([float(dimensions.get(name, 5.0)) for name in DIMENSION_NAMES])
        targets.append(float(row.performance_percentile))
    return features, targets


def calibrate_scoring(
    session: Session, user_id: uuid.UUID, *, activate: bool = True
) -> dict[str, Any]:
    """Fit dimension weights against realised performance."""
    assign_percentiles(session, user_id)
    features, targets = _feature_matrix(session, user_id)

    if len(features) < MIN_CALIBRATION_SAMPLES:
        return {
            "status": "insufficient_data",
            "sample_size": len(features),
            "required": MIN_CALIBRATION_SAMPLES,
        }

    import numpy as np

    matrix = np.array(features, dtype=float)
    observed = np.array(targets, dtype=float)

    # Ridge solution: (X'X + aI)^-1 X'y, on mean-centred features.
    centred = matrix - matrix.mean(axis=0)
    target_centred = observed - observed.mean()
    gram = centred.T @ centred + RIDGE_ALPHA * np.eye(centred.shape[1])
    coefficients = np.linalg.solve(gram, centred.T @ target_centred)

    # Negative weights are not meaningful in this model (a dimension cannot
    # make a clip worse by being higher), so clamp at zero, then normalise.
    raw = np.clip(coefficients, 0.0, None)
    if raw.sum() <= 0:
        return {"status": "degenerate_fit", "sample_size": len(features)}
    fitted = raw / raw.sum()

    # Bound the drift from the defaults so one window cannot rewrite the model.
    weights: dict[str, float] = {}
    for index, name in enumerate(DIMENSION_NAMES):
        default = DEFAULT_WEIGHTS[name]
        low, high = default * (1 - MAX_WEIGHT_DRIFT), default * (1 + MAX_WEIGHT_DRIFT)
        weights[name] = float(min(max(fitted[index], low), high))
    total = sum(weights.values())
    weights = {k: v / total for k, v in weights.items()}

    baseline_weights = np.array([DEFAULT_WEIGHTS[n] for n in DIMENSION_NAMES])
    calibrated_weights = np.array([weights[n] for n in DIMENSION_NAMES])
    baseline_prediction = matrix @ baseline_weights * 10.0
    calibrated_prediction = matrix @ calibrated_weights * 10.0
    baseline_mae = float(np.mean(np.abs(baseline_prediction - observed)))
    calibrated_mae = float(np.mean(np.abs(calibrated_prediction - observed)))
    correlation = (
        float(np.corrcoef(calibrated_prediction, observed)[0, 1])
        if len(observed) > 2
        else None
    )

    if activate:
        for previous in session.execute(
            select(ScoreCalibration).where(
                ScoreCalibration.user_id == user_id, ScoreCalibration.is_active.is_(True)
            )
        ).scalars():
            previous.is_active = False

    record = ScoreCalibration(
        user_id=user_id,
        weights=weights,
        sample_size=len(features),
        baseline_mae=baseline_mae,
        calibrated_mae=calibrated_mae,
        correlation=correlation,
        method="ridge",
        notes=(
            f"Fitted on {len(features)} published clips; MAE "
            f"{baseline_mae:.1f} -> {calibrated_mae:.1f}."
        ),
        is_active=activate,
    )
    session.add(record)
    session.flush()

    logger.info(
        "scoring calibrated",
        extra={
            "user_id": str(user_id),
            "samples": len(features),
            "baseline_mae": round(baseline_mae, 2),
            "calibrated_mae": round(calibrated_mae, 2),
        },
    )
    return {
        "status": "ok",
        "calibration_id": str(record.id),
        "sample_size": len(features),
        "baseline_mae": baseline_mae,
        "calibrated_mae": calibrated_mae,
        "correlation": correlation,
        "weights": weights,
    }


def active_weights(session: Session, user_id: uuid.UUID) -> dict[str, float] | None:
    record = session.execute(
        select(ScoreCalibration)
        .where(ScoreCalibration.user_id == user_id, ScoreCalibration.is_active.is_(True))
        .order_by(ScoreCalibration.created_at.desc())
        .limit(1)
    ).scalars().first()
    return dict(record.weights) if record else None


def account_summary(session: Session, user_id: uuid.UUID) -> dict[str, Any]:
    rows = latest_analytics(session, user_id)
    published = len(rows)
    totals = {
        "views": sum(r.views or 0 for r in rows),
        "likes": sum(r.likes or 0 for r in rows),
        "comments": sum(r.comments or 0 for r in rows),
    }
    completions = [r.completion_rate for r in rows if r.completion_rate is not None]
    errors = [abs(r.prediction_error) for r in rows if r.prediction_error is not None]

    correlation = None
    pairs = [
        (r.predicted_score, r.performance_percentile)
        for r in rows
        if r.predicted_score is not None and r.performance_percentile is not None
    ]
    if len(pairs) > 2:
        import numpy as np

        predicted = np.array([p for p, _ in pairs])
        actual = np.array([a for _, a in pairs])
        if predicted.std() > 0 and actual.std() > 0:
            correlation = float(np.corrcoef(predicted, actual)[0, 1])

    return {
        "clips_published": published,
        "total_views": totals["views"],
        "total_likes": totals["likes"],
        "total_comments": totals["comments"],
        "average_completion_rate": (
            sum(completions) / len(completions) if completions else None
        ),
        "mean_absolute_error": sum(errors) / len(errors) if errors else None,
        "score_correlation": correlation,
        "calibration_sample_size": len(pairs),
    }


def recent_window(days: int = 30) -> datetime:
    return _now() - timedelta(days=days)


def platform_of(job: PublishingJob) -> Platform:
    return Platform(job.platform)
