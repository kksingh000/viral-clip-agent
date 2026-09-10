"""Per-account settings."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.deps import CurrentUser, DbSession, RateLimited, Settings
from app.core.errors import ValidationFailure
from app.models.user import UserSettings
from app.schemas.misc import SettingsResponse, SettingsUpdateRequest
from app.services.scoring import DEFAULT_WEIGHTS, ScoringWeights
from app.services.trends import DEFAULT_TREND_WEIGHTS, resolve_weights

router = APIRouter(prefix="/settings", tags=["settings"], dependencies=[RateLimited])


@router.get("", response_model=SettingsResponse)
async def get_settings(user_settings: Settings) -> UserSettings:
    return user_settings


@router.put("", response_model=SettingsResponse)
async def update_settings(
    payload: SettingsUpdateRequest,
    user: CurrentUser,
    session: DbSession,
    user_settings: Settings,
) -> UserSettings:
    updates = payload.model_dump(exclude_unset=True)

    # Cross-field checks against the *stored* values, not only the submitted
    # ones, so a partial update cannot leave an incoherent configuration.
    minimum = updates.get("min_clip_seconds", user_settings.min_clip_seconds)
    maximum = updates.get("max_clip_seconds", user_settings.max_clip_seconds)
    if minimum >= maximum:
        raise ValidationFailure("min_clip_seconds must be less than max_clip_seconds.")

    approve = updates.get("auto_approve_score", user_settings.auto_approve_score)
    reject = updates.get("reject_below_score", user_settings.reject_below_score)
    if approve < reject:
        raise ValidationFailure(
            "auto_approve_score must be greater than or equal to reject_below_score."
        )

    if "scoring_weights" in updates:
        # Normalise and drop unknown keys so a typo cannot silently disable a
        # dimension.
        updates["scoring_weights"] = ScoringWeights.from_mapping(
            updates["scoring_weights"]
        ).weights
    if "trend_weights" in updates:
        updates["trend_weights"] = resolve_weights(updates["trend_weights"])

    for field, value in updates.items():
        setattr(user_settings, field, value)
    await session.commit()
    await session.refresh(user_settings)
    return user_settings


@router.get("/defaults", response_model=dict)
async def get_defaults() -> dict:
    """The default weights, so the UI can show what a reset would restore."""
    return {
        "scoring_weights": DEFAULT_WEIGHTS,
        "trend_weights": DEFAULT_TREND_WEIGHTS,
    }
