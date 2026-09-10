"""Aggregate v1 router."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.v1 import (
    analytics,
    auth,
    clips,
    integrations,
    jobs,
    settings,
    trends,
    videos,
)

api_router = APIRouter()
api_router.include_router(auth.router)
api_router.include_router(videos.router)
api_router.include_router(clips.router)
api_router.include_router(jobs.router)
api_router.include_router(trends.router)
api_router.include_router(settings.router)
api_router.include_router(analytics.router)
api_router.include_router(integrations.router)
