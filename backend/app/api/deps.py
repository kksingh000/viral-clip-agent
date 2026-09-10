"""Shared FastAPI dependencies: authentication, settings, rate limiting."""

from __future__ import annotations

import time
import uuid
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Annotated

from fastapi import Depends, Header, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.errors import AuthenticationError, ForbiddenError, RateLimitError
from app.core.logging import get_logger
from app.core.security import decode_token, hash_api_key
from app.database.session import get_db
from app.models.user import ApiKey, User, UserSettings

logger = get_logger(__name__)

bearer_scheme = HTTPBearer(auto_error=False)

DbSession = Annotated[AsyncSession, Depends(get_db)]


async def _user_from_bearer(session: AsyncSession, token: str) -> User:
    payload = decode_token(token, expected_type="access")
    try:
        user_id = uuid.UUID(str(payload["sub"]))
    except (KeyError, ValueError) as exc:
        raise AuthenticationError("Token subject is not a valid user id.") from exc
    user = await session.get(User, user_id)
    if user is None:
        raise AuthenticationError("The account for this token no longer exists.")
    return user


async def _user_from_api_key(session: AsyncSession, raw_key: str) -> User:
    digest = hash_api_key(raw_key)
    record = (
        await session.execute(select(ApiKey).where(ApiKey.key_hash == digest))
    ).scalars().first()
    if record is None or record.revoked_at is not None:
        raise AuthenticationError("Invalid or revoked API key.")
    record.last_used_at = datetime.now(timezone.utc)
    user = await session.get(User, record.user_id)
    if user is None:
        raise AuthenticationError("The account for this API key no longer exists.")
    return user


async def get_current_user(
    session: DbSession,
    credentials: Annotated[
        HTTPAuthorizationCredentials | None, Depends(bearer_scheme)
    ] = None,
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
) -> User:
    """Resolve the caller from a bearer token or an API key."""
    if credentials is not None and credentials.credentials:
        user = await _user_from_bearer(session, credentials.credentials)
    elif x_api_key:
        user = await _user_from_api_key(session, x_api_key)
    else:
        raise AuthenticationError(
            "Authentication required. Send an Authorization: Bearer header or "
            "an X-API-Key header."
        )
    if not user.is_active:
        raise ForbiddenError("This account is disabled.")
    return user


CurrentUser = Annotated[User, Depends(get_current_user)]


async def get_current_superuser(user: CurrentUser) -> User:
    if not user.is_superuser:
        raise ForbiddenError("This operation requires an administrator account.")
    return user


async def get_user_settings(session: DbSession, user: CurrentUser) -> UserSettings:
    """Fetch the caller's settings, creating defaults on first access."""
    record = (
        await session.execute(
            select(UserSettings).where(UserSettings.user_id == user.id)
        )
    ).scalars().first()
    if record is None:
        record = UserSettings(user_id=user.id)
        session.add(record)
        await session.flush()
    return record


Settings = Annotated[UserSettings, Depends(get_user_settings)]


# --------------------------------------------------------------- rate limiting
#: Stop the in-process limiter's key table growing without bound. Keys include
#: the request path, so a scanner hitting random URLs would otherwise leak
#: memory for the life of the process.
_MAX_TRACKED_KEYS = 20_000
_SWEEP_INTERVAL_SECONDS = 60.0


class _InMemoryRateLimiter:
    """Sliding-window limiter held in this process.

    Correct for a single replica; with several API replicas each holds its own
    counters, so the effective limit is multiplied by the replica count. That
    is why :class:`_RedisRateLimiter` is preferred whenever Redis is reachable.
    """

    def __init__(self) -> None:
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._last_sweep = time.monotonic()

    def _sweep(self, now: float, window: int) -> None:
        """Drop keys whose window has fully expired."""
        stale = [
            key
            for key, hits in self._hits.items()
            if not hits or now - hits[-1] > window
        ]
        for key in stale:
            del self._hits[key]
        self._last_sweep = now

    def check(self, key: str, limit: int, window: int) -> tuple[bool, int]:
        now = time.monotonic()
        if (
            now - self._last_sweep > _SWEEP_INTERVAL_SECONDS
            or len(self._hits) > _MAX_TRACKED_KEYS
        ):
            self._sweep(now, window)

        hits = self._hits[key]
        while hits and now - hits[0] > window:
            hits.popleft()
        if len(hits) >= limit:
            retry_after = int(window - (now - hits[0])) + 1
            return False, max(1, retry_after)
        hits.append(now)
        return True, 0


class _RedisRateLimiter:
    """Fixed-window limiter shared by every API replica.

    Uses INCR against a bucketed key: the first increment sets the expiry, so
    the key cleans itself up. Fixed rather than sliding windows because a
    single round trip matters on every request, and the boundary effect (up to
    2x the limit across a window edge) is acceptable for abuse protection.

    Any Redis failure falls back to the in-process limiter rather than either
    failing the request or letting it through unlimited.
    """

    def __init__(self, url: str, fallback: _InMemoryRateLimiter) -> None:
        import redis

        self._redis = redis.from_url(
            url, socket_connect_timeout=0.25, socket_timeout=0.25
        )
        self._fallback = fallback

    def check(self, key: str, limit: int, window: int) -> tuple[bool, int]:
        bucket = int(time.time() // window)
        redis_key = f"rl:{key}:{bucket}"
        try:
            pipeline = self._redis.pipeline()
            pipeline.incr(redis_key)
            pipeline.expire(redis_key, window)
            count, _ = pipeline.execute()
        except Exception as exc:  # noqa: BLE001 - never fail a request on this
            logger.warning(
                "rate limiter unavailable; using the in-process fallback",
                extra={"error": str(exc)},
            )
            return self._fallback.check(key, limit, window)

        if int(count) > limit:
            retry_after = window - int(time.time() % window)
            return False, max(1, retry_after)
        return True, 0


_memory_limiter = _InMemoryRateLimiter()
_limiter: _InMemoryRateLimiter | _RedisRateLimiter | None = None


def _get_limiter():
    """Choose a limiter once, preferring the shared one."""
    global _limiter
    if _limiter is not None:
        return _limiter
    try:
        _limiter = _RedisRateLimiter(settings.redis_url, _memory_limiter)
        # Fail fast here rather than on the first request.
        _limiter._redis.ping()
        logger.info("rate limiting backed by redis")
    except Exception:  # noqa: BLE001 - redis is optional
        _limiter = _memory_limiter
        logger.info("rate limiting is per-process (redis unreachable)")
    return _limiter


def reset_rate_limiter() -> None:
    """Test hook: forget the chosen backend and any counters."""
    global _limiter
    _limiter = None
    _memory_limiter._hits.clear()


async def rate_limit(request: Request) -> None:
    """Per-client request cap.

    Keyed by authenticated user where available, else by client address, so a
    shared NAT does not throttle every user behind it.
    """
    if not settings.rate_limit_enabled:
        return
    identity = request.headers.get("X-API-Key") or request.headers.get("Authorization")
    if identity:
        key = f"auth:{hash_api_key(identity)[:32]}"
    else:
        client = request.client.host if request.client else "unknown"
        key = f"ip:{client}"
    key = f"{key}:{request.url.path}"

    allowed, retry_after = _get_limiter().check(
        key, settings.rate_limit_requests, settings.rate_limit_window_seconds
    )
    if not allowed:
        raise RateLimitError(
            "Too many requests. Slow down and retry.",
            details={"retry_after_seconds": retry_after},
        )


RateLimited = Depends(rate_limit)
