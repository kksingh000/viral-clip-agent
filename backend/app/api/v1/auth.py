"""Authentication endpoints."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, status
from sqlalchemy import select

from app.api.deps import CurrentUser, DbSession
from app.core.config import settings
from app.core.errors import (
    AuthenticationError,
    ConflictError,
    ForbiddenError,
    NotFoundError,
)
from app.core.logging import get_logger
from app.core.security import (
    create_access_token,
    create_refresh_token,
    decode_token,
    generate_api_key,
    hash_password,
    verify_password,
)
from app.models.user import ApiKey, User, UserSettings
from app.schemas.auth import (
    ApiKeyCreatedResponse,
    ApiKeyCreateRequest,
    ApiKeyResponse,
    LoginRequest,
    RefreshRequest,
    RegisterRequest,
    TokenResponse,
    UserResponse,
)
from app.schemas.common import MessageResponse

logger = get_logger(__name__)
router = APIRouter(prefix="/auth", tags=["auth"])


def _tokens(user: User) -> TokenResponse:
    return TokenResponse(
        access_token=create_access_token(str(user.id)),
        refresh_token=create_refresh_token(str(user.id)),
        expires_in=settings.access_token_expire_minutes * 60,
    )


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def register(payload: RegisterRequest, session: DbSession) -> TokenResponse:
    if not settings.allow_registration:
        raise ForbiddenError("Self-registration is disabled on this deployment.")

    email = payload.email.lower()
    existing = (
        await session.execute(select(User).where(User.email == email))
    ).scalars().first()
    if existing is not None:
        raise ConflictError("An account with that email already exists.")

    user = User(
        email=email,
        hashed_password=hash_password(payload.password),
        full_name=payload.full_name,
    )
    session.add(user)
    await session.flush()
    session.add(UserSettings(user_id=user.id))
    await session.commit()
    logger.info("user registered", extra={"user_id": str(user.id)})
    return _tokens(user)


@router.post("/login", response_model=TokenResponse)
async def login(payload: LoginRequest, session: DbSession) -> TokenResponse:
    user = (
        await session.execute(select(User).where(User.email == payload.email.lower()))
    ).scalars().first()
    # Verify against a dummy hash when the user is missing so the response time
    # does not reveal whether the address is registered.
    stored = user.hashed_password if user else hash_password("not-a-real-password")
    if not verify_password(payload.password, stored) or user is None:
        raise AuthenticationError("Incorrect email or password.")
    if not user.is_active:
        raise ForbiddenError("This account is disabled.")

    user.last_login_at = datetime.now(timezone.utc)
    await session.commit()
    return _tokens(user)


@router.post("/refresh", response_model=TokenResponse)
async def refresh(payload: RefreshRequest, session: DbSession) -> TokenResponse:
    decoded = decode_token(payload.refresh_token, expected_type="refresh")
    try:
        user_id = uuid.UUID(str(decoded["sub"]))
    except ValueError as exc:
        raise AuthenticationError("Malformed refresh token.") from exc
    user = await session.get(User, user_id)
    if user is None or not user.is_active:
        raise AuthenticationError("This refresh token is no longer valid.")
    return _tokens(user)


@router.get("/me", response_model=UserResponse)
async def me(user: CurrentUser) -> User:
    return user


@router.get("/api-keys", response_model=list[ApiKeyResponse])
async def list_api_keys(user: CurrentUser, session: DbSession) -> list[ApiKey]:
    rows = await session.execute(
        select(ApiKey).where(ApiKey.user_id == user.id).order_by(ApiKey.created_at.desc())
    )
    return list(rows.scalars())


@router.post(
    "/api-keys",
    response_model=ApiKeyCreatedResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_api_key(
    payload: ApiKeyCreateRequest, user: CurrentUser, session: DbSession
) -> ApiKeyCreatedResponse:
    raw, digest = generate_api_key()
    record = ApiKey(
        user_id=user.id,
        name=payload.name,
        key_prefix=raw[:12],
        key_hash=digest,
    )
    session.add(record)
    await session.commit()
    await session.refresh(record)
    # The plaintext key is returned exactly once and never stored.
    return ApiKeyCreatedResponse(
        id=record.id,
        name=record.name,
        key_prefix=record.key_prefix,
        created_at=record.created_at,
        last_used_at=None,
        revoked_at=None,
        api_key=raw,
    )


@router.delete("/api-keys/{key_id}", response_model=MessageResponse)
async def revoke_api_key(
    key_id: uuid.UUID, user: CurrentUser, session: DbSession
) -> MessageResponse:
    record = await session.get(ApiKey, key_id)
    if record is None or record.user_id != user.id:
        raise NotFoundError("API key not found.")
    record.revoked_at = datetime.now(timezone.utc)
    await session.commit()
    return MessageResponse(message="API key revoked.")
