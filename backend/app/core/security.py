"""Password hashing and JWT issuance/verification."""

from __future__ import annotations

import hashlib
import hmac
import secrets
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Literal

import jwt
from jwt import InvalidTokenError

from app.core.config import settings
from app.core.errors import AuthenticationError

TokenType = Literal["access", "refresh"]

_PBKDF2_ROUNDS = 390_000
MIN_PASSWORD_LENGTH = 8


def hash_password(password: str) -> str:
    """Hash a password with PBKDF2-HMAC-SHA256.

    PBKDF2 from the standard library avoids a native build step in the
    container image while remaining sound at this iteration count. Swap in
    argon2 by replacing this pair of functions.
    """
    if not password or len(password) < MIN_PASSWORD_LENGTH:
        raise ValueError(
            f"Password must be at least {MIN_PASSWORD_LENGTH} characters long."
        )
    salt = secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, _PBKDF2_ROUNDS)
    return f"pbkdf2_sha256${_PBKDF2_ROUNDS}${salt.hex()}${dk.hex()}"


def verify_password(password: str, encoded: str) -> bool:
    try:
        algorithm, rounds_s, salt_hex, hash_hex = encoded.split("$")
    except ValueError:
        return False
    if algorithm != "pbkdf2_sha256":
        return False
    try:
        rounds = int(rounds_s)
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(hash_hex)
    except ValueError:
        return False
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, rounds)
    return hmac.compare_digest(dk, expected)


def _create_token(subject: str, token_type: TokenType, expires: timedelta) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": subject,
        "type": token_type,
        "iat": int(now.timestamp()),
        "exp": int((now + expires).timestamp()),
        "jti": uuid.uuid4().hex,
    }
    return jwt.encode(payload, settings.secret_key, algorithm=settings.jwt_algorithm)


def create_access_token(subject: str) -> str:
    return _create_token(
        subject, "access", timedelta(minutes=settings.access_token_expire_minutes)
    )


def create_refresh_token(subject: str) -> str:
    return _create_token(
        subject, "refresh", timedelta(days=settings.refresh_token_expire_days)
    )


def decode_token(token: str, *, expected_type: TokenType = "access") -> dict[str, Any]:
    try:
        payload = jwt.decode(
            token, settings.secret_key, algorithms=[settings.jwt_algorithm]
        )
    except InvalidTokenError as exc:
        raise AuthenticationError(f"Invalid token: {exc}") from exc
    if payload.get("type") != expected_type:
        raise AuthenticationError(
            "Token type mismatch: expected " + expected_type + "."
        )
    if not payload.get("sub"):
        raise AuthenticationError("Token is missing a subject.")
    return payload


def generate_api_key() -> tuple[str, str]:
    """Return ``(plaintext_key, sha256_digest)``; only the digest is stored."""
    raw = "vca_" + secrets.token_urlsafe(32)
    return raw, hashlib.sha256(raw.encode()).hexdigest()


def hash_api_key(raw: str) -> str:
    return hashlib.sha256(raw.encode()).hexdigest()
