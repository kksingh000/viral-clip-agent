"""Signed media delivery for the local storage backend.

Local storage is not a public directory. Object URLs produced by
:class:`LocalStorageProvider` carry an HMAC signature and an expiry, and this
route refuses to serve anything without a valid one -- so local development
behaves like S3 presigned URLs rather than like an open file server.
"""

from __future__ import annotations

import mimetypes
from typing import Annotated

from fastapi import APIRouter, Query
from fastapi.responses import FileResponse

from app.core.config import settings
from app.core.errors import ForbiddenError, NotFoundError
from app.providers.registry import get_storage_provider
from app.providers.storage.local_storage import (
    LocalStorageProvider,
    _sanitize_key,
    verify_signature,
)

router = APIRouter(prefix="/media", tags=["media"], include_in_schema=False)


@router.get("/{key:path}")
async def serve_media(
    key: str,
    expires: Annotated[int, Query()],
    signature: Annotated[str, Query(max_length=128)],
) -> FileResponse:
    if settings.storage_provider != "local":
        raise NotFoundError("Media is served directly by the object store.")

    clean = _sanitize_key(key)
    if not verify_signature(clean, expires, signature):
        raise ForbiddenError("This media link is invalid or has expired.")

    storage = get_storage_provider()
    if not isinstance(storage, LocalStorageProvider):  # pragma: no cover
        raise NotFoundError("Local media delivery is not enabled.")

    path = storage.local_path(clean)
    if path is None:
        raise NotFoundError("Media not found.")

    content_type, _ = mimetypes.guess_type(path.name)
    return FileResponse(
        path,
        media_type=content_type or "application/octet-stream",
        headers={"Cache-Control": "private, max-age=300"},
    )
