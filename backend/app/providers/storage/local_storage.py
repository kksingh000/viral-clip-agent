"""Filesystem-backed storage for local development and single-node deploys."""

from __future__ import annotations

import base64
import hashlib
import hmac
import shutil
import time
from pathlib import Path
from typing import BinaryIO
from urllib.parse import quote, urlencode

from app.core.config import settings
from app.core.errors import NotFoundError, ValidationFailure
from app.core.logging import get_logger
from app.providers.base import StorageProvider, StoredObject

logger = get_logger(__name__)

_CHUNK = 1024 * 1024


def _sanitize_key(key: str) -> str:
    """Reject traversal and absolute keys before they touch the filesystem."""
    key = key.replace("\\", "/").strip("/")
    if not key:
        raise ValidationFailure("Storage key must not be empty.")
    parts = [p for p in key.split("/") if p not in ("", ".")]
    if any(p == ".." for p in parts):
        raise ValidationFailure(f"Storage key escapes the root: {key!r}")
    if any(":" in p for p in parts):
        raise ValidationFailure(f"Storage key contains a drive separator: {key!r}")
    return "/".join(parts)


class LocalStorageProvider(StorageProvider):
    name = "local"

    def __init__(self, root: Path | None = None, base_url: str | None = None) -> None:
        self.root = Path(root or settings.storage_local_root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.base_url = (base_url or settings.storage_public_base_url).rstrip("/")

    def _path(self, key: str) -> Path:
        path = (self.root / _sanitize_key(key)).resolve()
        # Defence in depth: the resolved path must still sit under the root.
        if not str(path).startswith(str(self.root)):
            raise ValidationFailure(f"Storage key escapes the root: {key!r}")
        return path

    # ------------------------------------------------------------------ write
    def put_file(
        self, key: str, path: Path, *, content_type: str | None = None
    ) -> StoredObject:
        destination = self._path(key)
        destination.parent.mkdir(parents=True, exist_ok=True)
        source = Path(path)
        if source.resolve() != destination:
            shutil.copyfile(source, destination)
        digest = _sha256(destination)
        return StoredObject(
            key=_sanitize_key(key),
            size_bytes=destination.stat().st_size,
            content_type=content_type,
            checksum_sha256=digest,
        )

    def put_stream(
        self,
        key: str,
        stream: BinaryIO,
        *,
        content_type: str | None = None,
        max_bytes: int | None = None,
    ) -> StoredObject:
        destination = self._path(key)
        destination.parent.mkdir(parents=True, exist_ok=True)
        hasher = hashlib.sha256()
        written = 0
        try:
            with destination.open("wb") as handle:
                while chunk := stream.read(_CHUNK):
                    written += len(chunk)
                    if max_bytes is not None and written > max_bytes:
                        raise ValidationFailure(
                            f"Upload exceeds the {max_bytes} byte limit."
                        )
                    hasher.update(chunk)
                    handle.write(chunk)
        except Exception:
            destination.unlink(missing_ok=True)
            raise
        return StoredObject(
            key=_sanitize_key(key),
            size_bytes=written,
            content_type=content_type,
            checksum_sha256=hasher.hexdigest(),
        )

    # ------------------------------------------------------------------- read
    def get_to_path(self, key: str, destination: Path) -> Path:
        source = self._path(key)
        if not source.exists():
            raise NotFoundError(f"Object not found: {key}")
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source != destination.resolve():
            shutil.copyfile(source, destination)
        return destination

    def exists(self, key: str) -> bool:
        return self._path(key).exists()

    def delete(self, key: str) -> None:
        self._path(key).unlink(missing_ok=True)

    def local_path(self, key: str) -> Path | None:
        path = self._path(key)
        return path if path.exists() else None

    # ------------------------------------------------------------------- urls
    def signed_url(self, key: str, *, expires_in: int | None = None) -> str:
        """HMAC-signed URL served by :mod:`app.api.v1.media`.

        Local storage is not public: the media route verifies this signature
        before streaming a file, so object URLs behave like S3 presigned URLs.
        """
        clean = _sanitize_key(key)
        expiry = int(time.time()) + int(expires_in or settings.signed_url_ttl_seconds)
        signature = sign_key(clean, expiry)
        query = urlencode({"expires": expiry, "signature": signature})
        return f"{self.base_url}/{quote(clean)}?{query}"


def sign_key(key: str, expires: int) -> str:
    message = f"{key}:{expires}".encode()
    digest = hmac.new(settings.secret_key.encode(), message, hashlib.sha256).digest()
    return base64.urlsafe_b64encode(digest).decode().rstrip("=")


def verify_signature(key: str, expires: int, signature: str) -> bool:
    if expires < int(time.time()):
        return False
    return hmac.compare_digest(sign_key(key, expires), signature)


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            hasher.update(chunk)
    return hasher.hexdigest()
