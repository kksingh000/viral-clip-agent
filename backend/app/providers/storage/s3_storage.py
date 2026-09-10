"""S3 / S3-compatible object storage (AWS S3, MinIO, R2, ...)."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, BinaryIO

from app.core.config import settings
from app.core.errors import NotFoundError, ProviderError, ValidationFailure
from app.core.logging import get_logger
from app.providers.base import StorageProvider, StoredObject
from app.providers.storage.local_storage import _sanitize_key

logger = get_logger(__name__)

_CHUNK = 8 * 1024 * 1024


class S3StorageProvider(StorageProvider):
    name = "s3"

    def __init__(
        self,
        bucket: str | None = None,
        *,
        region: str | None = None,
        endpoint_url: str | None = None,
    ) -> None:
        try:
            import boto3
            from botocore.config import Config
        except ImportError as exc:  # pragma: no cover
            raise ProviderError("The 'boto3' package is required for S3 storage.") from exc

        self.bucket = bucket or settings.s3_bucket
        if not self.bucket:
            raise ProviderError("S3_BUCKET is not configured.")
        self._client = boto3.client(
            "s3",
            region_name=region or settings.s3_region,
            endpoint_url=endpoint_url or settings.s3_endpoint_url,
            aws_access_key_id=settings.s3_access_key_id,
            aws_secret_access_key=settings.s3_secret_access_key,
            config=Config(signature_version="s3v4", retries={"max_attempts": 5}),
        )

    def _extra(self, content_type: str | None) -> dict[str, Any]:
        return {"ContentType": content_type} if content_type else {}

    def put_file(
        self, key: str, path: Path, *, content_type: str | None = None
    ) -> StoredObject:
        clean = _sanitize_key(key)
        source = Path(path)
        if not source.exists():
            raise NotFoundError(f"Local file not found: {path}")
        digest = _sha256(source)
        self._client.upload_file(
            str(source), self.bucket, clean, ExtraArgs=self._extra(content_type)
        )
        return StoredObject(
            key=clean,
            size_bytes=source.stat().st_size,
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
        clean = _sanitize_key(key)
        # Spool to a temp file so the size limit is enforced before any part is
        # committed, and so a checksum can be computed in one pass.
        import tempfile

        hasher = hashlib.sha256()
        written = 0
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = Path(tmp.name)
            while chunk := stream.read(1024 * 1024):
                written += len(chunk)
                if max_bytes is not None and written > max_bytes:
                    tmp_path.unlink(missing_ok=True)
                    raise ValidationFailure(f"Upload exceeds the {max_bytes} byte limit.")
                hasher.update(chunk)
                tmp.write(chunk)
        try:
            self._client.upload_file(
                str(tmp_path), self.bucket, clean, ExtraArgs=self._extra(content_type)
            )
        finally:
            tmp_path.unlink(missing_ok=True)
        return StoredObject(
            key=clean,
            size_bytes=written,
            content_type=content_type,
            checksum_sha256=hasher.hexdigest(),
        )

    def get_to_path(self, key: str, destination: Path) -> Path:
        clean = _sanitize_key(key)
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._client.download_file(self.bucket, clean, str(destination))
        except Exception as exc:  # noqa: BLE001
            raise NotFoundError(f"Object not found: {key} ({exc})") from exc
        return destination

    def exists(self, key: str) -> bool:
        try:
            self._client.head_object(Bucket=self.bucket, Key=_sanitize_key(key))
            return True
        except Exception:  # noqa: BLE001 - head_object raises for 404 and 403
            return False

    def delete(self, key: str) -> None:
        self._client.delete_object(Bucket=self.bucket, Key=_sanitize_key(key))

    def local_path(self, key: str) -> Path | None:
        return None

    def signed_url(self, key: str, *, expires_in: int | None = None) -> str:
        return self._client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket, "Key": _sanitize_key(key)},
            ExpiresIn=int(expires_in or settings.signed_url_ttl_seconds),
        )


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            hasher.update(chunk)
    return hasher.hexdigest()
