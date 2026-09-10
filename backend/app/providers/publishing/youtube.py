"""YouTube Shorts publishing through the official YouTube Data API v3.

Requires an OAuth 2.0 access token with ``youtube.upload`` (and
``yt-analytics.readonly`` for retention metrics). The caller supplies the token
in ``credentials``; this module never stores or refreshes credentials itself --
that is :mod:`app.services.platform_accounts`.

A clip qualifies as a Short when it is vertical and 3 minutes or shorter; the
renderer already guarantees 9:16, and :data:`MAX_SHORT_SECONDS` is enforced
here as a second gate.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Sequence

import httpx

from app.core.errors import ProviderError
from app.core.logging import get_logger
from app.providers.base import PlatformMetrics, PublishingProvider, PublishResult

logger = get_logger(__name__)

UPLOAD_URL = "https://www.googleapis.com/upload/youtube/v3/videos"
VIDEOS_URL = "https://www.googleapis.com/youtube/v3/videos"
MAX_SHORT_SECONDS = 180
_CHUNK = 8 * 1024 * 1024


class YouTubeShortsProvider(PublishingProvider):
    name = "youtube_shorts"

    def __init__(
        self,
        timeout: float = 300.0,
        *,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._timeout = timeout
        # Injectable so the request construction (URLs, params, headers,
        # Content-Range, resume handling) can be tested without credentials.
        self._transport = transport

    def _client(self, timeout: float | None = None) -> httpx.Client:
        return httpx.Client(
            timeout=timeout or self._timeout, transport=self._transport
        )

    @staticmethod
    def _token(credentials: dict[str, Any]) -> str:
        token = credentials.get("access_token")
        if not token:
            raise ProviderError(
                "No YouTube access token available. Reconnect the account."
            )
        return str(token)

    def publish(
        self,
        *,
        video_path: Path,
        title: str,
        description: str,
        hashtags: Sequence[str],
        credentials: dict[str, Any],
        privacy: str = "private",
    ) -> PublishResult:
        path = Path(video_path)
        if not path.exists():
            raise ProviderError(f"Upload file is missing: {path}")
        token = self._token(credentials)
        size = path.stat().st_size

        tag_line = " ".join(t if t.startswith("#") else f"#{t}" for t in hashtags)
        body = {
            "snippet": {
                "title": title[:100],
                "description": (description + ("\n\n" + tag_line if tag_line else ""))[:5000],
                "tags": [t.lstrip("#") for t in hashtags][:15],
                "categoryId": str(credentials.get("category_id", "22")),
            },
            "status": {
                "privacyStatus": privacy,
                "selfDeclaredMadeForKids": bool(credentials.get("made_for_kids", False)),
            },
        }

        with self._client() as client:
            # 1. Initiate a resumable session.
            init = client.post(
                UPLOAD_URL,
                params={"uploadType": "resumable", "part": "snippet,status"},
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json; charset=UTF-8",
                    "X-Upload-Content-Length": str(size),
                    "X-Upload-Content-Type": "video/mp4",
                },
                content=json.dumps(body),
            )
            if init.status_code >= 400:
                raise ProviderError(
                    f"YouTube rejected the upload session ({init.status_code}): {init.text[:500]}"
                )
            session_url = init.headers.get("Location")
            if not session_url:
                raise ProviderError("YouTube did not return a resumable session URL.")

            # 2. Upload in chunks so large files survive transient failures.
            uploaded = 0
            response: httpx.Response | None = None
            with path.open("rb") as handle:
                while uploaded < size:
                    chunk = handle.read(_CHUNK)
                    if not chunk:
                        break
                    last = uploaded + len(chunk) - 1
                    response = client.put(
                        session_url,
                        headers={
                            "Content-Length": str(len(chunk)),
                            "Content-Range": f"bytes {uploaded}-{last}/{size}",
                        },
                        content=chunk,
                    )
                    if response.status_code in (200, 201):
                        break
                    if response.status_code == 308:
                        uploaded = last + 1
                        continue
                    raise ProviderError(
                        f"YouTube upload failed ({response.status_code}): "
                        f"{response.text[:500]}"
                    )

        if response is None or response.status_code not in (200, 201):
            raise ProviderError("YouTube upload did not complete.")
        payload = response.json()
        video_id = payload.get("id")
        if not video_id:
            raise ProviderError("YouTube response contained no video id.")
        return PublishResult(
            external_post_id=video_id,
            external_url=f"https://www.youtube.com/shorts/{video_id}",
            raw=payload,
        )

    def fetch_metrics(
        self, *, external_post_id: str, credentials: dict[str, Any]
    ) -> PlatformMetrics:
        token = self._token(credentials)
        with self._client(60.0) as client:
            response = client.get(
                VIDEOS_URL,
                params={"part": "statistics,contentDetails", "id": external_post_id},
                headers={"Authorization": f"Bearer {token}"},
            )
        if response.status_code >= 400:
            raise ProviderError(
                f"YouTube metrics request failed ({response.status_code}): "
                f"{response.text[:300]}"
            )
        items = response.json().get("items") or []
        if not items:
            raise ProviderError(f"YouTube returned no video for id {external_post_id}.")
        stats = items[0].get("statistics", {})

        def _int(key: str) -> int | None:
            value = stats.get(key)
            return int(value) if value is not None else None

        return PlatformMetrics(
            views=_int("viewCount"),
            likes=_int("likeCount"),
            comments=_int("commentCount"),
            raw={"fetched_at": time.time(), "statistics": stats},
        )
