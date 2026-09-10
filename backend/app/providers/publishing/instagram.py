"""Instagram Reels publishing through the official Instagram Graph API.

The Graph API pulls the media from a URL rather than accepting a file upload,
so the caller must pass ``video_url`` in ``credentials`` -- a signed, publicly
reachable URL produced by the storage provider. Publishing is a three-step
flow: create a container, poll until it finishes processing, then publish it.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Sequence

import httpx

from app.core.errors import ProviderError
from app.core.logging import get_logger
from app.providers.base import PlatformMetrics, PublishingProvider, PublishResult

logger = get_logger(__name__)

GRAPH_BASE = "https://graph.facebook.com/v21.0"
CONTAINER_POLL_SECONDS = 5
CONTAINER_TIMEOUT_SECONDS = 600


class InstagramReelsProvider(PublishingProvider):
    name = "instagram_reels"

    def __init__(
        self,
        timeout: float = 120.0,
        *,
        transport: httpx.BaseTransport | None = None,
        poll_seconds: float | None = None,
    ) -> None:
        self._timeout = timeout
        self._transport = transport
        # Overridable so tests do not sleep through the container poll.
        self._poll_seconds = (
            CONTAINER_POLL_SECONDS if poll_seconds is None else poll_seconds
        )

    def _client(self, timeout: float | None = None) -> httpx.Client:
        return httpx.Client(
            timeout=timeout or self._timeout, transport=self._transport
        )

    @staticmethod
    def _credentials(credentials: dict[str, Any]) -> tuple[str, str]:
        token = credentials.get("access_token")
        account_id = credentials.get("ig_user_id") or credentials.get("external_account_id")
        if not token or not account_id:
            raise ProviderError(
                "Instagram publishing needs an access token and an IG user id. "
                "Reconnect the account."
            )
        return str(token), str(account_id)

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
        token, account_id = self._credentials(credentials)
        video_url = credentials.get("video_url")
        if not video_url:
            raise ProviderError(
                "Instagram pulls media from a URL. Provide a signed 'video_url' "
                "reachable from the public internet."
            )

        tag_line = " ".join(t if t.startswith("#") else f"#{t}" for t in hashtags)
        caption = (description or title).strip()
        if tag_line:
            caption = f"{caption}\n\n{tag_line}"

        with self._client() as client:
            container = client.post(
                f"{GRAPH_BASE}/{account_id}/media",
                data={
                    "media_type": "REELS",
                    "video_url": video_url,
                    "caption": caption[:2200],
                    "share_to_feed": str(bool(credentials.get("share_to_feed", True))).lower(),
                    "access_token": token,
                },
            )
            if container.status_code >= 400:
                raise ProviderError(
                    f"Instagram container creation failed ({container.status_code}): "
                    f"{container.text[:500]}"
                )
            container_id = container.json().get("id")
            if not container_id:
                raise ProviderError("Instagram returned no container id.")

            self._await_container(client, container_id, token)

            published = client.post(
                f"{GRAPH_BASE}/{account_id}/media_publish",
                data={"creation_id": container_id, "access_token": token},
            )
            if published.status_code >= 400:
                raise ProviderError(
                    f"Instagram publish failed ({published.status_code}): "
                    f"{published.text[:500]}"
                )
            payload = published.json()

        media_id = payload.get("id")
        if not media_id:
            raise ProviderError("Instagram response contained no media id.")
        return PublishResult(
            external_post_id=media_id,
            external_url=f"https://www.instagram.com/reel/{media_id}/",
            raw=payload,
        )

    def _await_container(self, client: httpx.Client, container_id: str, token: str) -> None:
        deadline = time.monotonic() + CONTAINER_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            status = client.get(
                f"{GRAPH_BASE}/{container_id}",
                params={"fields": "status_code,status", "access_token": token},
            )
            if status.status_code >= 400:
                raise ProviderError(
                    f"Instagram container status failed ({status.status_code}): "
                    f"{status.text[:300]}"
                )
            code = status.json().get("status_code")
            if code == "FINISHED":
                return
            if code == "ERROR":
                raise ProviderError(
                    f"Instagram failed to process the video: {status.json().get('status')}"
                )
            time.sleep(self._poll_seconds)
        raise ProviderError("Instagram container did not finish processing in time.")

    def fetch_metrics(
        self, *, external_post_id: str, credentials: dict[str, Any]
    ) -> PlatformMetrics:
        token, _ = self._credentials(credentials)
        with self._client(60.0) as client:
            response = client.get(
                f"{GRAPH_BASE}/{external_post_id}/insights",
                params={
                    "metric": "plays,likes,comments,shares,saved,ig_reels_avg_watch_time",
                    "access_token": token,
                },
            )
        if response.status_code >= 400:
            raise ProviderError(
                f"Instagram insights failed ({response.status_code}): {response.text[:300]}"
            )
        values: dict[str, Any] = {}
        for entry in response.json().get("data", []):
            series = entry.get("values") or [{}]
            values[entry.get("name", "")] = series[0].get("value")

        avg_watch_ms = values.get("ig_reels_avg_watch_time")
        return PlatformMetrics(
            views=values.get("plays"),
            likes=values.get("likes"),
            comments=values.get("comments"),
            shares=values.get("shares"),
            saves=values.get("saved"),
            average_view_duration=(avg_watch_ms / 1000.0) if avg_watch_ms else None,
            raw=values,
        )
