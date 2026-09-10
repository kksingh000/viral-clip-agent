"""Manual-export "publishing" provider.

Records the intent to publish and hands back a stable local reference. This is
the default: the platform never uploads anywhere on the user's behalf until an
account is explicitly connected.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Sequence

from app.core.errors import ProviderError
from app.providers.base import PlatformMetrics, PublishingProvider, PublishResult


class ManualExportProvider(PublishingProvider):
    name = "manual_export"
    is_manual = True

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
        if not Path(video_path).exists():
            raise ProviderError(f"Export file is missing: {video_path}")
        return PublishResult(
            external_post_id=f"manual-{uuid.uuid4().hex[:12]}",
            external_url=None,
            raw={
                "mode": "manual_export",
                "file": str(video_path),
                "title": title,
                "hashtags": list(hashtags),
                "note": "Downloaded for manual upload; nothing was posted.",
            },
        )

    def fetch_metrics(
        self, *, external_post_id: str, credentials: dict[str, Any]
    ) -> PlatformMetrics:
        # There is no platform to query for a manual export.
        return PlatformMetrics(raw={"mode": "manual_export"})
