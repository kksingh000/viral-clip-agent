"""Trend discovery over publicly available metadata.

**This module never downloads media.** It reads metadata through the official
YouTube Data API and records it. Every video it creates is stored with
``authorization_status = UNKNOWN`` and ``status = DISCOVERED``; the processing
pipeline refuses to touch such a record until a human establishes rights.
Discovery tells you what is moving -- it does not tell you what you may
republish, and the API is deliberately shaped so the two cannot be confused.
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Sequence

import httpx
from slugify import slugify
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.agents.base import AgentContext
from app.agents.trend_agent import TopicPayload, TrendAgent
from app.core.config import settings
from app.core.errors import ProviderError
from app.core.logging import get_logger
from app.models.enums import AuthorizationStatus, VideoSource, VideoStatus
from app.models.trend import Topic, TopicMention
from app.models.user import UserSettings
from app.models.video import Channel, Video, VideoMetric
from app.services.trends import (
    MetricSnapshot,
    compute_metrics,
    compute_trend_score,
    topic_trend_score,
)

logger = get_logger(__name__)

API_BASE = "https://www.googleapis.com/youtube/v3"
#: Videos discovered longer ago than this are not re-polled for metrics.
METRIC_REFRESH_WINDOW_HOURS = 96
MAX_PAGE_SIZE = 50

_ISO_DURATION = re.compile(
    r"P(?:(?P<days>\d+)D)?T?(?:(?P<hours>\d+)H)?(?:(?P<minutes>\d+)M)?(?:(?P<seconds>\d+)S)?"
)


def parse_iso8601_duration(value: str | None) -> float | None:
    """Convert an ISO-8601 duration (``PT4M13S``) to seconds."""
    if not value:
        return None
    match = _ISO_DURATION.fullmatch(value.strip())
    if not match:
        return None
    parts = {k: int(v) for k, v in match.groupdict(default="0").items()}
    return float(
        parts["days"] * 86400
        + parts["hours"] * 3600
        + parts["minutes"] * 60
        + parts["seconds"]
    )


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


class YouTubeMetadataClient:
    """Read-only client for the official YouTube Data API v3."""

    def __init__(
        self,
        api_key: str | None = None,
        timeout: float = 30.0,
        *,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.api_key = api_key or settings.youtube_api_key
        if not self.api_key:
            raise ProviderError(
                "YOUTUBE_API_KEY is not configured. Trend discovery reads the "
                "official YouTube Data API and cannot run without a key."
            )
        self._timeout = timeout
        self._transport = transport

    def _get(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        params = {**params, "key": self.api_key}
        with httpx.Client(timeout=self._timeout, transport=self._transport) as client:
            response = client.get(f"{API_BASE}/{path}", params=params)
        if response.status_code == 403:
            raise ProviderError(
                "YouTube API returned 403. The key may be missing quota or the "
                f"Data API may not be enabled: {response.text[:300]}"
            )
        if response.status_code >= 400:
            raise ProviderError(
                f"YouTube API error {response.status_code}: {response.text[:300]}"
            )
        return response.json()

    def most_popular(
        self, *, region: str = "US", category_id: str | None = None, limit: int = 50
    ) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        page_token: str | None = None
        while len(items) < limit:
            params: dict[str, Any] = {
                "part": "snippet,statistics,contentDetails",
                "chart": "mostPopular",
                "regionCode": region,
                "maxResults": min(MAX_PAGE_SIZE, limit - len(items)),
            }
            if category_id:
                params["videoCategoryId"] = category_id
            if page_token:
                params["pageToken"] = page_token
            payload = self._get("videos", params)
            items.extend(payload.get("items", []))
            page_token = payload.get("nextPageToken")
            if not page_token:
                break
        return items[:limit]

    def search_recent(
        self, query: str, *, region: str = "US", limit: int = 50, days: int = 3
    ) -> list[dict[str, Any]]:
        published_after = (
            datetime.now(timezone.utc) - timedelta(days=days)
        ).strftime("%Y-%m-%dT%H:%M:%SZ")
        payload = self._get(
            "search",
            {
                "part": "snippet",
                "q": query,
                "type": "video",
                "order": "viewCount",
                "regionCode": region,
                "publishedAfter": published_after,
                "maxResults": min(MAX_PAGE_SIZE, limit),
            },
        )
        ids = [
            item["id"]["videoId"]
            for item in payload.get("items", [])
            if item.get("id", {}).get("videoId")
        ]
        return self.videos_by_id(ids)

    def videos_by_id(self, video_ids: Sequence[str]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for chunk in _chunks(list(video_ids), MAX_PAGE_SIZE):
            if not chunk:
                continue
            payload = self._get(
                "videos",
                {
                    "part": "snippet,statistics,contentDetails",
                    "id": ",".join(chunk),
                },
            )
            out.extend(payload.get("items", []))
        return out

    def channels_by_id(self, channel_ids: Sequence[str]) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for chunk in _chunks(sorted(set(channel_ids)), MAX_PAGE_SIZE):
            if not chunk:
                continue
            payload = self._get(
                "channels", {"part": "snippet,statistics", "id": ",".join(chunk)}
            )
            for item in payload.get("items", []):
                out[item["id"]] = item
        return out


def _chunks(values: list[str], size: int) -> Iterable[list[str]]:
    for index in range(0, len(values), size):
        yield values[index : index + size]


# ------------------------------------------------------------------ upserting
def _upsert_channel(session: Session, payload: dict[str, Any] | None, external_id: str,
                    fallback_title: str) -> Channel:
    channel = session.execute(
        select(Channel).where(
            Channel.platform == "youtube", Channel.external_id == external_id
        )
    ).scalars().first()
    if channel is None:
        channel = Channel(
            platform="youtube", external_id=external_id, title=fallback_title
        )
        session.add(channel)
    if payload:
        snippet = payload.get("snippet", {})
        statistics = payload.get("statistics", {})
        channel.title = snippet.get("title") or channel.title
        channel.handle = snippet.get("customUrl")
        channel.description = snippet.get("description")
        channel.country = snippet.get("country")
        channel.thumbnail_url = (
            snippet.get("thumbnails", {}).get("default", {}).get("url")
        )
        channel.subscriber_count = _as_int(statistics.get("subscriberCount"))
        channel.video_count = _as_int(statistics.get("videoCount"))
        channel.view_count = _as_int(statistics.get("viewCount"))
        channel.metrics_fetched_at = datetime.now(timezone.utc)
    session.flush()
    return channel


def _upsert_video(
    session: Session,
    *,
    user_id: uuid.UUID,
    item: dict[str, Any],
    channel: Channel | None,
) -> tuple[Video, bool]:
    external_id = item.get("id")
    snippet = item.get("snippet", {})
    statistics = item.get("statistics", {})
    details = item.get("contentDetails", {})

    video = session.execute(
        select(Video).where(
            Video.user_id == user_id,
            Video.source == VideoSource.DISCOVERY,
            Video.external_id == external_id,
        )
    ).scalars().first()
    created = video is None

    if video is None:
        video = Video(
            user_id=user_id,
            source=VideoSource.DISCOVERY,
            external_id=external_id,
            # Discovery establishes no rights whatsoever.
            authorization_status=AuthorizationStatus.UNKNOWN,
            status=VideoStatus.DISCOVERED,
            title=snippet.get("title") or "(untitled)",
        )
        session.add(video)

    video.channel_id = channel.id if channel else None
    video.title = snippet.get("title") or video.title
    video.description = snippet.get("description")
    video.creator_name = snippet.get("channelTitle")
    video.category = snippet.get("categoryId")
    video.language = snippet.get("defaultAudioLanguage") or snippet.get("defaultLanguage")
    video.tags = list(snippet.get("tags") or [])[:50]
    video.published_at = _parse_datetime(snippet.get("publishedAt"))
    video.source_url = f"https://www.youtube.com/watch?v={external_id}"
    video.thumbnail_url = (
        snippet.get("thumbnails", {}).get("high", {}).get("url")
        or snippet.get("thumbnails", {}).get("default", {}).get("url")
    )
    video.duration_seconds = parse_iso8601_duration(details.get("duration"))
    video.view_count = _as_int(statistics.get("viewCount"))
    video.like_count = _as_int(statistics.get("likeCount"))
    video.comment_count = _as_int(statistics.get("commentCount"))
    session.flush()
    return video, created


def _record_snapshot(session: Session, video: Video, *, weights: dict | None) -> float:
    now = datetime.now(timezone.utc)
    snapshot = VideoMetric(
        video_id=video.id,
        captured_at=now,
        view_count=video.view_count,
        like_count=video.like_count,
        comment_count=video.comment_count,
    )
    session.add(snapshot)
    session.flush()

    history = list(
        session.execute(
            select(VideoMetric)
            .where(VideoMetric.video_id == video.id)
            .order_by(VideoMetric.captured_at.desc())
            .limit(6)
        ).scalars()
    )
    metrics = compute_metrics(
        published_at=video.published_at,
        snapshots=[
            MetricSnapshot(
                captured_at=row.captured_at,
                view_count=row.view_count,
                like_count=row.like_count,
                comment_count=row.comment_count,
            )
            for row in reversed(history)
        ],
        now=now,
    )
    result = compute_trend_score(metrics, weights=weights, topic_trend=0.0)

    snapshot.view_velocity = metrics.view_velocity
    snapshot.engagement_rate = metrics.engagement_rate
    snapshot.comment_rate = metrics.comment_rate
    snapshot.growth_acceleration = metrics.growth_acceleration
    snapshot.trend_score = result.score

    video.trend_score = result.score
    video.trend_breakdown = result.to_dict()
    session.flush()
    return result.score


# -------------------------------------------------------------------- topics
def _upsert_topics(
    session: Session,
    videos: Sequence[Video],
    *,
    region: str,
    context: AgentContext,
) -> list[Topic]:
    if not videos:
        return []

    known = [
        row.label
        for row in session.execute(select(Topic).order_by(Topic.last_seen_at.desc().nullslast()).limit(80)).scalars()
    ]
    payload = TopicPayload(
        videos=[
            {
                "index": index,
                "title": video.title,
                "channel": video.creator_name,
                "category": video.category,
                "tags": video.tags[:12],
                "description": (video.description or "")[:400],
            }
            for index, video in enumerate(videos)
        ],
        region=region,
        known_topics=known,
        max_topics=15,
    )
    result = TrendAgent().run(payload, context=context)

    now = datetime.now(timezone.utc)
    touched: list[Topic] = []
    for detected in result.output.topics:
        slug = slugify(detected.label)[:160]
        if not slug:
            continue
        topic = session.execute(select(Topic).where(Topic.slug == slug)).scalars().first()
        if topic is None:
            topic = Topic(
                slug=slug,
                label=detected.label,
                kind=detected.kind,
                region=region,
                first_seen_at=now,
            )
            session.add(topic)
            session.flush()

        matched = [videos[i] for i in detected.video_indexes if i < len(videos)]
        previous_count = topic.video_count
        new_links = 0
        for video in matched:
            exists = session.execute(
                select(TopicMention).where(
                    TopicMention.topic_id == topic.id, TopicMention.video_id == video.id
                )
            ).scalars().first()
            if exists is None:
                session.add(
                    TopicMention(
                        topic_id=topic.id,
                        video_id=video.id,
                        matched_terms=detected.keywords[:8],
                    )
                )
                new_links += 1

        topic.keywords = sorted(set(topic.keywords) | set(detected.keywords))[:20]
        topic.description = detected.rationale or topic.description
        topic.video_count = previous_count + new_links
        topic.total_views += sum(v.view_count or 0 for v in matched)
        topic.last_seen_at = now
        score, growth = topic_trend_score(
            video_count=topic.video_count,
            previous_video_count=previous_count,
            total_views=topic.total_views,
        )
        topic.trend_score = score
        topic.growth_rate = growth
        topic.history = (topic.history or [])[-29:] + [
            {
                "at": now.isoformat(),
                "video_count": topic.video_count,
                "trend_score": round(score, 2),
            }
        ]
        touched.append(topic)
    session.flush()
    return touched


# ----------------------------------------------------------------- entrypoints
def run_discovery(
    session: Session,
    *,
    user_id: uuid.UUID,
    region: str = "US",
    category_id: str | None = None,
    max_results: int = 50,
    query: str | None = None,
    context: AgentContext | None = None,
) -> dict[str, Any]:
    """Discover trending videos and record metadata only."""
    context = context or AgentContext()
    client = YouTubeMetadataClient()

    items = (
        client.search_recent(query, region=region, limit=max_results)
        if query
        else client.most_popular(
            region=region, category_id=category_id, limit=max_results
        )
    )
    if not items:
        return {"discovered": 0, "created": 0, "topics": 0, "region": region}

    channel_ids = [
        item.get("snippet", {}).get("channelId")
        for item in items
        if item.get("snippet", {}).get("channelId")
    ]
    try:
        channel_payloads = client.channels_by_id(channel_ids)
    except ProviderError as exc:
        logger.warning("channel lookup failed", extra={"error": str(exc)})
        channel_payloads = {}

    user_settings = session.execute(
        select(UserSettings).where(UserSettings.user_id == user_id)
    ).scalars().first()
    weights = user_settings.trend_weights if user_settings else None

    videos: list[Video] = []
    created_count = 0
    for item in items:
        snippet = item.get("snippet", {})
        channel_external = snippet.get("channelId")
        channel = (
            _upsert_channel(
                session,
                channel_payloads.get(channel_external),
                channel_external,
                snippet.get("channelTitle") or "(unknown channel)",
            )
            if channel_external
            else None
        )
        video, created = _upsert_video(
            session, user_id=user_id, item=item, channel=channel
        )
        created_count += int(created)
        _record_snapshot(session, video, weights=weights)
        videos.append(video)

    topics = _upsert_topics(session, videos, region=region, context=context)

    # Fold topic heat back into each video's trend score.
    topic_heat = {t.id: t.trend_score / 100.0 for t in topics}
    for video in videos:
        mentions = session.execute(
            select(TopicMention).where(TopicMention.video_id == video.id)
        ).scalars()
        heats = [topic_heat.get(m.topic_id, 0.0) for m in mentions]
        if not heats:
            continue
        breakdown = dict(video.trend_breakdown or {})
        metrics = compute_metrics(
            published_at=video.published_at,
            snapshots=[
                MetricSnapshot(
                    captured_at=row.captured_at,
                    view_count=row.view_count,
                    like_count=row.like_count,
                    comment_count=row.comment_count,
                )
                for row in session.execute(
                    select(VideoMetric)
                    .where(VideoMetric.video_id == video.id)
                    .order_by(VideoMetric.captured_at)
                ).scalars()
            ],
        )
        rescored = compute_trend_score(
            metrics, weights=weights, topic_trend=max(heats)
        )
        video.trend_score = rescored.score
        video.trend_breakdown = {**breakdown, **rescored.to_dict()}
    session.flush()

    logger.info(
        "discovery complete",
        extra={
            "region": region,
            "videos": len(videos),
            "created": created_count,
            "topics": len(topics),
        },
    )
    return {
        "discovered": len(videos),
        "created": created_count,
        "topics": len(topics),
        "region": region,
        "note": (
            "Metadata only. Every discovered video is stored with "
            "authorization_status=UNKNOWN and cannot be processed until rights "
            "are established."
        ),
    }


def refresh_metrics(session: Session, *, limit: int = 200) -> dict[str, Any]:
    """Re-poll metrics for recently discovered videos.

    Velocity and acceleration need a second observation; without this the
    trend score is just a snapshot ranking.
    """
    if not settings.youtube_api_key:
        return {"status": "skipped", "reason": "YOUTUBE_API_KEY is not configured"}

    cutoff = datetime.now(timezone.utc) - timedelta(hours=METRIC_REFRESH_WINDOW_HOURS)
    videos = list(
        session.execute(
            select(Video)
            .where(
                Video.source == VideoSource.DISCOVERY,
                Video.external_id.is_not(None),
                Video.created_at >= cutoff,
            )
            .order_by(Video.trend_score.desc().nullslast())
            .limit(limit)
        ).scalars()
    )
    if not videos:
        return {"refreshed": 0}

    client = YouTubeMetadataClient()
    by_external = {v.external_id: v for v in videos if v.external_id}
    items = client.videos_by_id(list(by_external))

    refreshed = 0
    for item in items:
        video = by_external.get(item.get("id"))
        if video is None:
            continue
        statistics = item.get("statistics", {})
        video.view_count = _as_int(statistics.get("viewCount"))
        video.like_count = _as_int(statistics.get("likeCount"))
        video.comment_count = _as_int(statistics.get("commentCount"))
        _record_snapshot(session, video, weights=None)
        refreshed += 1

    logger.info("trend metrics refreshed", extra={"videos": refreshed})
    return {"refreshed": refreshed}
