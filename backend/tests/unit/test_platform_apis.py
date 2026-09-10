"""Third-party API request construction, verified up to the network boundary.

These integrations cannot be exercised against live endpoints without real
credentials and real accounts. What *can* be verified without either is
everything this code is actually responsible for: the URLs it calls, the
parameters and headers it sends, how it drives a multi-step protocol, and how
it behaves when the far side returns an error.

``httpx.MockTransport`` intercepts at the transport layer, so the real client
code runs unchanged.
"""

from __future__ import annotations

import json

import httpx
import pytest

from app.core.errors import ProviderError
from app.providers.publishing.instagram import GRAPH_BASE, InstagramReelsProvider
from app.providers.publishing.youtube import (
    UPLOAD_URL,
    VIDEOS_URL,
    YouTubeShortsProvider,
)
from app.services.discovery import YouTubeMetadataClient

SESSION_URL = "https://upload.example.invalid/resumable/abc123"


def recorder():
    """Collect every intercepted request for assertion."""
    return []


@pytest.fixture
def clip_file(tmp_path):
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"\x00" * 4096)
    return path


# --------------------------------------------------------------------- YouTube
class TestYouTubeUpload:
    def _provider(self, handler):
        return YouTubeShortsProvider(transport=httpx.MockTransport(handler))

    def test_resumable_upload_flow(self, clip_file):
        seen = recorder()

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            if request.method == "POST":
                return httpx.Response(200, headers={"Location": SESSION_URL})
            return httpx.Response(200, json={"id": "yt-video-1"})

        result = self._provider(handler).publish(
            video_path=clip_file,
            title="A title",
            description="A description",
            hashtags=["compounding", "#investing"],
            credentials={"access_token": "token-abc"},
            privacy="unlisted",
        )

        assert result.external_post_id == "yt-video-1"
        assert result.external_url == "https://www.youtube.com/shorts/yt-video-1"

        init, upload = seen
        assert str(init.url).startswith(UPLOAD_URL)
        assert init.url.params["uploadType"] == "resumable"
        assert init.url.params["part"] == "snippet,status"
        assert init.headers["Authorization"] == "Bearer token-abc"
        assert init.headers["X-Upload-Content-Length"] == str(clip_file.stat().st_size)

        body = json.loads(init.content)
        assert body["status"]["privacyStatus"] == "unlisted"
        # Tags are sent without the leading '#'.
        assert body["snippet"]["tags"] == ["compounding", "investing"]
        # ...but the description keeps them as hashtags.
        assert "#compounding" in body["snippet"]["description"]

        assert upload.method == "PUT"
        assert str(upload.url) == SESSION_URL
        size = clip_file.stat().st_size
        assert upload.headers["Content-Range"] == f"bytes 0-{size - 1}/{size}"

    def test_308_resumes_from_the_next_byte(self, tmp_path):
        """A large file uploads in chunks; a 308 means "keep going"."""
        big = tmp_path / "big.mp4"
        big.write_bytes(b"\x01" * (20 * 1024 * 1024))  # 20 MiB -> 3 chunks
        ranges: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "POST":
                return httpx.Response(200, headers={"Location": SESSION_URL})
            ranges.append(request.headers["Content-Range"])
            if len(ranges) < 3:
                return httpx.Response(308)
            return httpx.Response(201, json={"id": "yt-big"})

        result = self._provider(handler).publish(
            video_path=big,
            title="t",
            description="d",
            hashtags=[],
            credentials={"access_token": "tok"},
        )
        assert result.external_post_id == "yt-big"
        assert len(ranges) == 3
        size = big.stat().st_size
        assert ranges[0] == f"bytes 0-{8 * 1024 * 1024 - 1}/{size}"
        # Each chunk continues exactly where the last one stopped.
        starts = [int(r.split()[1].split("-")[0]) for r in ranges]
        ends = [int(r.split()[1].split("-")[1].split("/")[0]) for r in ranges]
        assert starts[1] == ends[0] + 1
        assert starts[2] == ends[1] + 1
        assert ends[-1] == size - 1

    def test_title_is_truncated_to_the_platform_limit(self, clip_file):
        captured: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "POST":
                captured.update(json.loads(request.content))
                return httpx.Response(200, headers={"Location": SESSION_URL})
            return httpx.Response(200, json={"id": "x"})

        self._provider(handler).publish(
            video_path=clip_file,
            title="T" * 300,
            description="D" * 9000,
            hashtags=[f"tag{i}" for i in range(40)],
            credentials={"access_token": "tok"},
        )
        assert len(captured["snippet"]["title"]) == 100
        assert len(captured["snippet"]["description"]) <= 5000
        assert len(captured["snippet"]["tags"]) == 15

    def test_missing_token_is_refused_before_any_request(self, clip_file):
        def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
            raise AssertionError("no request should be made")

        with pytest.raises(ProviderError, match="access token"):
            self._provider(handler).publish(
                video_path=clip_file,
                title="t",
                description="d",
                hashtags=[],
                credentials={},
            )

    def test_missing_file_is_refused(self, tmp_path):
        def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
            raise AssertionError("no request should be made")

        with pytest.raises(ProviderError, match="missing"):
            self._provider(handler).publish(
                video_path=tmp_path / "nope.mp4",
                title="t",
                description="d",
                hashtags=[],
                credentials={"access_token": "tok"},
            )

    def test_a_session_without_a_location_header_fails_loudly(self, clip_file):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200)  # no Location

        with pytest.raises(ProviderError, match="resumable session"):
            self._provider(handler).publish(
                video_path=clip_file,
                title="t",
                description="d",
                hashtags=[],
                credentials={"access_token": "tok"},
            )

    def test_an_api_error_surfaces_the_status_and_body(self, clip_file):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(403, text="quotaExceeded")

        with pytest.raises(ProviderError, match="403"):
            self._provider(handler).publish(
                video_path=clip_file,
                title="t",
                description="d",
                hashtags=[],
                credentials={"access_token": "tok"},
            )

    def test_a_response_without_an_id_fails(self, clip_file):
        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "POST":
                return httpx.Response(200, headers={"Location": SESSION_URL})
            return httpx.Response(200, json={})

        with pytest.raises(ProviderError, match="no video id"):
            self._provider(handler).publish(
                video_path=clip_file,
                title="t",
                description="d",
                hashtags=[],
                credentials={"access_token": "tok"},
            )


class TestYouTubeMetrics:
    def test_statistics_are_parsed(self):
        def handler(request: httpx.Request) -> httpx.Response:
            assert str(request.url).startswith(VIDEOS_URL)
            assert request.url.params["id"] == "vid-1"
            assert request.headers["Authorization"] == "Bearer tok"
            return httpx.Response(
                200,
                json={
                    "items": [
                        {
                            "statistics": {
                                "viewCount": "12345",
                                "likeCount": "678",
                                "commentCount": "90",
                            }
                        }
                    ]
                },
            )

        metrics = YouTubeShortsProvider(
            transport=httpx.MockTransport(handler)
        ).fetch_metrics(external_post_id="vid-1", credentials={"access_token": "tok"})
        assert (metrics.views, metrics.likes, metrics.comments) == (12345, 678, 90)

    def test_an_unknown_video_raises(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"items": []})

        with pytest.raises(ProviderError, match="no video"):
            YouTubeShortsProvider(
                transport=httpx.MockTransport(handler)
            ).fetch_metrics(external_post_id="gone", credentials={"access_token": "t"})


# ------------------------------------------------------------------- Instagram
class TestInstagramPublish:
    def _provider(self, handler):
        return InstagramReelsProvider(
            transport=httpx.MockTransport(handler), poll_seconds=0
        )

    def _credentials(self, **extra):
        return {
            "access_token": "ig-token",
            "ig_user_id": "17841400000000000",
            "video_url": "https://media.example.invalid/clip.mp4?sig=abc",
            **extra,
        }

    def test_container_publish_flow(self, clip_file):
        seen = recorder()

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            path = request.url.path
            if path.endswith("/media"):
                return httpx.Response(200, json={"id": "container-1"})
            if path.endswith("/media_publish"):
                return httpx.Response(200, json={"id": "media-9"})
            return httpx.Response(200, json={"status_code": "FINISHED"})

        result = self._provider(handler).publish(
            video_path=clip_file,
            title="A title",
            description="A description",
            hashtags=["compounding"],
            credentials=self._credentials(),
        )
        assert result.external_post_id == "media-9"
        assert "instagram.com/reel/media-9" in result.external_url

        create = seen[0]
        assert str(create.url).startswith(f"{GRAPH_BASE}/17841400000000000/media")
        body = create.content.decode()
        assert "media_type=REELS" in body
        # The API pulls the file from a URL; it is never uploaded here.
        assert "video_url=" in body
        assert "%23compounding" in body  # '#compounding' url-encoded

        assert seen[-1].url.path.endswith("/media_publish")

    def test_it_waits_for_the_container_to_finish(self, clip_file):
        statuses = ["IN_PROGRESS", "IN_PROGRESS", "FINISHED"]
        polls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            path = request.url.path
            if path.endswith("/media"):
                return httpx.Response(200, json={"id": "c1"})
            if path.endswith("/media_publish"):
                return httpx.Response(200, json={"id": "m1"})
            index = min(polls["n"], len(statuses) - 1)
            polls["n"] += 1
            return httpx.Response(200, json={"status_code": statuses[index]})

        result = self._provider(handler).publish(
            video_path=clip_file,
            title="t",
            description="d",
            hashtags=[],
            credentials=self._credentials(),
        )
        assert result.external_post_id == "m1"
        assert polls["n"] >= 3, "should poll until FINISHED"

    def test_a_processing_error_is_not_published(self, clip_file):
        attempted = {"publish": False}

        def handler(request: httpx.Request) -> httpx.Response:
            path = request.url.path
            if path.endswith("/media"):
                return httpx.Response(200, json={"id": "c1"})
            if path.endswith("/media_publish"):
                attempted["publish"] = True
                return httpx.Response(200, json={"id": "should-not-happen"})
            return httpx.Response(
                200, json={"status_code": "ERROR", "status": "bad codec"}
            )

        with pytest.raises(ProviderError, match="failed to process"):
            self._provider(handler).publish(
                video_path=clip_file,
                title="t",
                description="d",
                hashtags=[],
                credentials=self._credentials(),
            )
        assert attempted["publish"] is False

    def test_a_missing_video_url_is_refused_before_any_request(self, clip_file):
        def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
            raise AssertionError("no request should be made")

        credentials = self._credentials()
        credentials.pop("video_url")
        with pytest.raises(ProviderError, match="video_url"):
            self._provider(handler).publish(
                video_path=clip_file,
                title="t",
                description="d",
                hashtags=[],
                credentials=credentials,
            )

    def test_missing_credentials_are_refused(self, clip_file):
        def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
            raise AssertionError("no request should be made")

        with pytest.raises(ProviderError, match="access token"):
            self._provider(handler).publish(
                video_path=clip_file,
                title="t",
                description="d",
                hashtags=[],
                credentials={"video_url": "https://x.invalid/a.mp4"},
            )

    def test_insights_are_parsed_and_watch_time_converted(self):
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path.endswith("/insights")
            return httpx.Response(
                200,
                json={
                    "data": [
                        {"name": "plays", "values": [{"value": 5000}]},
                        {"name": "likes", "values": [{"value": 300}]},
                        {"name": "saved", "values": [{"value": 42}]},
                        {
                            "name": "ig_reels_avg_watch_time",
                            "values": [{"value": 8200}],  # milliseconds
                        },
                    ]
                },
            )

        metrics = InstagramReelsProvider(
            transport=httpx.MockTransport(handler)
        ).fetch_metrics(
            external_post_id="m1",
            credentials={"access_token": "t", "ig_user_id": "1"},
        )
        assert metrics.views == 5000
        assert metrics.likes == 300
        assert metrics.saves == 42
        assert metrics.average_view_duration == pytest.approx(8.2)


# ------------------------------------------------------------------- discovery
class TestYouTubeDiscoveryClient:
    def _client(self, handler):
        return YouTubeMetadataClient(
            api_key="key-123", transport=httpx.MockTransport(handler)
        )

    def test_most_popular_sends_the_documented_parameters(self):
        seen = recorder()

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return httpx.Response(200, json={"items": [{"id": "a"}]})

        items = self._client(handler).most_popular(region="IN", limit=10)
        assert items == [{"id": "a"}]
        params = seen[0].url.params
        assert params["chart"] == "mostPopular"
        assert params["regionCode"] == "IN"
        assert params["maxResults"] == "10"
        assert params["key"] == "key-123"
        assert "statistics" in params["part"]

    def test_it_pages_until_the_limit_is_reached(self):
        pages = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            pages["n"] += 1
            if pages["n"] == 1:
                return httpx.Response(
                    200,
                    json={
                        "items": [{"id": f"v{i}"} for i in range(50)],
                        "nextPageToken": "page2",
                    },
                )
            return httpx.Response(200, json={"items": [{"id": "v50"}]})

        items = self._client(handler).most_popular(limit=51)
        assert len(items) == 51
        assert pages["n"] == 2

    def test_it_stops_when_there_is_no_next_page(self):
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            return httpx.Response(200, json={"items": [{"id": "only"}]})

        items = self._client(handler).most_popular(limit=200)
        assert items == [{"id": "only"}]
        assert calls["n"] == 1, "must not loop forever without a page token"

    def test_search_resolves_ids_to_full_records(self):
        seen = recorder()

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            if request.url.path.endswith("/search"):
                return httpx.Response(
                    200,
                    json={
                        "items": [
                            {"id": {"videoId": "x1"}},
                            {"id": {"kind": "channel"}},  # ignored
                        ]
                    },
                )
            return httpx.Response(200, json={"items": [{"id": "x1"}]})

        items = self._client(handler).search_recent("ai agents", limit=5)
        assert items == [{"id": "x1"}]
        search = seen[0].url.params
        assert search["q"] == "ai agents"
        assert search["type"] == "video"
        assert "publishedAfter" in search
        assert seen[1].url.params["id"] == "x1"

    def test_ids_are_batched_within_the_api_limit(self):
        batches: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            batches.append(len(request.url.params["id"].split(",")))
            return httpx.Response(200, json={"items": []})

        self._client(handler).videos_by_id([f"v{i}" for i in range(120)])
        assert batches == [50, 50, 20]

    def test_a_403_explains_the_likely_cause(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(403, text="quotaExceeded")

        with pytest.raises(ProviderError, match="quota"):
            self._client(handler).most_popular()

    def test_other_errors_surface_the_status(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="boom")

        with pytest.raises(ProviderError, match="500"):
            self._client(handler).most_popular()

    def test_an_empty_id_list_makes_no_request(self):
        def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
            raise AssertionError("no request should be made")

        assert self._client(handler).videos_by_id([]) == []

    def test_a_missing_key_is_refused_at_construction(self, monkeypatch):
        from app.core.config import settings

        monkeypatch.setattr(settings, "youtube_api_key", None)
        with pytest.raises(ProviderError, match="YOUTUBE_API_KEY"):
            YouTubeMetadataClient()
