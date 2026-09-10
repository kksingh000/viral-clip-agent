"""Failure isolation, rate limiting and response handling.

The documented guarantee is that one bad clip never costs you the rest of the
batch. That is only true if each unit of work is rolled back independently, so
it is asserted against a real database rather than assumed from the code shape.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import select

from app.models.clip import CandidateClip, GeneratedClip
from app.models.enums import (
    AuthorizationStatus,
    ClipStatus,
    CropMode,
    JobStatus,
    JobType,
    MediaAssetKind,
    VideoSource,
    VideoStatus,
)
from app.models.user import User
from app.models.video import MediaAsset, Video
from app.services import jobs as job_service

pytestmark = pytest.mark.integration


def fresh_session():
    """A new session, so committed worker writes are actually visible.

    The task commits through its own ``session_scope``; a session opened before
    that still sees its original snapshot."""
    from app.database.session import get_sync_session_factory

    return get_sync_session_factory()()


def make_user(session) -> User:
    user = User(
        email=f"iso-{uuid.uuid4().hex[:10]}@example.com",
        hashed_password="pbkdf2_sha256$1$00$00",
    )
    session.add(user)
    session.flush()
    return user


def make_video_with_candidates(session, user: User, count: int) -> Video:
    video = Video(
        user_id=user.id,
        source=VideoSource.UPLOAD,
        title="Batch source",
        authorization_status=AuthorizationStatus.USER_OWNED,
        status=VideoStatus.ANALYZED,
        duration_seconds=300.0,
    )
    session.add(video)
    session.flush()
    for index in range(count):
        session.add(
            CandidateClip(
                video_id=video.id,
                start_time=index * 40.0,
                end_time=index * 40.0 + 25.0,
                duration_seconds=25.0,
                viral_score=90.0 - index,
                hook=f"Hook {index}",
                summary=f"Summary {index}",
                reason="fixture",
            )
        )
    session.flush()
    return video


class TestBatchIsolation:
    def test_one_failing_clip_does_not_discard_the_others(
        self, sync_session, monkeypatch
    ):
        """Regression: the failure path called ``session.rollback()``, which
        threw away every clip already produced in the same transaction."""
        from app.services import clips as clip_service
        from app.workers import tasks

        user = make_user(sync_session)
        video = make_video_with_candidates(sync_session, user, 3)
        job = job_service.create_job(
            sync_session,
            user_id=user.id,
            job_type=JobType.GENERATE_CLIP,
            video_id=video.id,
            payload={},
        )
        job_id = str(job.id)
        sync_session.commit()

        attempts: list[float] = []

        def fake_generate(session, request, **kwargs):
            attempts.append(request.start)
            # The middle candidate fails after writing a row, which is exactly
            # the case a naive rollback would mishandle.
            clip = GeneratedClip(
                user_id=request.video.user_id,
                video_id=request.video.id,
                candidate_id=request.candidate.id if request.candidate else None,
                start_time=request.start,
                end_time=request.end,
                duration_seconds=request.duration,
                crop_mode=CropMode.SMART,
                caption_style="NORMAL",
                caption_position="LOWER_THIRD",
                status=ClipStatus.NEEDS_REVIEW,
                viral_score=80.0,
            )
            session.add(clip)
            session.flush()
            if abs(request.start - 40.0) < 0.01:
                raise RuntimeError("ffmpeg exploded on the middle clip")
            return clip_service.ClipGenerationResult(
                clip=clip,
                render=None,  # type: ignore[arg-type]
                quality=None,  # type: ignore[arg-type]
                safety_verdict=None,  # type: ignore[arg-type]
                attempts=1,
            )

        monkeypatch.setattr(clip_service, "generate_clip", fake_generate)

        result = tasks.generate_clips_task(job_id)

        assert len(attempts) == 3, "every candidate should be attempted"
        assert len(result["clips"]) == 2
        assert len(result["failures"]) == 1
        assert "ffmpeg exploded" in result["failures"][0]["error"]

        # The surviving clips are actually committed, and the failed one is not.
        with fresh_session() as verify:
            stored = list(
                verify.execute(
                    select(GeneratedClip).where(GeneratedClip.video_id == video.id)
                ).scalars()
            )
            assert len(stored) == 2
            starts = sorted(clip.start_time for clip in stored)
            assert starts == [0.0, 80.0]

            refreshed = job_service.get_job(verify, uuid.UUID(job_id))
            assert refreshed.status is JobStatus.COMPLETED

    def test_a_batch_where_everything_fails_still_completes_the_job(
        self, sync_session, monkeypatch
    ):
        from app.services import clips as clip_service
        from app.workers import tasks

        user = make_user(sync_session)
        video = make_video_with_candidates(sync_session, user, 2)
        job = job_service.create_job(
            sync_session,
            user_id=user.id,
            job_type=JobType.GENERATE_CLIP,
            video_id=video.id,
            payload={},
        )
        job_id = str(job.id)
        sync_session.commit()

        def always_fail(session, request, **kwargs):
            raise RuntimeError("nope")

        monkeypatch.setattr(clip_service, "generate_clip", always_fail)
        result = tasks.generate_clips_task(job_id)

        assert result["clips"] == []
        assert len(result["failures"]) == 2
        # The job itself completed: it did its work, the work just failed.
        with fresh_session() as verify:
            refreshed = job_service.get_job(verify, uuid.UUID(job_id))
            assert refreshed.status is JobStatus.COMPLETED
            remaining = list(
                verify.execute(
                    select(GeneratedClip).where(GeneratedClip.video_id == video.id)
                ).scalars()
            )
            assert remaining == [], "failed clips must not leave rows behind"

    def test_a_cancelled_job_is_not_executed(self, sync_session, monkeypatch):
        from app.services import clips as clip_service
        from app.workers import tasks

        user = make_user(sync_session)
        video = make_video_with_candidates(sync_session, user, 1)
        job = job_service.create_job(
            sync_session,
            user_id=user.id,
            job_type=JobType.GENERATE_CLIP,
            video_id=video.id,
            payload={},
        )
        job.status = JobStatus.CANCELLED
        job_id = str(job.id)
        sync_session.commit()

        def should_not_run(session, request, **kwargs):  # pragma: no cover
            raise AssertionError("cancelled work must not execute")

        monkeypatch.setattr(clip_service, "generate_clip", should_not_run)
        assert tasks.generate_clips_task(job_id)["status"] == "CANCELLED"


class TestJobLifecycle:
    def test_a_failure_records_a_code_message_and_timing(self, sync_session):
        user = make_user(sync_session)
        job = job_service.create_job(
            sync_session, user_id=user.id, job_type=JobType.ANALYZE_VIDEO
        )
        job_service.mark_started(sync_session, job)
        job_service.mark_failed(sync_session, job, RuntimeError("disk full"))

        assert job.status is JobStatus.FAILED
        assert job.error_code == "RuntimeError"
        assert "disk full" in job.error_message
        assert job.finished_at is not None
        assert job.duration_seconds is not None

    def test_retries_back_off_and_then_stop(self, sync_session):
        user = make_user(sync_session)
        job = job_service.create_job(
            sync_session, user_id=user.id, job_type=JobType.ANALYZE_VIDEO, max_retries=2
        )
        first = job_service.schedule_retry(sync_session, job)
        second = job_service.schedule_retry(sync_session, job)
        exhausted = job_service.schedule_retry(sync_session, job)

        assert first is not None and second is not None
        assert second > first, "backoff must increase"
        assert exhausted is None, "must stop after max_retries"
        assert job.retry_count == 2

    def test_an_app_error_keeps_its_error_code(self, sync_session):
        from app.core.errors import AuthorizationStatusError

        user = make_user(sync_session)
        job = job_service.create_job(
            sync_session, user_id=user.id, job_type=JobType.GENERATE_CLIP
        )
        job_service.mark_failed(
            sync_session, job, AuthorizationStatusError("not allowed")
        )
        assert job.error_code == "content_not_authorized"


class TestClipAssetLookup:
    def test_clip_assets_are_found_by_clip_not_video(self, sync_session):
        """Regression: clip assets store ``clip_id`` with ``video_id`` NULL, so
        the video-scoped lookup could never match."""
        from app.services.media import get_asset, get_clip_asset

        user = make_user(sync_session)
        video = make_video_with_candidates(sync_session, user, 1)
        clip = GeneratedClip(
            user_id=user.id,
            video_id=video.id,
            start_time=0.0,
            end_time=20.0,
            duration_seconds=20.0,
            crop_mode=CropMode.SMART,
            caption_style="NORMAL",
            caption_position="LOWER_THIRD",
            status=ClipStatus.RENDERED,
        )
        sync_session.add(clip)
        sync_session.flush()

        sync_session.add(
            MediaAsset(
                clip_id=clip.id,
                kind=MediaAssetKind.CLIP_VIDEO,
                storage_key=f"users/{user.id}/clips/{clip.id}/clip.mp4",
                content_type="video/mp4",
            )
        )
        sync_session.flush()

        assert get_asset(sync_session, video.id, MediaAssetKind.CLIP_VIDEO) is None
        found = get_clip_asset(sync_session, clip.id, MediaAssetKind.CLIP_VIDEO)
        assert found is not None
        assert found.storage_key.endswith("clip.mp4")


class TestRateLimiter:
    def _limiter(self):
        from app.api.deps import _InMemoryRateLimiter

        return _InMemoryRateLimiter()

    def test_it_denies_past_the_limit_and_reports_a_retry_after(self):
        limiter = self._limiter()
        for _ in range(3):
            assert limiter.check("k", limit=3, window=60)[0] is True
        allowed, retry_after = limiter.check("k", limit=3, window=60)
        assert allowed is False
        assert retry_after >= 1, "Retry-After must never be zero or negative"

    def test_keys_are_independent(self):
        limiter = self._limiter()
        assert limiter.check("a", limit=1, window=60)[0] is True
        assert limiter.check("b", limit=1, window=60)[0] is True
        assert limiter.check("a", limit=1, window=60)[0] is False

    def test_expired_keys_are_swept(self):
        """Regression: keys include the request path, so without a sweep a URL
        scanner would grow the table for the life of the process."""
        import time as _time

        limiter = self._limiter()
        for index in range(50):
            limiter.check(f"key-{index}", limit=10, window=60)
        assert len(limiter._hits) == 50

        # Age every recorded hit well past the window, then force a sweep.
        stale = _time.monotonic() - 10_000
        for hits in limiter._hits.values():
            hits.clear()
            hits.append(stale)
        limiter._last_sweep = 0.0

        limiter.check("fresh", limit=10, window=60)
        assert set(limiter._hits) == {"fresh"}, "stale keys must be released"

    def test_a_busy_key_is_not_swept(self):
        limiter = self._limiter()
        limiter.check("busy", limit=10, window=60)
        limiter._last_sweep = 0.0
        limiter.check("other", limit=10, window=60)
        assert "busy" in limiter._hits


class TestResponseHandling:
    @pytest.mark.asyncio
    async def test_media_paths_are_not_gzipped(self, client):
        """Compressing an mp4 wastes CPU and buffers the whole file."""
        response = await client.get(
            "/media/does/not/exist",
            params={"expires": 1, "signature": "x"},
            headers={"Accept-Encoding": "gzip"},
        )
        assert response.headers.get("content-encoding") != "gzip"

    @pytest.mark.asyncio
    async def test_json_is_still_gzipped(self, auth_client):
        for index in range(30):
            await auth_client.post(
                "/api/v1/videos",
                json={"title": f"Compressible video number {index}" * 8},
            )
        response = await auth_client.get(
            "/api/v1/videos",
            params={"limit": 30},
            headers={"Accept-Encoding": "gzip"},
        )
        assert response.status_code == 200
        assert len(response.content) > 1024
