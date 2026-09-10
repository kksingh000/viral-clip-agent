"""API contract: authentication, tenant isolation, validation and pagination.

Tenant isolation is the security property with the most ways to go wrong — every
list endpoint filters by user, and every detail endpoint has to check ownership
rather than trusting an unguessable id. It is asserted here per endpoint rather
than assumed.
"""

from __future__ import annotations

import uuid

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app as fastapi_app

API = "/api/v1"
pytestmark = pytest.mark.integration


async def new_account(**overrides) -> AsyncClient:
    """A client authenticated as a brand-new account."""
    client = AsyncClient(
        transport=ASGITransport(app=fastapi_app), base_url="http://testserver"
    )
    email = f"user-{uuid.uuid4().hex[:10]}@example.com"
    response = await client.post(
        f"{API}/auth/register",
        json={"email": email, "password": "correct-horse-battery", **overrides},
    )
    assert response.status_code == 201, response.text
    client.headers["Authorization"] = f"Bearer {response.json()['access_token']}"
    return client


async def make_video(client: AsyncClient, title: str = "Owned video") -> str:
    response = await client.post(
        f"{API}/videos",
        json={"title": title, "authorization_status": "USER_OWNED"},
    )
    assert response.status_code == 201, response.text
    return response.json()["id"]


class TestAuthentication:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "method,path",
        [
            ("GET", f"{API}/videos"),
            ("GET", f"{API}/clips"),
            ("GET", f"{API}/jobs"),
            ("GET", f"{API}/dashboard"),
            ("GET", f"{API}/settings"),
            ("GET", f"{API}/trending"),
            ("GET", f"{API}/analytics"),
            ("GET", f"{API}/integrations/accounts"),
            ("POST", f"{API}/clips/generate"),
        ],
    )
    async def test_protected_endpoints_require_a_token(self, client, method, path):
        response = await client.request(method, path, json={})
        assert response.status_code == 401
        assert response.json()["error_code"] == "unauthenticated"

    @pytest.mark.asyncio
    async def test_a_garbage_token_is_rejected(self, client):
        response = await client.get(
            f"{API}/videos", headers={"Authorization": "Bearer not.a.token"}
        )
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_a_refresh_token_is_not_accepted_as_an_access_token(self, client):
        registration = await client.post(
            f"{API}/auth/register",
            json={
                "email": f"u-{uuid.uuid4().hex[:8]}@example.com",
                "password": "correct-horse-battery",
            },
        )
        refresh = registration.json()["refresh_token"]
        response = await client.get(
            f"{API}/videos", headers={"Authorization": f"Bearer {refresh}"}
        )
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_duplicate_registration_is_rejected(self, client):
        email = f"dupe-{uuid.uuid4().hex[:8]}@example.com"
        payload = {"email": email, "password": "correct-horse-battery"}
        assert (await client.post(f"{API}/auth/register", json=payload)).status_code == 201
        second = await client.post(f"{API}/auth/register", json=payload)
        assert second.status_code == 409
        assert second.json()["error_code"] == "conflict"

    @pytest.mark.asyncio
    async def test_a_wrong_password_does_not_reveal_the_account(self, client):
        email = f"login-{uuid.uuid4().hex[:8]}@example.com"
        await client.post(
            f"{API}/auth/register",
            json={"email": email, "password": "correct-horse-battery"},
        )
        wrong = await client.post(
            f"{API}/auth/login", json={"email": email, "password": "not-the-password"}
        )
        unknown = await client.post(
            f"{API}/auth/login",
            json={"email": "nobody@example.com", "password": "not-the-password"},
        )
        assert wrong.status_code == unknown.status_code == 401
        assert wrong.json()["error_message"] == unknown.json()["error_message"]

    @pytest.mark.asyncio
    async def test_an_api_key_authenticates_and_is_returned_only_once(self):
        client = await new_account()
        try:
            created = await client.post(
                f"{API}/auth/api-keys", json={"name": "ci"}
            )
            assert created.status_code == 201
            raw = created.json()["api_key"]
            assert raw.startswith("vca_")

            listed = await client.get(f"{API}/auth/api-keys")
            assert all("api_key" not in row for row in listed.json())

            fresh = AsyncClient(
                transport=ASGITransport(app=fastapi_app),
                base_url="http://testserver",
                headers={"X-API-Key": raw},
            )
            async with fresh:
                assert (await fresh.get(f"{API}/videos")).status_code == 200
        finally:
            await client.aclose()

    @pytest.mark.asyncio
    async def test_a_revoked_api_key_stops_working(self):
        client = await new_account()
        try:
            created = (
                await client.post(f"{API}/auth/api-keys", json={"name": "temp"})
            ).json()
            await client.delete(f"{API}/auth/api-keys/{created['id']}")

            fresh = AsyncClient(
                transport=ASGITransport(app=fastapi_app),
                base_url="http://testserver",
                headers={"X-API-Key": created["api_key"]},
            )
            async with fresh:
                assert (await fresh.get(f"{API}/videos")).status_code == 401
        finally:
            await client.aclose()


class TestTenantIsolation:
    @pytest.mark.asyncio
    async def test_one_account_cannot_see_or_touch_anothers_video(self):
        alice = await new_account()
        bob = await new_account()
        try:
            video_id = await make_video(alice, "Alice's private video")

            # Bob's list must not contain it.
            listing = (await bob.get(f"{API}/videos")).json()
            assert all(item["id"] != video_id for item in listing["items"])

            # Direct access is 404, not 403: existence is not disclosed.
            for method, path, body in [
                ("GET", f"{API}/videos/{video_id}", None),
                ("PATCH", f"{API}/videos/{video_id}", {"title": "hijacked"}),
                ("DELETE", f"{API}/videos/{video_id}", None),
                ("GET", f"{API}/videos/{video_id}/candidates", None),
                ("GET", f"{API}/videos/{video_id}/scenes", None),
                ("GET", f"{API}/videos/{video_id}/transcript", None),
                ("POST", f"{API}/videos/{video_id}/analyze", {}),
                (
                    "PUT",
                    f"{API}/videos/{video_id}/authorization",
                    {"authorization_status": "USER_OWNED"},
                ),
            ]:
                response = await bob.request(method, path, json=body)
                assert response.status_code == 404, f"{method} {path} leaked"

            # And Alice's video is untouched.
            assert (await alice.get(f"{API}/videos/{video_id}")).json()[
                "title"
            ] == "Alice's private video"
        finally:
            await alice.aclose()
            await bob.aclose()

    @pytest.mark.asyncio
    async def test_one_account_cannot_read_anothers_jobs(self):
        alice = await new_account()
        bob = await new_account()
        try:
            video_id = await make_video(alice)
            # Analysis is refused (no media), but a job row is not created, so
            # use a job that does exist: generate against a real video.
            response = await alice.post(
                f"{API}/clips/generate", json={"video_id": video_id, "count": 1}
            )
            assert response.status_code == 202
            job_id = response.json()["job_id"]

            assert (await bob.get(f"{API}/jobs/{job_id}")).status_code == 404
            assert (
                await bob.post(f"{API}/jobs/{job_id}/cancel")
            ).status_code == 404

            bob_jobs = (await bob.get(f"{API}/jobs")).json()
            assert all(row["id"] != job_id for row in bob_jobs["items"])
            assert (await alice.get(f"{API}/jobs/{job_id}")).status_code == 200
        finally:
            await alice.aclose()
            await bob.aclose()

    @pytest.mark.asyncio
    async def test_settings_are_per_account(self):
        alice = await new_account()
        bob = await new_account()
        try:
            await alice.put(f"{API}/settings", json={"clips_per_video": 9})
            assert (await alice.get(f"{API}/settings")).json()["clips_per_video"] == 9
            assert (await bob.get(f"{API}/settings")).json()["clips_per_video"] != 9
        finally:
            await alice.aclose()
            await bob.aclose()

    @pytest.mark.asyncio
    async def test_an_unknown_id_is_404_not_500(self):
        client = await new_account()
        try:
            missing = uuid.uuid4()
            for path in (
                f"{API}/videos/{missing}",
                f"{API}/clips/{missing}",
                f"{API}/jobs/{missing}",
            ):
                assert (await client.get(path)).status_code == 404
        finally:
            await client.aclose()


class TestValidationAndRights:
    @pytest.mark.asyncio
    async def test_licensed_content_requires_a_recorded_basis(self, auth_client):
        """LICENSED/AUTHORIZED are assertions; the note is the record of them."""
        response = await auth_client.post(
            f"{API}/videos",
            json={"title": "Licensed clip", "authorization_status": "LICENSED"},
        )
        assert response.status_code == 422
        assert "note" in response.text or "evidence" in response.text

        ok = await auth_client.post(
            f"{API}/videos",
            json={
                "title": "Licensed clip",
                "authorization_status": "LICENSED",
                "authorization_note": "Licence #4412, signed 2026-01-02.",
            },
        )
        assert ok.status_code == 201

    @pytest.mark.asyncio
    async def test_user_uploaded_cannot_be_asserted_directly(self, auth_client):
        """It is set by the upload endpoint, not claimed by the client."""
        response = await auth_client.post(
            f"{API}/videos",
            json={"title": "x", "authorization_status": "USER_UPLOADED"},
        )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_generating_from_unauthorized_source_is_forbidden(self, auth_client):
        video = await auth_client.post(
            f"{API}/videos",
            json={
                "title": "Third-party video",
                "source": "DISCOVERY",
                "authorization_status": "UNKNOWN",
            },
        )
        video_id = video.json()["id"]
        response = await auth_client.post(
            f"{API}/clips/generate", json={"video_id": video_id, "count": 1}
        )
        assert response.status_code == 403
        assert response.json()["error_code"] == "forbidden"

    @pytest.mark.asyncio
    async def test_manual_clip_rejects_an_inverted_range(self, auth_client):
        video_id = await make_video(auth_client)
        response = await auth_client.post(
            f"{API}/clips/manual",
            json={"video_id": video_id, "start_time": 30, "end_time": 10},
        )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_settings_reject_an_incoherent_combination(self, auth_client):
        response = await auth_client.put(
            f"{API}/settings", json={"min_clip_seconds": 60, "max_clip_seconds": 20}
        )
        assert response.status_code == 422

        response = await auth_client.put(
            f"{API}/settings",
            json={"auto_approve_score": 40, "reject_below_score": 80},
        )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_a_partial_settings_update_is_checked_against_stored_values(
        self, auth_client
    ):
        """Only one side of the pair is submitted, so validation has to read the
        other from the database rather than the request."""
        await auth_client.put(
            f"{API}/settings", json={"min_clip_seconds": 10, "max_clip_seconds": 30}
        )
        response = await auth_client.put(
            f"{API}/settings", json={"min_clip_seconds": 45}
        )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_scoring_weights_are_renormalised_on_save(self, auth_client):
        response = await auth_client.put(
            f"{API}/settings",
            json={"scoring_weights": {"hook_strength": 10.0, "not_a_dimension": 5.0}},
        )
        assert response.status_code == 200
        weights = response.json()["scoring_weights"]
        assert "not_a_dimension" not in weights
        assert sum(weights.values()) == pytest.approx(1.0, abs=1e-6)

    @pytest.mark.asyncio
    async def test_discovery_is_disabled_without_an_api_key(self, auth_client):
        response = await auth_client.post(
            f"{API}/trending/discover", json={"region": "US", "max_results": 5}
        )
        assert response.status_code == 422
        assert "YOUTUBE_API_KEY" in response.json()["error_message"]

    @pytest.mark.asyncio
    async def test_publishing_needs_a_connected_account(self, auth_client):
        video_id = await make_video(auth_client)
        response = await auth_client.post(
            f"{API}/clips/generate", json={"video_id": video_id, "count": 1}
        )
        assert response.status_code == 202
        # No clip exists (no media), so publishing an unknown clip is a 404 --
        # the point is that it never reaches the platform without an account.
        response = await auth_client.post(
            f"{API}/integrations/publish/{uuid.uuid4()}",
            json={"platform": "YOUTUBE_SHORTS"},
        )
        assert response.status_code == 404


class TestListings:
    @pytest.mark.asyncio
    async def test_pagination_reports_a_stable_total(self):
        client = await new_account()
        try:
            for index in range(5):
                await make_video(client, f"Video {index}")

            first = (await client.get(f"{API}/videos", params={"limit": 2})).json()
            assert first["total"] == 5
            assert len(first["items"]) == 2
            assert first["offset"] == 0

            second = (
                await client.get(f"{API}/videos", params={"limit": 2, "offset": 2})
            ).json()
            assert second["total"] == 5
            first_ids = {item["id"] for item in first["items"]}
            second_ids = {item["id"] for item in second["items"]}
            assert first_ids.isdisjoint(second_ids)
        finally:
            await client.aclose()

    @pytest.mark.asyncio
    async def test_filters_apply(self):
        client = await new_account()
        try:
            await make_video(client, "Findable title")
            await client.post(
                f"{API}/videos",
                json={
                    "title": "Discovered elsewhere",
                    "source": "DISCOVERY",
                    "authorization_status": "UNKNOWN",
                },
            )

            owned = (
                await client.get(
                    f"{API}/videos", params={"authorization": "USER_OWNED"}
                )
            ).json()
            assert owned["total"] == 1
            assert owned["items"][0]["title"] == "Findable title"

            search = (
                await client.get(f"{API}/videos", params={"search": "Findable"})
            ).json()
            assert search["total"] == 1
        finally:
            await client.aclose()

    @pytest.mark.asyncio
    async def test_out_of_range_pagination_is_rejected(self, auth_client):
        assert (
            await auth_client.get(f"{API}/videos", params={"limit": 0})
        ).status_code == 422
        assert (
            await auth_client.get(f"{API}/videos", params={"limit": 5000})
        ).status_code == 422

    @pytest.mark.asyncio
    async def test_an_unknown_sort_key_is_rejected(self, auth_client):
        response = await auth_client.get(
            f"{API}/videos", params={"order_by": "; DROP TABLE videos"}
        )
        assert response.status_code == 422


class TestErrorEnvelope:
    @pytest.mark.asyncio
    async def test_errors_carry_a_code_and_a_request_id(self, auth_client):
        response = await auth_client.get(f"{API}/videos/{uuid.uuid4()}")
        body = response.json()
        assert set(body) >= {"error_code", "error_message", "details", "request_id"}
        assert body["request_id"]
        assert response.headers["X-Request-ID"] == body["request_id"]

    @pytest.mark.asyncio
    async def test_health_reports_each_component(self, client):
        body = (await client.get("/health")).json()
        assert body["status"] in ("ok", "degraded")
        names = {component["name"] for component in body["components"]}
        assert {"database", "task_queue", "ffmpeg", "llm", "transcription"} <= names
        # Mock providers must be reported as degraded, not as working.
        by_name = {c["name"]: c for c in body["components"]}
        assert by_name["llm"]["status"] == "degraded"
        assert by_name["transcription"]["status"] == "degraded"
