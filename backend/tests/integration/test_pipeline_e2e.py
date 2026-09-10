"""End-to-end pipeline test.

Exercises the path a real user takes:

    register -> register a video -> upload media -> analyse
      -> inspect candidates -> generate a clip -> quality + safety
      -> approve -> download

A transcript sidecar is planted so the moment detector reads real words; that
is the supported way to run the pipeline without a transcription model.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.fixtures.transcripts import build_transcript

pytestmark = [pytest.mark.integration, pytest.mark.slow]

# Timed to sit inside the 90s sample clip and to contain a clear hook/payoff.
TRANSCRIPT_TEXT = """\
Okay so welcome back to the channel, please subscribe before we start.
Most people completely misunderstand how compounding works. They think it is \
about patience. It is not. It is about survival.
The reason is simple. Compounding only works if you never interrupt it, and \
almost everyone interrupts it. One bad year of panic selling erases nine good \
years of returns.
So what do you actually do? You automate the decision once, and then you stop \
touching it. That is the whole strategy.
Anyway that is basically all I wanted to say about that today.
"""


def _plant_transcript_sidecar(video_id: str) -> Path:
    """Write the sidecar the mock transcription provider reads."""
    from app.services.storage_paths import video_workdir

    transcript = build_transcript(TRANSCRIPT_TEXT)
    payload = {
        "language": transcript.language,
        "duration": transcript.duration,
        "segments": [
            {
                "start": chunk.start,
                "end": chunk.end,
                "text": chunk.text,
                "words": [w.to_dict() for w in chunk.words],
            }
            for chunk in transcript.chunks
        ],
    }
    path = video_workdir(video_id) / "audio.transcript.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@pytest.mark.asyncio
async def test_full_pipeline(auth_client, sample_video_path):
    api = "/api/v1"

    # 1. Register the video, asserting ownership.
    response = await auth_client.post(
        f"{api}/videos",
        json={
            "title": "How compounding actually works",
            "source": "UPLOAD",
            "creator_name": "Test Creator",
            "category": "Education",
            "language": "en",
            "authorization_status": "USER_OWNED",
            "authorization_note": "Recorded by the account holder.",
        },
    )
    assert response.status_code == 201, response.text
    video = response.json()
    video_id = video["id"]
    assert video["authorization_status"] == "USER_OWNED"
    assert video["has_media"] is False

    # 2. Upload the media.
    with open(sample_video_path, "rb") as handle:
        response = await auth_client.post(
            f"{api}/videos/{video_id}/upload",
            files={"file": ("sample.mp4", handle, "video/mp4")},
        )
    assert response.status_code == 201, response.text
    assert response.json()["has_media"] is True

    # 3. Plant the transcript so moment detection has real words to read.
    _plant_transcript_sidecar(video_id)

    # 4. Analyse (eager mode: the task runs inline).
    response = await auth_client.post(f"{api}/videos/{video_id}/analyze", json={})
    assert response.status_code == 202, response.text
    job_id = response.json()["job_id"]

    job = (await auth_client.get(f"{api}/jobs/{job_id}")).json()
    assert job["status"] == "COMPLETED", job
    assert job["result"]["segments"] > 0
    assert job["result"]["transcript_is_synthetic"] is False

    # 5. Transcript and scenes are persisted.
    transcript = (await auth_client.get(f"{api}/videos/{video_id}/transcript")).json()
    assert transcript["word_count"] > 40
    assert transcript["is_synthetic"] is False
    assert len(transcript["segments"]) > 3

    scenes = (await auth_client.get(f"{api}/videos/{video_id}/scenes")).json()
    assert len(scenes) >= 1

    # 6. Candidates are ranked and explain themselves.
    candidates = (await auth_client.get(f"{api}/videos/{video_id}/candidates")).json()
    assert candidates, "no candidate moments were produced"
    top = candidates[0]
    assert top["viral_score"] > 0
    assert top["reason"]
    assert "dimensions" in top["score_breakdown"]
    assert candidates == sorted(candidates, key=lambda c: -c["viral_score"])
    # The boilerplate opening must not be the top candidate.
    assert "subscribe" not in (top["transcript_excerpt"] or "").lower()

    # 7. Render the best candidate.
    response = await auth_client.post(
        f"{api}/clips/generate",
        json={"video_id": video_id, "candidate_ids": [top["id"]]},
    )
    assert response.status_code == 202, response.text
    job = (await auth_client.get(f"{api}/jobs/{response.json()['job_id']}")).json()
    assert job["status"] == "COMPLETED", job
    assert job["result"]["clips"], job["result"]

    # 8. The clip is a real 9:16 file with copy and checks attached.
    clip_id = job["result"]["clips"][0]
    clip = (await auth_client.get(f"{api}/clips/{clip_id}")).json()
    assert clip["width"] == 1080 and clip["height"] == 1920
    assert clip["duration_seconds"] > 10
    assert clip["media_url"]
    assert clip["title"]
    assert clip["quality_checks"], "no quality check was recorded"
    assert clip["safety_checks"], "no safety check was recorded"
    # With no model configured, safety must never come back SAFE.
    assert clip["safety_checks"][0]["verdict"] in ("REVIEW", "BLOCK")
    assert clip["status"] in ("NEEDS_REVIEW", "APPROVED", "REJECTED")

    # 9. Approve and download.
    response = await auth_client.post(
        f"{api}/clips/{clip_id}/approve", json={"note": "Looks good."}
    )
    assert response.status_code == 200, response.text
    assert response.json()["status"] == "APPROVED"

    response = await auth_client.get(f"{api}/clips/{clip_id}/download")
    assert response.status_code == 200
    assert response.headers["content-type"] == "video/mp4"
    assert len(response.content) > 50_000

    # 10. The dashboard reflects the run.
    dashboard = (await auth_client.get(f"{api}/dashboard")).json()
    assert dashboard["clips_generated"] >= 1
    assert dashboard["clips_approved"] >= 1


@pytest.mark.asyncio
async def test_unauthorized_video_cannot_be_analysed(auth_client, sample_video_path):
    """The rights gate is enforced by the API, not only by the UI."""
    api = "/api/v1"
    response = await auth_client.post(
        f"{api}/videos",
        json={
            "title": "Someone else's video",
            "source": "DISCOVERY",
            "authorization_status": "UNKNOWN",
        },
    )
    video_id = response.json()["id"]

    response = await auth_client.post(f"{api}/videos/{video_id}/analyze", json={})
    assert response.status_code == 422
    body = response.json()
    assert body["error_code"] == "validation_error"
    assert "authorization_status" in body["error_message"]


@pytest.mark.asyncio
async def test_upload_rejects_unsupported_type(auth_client):
    api = "/api/v1"
    response = await auth_client.post(
        f"{api}/videos",
        json={"title": "Bad upload", "authorization_status": "USER_OWNED"},
    )
    video_id = response.json()["id"]

    response = await auth_client.post(
        f"{api}/videos/{video_id}/upload",
        files={"file": ("payload.exe", b"MZ\x90\x00", "application/x-msdownload")},
    )
    assert response.status_code == 422
    assert response.json()["error_code"] == "validation_error"
