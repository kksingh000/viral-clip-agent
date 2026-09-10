"""Video analysis pipeline.

Runs inside a Celery worker on a synchronous session:

    resolve media -> probe -> extract audio -> transcribe -> detect scenes
      -> segment transcript -> persist -> find moments -> snap/score/dedupe
      -> persist candidates

Each stage is cached: re-analysing a video reuses the extracted audio, the
transcript and the scene list unless ``force`` is set. That is the difference
between a cheap re-run and paying for transcription twice.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.agents.base import AgentContext
from app.agents.viral_moment_agent import MomentSearchPayload, ViralMomentAgent
from app.core.config import settings
from app.core.errors import MediaAnalysisError, TranscriptionError
from app.core.logging import get_logger
from app.models.clip import CandidateClip
from app.models.enums import (
    CandidateStatus,
    MediaAssetKind,
    VideoStatus,
)
from app.models.job import ProcessingJob
from app.models.transcript import Scene, Transcript, TranscriptSegment
from app.models.user import UserSettings
from app.models.video import Video
from app.providers.base import TranscriptionResult, TranscriptWord
from app.providers.registry import get_storage_provider, get_transcription_provider
from app.services import jobs as job_service
from app.services import media as media_service
from app.services.clip_selection import (
    ScoredCandidate,
    build_scored_candidate,
    deduplicate,
    ensure_standalone_opening,
    extension_limits,
    snap_boundaries,
)
from app.services.scoring import ScoringWeights
from app.services.segmentation import Segment, segment_transcript
from app.services.storage_paths import source_audio_key, video_workdir
from app.video.audio import extract_audio, profile_audio
from app.video.ffmpeg import probe
from app.video.scenes import detect_scenes

logger = get_logger(__name__)


@dataclass(slots=True)
class AnalysisResult:
    video_id: uuid.UUID
    transcript_id: uuid.UUID | None
    segment_count: int
    scene_count: int
    candidate_count: int
    duplicates_dropped: int
    language: str | None
    transcript_is_synthetic: bool
    used_fallback: bool
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_id": str(self.video_id),
            "transcript_id": str(self.transcript_id) if self.transcript_id else None,
            "segments": self.segment_count,
            "scenes": self.scene_count,
            "candidates": self.candidate_count,
            "duplicates_dropped": self.duplicates_dropped,
            "language": self.language,
            "transcript_is_synthetic": self.transcript_is_synthetic,
            "used_fallback": self.used_fallback,
            "warnings": self.warnings,
        }


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _progress(
    session: Session, job: ProcessingJob | None, value: float, stage: str
) -> None:
    if job is not None:
        job_service.update_progress(session, job, progress=value, stage=stage)


def analyze_video(
    session: Session,
    video: Video,
    *,
    job: ProcessingJob | None = None,
    user_settings: UserSettings | None = None,
    force: bool = False,
    context: AgentContext | None = None,
) -> AnalysisResult:
    """Analyse one authorized video end to end."""
    media_service.assert_processable(video)
    context = context or AgentContext(job_id=str(job.id) if job else None)
    warnings: list[str] = []

    video.status = VideoStatus.ANALYZING
    video.status_detail = None
    session.flush()

    # ------------------------------------------------------------- 1. media
    _progress(session, job, 0.02, "resolving media")
    source = media_service.resolve_source_media(session, video)
    workdir = video_workdir(video.id)

    _progress(session, job, 0.06, "probing")
    info = probe(source)
    _apply_probe(video, info)
    session.flush()

    if info.duration < 5.0:
        raise MediaAnalysisError(
            f"Video is only {info.duration:.1f}s long; there is nothing to clip."
        )

    # ------------------------------------------------------------- 2. audio
    audio_path = workdir / "audio.wav"
    if force or not audio_path.exists():
        _progress(session, job, 0.10, "extracting audio")
        if not info.has_audio:
            raise MediaAnalysisError(
                "The video has no audio track. Moment detection needs speech."
            )
        extract_audio(source, audio_path)
        _store_audio_asset(session, video, audio_path)

    _progress(session, job, 0.16, "profiling audio")
    audio_profile = profile_audio(audio_path)

    # -------------------------------------------------------- 3. transcript
    existing = session.execute(
        select(Transcript).where(Transcript.video_id == video.id)
    ).scalars().first()

    if existing is not None and not force:
        logger.info("reusing cached transcript", extra={"video_id": str(video.id)})
        transcript_row = existing
        transcript_result = _transcript_from_rows(session, transcript_row)
        synthetic = bool(transcript_row.provider_metadata.get("synthetic"))
    else:
        _progress(session, job, 0.20, "transcribing")
        provider = get_transcription_provider()
        language = _preferred_language(user_settings, video)
        try:
            transcript_result = provider.transcribe(audio_path, language=language)
        except TranscriptionError:
            raise
        except Exception as exc:  # noqa: BLE001 - normalise provider failures
            raise TranscriptionError(f"Transcription failed: {exc}") from exc
        synthetic = bool(transcript_result.metadata.get("synthetic"))
        if synthetic:
            warnings.append(
                "No transcription provider is configured. Segment timings are "
                "real but the words are placeholders, so moment detection "
                "cannot read what is said."
            )
        if existing is not None:
            session.delete(existing)
            session.flush()
        transcript_row = None  # created after segmentation

    # ------------------------------------------------------------ 4. scenes
    _progress(session, job, 0.55, "detecting scenes")
    scene_rows = list(
        session.execute(
            select(Scene).where(Scene.video_id == video.id).order_by(Scene.start_time)
        ).scalars()
    )
    if force or not scene_rows:
        spans = detect_scenes(source, duration=info.duration)
        session.execute(delete(Scene).where(Scene.video_id == video.id))
        scene_rows = [
            Scene(
                video_id=video.id,
                index=span.index,
                start_time=span.start,
                end_time=span.end,
                change_score=span.change_score,
            )
            for span in spans
        ]
        session.add_all(scene_rows)
        session.flush()
    scene_boundaries = [row.start_time for row in scene_rows if row.start_time > 0]

    # ------------------------------------------------------- 5. segmentation
    _progress(session, job, 0.62, "segmenting transcript")
    segments = segment_transcript(
        transcript_result, scene_boundaries=scene_boundaries
    )
    if transcript_row is None:
        transcript_row = _persist_transcript(
            session, video, transcript_result, segments, synthetic=synthetic
        )

    # ---------------------------------------------------------- 6. moments
    _progress(session, job, 0.70, "finding viral moments")
    resolved = _resolve_settings(user_settings)
    agent = ViralMomentAgent()
    payload = MomentSearchPayload(
        video_title=video.title,
        video_duration=info.duration,
        segments=segments,
        min_clip_seconds=resolved["min_clip_seconds"],
        max_clip_seconds=resolved["max_clip_seconds"],
        max_candidates=min(settings.max_candidates_per_video, resolved["clips_per_video"] * 3),
        creator=video.creator_name,
        category=video.category,
        language=transcript_result.language,
        transcript_is_synthetic=synthetic,
    )
    agent_result = agent.run_windowed(payload, context=context)
    warnings.extend(agent_result.warnings)

    if job is not None:
        job_service.record_usage(
            session,
            job,
            input_tokens=context.usage.input_tokens,
            output_tokens=context.usage.output_tokens,
            calls=context.calls,
            cost_usd=context.cost_usd,
        )

    # ------------------------------------------- 7. snap, score, de-duplicate
    _progress(session, job, 0.85, "scoring candidates")
    words = transcript_result.words
    weights = ScoringWeights.from_mapping(
        user_settings.scoring_weights if user_settings else None
    )
    scored: list[ScoredCandidate] = []
    for candidate in agent_result.output.candidates:
        # Growth must not swallow a channel intro or a rambling sign-off just
        # to reach the minimum duration.
        limits = extension_limits(segments, candidate.start_time, candidate.end_time)
        span = snap_boundaries(
            candidate.start_time,
            candidate.end_time,
            words,
            scene_boundaries=scene_boundaries,
            min_duration=resolved["min_clip_seconds"],
            max_duration=resolved["max_clip_seconds"],
            total_duration=info.duration,
            limits=limits,
        )
        span = ensure_standalone_opening(
            span,
            words,
            max_duration=resolved["max_clip_seconds"],
            limits=limits,
        )
        scored.append(
            build_scored_candidate(
                span,
                candidate.dimensions,
                hook=candidate.hook,
                summary=candidate.summary,
                reason=candidate.reason,
                weights=weights,
                min_duration=resolved["min_clip_seconds"],
                max_duration=resolved["max_clip_seconds"],
                silence_ratio=audio_profile.silence_ratio,
                has_audio=info.has_audio,
            )
        )

    kept, dropped = deduplicate(scored)
    kept = [c for c in kept if c.duration > 1.0]

    # ------------------------------------------------------- 8. persistence
    _progress(session, job, 0.94, "saving candidates")
    session.execute(delete(CandidateClip).where(CandidateClip.video_id == video.id))
    model_name = agent_result.model
    for rank, candidate in enumerate(kept, start=1):
        session.add(
            _candidate_row(video, candidate, rank=rank, model_name=model_name)
        )
    for candidate in dropped:
        session.add(
            _candidate_row(
                video,
                candidate,
                rank=None,
                model_name=model_name,
                status=CandidateStatus.DISCARDED_DUPLICATE,
            )
        )

    video.status = VideoStatus.ANALYZED
    video.analyzed_at = _now()
    video.status_detail = "; ".join(warnings)[:2000] or None
    session.flush()

    logger.info(
        "analysis complete",
        extra={
            "video_id": str(video.id),
            "segments": len(segments),
            "scenes": len(scene_rows),
            "candidates": len(kept),
            "dropped": len(dropped),
            "fallback": agent_result.used_fallback,
        },
    )
    return AnalysisResult(
        video_id=video.id,
        transcript_id=transcript_row.id if transcript_row else None,
        segment_count=len(segments),
        scene_count=len(scene_rows),
        candidate_count=len(kept),
        duplicates_dropped=len(dropped),
        language=transcript_result.language,
        transcript_is_synthetic=synthetic,
        used_fallback=agent_result.used_fallback,
        warnings=warnings,
    )


# --------------------------------------------------------------------- pieces
def _apply_probe(video: Video, info) -> None:
    video.duration_seconds = info.duration
    video.width = info.width
    video.height = info.height
    video.fps = info.fps
    video.video_codec = info.video.codec_name if info.video else None
    video.audio_codec = info.audio.codec_name if info.audio else None
    video.audio_available = info.has_audio
    video.file_size_bytes = info.size_bytes


def _store_audio_asset(session: Session, video: Video, audio_path) -> None:
    storage = get_storage_provider()
    key = source_audio_key(video.user_id, video.id)
    try:
        stored = storage.put_file(key, audio_path, content_type="audio/wav")
    except Exception as exc:  # noqa: BLE001 - the local copy is what matters
        logger.warning("could not store extracted audio", extra={"error": str(exc)})
        return
    existing = media_service.get_asset(session, video.id, MediaAssetKind.SOURCE_AUDIO)
    if existing is None:
        media_service.register_asset(
            session,
            kind=MediaAssetKind.SOURCE_AUDIO,
            storage_key=stored.key,
            video_id=video.id,
            content_type="audio/wav",
            size_bytes=stored.size_bytes,
            checksum_sha256=stored.checksum_sha256,
        )


def _persist_transcript(
    session: Session,
    video: Video,
    result: TranscriptionResult,
    segments: list[Segment],
    *,
    synthetic: bool,
) -> Transcript:
    transcript = Transcript(
        video_id=video.id,
        provider=result.provider,
        model=result.model,
        language=result.language,
        language_confidence=result.language_probability,
        duration_seconds=result.duration,
        word_count=len(result.words),
        text=result.text,
        provider_metadata={**result.metadata, "synthetic": synthetic},
    )
    session.add(transcript)
    session.flush()

    for segment in segments:
        signals = segment.signals
        session.add(
            TranscriptSegment(
                transcript_id=transcript.id,
                index=segment.index,
                start_time=segment.start,
                end_time=segment.end,
                text=segment.text,
                speaker=segment.speaker,
                words=[w.to_dict() for w in segment.words],
                words_per_second=signals.words_per_second if signals else None,
                leading_pause=signals.leading_pause if signals else None,
                trailing_pause=signals.trailing_pause if signals else None,
                prefilter_score=signals.text.prefilter_score if signals else None,
                signals=signals.to_dict() if signals else {},
            )
        )
    session.flush()
    return transcript


def _transcript_from_rows(session: Session, transcript: Transcript) -> TranscriptionResult:
    """Rebuild the provider-shaped result from stored rows, so the cached path
    and the fresh path feed identical data downstream."""
    from app.providers.base import TranscriptChunk

    rows = list(
        session.execute(
            select(TranscriptSegment)
            .where(TranscriptSegment.transcript_id == transcript.id)
            .order_by(TranscriptSegment.index)
        ).scalars()
    )
    chunks = [
        TranscriptChunk(
            start=row.start_time,
            end=row.end_time,
            text=row.text,
            speaker=row.speaker,
            words=[
                TranscriptWord(
                    word=str(w.get("word", "")),
                    start=float(w.get("start", 0.0)),
                    end=float(w.get("end", 0.0)),
                    probability=w.get("probability"),
                )
                for w in (row.words or [])
            ],
        )
        for row in rows
    ]
    return TranscriptionResult(
        language=transcript.language,
        language_probability=transcript.language_confidence,
        duration=transcript.duration_seconds or 0.0,
        chunks=chunks,
        provider=transcript.provider,
        model=transcript.model,
        metadata=dict(transcript.provider_metadata or {}),
    )


def _candidate_row(
    video: Video,
    candidate: ScoredCandidate,
    *,
    rank: int | None,
    model_name: str | None,
    status: CandidateStatus = CandidateStatus.PROPOSED,
) -> CandidateClip:
    return CandidateClip(
        video_id=video.id,
        start_time=candidate.start,
        end_time=candidate.end,
        duration_seconds=candidate.duration,
        context_lead_in=candidate.context_lead_in,
        raw_moment_start=candidate.raw_start,
        hook=candidate.hook,
        summary=candidate.summary,
        reason=candidate.reason,
        transcript_excerpt=candidate.text[:4000],
        viral_score=candidate.viral_score,
        score_breakdown={
            **candidate.score.to_dict(),
            "adjustments": candidate.adjustments,
        },
        status=status,
        rank=rank,
        model_name=model_name,
    )


def _preferred_language(
    user_settings: UserSettings | None, video: Video
) -> str | None:
    if video.language:
        return video.language
    if user_settings and user_settings.preferred_languages:
        first = user_settings.preferred_languages[0]
        # "auto" is the explicit way to ask for detection.
        return None if str(first).lower() == "auto" else str(first)
    return settings.transcription_default_language


def _resolve_settings(user_settings: UserSettings | None) -> dict[str, Any]:
    if user_settings is None:
        return {
            "min_clip_seconds": 15.0,
            "max_clip_seconds": 60.0,
            "clips_per_video": settings.default_clips_per_video,
        }
    return {
        "min_clip_seconds": user_settings.min_clip_seconds,
        "max_clip_seconds": user_settings.max_clip_seconds,
        "clips_per_video": user_settings.clips_per_video,
    }
