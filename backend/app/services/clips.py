"""Clip generation: framing, rendering, packaging, QC and safety.

The generation loop is:

    plan crop -> build captions -> render -> inspect -> (auto-correct & retry)
      -> safety review -> package copy -> store -> route for approval

Auto-correction is bounded by ``UserSettings.max_regeneration_attempts``; when
the corrections are exhausted the clip is kept and routed to a human with the
outstanding issues attached, rather than being silently discarded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.agents.base import AgentContext
from app.agents.hook_agent import CopyPayload, HookAgent
from app.agents.quality_agent import QualityPayload, QualityVerdict, VideoQualityAgent
from app.agents.safety_agent import ContentSafetyAgent, SafetyPayload, verdict_of
from app.core.errors import NotFoundError, ValidationFailure
from app.core.logging import get_logger
from app.models.check import ContentSafetyCheck, QualityCheck
from app.models.clip import CandidateClip, ClipVariant, GeneratedClip
from app.models.enums import (
    CandidateStatus,
    CaptionPosition,
    CaptionStyle,
    ClipStatus,
    CropMode,
    MediaAssetKind,
    QualityStatus,
    SafetyVerdict,
)
from app.models.job import ProcessingJob
from app.models.transcript import Transcript, TranscriptSegment
from app.models.user import UserSettings
from app.models.video import Video
from app.providers.registry import get_storage_provider, get_vision_provider
from app.services import jobs as job_service
from app.services import media as media_service
from app.services.quality import inspect_clip, perceptual_hash
from app.services.storage_paths import (
    clip_subtitle_key,
    clip_thumbnail_key,
    clip_variant_key,
    clip_video_key,
    job_workdir,
)
from app.video.captions import CaptionStyleConfig, CaptionWord, words_from_segments
from app.video.ffmpeg import probe
from app.video.renderer import HookOverlay, RenderRequest, RenderResult, render_clip
from app.video.tracking import CropPlan, SplitCropPlan, plan_crop, plan_split_crop

logger = get_logger(__name__)

#: Frames per second sampled for face tracking. Enough to follow a speaker
#: change without decoding the whole clip.
CROP_SAMPLE_FPS = 3.0


@dataclass(slots=True)
class ClipRequest:
    """Everything needed to render one clip, independent of the database."""

    video: Video
    start: float
    end: float
    candidate: CandidateClip | None = None
    crop_mode: CropMode | None = None
    caption_style: CaptionStyle | None = None
    caption_position: CaptionPosition | None = None
    caption_overrides: dict[str, Any] = field(default_factory=dict)
    hook_text: str | None = None
    include_hook_overlay: bool = True
    width: int | None = None
    height: int | None = None
    fps: int | None = None

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass(slots=True)
class ClipGenerationResult:
    clip: GeneratedClip
    render: RenderResult
    quality: QualityVerdict
    safety_verdict: SafetyVerdict
    attempts: int
    corrections: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _now() -> datetime:
    return datetime.now(timezone.utc)


# --------------------------------------------------------------------- inputs
def caption_words_for(
    session: Session, video: Video, start: float, end: float
) -> list[CaptionWord]:
    """Clip-relative caption words from the stored transcript."""
    transcript = session.execute(
        select(Transcript).where(Transcript.video_id == video.id)
    ).scalars().first()
    if transcript is None:
        return []
    rows = list(
        session.execute(
            select(TranscriptSegment)
            .where(
                TranscriptSegment.transcript_id == transcript.id,
                TranscriptSegment.end_time > start,
                TranscriptSegment.start_time < end,
            )
            .order_by(TranscriptSegment.index)
        ).scalars()
    )
    if bool((transcript.provider_metadata or {}).get("synthetic")):
        # Placeholder text must never be burned into a video.
        return []
    words = words_from_segments(({"words": row.words} for row in rows), offset=start)
    return [
        w
        for w in words
        if w.end > 0 and w.start < (end - start) + 0.05
    ]


def transcript_text_for(
    session: Session, video: Video, start: float, end: float
) -> str:
    transcript = session.execute(
        select(Transcript).where(Transcript.video_id == video.id)
    ).scalars().first()
    if transcript is None:
        return ""
    if bool((transcript.provider_metadata or {}).get("synthetic")):
        return ""
    rows = session.execute(
        select(TranscriptSegment)
        .where(
            TranscriptSegment.transcript_id == transcript.id,
            TranscriptSegment.end_time > start,
            TranscriptSegment.start_time < end,
        )
        .order_by(TranscriptSegment.index)
    ).scalars()
    return " ".join(row.text for row in rows).strip()


@dataclass(slots=True)
class Framing:
    """The framing decision for one clip.

    Exactly one of ``crop`` / ``split`` is set for a given mode; ``BLUR_PAD``
    sets neither because the renderer needs no window for it.
    """

    mode: CropMode
    crop: CropPlan | None = None
    split: SplitCropPlan | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def face_driven(self) -> bool:
        if self.split is not None:
            return True
        return bool(self.crop and self.crop.face_driven)

    def to_dict(self) -> dict[str, Any]:
        if self.split is not None:
            return self.split.to_dict()
        return self.crop.to_dict() if self.crop else {"layout": "blur_pad"}


def _sample_frames(source: Path, start: float, end: float) -> list:
    """Sample frames for face tracking, rebased onto clip-relative time."""
    try:
        frames = get_vision_provider().analyze_frames(
            source, start=start, end=end, sample_fps=CROP_SAMPLE_FPS
        )
    except Exception as exc:  # noqa: BLE001 - framing must not fail the clip
        logger.warning(
            "face analysis failed; falling back to a centred crop",
            extra={"error": str(exc)},
        )
        return []
    for frame in frames:
        frame.timestamp = max(0.0, frame.timestamp - start)
    return frames


def build_framing(
    source: Path,
    *,
    start: float,
    end: float,
    crop_mode: CropMode,
    width: int,
    height: int,
    target_width: int,
    target_height: int,
    scene_boundaries: Sequence[float] = (),
) -> Framing:
    """Plan the vertical framing for one clip."""
    if crop_mode is CropMode.BLUR_PAD:
        return Framing(mode=crop_mode)

    relative_scenes = [b - start for b in scene_boundaries if start <= b <= end]
    frames = (
        _sample_frames(source, start, end)
        if crop_mode in (CropMode.SMART, CropMode.SPLIT_SPEAKERS)
        else []
    )

    if crop_mode is CropMode.SPLIT_SPEAKERS:
        split = plan_split_crop(
            frames,
            source_width=width,
            source_height=height,
            target_width=target_width,
            target_height=target_height,
            scene_boundaries=relative_scenes,
        )
        if split is not None:
            return Framing(mode=crop_mode, split=split, notes=list(split.notes))
        # One speaker (or none) is not a split screen. Fall back to following
        # the single subject rather than showing the same face twice.
        logger.info("split framing needs two speakers; falling back to smart crop")
        crop_mode = CropMode.SMART
        fallback_note = "only one speaker detected; used single-subject framing"
    else:
        fallback_note = None

    plan = plan_crop(
        frames,
        source_width=width,
        source_height=height,
        target_aspect=target_width / target_height,
        scene_boundaries=relative_scenes,
    )
    if crop_mode is CropMode.CENTER:
        centre = max(0.0, (width - plan.crop_width) / 2)
        plan.keyframes = [
            plan.keyframes[0].__class__(t=0.0, x=centre, y=plan.keyframes[0].y)
        ]
        plan.face_driven = False
        plan.notes.append("centre crop requested")
    if fallback_note:
        plan.notes.append(fallback_note)
    return Framing(mode=crop_mode, crop=plan, notes=list(plan.notes))


# ------------------------------------------------------------------ rendering
def generate_clip(
    session: Session,
    request: ClipRequest,
    *,
    user_settings: UserSettings | None = None,
    job: ProcessingJob | None = None,
    context: AgentContext | None = None,
) -> ClipGenerationResult:
    """Render, inspect, correct and package one clip."""
    video = request.video
    media_service.assert_processable(video)
    if request.duration <= 1.0:
        raise ValidationFailure(
            f"Clip duration must be greater than 1s, got {request.duration:.2f}s."
        )

    context = context or AgentContext(job_id=str(job.id) if job else None)
    settings_view = _settings_view(user_settings)
    source = media_service.resolve_source_media(session, video)
    info = probe(source)

    crop_mode = request.crop_mode or settings_view["crop_mode"]
    caption_style = request.caption_style or settings_view["caption_style"]
    caption_position = request.caption_position or settings_view["caption_position"]
    out_width = request.width or settings_view["output_width"]
    out_height = request.height or settings_view["output_height"]

    clip = GeneratedClip(
        user_id=video.user_id,
        video_id=video.id,
        candidate_id=request.candidate.id if request.candidate else None,
        start_time=request.start,
        end_time=request.end,
        duration_seconds=request.duration,
        crop_mode=crop_mode,
        caption_style=caption_style,
        caption_position=caption_position,
        status=ClipStatus.RENDERING,
        viral_score=request.candidate.viral_score if request.candidate else None,
    )
    session.add(clip)
    session.flush()

    workdir = job_workdir(job.id if job else clip.id)
    scene_boundaries = _scene_boundaries(session, video)

    caption_config = CaptionStyleConfig.for_style(
        caption_style,
        caption_position,
        **{**settings_view["caption_overrides"], **request.caption_overrides},
    )
    words = caption_words_for(session, video, request.start, request.end)
    clip_text = transcript_text_for(session, video, request.start, request.end)

    # ------------------------------------------------------------- packaging
    if job is not None:
        job_service.update_progress(session, job, progress=0.15, stage="writing copy")
    copy_result = HookAgent().run(
        CopyPayload(
            transcript=clip_text,
            duration=request.duration,
            summary=request.candidate.summary if request.candidate else "",
            existing_hook=request.hook_text
            or (request.candidate.hook if request.candidate else ""),
            source_title=video.title,
            creator=video.creator_name,
            category=video.category,
            language=video.language,
        ),
        context=context,
    )
    copy = copy_result.output
    hook_text = request.hook_text or (copy.hooks[0].text if copy.hooks else None)

    # ---------------------------------------------------------- render loop
    max_attempts = max(1, settings_view["max_regeneration_attempts"] + 1)
    corrections: list[str] = []
    warnings: list[str] = list(copy_result.warnings)
    attempt = 0
    start, end = request.start, request.end
    render: RenderResult | None = None
    verdict: QualityVerdict | None = None
    last_framing: Framing | None = None

    while attempt < max_attempts:
        attempt += 1
        if job is not None:
            job_service.update_progress(
                session, job, progress=0.2 + 0.5 * (attempt / max_attempts),
                stage=f"rendering (attempt {attempt})",
            )
        attempt_dir = workdir / f"attempt_{attempt}"
        framing = last_framing = build_framing(
            source,
            start=start,
            end=end,
            crop_mode=crop_mode,
            width=info.width or out_width,
            height=info.height or out_height,
            target_width=out_width,
            target_height=out_height,
            scene_boundaries=scene_boundaries,
        )
        relative_words = _shift_words(words, request.start, start)
        render = render_clip(
            RenderRequest(
                source=source,
                output=attempt_dir / "clip.mp4",
                start=start,
                end=end,
                workdir=attempt_dir,
                crop_mode=crop_mode,
                crop_plan=framing.crop,
                split_plan=framing.split,
                caption_words=relative_words,
                caption_config=caption_config,
                hook=(
                    HookOverlay(text=hook_text)
                    if hook_text and request.include_hook_overlay
                    else None
                ),
                width=out_width,
                height=out_height,
                fps=request.fps or settings_view["output_fps"],
            )
        )
        warnings.extend(render.warnings)

        if job is not None:
            job_service.update_progress(
                session, job, progress=0.75, stage="quality control"
            )
        cues = [c.to_dict() for c in (render.captions.cues if render.captions else [])]
        technical = inspect_clip(
            render.output,
            expected_duration=end - start,
            expected_width=out_width,
            expected_height=out_height,
            min_duration=settings_view["min_clip_seconds"],
            max_duration=settings_view["max_clip_seconds"],
            caption_cues=cues,
            caption_font_size=(
                caption_config.font_size if caption_style is not CaptionStyle.NONE else None
            ),
            caption_y_offset=render.captions.band.y_offset if render.captions else None,
            caption_band_height=render.captions.band.height if render.captions else None,
        )
        verdict = VideoQualityAgent().evaluate(
            QualityPayload(
                transcript=clip_text,
                duration=render.duration,
                technical=technical,
                caption_text=" ".join(c.text for c in (render.captions.cues if render.captions else [])),
                hook_text=hook_text or "",
                crop_notes=framing.notes or ["blurred padding"],
                face_driven_crop=framing.face_driven,
            ),
            threshold=settings_view["quality_threshold"],
            context=context,
        )
        session.add(
            QualityCheck(
                clip_id=clip.id,
                attempt=attempt,
                status=verdict.status,
                score=verdict.score,
                threshold=verdict.threshold,
                issues=[i.to_dict() for i in verdict.issues],
                recommendations=verdict.recommendations,
                technical=verdict.technical.to_dict(),
                auto_corrected=False,
            )
        )
        session.flush()

        if verdict.status is not QualityStatus.FAIL or attempt >= max_attempts:
            break

        applied, start, end, caption_config = _apply_corrections(
            verdict,
            start=start,
            end=end,
            caption_config=caption_config,
            min_duration=settings_view["min_clip_seconds"],
            max_duration=settings_view["max_clip_seconds"],
            source_duration=info.duration,
        )
        if not applied:
            logger.info(
                "no automatic correction available; keeping the clip for review",
                extra={"clip_id": str(clip.id), "score": round(verdict.score, 1)},
            )
            break
        corrections.extend(applied)
        clip.regeneration_count += 1
        logger.info(
            "regenerating clip after quality failure",
            extra={"clip_id": str(clip.id), "attempt": attempt, "fixes": applied},
        )

    assert render is not None and verdict is not None  # loop always runs once

    # -------------------------------------------------------------- safety
    if job is not None:
        job_service.update_progress(session, job, progress=0.85, stage="safety review")
    safety = ContentSafetyAgent().run(
        SafetyPayload(
            transcript=clip_text,
            duration=render.duration,
            title=copy.youtube_title,
            hook_text=hook_text or "",
            source_title=video.title,
            language=video.language,
            authorization_status=str(video.authorization_status),
        ),
        context=context,
    )
    safety_output = safety.output
    safety_verdict = verdict_of(safety_output)
    session.add(
        ContentSafetyCheck(
            clip_id=clip.id,
            verdict=safety_verdict,
            category_scores=safety_output.category_scores,
            flagged_categories=safety_output.flagged_categories,
            rationale=safety_output.rationale,
            evidence=[{"quote": q} for q in safety_output.evidence],
            model_name=safety.model,
        )
    )

    # ------------------------------------------------------------- storage
    if job is not None:
        job_service.update_progress(session, job, progress=0.92, stage="uploading")
    _store_outputs(session, clip, render)

    # -------------------------------------------------------------- finish
    clip.start_time = start
    clip.end_time = end
    clip.duration_seconds = render.duration
    clip.width = render.width
    clip.height = render.height
    clip.fps = render.fps
    clip.file_size_bytes = render.size_bytes
    clip.quality_score = verdict.score
    clip.safety_verdict = safety_verdict.value
    clip.title = copy.youtube_title
    clip.hook_text = hook_text
    clip.hook_alternatives = [
        {"text": h.text, "rationale": h.rationale, "confidence": h.confidence}
        for h in copy.hooks
    ]
    clip.description = copy.description
    clip.instagram_caption = copy.instagram_caption
    clip.hashtags = copy.hashtags
    clip.keywords = copy.keywords
    clip.transcript_text = clip_text
    clip.captions = [c.to_dict() for c in (render.captions.cues if render.captions else [])]
    # Record the framing that was actually rendered, so the editor can show
    # the camera path and a regeneration can reproduce it.
    clip.crop_keyframes = (
        [k.to_dict() for k in last_framing.crop.keyframes]
        if last_framing and last_framing.crop
        else []
    )
    clip.render_config = {
        "crop_mode": crop_mode.value,
        "caption_style": caption_style.value,
        "caption_position": caption_position.value,
        "output": [out_width, out_height],
        "fps": render.fps,
        "filter_graph": render.filter_graph,
        "corrections": corrections,
        "attempts": attempt,
        "crop": last_framing.to_dict() if last_framing else None,
    }
    clip.perceptual_hash = perceptual_hash(render.output)
    clip.status = _route_status(
        clip,
        verdict=verdict,
        safety=safety_verdict,
        settings_view=settings_view,
    )
    clip.status_detail = "; ".join(warnings)[:2000] or None
    if request.candidate is not None:
        request.candidate.status = CandidateStatus.GENERATED

    if job is not None:
        job_service.record_usage(
            session,
            job,
            input_tokens=context.usage.input_tokens,
            output_tokens=context.usage.output_tokens,
            calls=context.calls,
            cost_usd=context.cost_usd,
        )
    session.flush()

    logger.info(
        "clip generated",
        extra={
            "clip_id": str(clip.id),
            "status": clip.status,
            "quality": round(verdict.score, 1),
            "safety": safety_verdict.value,
            "attempts": attempt,
        },
    )
    return ClipGenerationResult(
        clip=clip,
        render=render,
        quality=verdict,
        safety_verdict=safety_verdict,
        attempts=attempt,
        corrections=corrections,
        warnings=warnings,
    )


# ------------------------------------------------------------------- helpers
def _settings_view(user_settings: UserSettings | None) -> dict[str, Any]:
    if user_settings is None:
        return {
            "crop_mode": CropMode.SMART,
            "caption_style": CaptionStyle.WORD_HIGHLIGHT,
            "caption_position": CaptionPosition.LOWER_THIRD,
            "caption_overrides": {},
            "output_width": 1080,
            "output_height": 1920,
            "output_fps": None,
            "min_clip_seconds": 15.0,
            "max_clip_seconds": 60.0,
            "quality_threshold": 75.0,
            "max_regeneration_attempts": 2,
            "auto_approve_score": 90.0,
            "reject_below_score": 70.0,
            "block_on_safety_review": True,
        }
    return {
        "crop_mode": CropMode(user_settings.crop_mode),
        "caption_style": CaptionStyle(user_settings.caption_style),
        "caption_position": CaptionPosition(user_settings.caption_position),
        "caption_overrides": dict(user_settings.caption_overrides or {}),
        "output_width": user_settings.output_width,
        "output_height": user_settings.output_height,
        "output_fps": user_settings.output_fps,
        "min_clip_seconds": user_settings.min_clip_seconds,
        "max_clip_seconds": user_settings.max_clip_seconds,
        "quality_threshold": user_settings.quality_threshold,
        "max_regeneration_attempts": user_settings.max_regeneration_attempts,
        "auto_approve_score": user_settings.auto_approve_score,
        "reject_below_score": user_settings.reject_below_score,
        "block_on_safety_review": user_settings.block_on_safety_review,
    }


def _scene_boundaries(session: Session, video: Video) -> list[float]:
    from app.models.transcript import Scene

    rows = session.execute(
        select(Scene.start_time).where(Scene.video_id == video.id).order_by(Scene.start_time)
    ).scalars()
    return [value for value in rows if value and value > 0]


def _shift_words(
    words: Sequence[CaptionWord], original_start: float, new_start: float
) -> list[CaptionWord]:
    """Re-base caption timings when a correction moved the clip start."""
    delta = new_start - original_start
    if abs(delta) < 1e-6:
        return list(words)
    return [
        CaptionWord(text=w.text, start=w.start - delta, end=w.end - delta)
        for w in words
        if w.end - delta > 0
    ]


def _apply_corrections(
    verdict: QualityVerdict,
    *,
    start: float,
    end: float,
    caption_config: CaptionStyleConfig,
    min_duration: float,
    max_duration: float,
    source_duration: float,
) -> tuple[list[str], float, float, CaptionStyleConfig]:
    """Translate quality issues into concrete changes for the next attempt."""
    applied: list[str] = []
    for action in verdict.auto_fixes:
        if action == "trim_start":
            leading = max(
                verdict.technical.leading_black, verdict.technical.leading_silence
            )
            shift = min(leading, max(0.0, (end - start) - min_duration))
            if shift > 0.2:
                start += shift
                applied.append(f"trimmed {shift:.2f}s of dead air from the start")
        elif action == "trim":
            overshoot = (end - start) - max_duration
            if overshoot > 0:
                end -= overshoot
                applied.append(f"trimmed {overshoot:.2f}s to fit the maximum duration")
        elif action == "extend_end":
            room = min(2.5, max(0.0, source_duration - end))
            if room > 0.3 and (end + room - start) <= max_duration:
                end += room
                applied.append(f"extended the ending by {room:.2f}s")
        elif action == "extend_context":
            room = min(3.0, start)
            if room > 0.3 and (end - (start - room)) <= max_duration:
                start -= room
                applied.append(f"extended the opening by {room:.2f}s for context")
        elif action == "enlarge_captions":
            caption_config.font_size = int(caption_config.font_size * 1.25)
            applied.append(f"increased caption size to {caption_config.font_size}px")
        elif action == "reposition_captions":
            caption_config.position = CaptionPosition.LOWER_THIRD
            caption_config.safe_area_ratio = max(0.16, caption_config.safe_area_ratio)
            applied.append("moved captions inside the platform-safe area")
        elif action in ("normalize_audio", "rescale", "regenerate_captions"):
            # These are already unconditional in the renderer; re-running is
            # the correction.
            applied.append(f"re-rendered to address {action}")
    return applied, start, end, caption_config


def _route_status(
    clip: GeneratedClip,
    *,
    verdict: QualityVerdict,
    safety: SafetyVerdict,
    settings_view: dict[str, Any],
) -> ClipStatus:
    """Decide whether a clip is auto-approved, queued for review, or rejected.

    Safety always wins: a BLOCK is rejected regardless of score, and a REVIEW
    verdict prevents auto-approval when the user has asked it to.
    """
    if safety is SafetyVerdict.BLOCK:
        clip.review_note = "Blocked by the content safety review."
        return ClipStatus.REJECTED
    if verdict.status is QualityStatus.FAIL:
        clip.review_note = (
            f"Quality score {verdict.score:.0f} is below the "
            f"{verdict.threshold:.0f} threshold."
        )
        return ClipStatus.NEEDS_REVIEW

    score = clip.viral_score if clip.viral_score is not None else 0.0
    if score < settings_view["reject_below_score"]:
        clip.review_note = (
            f"Viral score {score:.0f} is below the "
            f"{settings_view['reject_below_score']:.0f} rejection threshold."
        )
        return ClipStatus.NEEDS_REVIEW
    if (
        score >= settings_view["auto_approve_score"]
        and not (safety is SafetyVerdict.REVIEW and settings_view["block_on_safety_review"])
    ):
        clip.auto_approved = True
        clip.approved_at = _now()
        return ClipStatus.APPROVED
    return ClipStatus.NEEDS_REVIEW


def _store_outputs(session: Session, clip: GeneratedClip, render: RenderResult) -> None:
    storage = get_storage_provider()
    key = clip_video_key(clip.user_id, clip.id)
    stored = storage.put_file(key, render.output, content_type="video/mp4")
    clip.storage_key = stored.key
    media_service.register_asset(
        session,
        kind=MediaAssetKind.CLIP_VIDEO,
        storage_key=stored.key,
        clip_id=clip.id,
        content_type="video/mp4",
        size_bytes=stored.size_bytes,
        checksum_sha256=stored.checksum_sha256,
    )

    if render.thumbnail and Path(render.thumbnail).exists():
        thumb = storage.put_file(
            clip_thumbnail_key(clip.user_id, clip.id),
            render.thumbnail,
            content_type="image/jpeg",
        )
        clip.thumbnail_storage_key = thumb.key
        media_service.register_asset(
            session,
            kind=MediaAssetKind.THUMBNAIL,
            storage_key=thumb.key,
            clip_id=clip.id,
            content_type="image/jpeg",
            size_bytes=thumb.size_bytes,
        )

    for subtitle in render.subtitle_files:
        path = Path(subtitle)
        if not path.exists():
            continue
        stored_sub = storage.put_file(
            clip_subtitle_key(clip.user_id, clip.id, path.suffix),
            path,
            content_type="text/plain",
        )
        if path.suffix == ".srt":
            clip.subtitle_storage_key = stored_sub.key
        media_service.register_asset(
            session,
            kind=MediaAssetKind.SUBTITLE,
            storage_key=stored_sub.key,
            clip_id=clip.id,
            content_type="text/plain",
            size_bytes=stored_sub.size_bytes,
        )


# ------------------------------------------------------------------ variants
VARIANT_RECIPES: dict[str, dict[str, Any]] = {
    "A": {
        "description": "Original cut with the default caption style.",
    },
    "B": {
        "description": "Stronger contextual opening: starts ~2s earlier.",
        "start_delta": -2.0,
    },
    "C": {
        "description": "Alternative framing and caption style.",
        "crop_mode": CropMode.BLUR_PAD,
        "caption_style": CaptionStyle.BOLD,
    },
}


def generate_variants(
    session: Session,
    clip: GeneratedClip,
    *,
    labels: Sequence[str] = ("B", "C"),
    user_settings: UserSettings | None = None,
    job: ProcessingJob | None = None,
) -> list[ClipVariant]:
    """Render alternative treatments of the same moment.

    Variants change framing, captions and how much lead-in there is. They never
    change what the speaker says, and they never re-order speech.
    """
    video = session.get(Video, clip.video_id)
    if video is None:
        raise NotFoundError(f"Video {clip.video_id} no longer exists.")
    media_service.assert_processable(video)

    source = media_service.resolve_source_media(session, video)
    info = probe(source)
    settings_view = _settings_view(user_settings)
    scene_boundaries = _scene_boundaries(session, video)
    created: list[ClipVariant] = []

    for label in labels:
        recipe = VARIANT_RECIPES.get(label)
        if recipe is None:
            continue
        start = max(0.0, clip.start_time + float(recipe.get("start_delta", 0.0)))
        end = min(info.duration, clip.end_time + float(recipe.get("end_delta", 0.0)))
        if end - start <= 1.0:
            continue

        crop_mode = recipe.get("crop_mode", CropMode(clip.crop_mode))
        caption_style = recipe.get("caption_style", CaptionStyle(clip.caption_style))
        variant = ClipVariant(
            clip_id=clip.id,
            label=label,
            description=str(recipe.get("description", "")),
            start_time=start,
            end_time=end,
            crop_mode=crop_mode,
            caption_style=caption_style,
            hook_text=clip.hook_text,
            status=ClipStatus.RENDERING,
        )
        session.add(variant)
        session.flush()

        # Keyed by the clip, not a fresh UUID: a random directory per call
        # leaves orphaned scratch space behind on every regeneration.
        workdir = job_workdir(job.id if job else clip.id) / f"variant_{label}"
        framing = build_framing(
            source,
            start=start,
            end=end,
            crop_mode=crop_mode,
            width=info.width or settings_view["output_width"],
            height=info.height or settings_view["output_height"],
            target_width=settings_view["output_width"],
            target_height=settings_view["output_height"],
            scene_boundaries=scene_boundaries,
        )
        # The upload is inside the try as well: an object-store outage is at
        # least as likely as an ffmpeg failure, and it must fail this variant
        # rather than abort the ones still queued behind it.
        try:
            render = render_clip(
                RenderRequest(
                    source=source,
                    output=workdir / f"variant_{label}.mp4",
                    start=start,
                    end=end,
                    workdir=workdir,
                    crop_mode=crop_mode,
                    crop_plan=framing.crop,
                    split_plan=framing.split,
                    caption_words=caption_words_for(session, video, start, end),
                    caption_config=CaptionStyleConfig.for_style(
                        caption_style, CaptionPosition(clip.caption_position)
                    ),
                    hook=HookOverlay(text=clip.hook_text) if clip.hook_text else None,
                    width=settings_view["output_width"],
                    height=settings_view["output_height"],
                )
            )
            stored = get_storage_provider().put_file(
                clip_variant_key(clip.user_id, clip.id, label),
                render.output,
                content_type="video/mp4",
            )
        except Exception as exc:  # noqa: BLE001 - one variant must not fail the rest
            variant.status = ClipStatus.FAILED
            variant.description = f"{variant.description} (failed: {exc})"[:1000]
            session.flush()
            logger.warning(
                "variant failed",
                extra={"clip_id": str(clip.id), "label": label, "error": str(exc)},
            )
            continue

        variant.storage_key = stored.key
        variant.status = ClipStatus.RENDERED
        variant.render_config = {
            "crop_mode": crop_mode.value,
            "caption_style": caption_style.value,
            "duration": render.duration,
        }
        session.flush()
        created.append(variant)

    return created
