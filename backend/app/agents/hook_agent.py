"""HookAgent -- on-screen hooks and platform packaging.

The hard rule enforced here in code (not only in the prompt) is that generated
copy must stay faithful to the clip: :meth:`HookAgent.postprocess` drops hooks
that assert a number the transcript never mentions, and truncates anything
over the platform limits.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from pydantic import Field, field_validator

from app.agents.base import AgentOutput, BaseAgent, compact_json
from app.ai.prompts.copy import SYSTEM
from app.core.logging import get_logger
from app.services.signals import analyze_text

logger = get_logger(__name__)

YOUTUBE_TITLE_LIMIT = 100
INSTAGRAM_CAPTION_LIMIT = 300
HOOK_CHAR_LIMIT = 60
_NUMBER_RE = re.compile(r"\d[\d,.]*")


class HookOption(AgentOutput):
    text: str = Field(min_length=2, max_length=120)
    rationale: str = Field(default="", max_length=400)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)

    @field_validator("text", "rationale")
    @classmethod
    def _strip(cls, value: str) -> str:
        return value.strip()


class CopyOutput(AgentOutput):
    hooks: list[HookOption] = Field(default_factory=list, max_length=5)
    youtube_title: str = Field(default="", max_length=140)
    instagram_caption: str = Field(default="", max_length=400)
    description: str = Field(default="", max_length=1200)
    hashtags: list[str] = Field(default_factory=list, max_length=12)
    keywords: list[str] = Field(default_factory=list, max_length=15)


@dataclass(slots=True)
class CopyPayload:
    transcript: str
    duration: float
    summary: str = ""
    existing_hook: str = ""
    source_title: str = ""
    creator: str | None = None
    category: str | None = None
    language: str | None = None
    trending_topics: list[str] = field(default_factory=list)


class HookAgent(BaseAgent[CopyPayload, CopyOutput]):
    name = "hook_agent"
    output_model = CopyOutput
    temperature = 0.7
    #: Packaging is a small, well-specified task -- the cheaper tier is enough.
    prefers_cheap_model = True

    def build_system_prompt(self, payload: CopyPayload) -> str:
        return SYSTEM

    def build_user_content(self, payload: CopyPayload) -> str:
        return compact_json(
            {
                "clip": {
                    "duration_seconds": round(payload.duration, 2),
                    "transcript": payload.transcript,
                    "summary": payload.summary,
                },
                "source": {
                    "title": payload.source_title,
                    "creator": payload.creator,
                    "category": payload.category,
                },
                "language": payload.language,
                "existing_hook": payload.existing_hook,
                "trending_topics": payload.trending_topics,
            }
        )

    # ---------------------------------------------------------- faithfulness
    def postprocess(self, output: CopyOutput, payload: CopyPayload) -> CopyOutput:
        transcript_numbers = set(_NUMBER_RE.findall(payload.transcript))
        kept: list[HookOption] = []
        for hook in output.hooks:
            invented = set(_NUMBER_RE.findall(hook.text)) - transcript_numbers
            if invented:
                logger.warning(
                    "dropped hook asserting a number absent from the clip",
                    extra={"hook": hook.text, "numbers": ",".join(sorted(invented))},
                )
                continue
            hook.text = _truncate(hook.text, HOOK_CHAR_LIMIT)
            kept.append(hook)
        if not kept and output.hooks:
            # Everything was rejected; fall back rather than shipping nothing.
            kept = [HookOption(text=self._safe_hook(payload), confidence=0.25)]
        output.hooks = sorted(kept, key=lambda h: h.confidence, reverse=True)[:5]

        output.youtube_title = _truncate(
            output.youtube_title or self._safe_hook(payload), YOUTUBE_TITLE_LIMIT
        )
        output.instagram_caption = _truncate(
            output.instagram_caption or output.youtube_title, INSTAGRAM_CAPTION_LIMIT
        )
        output.hashtags = _clean_hashtags(output.hashtags)
        output.keywords = [k.strip().lower() for k in output.keywords if k.strip()][:10]
        return output

    # ------------------------------------------------------------- fallback
    def fallback(self, payload: CopyPayload) -> CopyOutput:
        """Deterministic packaging derived from the clip's own words.

        Every field is extracted from the transcript, never invented, so the
        no-model path produces honest (if plain) copy.
        """
        sentences = _sentences(payload.transcript)
        opening = payload.existing_hook or (sentences[0] if sentences else "")
        opening = _truncate(opening, HOOK_CHAR_LIMIT)

        hooks: list[HookOption] = []
        if opening:
            hooks.append(
                HookOption(
                    text=opening,
                    rationale="First line of the clip, used verbatim.",
                    confidence=0.4,
                )
            )
        # A question or a marker-dense sentence often works better than the
        # literal first line.
        ranked = sorted(
            sentences[1:6],
            key=lambda s: analyze_text(s).prefilter_score,
            reverse=True,
        )
        for sentence in ranked[:2]:
            candidate = _truncate(sentence, HOOK_CHAR_LIMIT)
            if candidate and candidate not in {h.text for h in hooks}:
                hooks.append(
                    HookOption(
                        text=candidate,
                        rationale="High-signal line from the clip (no LLM available).",
                        confidence=0.3,
                    )
                )
        if not hooks:
            hooks = [
                HookOption(
                    text=_truncate(payload.summary or payload.source_title, HOOK_CHAR_LIMIT)
                    or "Clip",
                    rationale="No usable transcript text.",
                    confidence=0.1,
                )
            ]

        title = _truncate(
            payload.summary or opening or payload.source_title, YOUTUBE_TITLE_LIMIT
        )
        keywords = sorted(
            {w for w in _keywords(payload.transcript)},
        )[:8]
        return CopyOutput(
            hooks=hooks,
            youtube_title=title,
            instagram_caption=_truncate(
                f"{opening}\n\nFrom: {payload.source_title}".strip(),
                INSTAGRAM_CAPTION_LIMIT,
            ),
            description=(
                payload.summary
                or _truncate(payload.transcript, 280)
                or "Clip from the source video."
            ),
            hashtags=_clean_hashtags(keywords[:5]),
            keywords=keywords,
        )

    @staticmethod
    def _safe_hook(payload: CopyPayload) -> str:
        sentences = _sentences(payload.transcript)
        return _truncate(
            payload.existing_hook or (sentences[0] if sentences else payload.summary),
            HOOK_CHAR_LIMIT,
        )


# --------------------------------------------------------------------- helpers
def _truncate(text: str, limit: int) -> str:
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    cut = text[:limit].rstrip()
    if " " in cut:
        cut = cut[: cut.rfind(" ")]
    return cut.rstrip(" ,;:-") + "..."


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?।])\s+", (text or "").strip())
    return [p.strip() for p in parts if len(p.strip()) > 8]


def _clean_hashtags(tags: list[str]) -> list[str]:
    #: Generic reach tags add no targeting and platforms discount them.
    generic = {"viral", "fyp", "foryou", "foryoupage", "trending", "explore", "reels", "shorts"}
    out: list[str] = []
    for tag in tags:
        cleaned = re.sub(r"[^0-9a-zऀ-ॿ]", "", str(tag).lower().lstrip("#"))
        if len(cleaned) < 3 or cleaned in generic or cleaned in out:
            continue
        out.append(cleaned)
    return out[:8]


def _keywords(text: str) -> set[str]:
    from app.services.signals import keyword_set

    return keyword_set(text, minimum_length=5, limit=20)
