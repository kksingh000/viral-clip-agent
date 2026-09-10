"""TrendAgent -- topic intelligence over discovered video metadata.

The ranking arithmetic lives in :mod:`app.services.trends`. This agent does the
one part that needs language understanding: reading a batch of titles and
descriptions and naming the *topics* they are collectively about -- merging
"AI coding agents", "agentic coding" and "Claude Code workflow" into one topic
rather than three keyword rows.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from pydantic import Field, field_validator
from slugify import slugify

from app.agents.base import AgentOutput, BaseAgent, compact_json
from app.core.logging import get_logger

logger = get_logger(__name__)

TOPIC_KINDS = ("topic", "event", "meme", "debate", "creator", "product", "other")


class DetectedTopic(AgentOutput):
    label: str = Field(min_length=2, max_length=120)
    kind: str = Field(default="topic")
    keywords: list[str] = Field(default_factory=list, max_length=12)
    video_indexes: list[int] = Field(
        default_factory=list,
        max_length=200,
        description="Indexes into the supplied videos array.",
    )
    rationale: str = Field(default="", max_length=400)

    @field_validator("kind")
    @classmethod
    def _known_kind(cls, value: str) -> str:
        lowered = (value or "topic").strip().lower()
        return lowered if lowered in TOPIC_KINDS else "other"


class TopicOutput(AgentOutput):
    topics: list[DetectedTopic] = Field(default_factory=list, max_length=30)


@dataclass(slots=True)
class TopicPayload:
    videos: list[dict[str, Any]] = field(default_factory=list)
    region: str | None = None
    known_topics: list[str] = field(default_factory=list)
    max_topics: int = 15


SYSTEM = """\
# Role

You are TrendAgent. You read a batch of recently discovered video metadata and
identify the topics the batch is collectively about.

# Objective

Produce a small set of *topics* -- not keywords. "AI coding agents", "agentic
coding assistants" and "I replaced my dev team with Claude" belong to one
topic, not three.

# Input

- `videos`: an array of `{index, title, channel, category, tags, description}`
- `known_topics`: topics already in the database; reuse a label verbatim when
  a video belongs to an existing topic rather than inventing a near-duplicate
- `region`: the discovery region, if any

# Output

For each topic: a short human-readable `label`, a `kind` (topic, event, meme,
debate, creator, product, other), 3-8 `keywords` that would match future
videos on the same subject, the `video_indexes` that belong to it, and a
one-sentence `rationale`.

# Constraints

- A topic must cover at least two videos, unless it is clearly a breaking
  event.
- Labels are 2-6 words, specific enough to be actionable ("AI coding agents",
  not "technology").
- Reuse a `known_topics` label exactly when it applies.
- A video may belong to more than one topic.
- Return at most `max_topics` topics, ordered by how many videos they cover.
- Work only from the metadata supplied. Do not speculate about the video
  contents beyond what the title, tags and description state.

# Failure behaviour

If the batch has no coherent subject, return an empty `topics` array. Do not
invent topics to fill the response.
"""


class TrendAgent(BaseAgent[TopicPayload, TopicOutput]):
    name = "trend_agent"
    output_model = TopicOutput
    temperature = 0.2
    prefers_cheap_model = True

    def build_system_prompt(self, payload: TopicPayload) -> str:
        return SYSTEM

    def build_user_content(self, payload: TopicPayload) -> str:
        return compact_json(
            {
                "region": payload.region,
                "max_topics": payload.max_topics,
                "known_topics": payload.known_topics,
                "videos": payload.videos,
            }
        )

    def postprocess(self, output: TopicOutput, payload: TopicPayload) -> TopicOutput:
        valid_range = range(len(payload.videos))
        cleaned: list[DetectedTopic] = []
        seen: set[str] = set()
        for topic in output.topics:
            slug = slugify(topic.label)
            if not slug or slug in seen:
                continue
            seen.add(slug)
            topic.video_indexes = sorted(
                {i for i in topic.video_indexes if i in valid_range}
            )
            topic.keywords = [
                k.strip().lower() for k in topic.keywords if k and k.strip()
            ][:8]
            cleaned.append(topic)
        cleaned.sort(key=lambda t: len(t.video_indexes), reverse=True)
        output.topics = cleaned[: payload.max_topics]
        return output

    # -------------------------------------------------------------- fallback
    def fallback(self, payload: TopicPayload) -> TopicOutput:
        """N-gram clustering over titles and tags.

        Finds repeated 2- and 3-word phrases across the batch and groups the
        videos that contain them. It cannot merge synonyms the way the model
        can, but it does surface genuinely repeated subjects.
        """
        documents: list[tuple[int, str, set[str]]] = []
        for index, video in enumerate(payload.videos):
            text = " ".join(
                str(video.get(key) or "")
                for key in ("title", "category", "description")
            )
            tags = {str(t).lower().strip() for t in (video.get("tags") or []) if t}
            documents.append((index, text.lower(), tags))

        counter: Counter[str] = Counter()
        phrase_videos: dict[str, set[int]] = {}
        for index, text, tags in documents:
            for phrase in _phrases(text) | tags:
                counter[phrase] += 1
                phrase_videos.setdefault(phrase, set()).add(index)

        topics: list[DetectedTopic] = []
        claimed: set[str] = set()
        claimed_videos: list[set[int]] = []
        for phrase, count in counter.most_common(200):
            if count < 2 or len(topics) >= payload.max_topics:
                continue
            words = set(phrase.split())
            # Skip a phrase that is a near-duplicate of one already taken,
            # either lexically or by covering the same videos. Without the
            # second test, "ai" is emitted alongside "coding agents" for the
            # identical pair of videos.
            if any(len(words & set(other.split())) >= len(words) for other in claimed):
                continue
            videos = phrase_videos[phrase]
            if any(videos <= existing for existing in claimed_videos):
                continue
            claimed.add(phrase)
            claimed_videos.append(videos)
            topics.append(
                DetectedTopic(
                    label=phrase.title()[:120],
                    kind="topic",
                    keywords=sorted(words)[:8],
                    video_indexes=sorted(phrase_videos[phrase]),
                    rationale=(
                        f"Phrase appears in {count} of {len(documents)} videos "
                        "(n-gram clustering; no LLM available)."
                    ),
                )
            )
        return TopicOutput(topics=topics)


_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "the", "and", "for", "with", "you", "your", "how", "why", "what", "this",
    "that", "from", "are", "was", "will", "can", "new", "best", "top", "video",
    "watch", "full", "part", "episode", "official", "live", "vs", "about",
    "into", "our", "his", "her", "their", "has", "have", "been", "not", "but",
}


def _phrases(text: str, *, sizes: tuple[int, ...] = (2, 3)) -> set[str]:
    tokens = [t for t in _TOKEN_RE.findall(text) if t not in _STOPWORDS and len(t) > 2]
    out: set[str] = set()
    for size in sizes:
        for i in range(len(tokens) - size + 1):
            out.add(" ".join(tokens[i : i + size]))
    return out
