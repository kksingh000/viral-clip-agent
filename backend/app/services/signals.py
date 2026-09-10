"""Deterministic linguistic signals.

These are the cheap filter in front of the model (see docs/architecture.md,
"Cost control"): they decide which parts of a long transcript are worth
spending tokens on, they give the fallback scorer something real to work with,
and they are logged next to the model's own scores so the two can be compared.

Lexicons cover English plus romanised Hindi/Hinglish, which is the product's
stated language range. They are intentionally small and explainable -- this is
a prior, not a classifier.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, fields
from typing import Any, Iterable, Sequence

# --------------------------------------------------------------------- lexicons
HOOK_OPENERS = (
    "most people", "nobody", "no one", "here is", "here's", "the biggest",
    "the truth about", "what they don't", "what nobody", "stop doing",
    "i was wrong", "this changed", "the reason", "the problem with",
    "you have been", "you've been", "never do", "everyone thinks",
    "let me tell you", "listen", "look", "the secret",
    # Hinglish
    "sabse bada", "kisi ne nahi", "log sochte", "sach ye hai", "asli baat",
    "aapko kisi ne", "yahi wajah", "sunno", "dekho",
)

CURIOSITY_MARKERS = (
    "but", "however", "actually", "turns out", "the catch", "here is why",
    "here's why", "the twist", "what if", "imagine", "wait", "surprisingly",
    "the thing is", "plot twist",
    "lekin", "magar", "asal mein", "socho", "pata hai",
)

PAYOFF_MARKERS = (
    "because", "that is why", "that's why", "the reason is", "so the answer",
    "which means", "in short", "bottom line", "the lesson", "so what you do",
    "the result", "and that is how", "and that's how",
    "isliye", "matlab", "iska matlab", "natija", "seekh",
)

EMOTION_WORDS = (
    "love", "hate", "terrified", "shocked", "furious", "heartbreaking",
    "incredible", "insane", "brutal", "devastating", "amazing", "painful",
    "embarrassing", "proud", "grateful", "angry", "scared", "excited",
    "unbelievable", "ridiculous", "crazy", "wild",
    "kamaal", "bakwas", "gussa", "darr", "khushi", "pagal",
)

SUPERLATIVES = (
    "biggest", "best", "worst", "never", "always", "only", "first", "fastest",
    "hardest", "easiest", "most important", "number one", "everyone", "no one",
    "sabse", "hamesha", "kabhi nahi", "sirf",
)

ADVICE_MARKERS = (
    "you should", "you need to", "do this", "try this", "step one", "first,",
    "the trick is", "here is how", "here's how", "make sure", "avoid",
    "aapko", "karna chahiye", "tarika", "kaise",
)

CONTROVERSY_MARKERS = (
    "unpopular opinion", "controversial", "disagree", "wrong about",
    "people will hate", "hot take", "nonsense", "myth", "lie",
    "galat", "jhooth",
)

STORY_MARKERS = (
    "when i was", "one day", "a few years ago", "last year", "so i",
    "and then", "at that moment", "i remember",
    "ek baar", "jab main", "phir",
)

#: Words that point outside the clip. A moment starting with these usually
#: needs context the viewer will not have -- either a pronoun with no
#: antecedent inside the clip, or a conjunction continuing a sentence the
#: viewer never heard ("And that is why it works").
DANGLING_REFERENCES = (
    "he", "she", "it", "they", "them", "this", "that", "these", "those",
    "there", "again", "also", "another", "such", "so",
    "and", "but", "because", "which", "then", "anyway", "plus", "however",
    "therefore", "although",
    "wo", "ye", "unhone", "iska", "uska", "lekin", "isliye", "phir",
)

#: Single-token fillers, matched against the tokenised text.
FILLER_WORDS = (
    "um", "uh", "uhh", "erm", "hmm", "like", "basically", "literally",
    "honestly", "anyway", "matlab", "yaar", "toh",
)

#: Multi-token fillers. These are matched as substrings -- keeping them in the
#: single-token tuple meant they could never match anything.
FILLER_PHRASES = (
    "you know", "i mean", "sort of", "kind of", "or something", "and stuff",
    "at the end of the day", "to be honest", "if that makes sense",
    "wanted to say about", "move on to the next", "basically all",
)

#: Channel boilerplate. It is fluent, confident speech, so the positive
#: lexicons happily score it -- but it is worthless as a standalone short.
BOILERPLATE_MARKERS = (
    "welcome back", "welcome to the channel", "subscribe", "hit the bell",
    "smash that like", "like and subscribe", "before we start",
    "before we begin", "link in the description", "link in the bio",
    "check out the description", "see you next time", "see you in the next",
    "thanks for watching", "that is all for today", "that's all for today",
    "sponsored by", "today's video", "in this video",
    "channel ko subscribe", "video pasand aaye", "agle video mein",
)

_WORD_RE = re.compile(r"[\w']+", re.UNICODE)
_SENTENCE_END = re.compile(r"[.!?।]")


# ---------------------------------------------------------------------- scoring
@dataclass(slots=True)
class TextSignals:
    """Explainable, model-free measurements of a piece of transcript."""

    word_count: int = 0
    char_count: int = 0
    sentence_count: int = 0
    question_count: int = 0
    number_count: int = 0
    hook_opener: bool = False
    curiosity_hits: int = 0
    payoff_hits: int = 0
    emotion_hits: int = 0
    superlative_hits: int = 0
    advice_hits: int = 0
    controversy_hits: int = 0
    story_hits: int = 0
    boilerplate_hits: int = 0
    filler_ratio: float = 0.0
    second_person_ratio: float = 0.0
    first_person_ratio: float = 0.0
    starts_with_dangling_reference: bool = False
    ends_mid_sentence: bool = True
    #: 0..1 prior on how promising this text is as a standalone short.
    prefilter_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        # ``slots=True`` dataclasses have no ``__dict__``; enumerate fields.
        return {
            field.name: (
                round(value, 4)
                if isinstance(value := getattr(self, field.name), float)
                else value
            )
            for field in fields(self)
        }


def _count_phrases(text: str, phrases: Iterable[str]) -> int:
    return sum(1 for phrase in phrases if phrase in text)


def analyze_text(text: str) -> TextSignals:
    """Measure one span of transcript."""
    stripped = text.strip()
    lowered = stripped.lower()
    words = _WORD_RE.findall(lowered)
    count = len(words)
    if count == 0:
        return TextSignals()

    first_words = " ".join(words[:6])
    # Single-token fillers plus multi-token filler phrases, each phrase
    # counted as the number of tokens it consumes.
    filler = sum(1 for w in words if w in FILLER_WORDS)
    for phrase in FILLER_PHRASES:
        occurrences = lowered.count(phrase)
        if occurrences:
            filler += occurrences * len(phrase.split())
    second_person = sum(1 for w in words if w in ("you", "your", "yours", "aap", "tum"))
    first_person = sum(1 for w in words if w in ("i", "me", "my", "mine", "we", "our", "main"))

    signals = TextSignals(
        word_count=count,
        char_count=len(stripped),
        sentence_count=max(1, len(_SENTENCE_END.findall(stripped))),
        question_count=stripped.count("?"),
        number_count=sum(1 for w in words if any(ch.isdigit() for ch in w)),
        hook_opener=any(lowered.startswith(h) or h in first_words for h in HOOK_OPENERS),
        curiosity_hits=_count_phrases(lowered, CURIOSITY_MARKERS),
        payoff_hits=_count_phrases(lowered, PAYOFF_MARKERS),
        emotion_hits=_count_phrases(lowered, EMOTION_WORDS),
        superlative_hits=_count_phrases(lowered, SUPERLATIVES),
        advice_hits=_count_phrases(lowered, ADVICE_MARKERS),
        controversy_hits=_count_phrases(lowered, CONTROVERSY_MARKERS),
        story_hits=_count_phrases(lowered, STORY_MARKERS),
        boilerplate_hits=_count_phrases(lowered, BOILERPLATE_MARKERS),
        filler_ratio=filler / count,
        second_person_ratio=second_person / count,
        first_person_ratio=first_person / count,
        starts_with_dangling_reference=bool(words) and words[0] in DANGLING_REFERENCES,
        ends_mid_sentence=not bool(_SENTENCE_END.search(stripped[-2:] or "")),
    )
    signals.prefilter_score = _prefilter(signals)
    return signals


#: Every span starts from here so that ranking stays informative: a span with
#: no positive markers should still be distinguishable from one that is
#: actively bad (filler, boilerplate, dangling opening).
PREFILTER_BASELINE = 0.28


def _prefilter(s: TextSignals) -> float:
    """Combine signals into a 0..1 prior.

    Deliberately shallow: it only has to rank spans well enough to choose which
    ones the model looks at, and to behave sanely when no model is available.
    Scores are centred on :data:`PREFILTER_BASELINE` rather than on zero -- an
    earlier version floored most spans at 0.0, which made every candidate tie
    and destroyed the ranking the fallback path depends on.
    """
    score = PREFILTER_BASELINE
    score += 0.18 if s.hook_opener else 0.0
    score += min(0.14, 0.05 * s.curiosity_hits)
    score += min(0.14, 0.05 * s.payoff_hits)
    score += min(0.12, 0.04 * s.emotion_hits)
    score += min(0.10, 0.035 * s.superlative_hits)
    score += min(0.10, 0.035 * s.advice_hits)
    score += min(0.08, 0.04 * s.controversy_hits)
    score += min(0.06, 0.03 * s.story_hits)
    score += min(0.06, 0.03 * s.question_count)
    score += min(0.05, 0.02 * s.number_count)
    score += min(0.08, s.second_person_ratio * 0.6)

    # Penalties
    score -= min(0.22, s.filler_ratio * 1.4)
    # Boilerplate is the strongest negative signal there is: a call to
    # subscribe is never the moment.
    score -= min(0.45, 0.22 * s.boilerplate_hits)
    if s.starts_with_dangling_reference:
        score -= 0.14
    if s.word_count < 12:
        score -= 0.10
    return max(0.0, min(1.0, score))


# --------------------------------------------------------- boundary heuristics
def needs_more_context(text: str) -> bool:
    """Would a cold viewer be lost at this opening line?"""
    signals = analyze_text(text)
    return signals.starts_with_dangling_reference or signals.word_count < 8


def is_sentence_end(text: str) -> bool:
    return bool(_SENTENCE_END.search(text.strip()[-2:] or ""))


def keyword_set(text: str, *, minimum_length: int = 4, limit: int = 40) -> set[str]:
    """Content words, used for cheap topic matching and duplicate detection."""
    stop = {
        "this", "that", "with", "from", "your", "have", "they", "them", "what",
        "when", "will", "just", "like", "been", "were", "into", "than", "then",
        "there", "their", "about", "would", "could", "should", "which", "these",
        "those", "because", "here", "very", "much", "more", "most", "some",
    }
    words = _WORD_RE.findall(text.lower())
    out = {w for w in words if len(w) >= minimum_length and w not in stop}
    return set(sorted(out)[:limit])


def jaccard(a: Sequence[str] | set[str], b: Sequence[str] | set[str]) -> float:
    set_a, set_b = set(a), set(b)
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


@dataclass(slots=True)
class SegmentSignals:
    """Text signals plus timing context for one transcript segment."""

    text: TextSignals
    words_per_second: float = 0.0
    leading_pause: float = 0.0
    trailing_pause: float = 0.0
    scene_aligned_start: bool = False
    scene_aligned_end: bool = False
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.text.to_dict(),
            "words_per_second": round(self.words_per_second, 3),
            "leading_pause": round(self.leading_pause, 3),
            "trailing_pause": round(self.trailing_pause, 3),
            "scene_aligned_start": self.scene_aligned_start,
            "scene_aligned_end": self.scene_aligned_end,
            **self.extras,
        }
