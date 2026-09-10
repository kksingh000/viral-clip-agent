"""ContentSafetyAgent.

Returns SAFE / REVIEW / BLOCK. The default when the agent cannot actually
assess the content is **REVIEW, never SAFE**: an unassessed clip must not slip
into an automated publishing workflow just because no model was configured.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from pydantic import Field

from app.agents.base import AgentOutput, BaseAgent, compact_json
from app.core.logging import get_logger
from app.models.enums import SafetyVerdict

logger = get_logger(__name__)

CATEGORIES = (
    "hate",
    "harassment",
    "sexual",
    "violence",
    "dangerous_instructions",
    "profanity",
    "misleading_claims",
    "copyright_concern",
    "sensitive_topic",
)

#: Scores at or above these thresholds drive the verdict.
BLOCK_THRESHOLD = 7.0
REVIEW_THRESHOLD = 4.0

#: Deterministic lexicon used only by the fallback path. Deliberately narrow:
#: it catches unambiguous cases and defers everything else to a human.
_STRONG_PROFANITY = re.compile(
    r"\b(f[u\*]ck\w*|sh[i\*]t\w*|c[u\*]nt|b[i\*]tch\w*|a[s\*]{2}hole)\b", re.I
)
_VIOLENCE = re.compile(
    r"\b(kill (him|her|them|yourself)|shoot (him|her|them)|stab|behead|"
    r"bomb making|how to make a bomb)\b",
    re.I,
)
_SELF_HARM = re.compile(r"\b(kill yourself|end your life|suicide method)\b", re.I)
_DANGEROUS = re.compile(
    r"\b(how to (make|build) (a )?(bomb|explosive|gun)|synthesi[sz]e "
    r"(meth|fentanyl)|bypass (the )?safety)\b",
    re.I,
)
_MEDICAL_FINANCIAL_CLAIM = re.compile(
    r"\b(cures? (cancer|diabetes|aids)|guaranteed returns?|risk[- ]free "
    r"(profit|money)|100% (safe|effective|guaranteed))\b",
    re.I,
)


class SafetyOutput(AgentOutput):
    verdict: str = Field(description="SAFE, REVIEW or BLOCK")
    category_scores: dict[str, float] = Field(
        default_factory=dict,
        description="One 0-10 score per category; 0 means clearly absent.",
    )
    flagged_categories: list[str] = Field(default_factory=list, max_length=12)
    rationale: str = Field(default="", max_length=1000)
    evidence: list[str] = Field(
        default_factory=list,
        max_length=8,
        description="Short quotations from the clip supporting each flag.",
    )


@dataclass(slots=True)
class SafetyPayload:
    transcript: str
    duration: float
    title: str = ""
    hook_text: str = ""
    source_title: str = ""
    language: str | None = None
    authorization_status: str = "UNKNOWN"
    visual_notes: list[str] = field(default_factory=list)


SYSTEM = """\
# Role

You are ContentSafetyAgent. You review a short-form clip before it is offered
for publication.

# Objective

Identify content that would breach platform policy, mislead viewers, or cause
harm -- and separate "must not publish" from "a person should look at this".

# Input

The clip transcript, its title and hook, the source video title, and the
rights status of the source material.

# Categories

Score each 0-10, where 0 means clearly absent and 10 means unambiguous and
severe:

- `hate`: attacks or demeans a protected group.
- `harassment`: targets a specific person abusively.
- `sexual`: explicit sexual content.
- `violence`: graphic violence or glorification of it.
- `dangerous_instructions`: actionable instructions that could cause serious harm.
- `profanity`: strong language (note: profanity alone is rarely a block).
- `misleading_claims`: medical, financial or factual claims stated as certain
  that are false or unsupported.
- `copyright_concern`: the clip appears to contain third-party music, film or
  footage beyond the speaker's own material.
- `sensitive_topic`: self-harm, addiction, grief, violence against children,
  or similar -- handled responsibly or not.

# Verdict

- `BLOCK`: any category is severe enough that publishing would be harmful or a
  clear policy breach.
- `REVIEW`: something a human should see before it goes out, including anything
  you are genuinely unsure about.
- `SAFE`: nothing of concern.

Prefer REVIEW over SAFE when uncertain. Prefer REVIEW over BLOCK unless the
harm is clear -- blocking legitimate content has a real cost too.

# Constraints

- Judge the clip, not the topic. Discussing addiction is not endorsing it;
  reporting on violence is not glorifying it.
- Quote evidence. Every flagged category must have a supporting quotation.
- Strong language in casual speech is `profanity`, not `harassment`.
- Do not flag `copyright_concern` merely because the source is a third party;
  flag it when the clip's own content appears to embed someone else's work.

# Failure behaviour

If the transcript is empty or unintelligible, return `REVIEW` with a rationale
saying the content could not be assessed. Never return `SAFE` for content you
could not read.
"""


class ContentSafetyAgent(BaseAgent[SafetyPayload, SafetyOutput]):
    name = "safety_agent"
    output_model = SafetyOutput
    temperature = 0.0
    prefers_cheap_model = True

    def build_system_prompt(self, payload: SafetyPayload) -> str:
        return SYSTEM

    def build_user_content(self, payload: SafetyPayload) -> str:
        return compact_json(
            {
                "transcript": payload.transcript,
                "title": payload.title,
                "hook_text": payload.hook_text,
                "source_title": payload.source_title,
                "duration_seconds": round(payload.duration, 2),
                "language": payload.language,
                "authorization_status": payload.authorization_status,
                "visual_notes": payload.visual_notes,
            }
        )

    def postprocess(self, output: SafetyOutput, payload: SafetyPayload) -> SafetyOutput:
        """Normalise the verdict and keep it consistent with the scores.

        A model that scores a category at 9 but returns SAFE is corrected here
        -- the arithmetic decision belongs in code.
        """
        scores = {
            key: max(0.0, min(10.0, float(value)))
            for key, value in (output.category_scores or {}).items()
            if key in CATEGORIES
        }
        for category in CATEGORIES:
            scores.setdefault(category, 0.0)
        output.category_scores = scores

        flagged = [k for k, v in scores.items() if v >= REVIEW_THRESHOLD]
        output.flagged_categories = sorted(
            set(output.flagged_categories) | set(flagged)
        )

        highest = max(scores.values()) if scores else 0.0
        declared = (output.verdict or "").strip().upper()
        if highest >= BLOCK_THRESHOLD:
            resolved = SafetyVerdict.BLOCK
        elif highest >= REVIEW_THRESHOLD:
            resolved = SafetyVerdict.REVIEW
        elif declared in {v.value for v in SafetyVerdict}:
            resolved = SafetyVerdict(declared)
        else:
            resolved = SafetyVerdict.REVIEW

        if declared and declared != resolved.value:
            logger.info(
                "safety verdict adjusted to match category scores",
                extra={"declared": declared, "resolved": resolved.value},
            )
        output.verdict = resolved.value
        return output

    def fallback(self, payload: SafetyPayload) -> SafetyOutput:
        """Lexicon screen. Never returns SAFE.

        Without a model the clip has not really been assessed, so the honest
        outcome is REVIEW: the human approval step exists for exactly this.
        """
        text = payload.transcript or ""
        scores = {category: 0.0 for category in CATEGORIES}
        evidence: list[str] = []

        if match := _STRONG_PROFANITY.search(text):
            scores["profanity"] = 5.0
            evidence.append(match.group(0))
        if match := _VIOLENCE.search(text):
            scores["violence"] = 7.5
            evidence.append(match.group(0))
        if match := _SELF_HARM.search(text):
            scores["sensitive_topic"] = 8.5
            evidence.append(match.group(0))
        if match := _DANGEROUS.search(text):
            scores["dangerous_instructions"] = 8.5
            evidence.append(match.group(0))
        if match := _MEDICAL_FINANCIAL_CLAIM.search(text):
            scores["misleading_claims"] = 6.0
            evidence.append(match.group(0))
        if payload.authorization_status in ("UNKNOWN", "NOT_AUTHORIZED"):
            scores["copyright_concern"] = 6.0
            evidence.append(f"authorization_status={payload.authorization_status}")

        highest = max(scores.values())
        verdict = (
            SafetyVerdict.BLOCK if highest >= BLOCK_THRESHOLD else SafetyVerdict.REVIEW
        )
        rationale = (
            "No safety model is configured, so this clip has not been assessed "
            "for meaning. A lexicon screen was applied and the clip is routed "
            "to human review."
        )
        if highest >= BLOCK_THRESHOLD:
            rationale = (
                "A lexicon screen matched high-severity terms. No safety model "
                "is configured, so this is a conservative block pending review."
            )
        return SafetyOutput(
            verdict=verdict.value,
            category_scores=scores,
            flagged_categories=[k for k, v in scores.items() if v >= REVIEW_THRESHOLD],
            rationale=rationale,
            evidence=evidence[:8],
        )


def verdict_of(output: SafetyOutput) -> SafetyVerdict:
    try:
        return SafetyVerdict(output.verdict)
    except ValueError:  # pragma: no cover - postprocess normalises this
        return SafetyVerdict.REVIEW


def summarise(output: SafetyOutput) -> dict[str, Any]:
    return {
        "verdict": output.verdict,
        "flagged": output.flagged_categories,
        "max_score": max(output.category_scores.values(), default=0.0),
    }
