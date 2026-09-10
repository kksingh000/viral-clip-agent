"""VideoQualityAgent -- combines measurement with judgement.

The technical half is :mod:`app.services.quality` (deterministic, no tokens).
The model half answers only what measurement cannot: does the clip make sense
on its own, does the transcript look mis-recognised, is the framing cutting
someone's face off, does it end abruptly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from pydantic import Field

from app.agents.base import (
    AgentContext,
    AgentOutput,
    AgentResult,
    BaseAgent,
    compact_json,
)
from app.core.logging import get_logger
from app.models.enums import QualitySeverity, QualityStatus
from app.services.quality import QualityIssue, TechnicalReport, technical_score

logger = get_logger(__name__)


class SemanticIssue(AgentOutput):
    check: str = Field(max_length=64)
    severity: str = Field(description="INFO, MINOR, MAJOR or CRITICAL")
    message: str = Field(max_length=400)

    def to_quality_issue(self) -> QualityIssue:
        try:
            severity = QualitySeverity(self.severity.upper())
        except ValueError:
            severity = QualitySeverity.MINOR
        return QualityIssue(
            check=f"semantic.{self.check}", severity=severity, message=self.message
        )


class QualityReview(AgentOutput):
    """Judgements the measurements cannot make."""

    makes_sense_standalone: bool
    transcription_looks_correct: bool
    ends_abruptly: bool
    opening_is_strong: bool
    issues: list[SemanticIssue] = Field(default_factory=list, max_length=10)
    recommendations: list[str] = Field(default_factory=list, max_length=6)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


@dataclass(slots=True)
class QualityPayload:
    transcript: str
    duration: float
    technical: TechnicalReport
    caption_text: str = ""
    hook_text: str = ""
    crop_notes: list[str] = field(default_factory=list)
    face_driven_crop: bool = False


@dataclass(slots=True)
class QualityVerdict:
    status: QualityStatus
    score: float
    threshold: float
    issues: list[QualityIssue]
    recommendations: list[str]
    technical: TechnicalReport
    used_fallback: bool = False

    @property
    def passed(self) -> bool:
        return self.status is not QualityStatus.FAIL

    @property
    def auto_fixes(self) -> list[str]:
        """Distinct fix actions, most severe first."""
        ordered = sorted(
            (i for i in self.issues if i.auto_fixable and i.fix_action),
            key=lambda i: i.weight,
            reverse=True,
        )
        seen: list[str] = []
        for issue in ordered:
            if issue.fix_action and issue.fix_action not in seen:
                seen.append(issue.fix_action)
        return seen

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "score": round(self.score, 1),
            "issues": [i.to_dict() for i in self.issues],
            "recommendations": self.recommendations,
            "technical": self.technical.to_dict(),
        }


SYSTEM = """\
# Role

You are VideoQualityAgent. A short-form clip has already been measured by
deterministic checks (resolution, loudness, black frames, caption timing). You
review only what measurement cannot decide.

# Objective

Decide whether this clip is fit to publish, judged as a viewer who has never
seen the source video would experience it.

# Input

- `transcript`: what is said in the clip
- `caption_text`: what the burned-in captions say
- `hook_text`: the on-screen opening overlay, if any
- `technical`: the measurements already taken
- `crop_notes`: what the framing engine did, and whether faces drove it

# What to judge

- `makes_sense_standalone`: could a stranger follow this with no other context?
- `transcription_looks_correct`: do the captions read like plausible speech, or
  are there obvious mis-recognitions (nonsense words, wrong homophones,
  sentences that do not parse)?
- `ends_abruptly`: does it stop mid-thought rather than on a payoff?
- `opening_is_strong`: do the first few words earn the next five seconds?

# Constraints

- Do not restate the technical measurements; they are already recorded.
- Severity: CRITICAL only for something that makes the clip unpublishable;
  MAJOR for something a viewer would notice immediately; MINOR for polish.
- Recommendations must be actionable and specific ("extend the start by ~2s to
  include the setup sentence"), not generic advice.
- Judge only from the text supplied. If you cannot tell, say so via a lower
  `confidence` rather than guessing.

# Failure behaviour

If the transcript is empty or unintelligible, set every boolean
conservatively (`makes_sense_standalone` false, `transcription_looks_correct`
false), add one CRITICAL issue explaining that the content could not be
assessed, and set `confidence` below 0.2.
"""


class VideoQualityAgent(BaseAgent[QualityPayload, QualityReview]):
    name = "quality_agent"
    output_model = QualityReview
    temperature = 0.1
    prefers_cheap_model = True

    def build_system_prompt(self, payload: QualityPayload) -> str:
        return SYSTEM

    def build_user_content(self, payload: QualityPayload) -> str:
        return compact_json(
            {
                "transcript": payload.transcript,
                "caption_text": payload.caption_text,
                "hook_text": payload.hook_text,
                "duration_seconds": round(payload.duration, 2),
                "technical": payload.technical.to_dict(),
                "crop_notes": payload.crop_notes,
                "face_driven_crop": payload.face_driven_crop,
            }
        )

    def fallback(self, payload: QualityPayload) -> QualityReview:
        """Structural judgement without a model.

        Uses the same signal machinery the selector uses, so the answers are
        consistent with how the clip was chosen in the first place.
        """
        from app.services.signals import analyze_text, is_sentence_end

        text = (payload.transcript or "").strip()
        signals = analyze_text(text)
        issues: list[SemanticIssue] = []

        standalone = bool(text) and not signals.starts_with_dangling_reference
        abrupt = bool(text) and not is_sentence_end(text)
        strong_open = signals.hook_opener or signals.question_count > 0

        if not text:
            issues.append(
                SemanticIssue(
                    check="no_transcript",
                    severity="CRITICAL",
                    message="No transcript available, so the clip could not be assessed.",
                )
            )
        if not standalone and text:
            issues.append(
                SemanticIssue(
                    check="needs_context",
                    severity="MAJOR",
                    message="The clip opens with a reference to something it never states.",
                )
            )
        if abrupt:
            issues.append(
                SemanticIssue(
                    check="abrupt_ending",
                    severity="MAJOR",
                    message="The clip ends mid-sentence.",
                )
            )

        return QualityReview(
            makes_sense_standalone=standalone,
            transcription_looks_correct=bool(text),
            ends_abruptly=abrupt,
            opening_is_strong=strong_open,
            issues=issues,
            recommendations=(
                ["Extend the start to include the preceding sentence."]
                if not standalone and text
                else []
            ),
            confidence=0.25,
        )

    # ------------------------------------------------------------- combining
    def evaluate(
        self,
        payload: QualityPayload,
        *,
        threshold: float,
        context: AgentContext | None = None,
    ) -> QualityVerdict:
        """Run both halves and produce the final verdict."""
        result: AgentResult[QualityReview] = self.run(payload, context=context)
        review = result.output

        issues = list(payload.technical.issues)
        issues.extend(issue.to_quality_issue() for issue in review.issues)

        if not review.makes_sense_standalone:
            issues.append(
                QualityIssue(
                    check="semantic.standalone_context",
                    severity=QualitySeverity.MAJOR,
                    message="A viewer without the source video would not follow this clip.",
                    auto_fixable=True,
                    fix_action="extend_context",
                )
            )
        if review.ends_abruptly:
            issues.append(
                QualityIssue(
                    check="semantic.abrupt_ending",
                    severity=QualitySeverity.MAJOR,
                    message="The clip ends before the thought is finished.",
                    auto_fixable=True,
                    fix_action="extend_end",
                )
            )
        if not review.transcription_looks_correct and payload.technical.caption_cue_count:
            issues.append(
                QualityIssue(
                    check="semantic.transcription",
                    severity=QualitySeverity.MAJOR,
                    message="The captions read as mis-recognised speech.",
                    auto_fixable=True,
                    fix_action="retranscribe",
                )
            )

        score = technical_score(issues)
        if any(i.severity is QualitySeverity.CRITICAL for i in issues):
            status = QualityStatus.FAIL
        elif score < threshold:
            status = QualityStatus.FAIL
        elif issues:
            status = QualityStatus.WARN
        else:
            status = QualityStatus.PASS

        verdict = QualityVerdict(
            status=status,
            score=score,
            threshold=threshold,
            issues=issues,
            recommendations=list(review.recommendations),
            technical=payload.technical,
            used_fallback=result.used_fallback,
        )
        logger.info(
            "quality verdict",
            extra={
                "status": status.value,
                "score": round(score, 1),
                "issues": len(issues),
                "fallback": result.used_fallback,
            },
        )
        return verdict


def summarise_issues(issues: Sequence[QualityIssue]) -> str:
    if not issues:
        return "no issues"
    return "; ".join(f"{i.check}({i.severity.value})" for i in issues[:6])
