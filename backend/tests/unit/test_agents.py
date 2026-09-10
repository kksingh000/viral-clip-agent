"""Agent contract: schema validation, repair, retry and fallback.

These exercise the *real* LLM code path using a scripted provider, so the
validation/repair/retry machinery is covered without a network call.
"""

from __future__ import annotations

import pytest

from app.agents.base import AgentContext, dereference_schema
from app.agents.hook_agent import CopyPayload, HookAgent
from app.agents.quality_agent import QualityPayload, VideoQualityAgent
from app.agents.safety_agent import ContentSafetyAgent, SafetyPayload
from app.agents.trend_agent import TopicPayload, TrendAgent
from app.agents.viral_moment_agent import MomentSearchPayload, ViralMomentAgent
from app.core.errors import LLMResponseError
from app.models.enums import QualityStatus, SafetyVerdict
from app.providers.llm.mock_provider import MockLLMProvider, ScriptedLLMProvider
from app.services.quality import QualityIssue, TechnicalReport
from app.services.scoring import DIMENSION_NAMES
from app.services.segmentation import segment_transcript
from tests.fixtures.transcripts import sample_transcript


def moment_payload(**overrides):
    transcript = sample_transcript()
    defaults = dict(
        video_title="How compounding actually works",
        video_duration=transcript.duration,
        segments=segment_transcript(transcript),
        min_clip_seconds=15.0,
        max_clip_seconds=60.0,
        max_candidates=5,
        language="en",
    )
    defaults.update(overrides)
    return MomentSearchPayload(**defaults)


def valid_candidate(start=29.0, end=48.0):
    return {
        "start_time": start,
        "end_time": end,
        "hook": "The reason is simple.",
        "summary": "Explains why compounding fails in practice.",
        "reason": "Opens on a claim and resolves it.",
        "context_lead_in_seconds": 0.0,
        "dimensions": {name: 7.0 for name in DIMENSION_NAMES},
    }


class TestSchemaHandling:
    def test_refs_are_inlined(self):
        schema = dereference_schema(
            ViralMomentAgent(llm=MockLLMProvider()).output_model.model_json_schema()
        )
        rendered = repr(schema)
        assert "$defs" not in rendered
        assert "$ref" not in rendered

    def test_every_agent_produces_a_usable_schema(self):
        for agent_class in (
            ViralMomentAgent,
            HookAgent,
            VideoQualityAgent,
            ContentSafetyAgent,
            TrendAgent,
        ):
            schema = agent_class(llm=MockLLMProvider()).json_schema()
            assert schema.get("type") == "object"
            assert "properties" in schema


class TestFallbackBehaviour:
    def test_mock_provider_routes_to_the_deterministic_path(self):
        result = ViralMomentAgent(llm=MockLLMProvider()).run(moment_payload())
        assert result.used_fallback is True
        assert result.model == "deterministic"
        assert result.attempts == 0
        assert any("fallback" in w for w in result.warnings)

    def test_mock_provider_never_fabricates_output(self):
        """The mock must raise rather than invent text, so a bug that skips the
        fallback fails loudly instead of shipping fake model output."""
        provider = MockLLMProvider()
        with pytest.raises(LLMResponseError):
            provider.complete_json(system="s", messages=[], schema={})
        with pytest.raises(LLMResponseError):
            provider.complete(system="s", messages=[])

    def test_fallback_still_produces_usable_candidates(self):
        result = ViralMomentAgent(llm=MockLLMProvider()).run(moment_payload())
        assert result.output.candidates
        for candidate in result.output.candidates:
            assert candidate.end_time > candidate.start_time
            assert candidate.hook
            assert "deterministic" in candidate.reason.lower()

    def test_synthetic_transcripts_skip_the_model_entirely(self):
        agent = ViralMomentAgent(llm=ScriptedLLMProvider([]))
        result = agent.run_windowed(
            moment_payload(transcript_is_synthetic=True), context=AgentContext()
        )
        assert result.used_fallback is True
        assert any("structural only" in w for w in result.warnings)


class TestRetryAndRepair:
    def test_valid_response_is_accepted_first_try(self):
        provider = ScriptedLLMProvider([{"candidates": [valid_candidate()]}])
        result = ViralMomentAgent(llm=provider).run(moment_payload())
        assert result.used_fallback is False
        assert result.attempts == 1
        assert len(result.output.candidates) == 1

    def test_invalid_response_is_repaired_on_the_second_attempt(self):
        bad = {"candidates": [{**valid_candidate(), "end_time": -5}]}
        provider = ScriptedLLMProvider([bad, {"candidates": [valid_candidate()]}])
        agent = ViralMomentAgent(llm=provider)
        agent.retry_backoff = 0.0

        result = agent.run(moment_payload())
        assert result.used_fallback is False
        assert result.attempts == 2
        # The repair turn must show the model what was wrong.
        assert "Validation errors" in provider.calls[1]["messages"][-1].content

    def test_exhausted_retries_fall_back_rather_than_raising(self):
        provider = ScriptedLLMProvider([{"nope": 1}, {"nope": 2}, {"nope": 3}])
        agent = ViralMomentAgent(llm=provider)
        agent.retry_backoff = 0.0
        agent.max_attempts = 3

        result = agent.run(moment_payload())
        assert result.used_fallback is True
        assert result.attempts == 3
        assert result.output.candidates  # degraded, not empty

    def test_provider_errors_are_retried(self):
        provider = ScriptedLLMProvider(
            [LLMResponseError("transient"), {"candidates": [valid_candidate()]}]
        )
        agent = ViralMomentAgent(llm=provider)
        agent.retry_backoff = 0.0

        result = agent.run(moment_payload())
        assert result.used_fallback is False
        assert result.attempts == 2

    def test_usage_is_accumulated_on_the_context(self):
        context = AgentContext()
        provider = ScriptedLLMProvider([{"candidates": [valid_candidate()]}])
        ViralMomentAgent(llm=provider).run(moment_payload(), context=context)
        assert context.calls == 1


class TestHookAgent:
    def test_hooks_asserting_invented_numbers_are_dropped(self):
        """A hook may not introduce a statistic the clip never states."""
        provider = ScriptedLLMProvider(
            [
                {
                    "hooks": [
                        {"text": "This made me 47000 dollars", "confidence": 0.9},
                        {"text": "Nobody explains this part", "confidence": 0.8},
                    ],
                    "youtube_title": "Compounding, explained",
                    "instagram_caption": "",
                    "description": "",
                    "hashtags": ["investing", "viral", "fyp"],
                    "keywords": ["compounding"],
                }
            ]
        )
        result = HookAgent(llm=provider).run(
            CopyPayload(
                transcript="Compounding only works if you never interrupt it.",
                duration=20.0,
            )
        )
        texts = [h.text for h in result.output.hooks]
        assert "This made me 47000 dollars" not in texts
        assert "Nobody explains this part" in texts

    def test_generic_reach_hashtags_are_stripped(self):
        provider = ScriptedLLMProvider(
            [
                {
                    "hooks": [{"text": "A real hook", "confidence": 0.7}],
                    "youtube_title": "T",
                    "instagram_caption": "",
                    "description": "",
                    "hashtags": ["viral", "fyp", "compounding", "investing"],
                    "keywords": [],
                }
            ]
        )
        result = HookAgent(llm=provider).run(
            CopyPayload(transcript="Some words here.", duration=20.0)
        )
        assert "viral" not in result.output.hashtags
        assert "fyp" not in result.output.hashtags
        assert "compounding" in result.output.hashtags

    def test_fallback_copy_comes_from_the_transcript(self):
        text = "Most people misunderstand compounding. It is about survival."
        result = HookAgent(llm=MockLLMProvider()).run(
            CopyPayload(transcript=text, duration=20.0)
        )
        assert result.output.hooks
        for hook in result.output.hooks:
            # Every hook must be a fragment of what was actually said.
            assert hook.text.rstrip(".").rstrip("...") in text or hook.text in text

    def test_titles_are_truncated_to_the_platform_limit(self):
        provider = ScriptedLLMProvider(
            [
                {
                    "hooks": [{"text": "Hook", "confidence": 0.5}],
                    "youtube_title": "word " * 40,
                    "instagram_caption": "x " * 400,
                    "description": "",
                    "hashtags": [],
                    "keywords": [],
                }
            ]
        )
        result = HookAgent(llm=provider).run(
            CopyPayload(transcript="Some words.", duration=20.0)
        )
        assert len(result.output.youtube_title) <= 100
        assert len(result.output.instagram_caption) <= 300


class TestSafetyAgent:
    def _payload(self, transcript: str, rights: str = "USER_OWNED") -> SafetyPayload:
        return SafetyPayload(
            transcript=transcript, duration=20.0, authorization_status=rights
        )

    def test_fallback_never_returns_safe(self):
        """Without a model the clip has not been assessed, so SAFE would be a
        false assurance."""
        result = ContentSafetyAgent(llm=MockLLMProvider()).run(
            self._payload("A perfectly ordinary sentence about gardening.")
        )
        assert result.output.verdict in ("REVIEW", "BLOCK")
        assert result.output.verdict != SafetyVerdict.SAFE.value

    def test_fallback_blocks_on_unambiguous_terms(self):
        result = ContentSafetyAgent(llm=MockLLMProvider()).run(
            self._payload("Here is exactly how to make a bomb at home.")
        )
        assert result.output.verdict == SafetyVerdict.BLOCK.value
        assert "dangerous_instructions" in result.output.flagged_categories

    def test_unknown_rights_raise_a_copyright_concern(self):
        result = ContentSafetyAgent(llm=MockLLMProvider()).run(
            self._payload("Ordinary speech.", rights="UNKNOWN")
        )
        assert "copyright_concern" in result.output.flagged_categories

    def test_declared_verdict_is_corrected_to_match_the_scores(self):
        """A model that scores 9 for hate but says SAFE must not be believed;
        the arithmetic decision belongs in code."""
        provider = ScriptedLLMProvider(
            [
                {
                    "verdict": "SAFE",
                    "category_scores": {"hate": 9.0},
                    "flagged_categories": [],
                    "rationale": "",
                    "evidence": [],
                }
            ]
        )
        result = ContentSafetyAgent(llm=provider).run(self._payload("..."))
        assert result.output.verdict == SafetyVerdict.BLOCK.value
        assert "hate" in result.output.flagged_categories

    def test_clean_scores_allow_safe(self):
        provider = ScriptedLLMProvider(
            [
                {
                    "verdict": "SAFE",
                    "category_scores": {"hate": 0.0, "profanity": 1.0},
                    "flagged_categories": [],
                    "rationale": "Nothing of concern.",
                    "evidence": [],
                }
            ]
        )
        result = ContentSafetyAgent(llm=provider).run(self._payload("..."))
        assert result.output.verdict == SafetyVerdict.SAFE.value


class TestQualityAgent:
    def _report(self, issues=None) -> TechnicalReport:
        return TechnicalReport(
            duration=30.0,
            width=1080,
            height=1920,
            fps=30.0,
            has_audio=True,
            size_bytes=2_000_000,
            aspect_ratio=1080 / 1920,
            integrated_lufs=-14.0,
            true_peak_db=-1.5,
            silence_ratio=0.05,
            leading_silence=0.1,
            black_ratio=0.0,
            leading_black=0.0,
            caption_cue_count=8,
            caption_coverage=0.8,
            issues=issues or [],
        )

    def test_a_clean_clip_passes(self):
        verdict = VideoQualityAgent(llm=MockLLMProvider()).evaluate(
            QualityPayload(
                transcript="A complete thought that ends properly.",
                duration=30.0,
                technical=self._report(),
            ),
            threshold=75.0,
        )
        assert verdict.status is QualityStatus.PASS
        assert verdict.score == 100.0

    def test_a_critical_technical_issue_fails_regardless_of_score(self):
        from app.models.enums import QualitySeverity

        verdict = VideoQualityAgent(llm=MockLLMProvider()).evaluate(
            QualityPayload(
                transcript="A complete thought that ends properly.",
                duration=30.0,
                technical=self._report(
                    [
                        QualityIssue(
                            check="audio_missing",
                            severity=QualitySeverity.CRITICAL,
                            message="No audio track.",
                        )
                    ]
                ),
            ),
            threshold=10.0,
        )
        assert verdict.status is QualityStatus.FAIL

    def test_a_dangling_opening_is_flagged_and_is_auto_fixable(self):
        verdict = VideoQualityAgent(llm=MockLLMProvider()).evaluate(
            QualityPayload(
                transcript="And that is why it works so well.",
                duration=30.0,
                technical=self._report(),
            ),
            threshold=75.0,
        )
        checks = [issue.check for issue in verdict.issues]
        assert "semantic.standalone_context" in checks
        assert "extend_context" in verdict.auto_fixes

    def test_an_abrupt_ending_is_flagged(self):
        verdict = VideoQualityAgent(llm=MockLLMProvider()).evaluate(
            QualityPayload(
                transcript="Compounding only works if you never",
                duration=30.0,
                technical=self._report(),
            ),
            threshold=75.0,
        )
        assert "semantic.abrupt_ending" in [i.check for i in verdict.issues]


class TestTrendAgent:
    def test_fallback_clusters_repeated_phrases(self):
        payload = TopicPayload(
            videos=[
                {"title": "AI coding agents changed my workflow", "tags": ["ai"]},
                {"title": "I replaced my team with AI coding agents", "tags": ["ai"]},
                {"title": "Sourdough starter guide", "tags": ["baking"]},
            ],
            max_topics=5,
        )
        result = TrendAgent(llm=MockLLMProvider()).run(payload)
        assert result.output.topics
        top = result.output.topics[0]
        assert top.video_indexes == [0, 1]

    def test_topics_covering_the_same_videos_are_not_duplicated(self):
        """Regression: "Ai" was emitted alongside "Coding Agents" for the same
        pair of videos."""
        payload = TopicPayload(
            videos=[
                {"title": "AI coding agents changed my workflow", "tags": ["ai"]},
                {"title": "I replaced my team with AI coding agents", "tags": ["ai"]},
            ],
            max_topics=5,
        )
        result = TrendAgent(llm=MockLLMProvider()).run(payload)
        covered = [tuple(t.video_indexes) for t in result.output.topics]
        assert len(covered) == len(set(covered))

    def test_out_of_range_indexes_are_discarded(self):
        provider = ScriptedLLMProvider(
            [
                {
                    "topics": [
                        {
                            "label": "AI coding agents",
                            "kind": "topic",
                            "keywords": ["ai", "agents"],
                            "video_indexes": [0, 1, 99],
                            "rationale": "",
                        }
                    ]
                }
            ]
        )
        payload = TopicPayload(videos=[{"title": "a"}, {"title": "b"}], max_topics=5)
        result = TrendAgent(llm=provider).run(payload)
        assert result.output.topics[0].video_indexes == [0, 1]

    def test_unknown_kind_is_normalised(self):
        provider = ScriptedLLMProvider(
            [
                {
                    "topics": [
                        {
                            "label": "Something",
                            "kind": "wildly-invented-kind",
                            "keywords": [],
                            "video_indexes": [0],
                            "rationale": "",
                        }
                    ]
                }
            ]
        )
        result = TrendAgent(llm=provider).run(
            TopicPayload(videos=[{"title": "a"}], max_topics=5)
        )
        assert result.output.topics[0].kind == "other"


def test_every_agent_declares_the_required_prompt_sections():
    """Spec: each prompt must state role, objective, output, constraints and
    failure behaviour."""
    cases = [
        (ViralMomentAgent, moment_payload()),
        (HookAgent, CopyPayload(transcript="x", duration=10)),
        (ContentSafetyAgent, SafetyPayload(transcript="x", duration=10)),
        (TrendAgent, TopicPayload(videos=[])),
    ]
    for agent_class, payload in cases:
        prompt = agent_class(llm=MockLLMProvider()).build_system_prompt(payload)
        lowered = prompt.lower()
        for section in ("# role", "# objective", "# constraints", "failure behaviour"):
            assert section in lowered, f"{agent_class.__name__} prompt missing {section}"


def test_user_content_is_json_only():
    """Payloads must be data, never instructions."""
    import json

    content = ViralMomentAgent(llm=MockLLMProvider()).build_user_content(
        moment_payload()
    )
    parsed = json.loads(content)
    assert set(parsed) >= {"video", "constraints", "segments"}
