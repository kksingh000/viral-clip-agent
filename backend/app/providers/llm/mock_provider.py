"""No-network LLM provider.

This provider never fabricates model output. It flags itself with
``is_mock = True``; agents see that and run their deterministic fallback
implementation instead, so the whole pipeline works end to end with no API key
(useful for CI, local development and offline demos) while remaining honest
about the fact that no model was consulted.

If an agent calls it anyway -- a bug -- it raises rather than inventing text.
"""

from __future__ import annotations

from typing import Any, Sequence

from app.core.errors import LLMResponseError
from app.providers.base import LLMMessage, LLMProvider, LLMResponse


class MockLLMProvider(LLMProvider):
    name = "mock"
    is_mock = True

    def __init__(self, default_model: str = "mock-deterministic") -> None:
        self._default_model = default_model
        self.calls: list[dict[str, Any]] = []

    def complete(
        self,
        *,
        system: str,
        messages: Sequence[LLMMessage],
        max_tokens: int | None = None,
        temperature: float | None = None,
        model: str | None = None,
    ) -> LLMResponse:
        self.calls.append({"system": system, "messages": list(messages)})
        raise LLMResponseError(
            "MockLLMProvider cannot generate text. The calling agent must use "
            "its deterministic fallback when provider.is_mock is True."
        )

    def complete_json(
        self,
        *,
        system: str,
        messages: Sequence[LLMMessage],
        schema: dict[str, Any],
        schema_name: str = "response",
        max_tokens: int | None = None,
        temperature: float | None = None,
        model: str | None = None,
    ) -> LLMResponse:
        self.calls.append(
            {"system": system, "messages": list(messages), "schema": schema_name}
        )
        raise LLMResponseError(
            "MockLLMProvider cannot generate structured output. The calling "
            "agent must use its deterministic fallback when provider.is_mock is True."
        )


class ScriptedLLMProvider(LLMProvider):
    """Returns pre-canned structured responses. Used by tests to exercise the
    real LLM code path (validation, repair, retry) without a network call."""

    name = "scripted"
    is_mock = False

    def __init__(self, responses: Sequence[dict[str, Any] | Exception]) -> None:
        self._responses = list(responses)
        self._index = 0
        self.calls: list[dict[str, Any]] = []

    def _next(self) -> dict[str, Any]:
        if self._index >= len(self._responses):
            raise LLMResponseError("ScriptedLLMProvider exhausted.")
        item = self._responses[self._index]
        self._index += 1
        if isinstance(item, Exception):
            raise item
        return item

    def complete(
        self,
        *,
        system: str,
        messages: Sequence[LLMMessage],
        max_tokens: int | None = None,
        temperature: float | None = None,
        model: str | None = None,
    ) -> LLMResponse:
        import json

        self.calls.append({"system": system, "messages": list(messages)})
        payload = self._next()
        return LLMResponse(text=json.dumps(payload), model="scripted")

    def complete_json(
        self,
        *,
        system: str,
        messages: Sequence[LLMMessage],
        schema: dict[str, Any],
        schema_name: str = "response",
        max_tokens: int | None = None,
        temperature: float | None = None,
        model: str | None = None,
    ) -> LLMResponse:
        import json

        self.calls.append(
            {"system": system, "messages": list(messages), "schema": schema_name}
        )
        payload = self._next()
        return LLMResponse(text=json.dumps(payload), model="scripted", parsed=payload)
