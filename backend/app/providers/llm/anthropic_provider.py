"""Anthropic implementation of :class:`LLMProvider`.

Structured output is obtained by forcing a single tool call whose input schema
is the response schema -- the model cannot then emit prose around the JSON.
"""

from __future__ import annotations

import json
from typing import Any, Sequence

from app.core.config import settings
from app.core.errors import LLMResponseError, ProviderError
from app.core.logging import get_logger
from app.providers.base import LLMMessage, LLMProvider, LLMResponse, LLMUsage

logger = get_logger(__name__)

# USD per million tokens. Unknown models fall back to zero so cost reporting
# degrades to "unmeasured" rather than to a wrong number.
_PRICING: dict[str, tuple[float, float]] = {
    "claude-opus-5": (15.0, 75.0),
    "claude-sonnet-5": (3.0, 15.0),
    "claude-haiku-4-5-20251001": (1.0, 5.0),
}


class AnthropicLLMProvider(LLMProvider):
    name = "anthropic"

    def __init__(self, api_key: str | None = None, default_model: str | None = None) -> None:
        try:
            from anthropic import Anthropic
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ProviderError(
                "The 'anthropic' package is required for LLM_PROVIDER=anthropic."
            ) from exc
        key = api_key or settings.anthropic_api_key
        if not key:
            raise ProviderError("ANTHROPIC_API_KEY is not configured.")
        self._client = Anthropic(api_key=key, timeout=settings.llm_timeout_seconds)
        self._default_model = default_model or settings.llm_model

    # ------------------------------------------------------------------ utils
    def _payload(
        self,
        system: str,
        messages: Sequence[LLMMessage],
        max_tokens: int | None,
        temperature: float | None,
        model: str | None,
    ) -> dict[str, Any]:
        return {
            "model": model or self._default_model,
            "system": system,
            "max_tokens": max_tokens or settings.llm_max_output_tokens,
            "temperature": (
                settings.llm_temperature if temperature is None else temperature
            ),
            "messages": [{"role": m.role, "content": m.content} for m in messages],
        }

    @staticmethod
    def _usage(response: Any) -> LLMUsage:
        usage = getattr(response, "usage", None)
        return LLMUsage(
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
        )

    # --------------------------------------------------------------- requests
    def complete(
        self,
        *,
        system: str,
        messages: Sequence[LLMMessage],
        max_tokens: int | None = None,
        temperature: float | None = None,
        model: str | None = None,
    ) -> LLMResponse:
        payload = self._payload(system, messages, max_tokens, temperature, model)
        try:
            response = self._client.messages.create(**payload)
        except Exception as exc:  # noqa: BLE001 - normalised to ProviderError
            raise ProviderError(f"Anthropic request failed: {exc}") from exc
        text = "".join(
            block.text for block in response.content if getattr(block, "type", "") == "text"
        )
        return LLMResponse(
            text=text,
            model=payload["model"],
            usage=self._usage(response),
            stop_reason=getattr(response, "stop_reason", None),
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
        payload = self._payload(system, messages, max_tokens, temperature, model)
        payload["tools"] = [
            {
                "name": schema_name,
                "description": f"Return the {schema_name} object. This is the only "
                "way to answer; never reply with prose.",
                "input_schema": schema,
            }
        ]
        payload["tool_choice"] = {"type": "tool", "name": schema_name}
        try:
            response = self._client.messages.create(**payload)
        except Exception as exc:  # noqa: BLE001
            raise ProviderError(f"Anthropic request failed: {exc}") from exc

        for block in response.content:
            if getattr(block, "type", "") == "tool_use" and block.name == schema_name:
                parsed = block.input
                if not isinstance(parsed, dict):
                    raise LLMResponseError("Tool input was not a JSON object.")
                return LLMResponse(
                    text=json.dumps(parsed),
                    model=payload["model"],
                    usage=self._usage(response),
                    stop_reason=getattr(response, "stop_reason", None),
                    parsed=parsed,
                )
        raise LLMResponseError(
            "Model did not return the required tool call.",
            details={"stop_reason": getattr(response, "stop_reason", None)},
        )

    def estimate_cost_usd(self, model: str, usage: LLMUsage) -> float:
        rates = _PRICING.get(model)
        if not rates:
            return 0.0
        return (
            usage.input_tokens * rates[0] + usage.output_tokens * rates[1]
        ) / 1_000_000
