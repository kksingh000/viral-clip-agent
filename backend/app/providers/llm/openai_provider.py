"""OpenAI-compatible implementation of :class:`LLMProvider`.

Also serves any OpenAI-compatible endpoint (vLLM, Ollama, Together, ...) via
``OPENAI_BASE_URL``.
"""

from __future__ import annotations

import json
from typing import Any, Sequence

from app.core.config import settings
from app.core.errors import LLMResponseError, ProviderError
from app.core.logging import get_logger
from app.providers.base import LLMMessage, LLMProvider, LLMResponse, LLMUsage

logger = get_logger(__name__)

_PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o": (2.5, 10.0),
    "gpt-4o-mini": (0.15, 0.6),
}


def _strict(schema: dict[str, Any]) -> dict[str, Any]:
    """OpenAI strict mode requires ``additionalProperties: false`` everywhere
    and every property listed in ``required``."""
    if not isinstance(schema, dict):
        return schema
    out = dict(schema)
    if out.get("type") == "object":
        out.setdefault("additionalProperties", False)
        props = out.get("properties") or {}
        out["properties"] = {k: _strict(v) for k, v in props.items()}
        out["required"] = list(out["properties"].keys())
    elif out.get("type") == "array" and "items" in out:
        out["items"] = _strict(out["items"])
    return out


class OpenAILLMProvider(LLMProvider):
    name = "openai"

    def __init__(self, api_key: str | None = None, default_model: str | None = None) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise ProviderError(
                "The 'openai' package is required for LLM_PROVIDER=openai."
            ) from exc
        key = api_key or settings.openai_api_key
        if not key:
            raise ProviderError("OPENAI_API_KEY is not configured.")
        self._client = OpenAI(
            api_key=key,
            base_url=settings.openai_base_url or None,
            timeout=settings.llm_timeout_seconds,
        )
        self._default_model = default_model or settings.llm_model

    def _messages(self, system: str, messages: Sequence[LLMMessage]) -> list[dict[str, str]]:
        return [{"role": "system", "content": system}] + [
            {"role": m.role, "content": m.content} for m in messages
        ]

    @staticmethod
    def _usage(response: Any) -> LLMUsage:
        usage = getattr(response, "usage", None)
        return LLMUsage(
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
        )

    def complete(
        self,
        *,
        system: str,
        messages: Sequence[LLMMessage],
        max_tokens: int | None = None,
        temperature: float | None = None,
        model: str | None = None,
    ) -> LLMResponse:
        chosen = model or self._default_model
        try:
            response = self._client.chat.completions.create(
                model=chosen,
                messages=self._messages(system, messages),
                max_tokens=max_tokens or settings.llm_max_output_tokens,
                temperature=(
                    settings.llm_temperature if temperature is None else temperature
                ),
            )
        except Exception as exc:  # noqa: BLE001
            raise ProviderError(f"OpenAI request failed: {exc}") from exc
        return LLMResponse(
            text=response.choices[0].message.content or "",
            model=chosen,
            usage=self._usage(response),
            stop_reason=response.choices[0].finish_reason,
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
        chosen = model or self._default_model
        try:
            response = self._client.chat.completions.create(
                model=chosen,
                messages=self._messages(system, messages),
                max_tokens=max_tokens or settings.llm_max_output_tokens,
                temperature=(
                    settings.llm_temperature if temperature is None else temperature
                ),
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema_name,
                        "strict": True,
                        "schema": _strict(schema),
                    },
                },
            )
        except Exception as exc:  # noqa: BLE001
            raise ProviderError(f"OpenAI request failed: {exc}") from exc

        text = response.choices[0].message.content or ""
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise LLMResponseError(f"Model returned invalid JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise LLMResponseError("Model returned JSON that is not an object.")
        return LLMResponse(
            text=text,
            model=chosen,
            usage=self._usage(response),
            stop_reason=response.choices[0].finish_reason,
            parsed=parsed,
        )

    def estimate_cost_usd(self, model: str, usage: LLMUsage) -> float:
        rates = _PRICING.get(model)
        if not rates:
            return 0.0
        return (
            usage.input_tokens * rates[0] + usage.output_tokens * rates[1]
        ) / 1_000_000
