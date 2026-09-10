"""Agent foundation.

Every agent follows the same contract:

* it declares a Pydantic **output model**, which is turned into the JSON schema
  the provider is constrained to;
* it builds a **system prompt** (role, objective, criteria, constraints,
  failure behaviour) and a **user payload** (data only, as JSON);
* it validates the response, **repairs** once by handing the validation errors
  back to the model, then **retries** with backoff;
* if the model cannot be used at all -- no provider configured, or every
  attempt failed -- it falls back to a **deterministic implementation** rather
  than inventing output or aborting the pipeline.

Token usage and estimated cost are accumulated on an :class:`AgentContext` so a
job can report exactly what it spent.
"""

from __future__ import annotations

import abc
import json
import time
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, ValidationError

from app.core.config import settings
from app.core.errors import LLMResponseError, ProviderError
from app.core.logging import get_logger
from app.providers.base import LLMMessage, LLMProvider, LLMUsage
from app.providers.registry import get_llm_provider

logger = get_logger(__name__)

class AgentOutput(BaseModel):
    """Base for every agent's structured response.

    Unknown keys are rejected. Pydantic ignores extras by default, which meant
    a completely malformed response could validate into an all-defaults object
    -- an empty ``candidates`` list reads as "the agent found nothing" when in
    fact the model returned garbage. Forbidding extras turns that into a
    validation error, which the repair-and-retry loop can act on.
    """

    model_config = ConfigDict(extra="forbid")


TOut = TypeVar("TOut", bound=BaseModel)
TIn = TypeVar("TIn")


@dataclass
class AgentContext:
    """Accumulates cost/usage across every agent call inside one job."""

    job_id: str | None = None
    usage: LLMUsage = field(default_factory=LLMUsage)
    calls: int = 0
    cost_usd: float = 0.0
    warnings: list[str] = field(default_factory=list)

    def record(self, provider: LLMProvider, model: str, usage: LLMUsage) -> None:
        self.usage = self.usage + usage
        self.calls += 1
        self.cost_usd += provider.estimate_cost_usd(model, usage)

    def warn(self, message: str) -> None:
        if message not in self.warnings:
            self.warnings.append(message)


@dataclass
class AgentResult(Generic[TOut]):
    output: TOut
    used_fallback: bool
    attempts: int
    model: str
    usage: LLMUsage = field(default_factory=LLMUsage)
    cost_usd: float = 0.0
    elapsed_seconds: float = 0.0
    warnings: list[str] = field(default_factory=list)


def dereference_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Inline ``$ref``/``$defs`` so the schema is portable across providers.

    Pydantic emits ``$defs`` for nested models. Anthropic tolerates them;
    OpenAI strict mode is fussier, and inlining sidesteps the difference
    entirely. Recursive models are not supported -- and none are used here.
    """
    defs = schema.get("$defs", {})

    def resolve(node: Any, depth: int = 0) -> Any:
        if depth > 20:  # guard against a cyclic model definition
            return {"type": "object"}
        if isinstance(node, dict):
            if "$ref" in node:
                name = str(node["$ref"]).rsplit("/", 1)[-1]
                target = defs.get(name)
                if target is None:
                    return {"type": "object"}
                merged = resolve(target, depth + 1)
                extras = {k: v for k, v in node.items() if k != "$ref"}
                return {**merged, **extras} if extras else merged
            return {k: resolve(v, depth + 1) for k, v in node.items() if k != "$defs"}
        if isinstance(node, list):
            return [resolve(item, depth + 1) for item in node]
        return node

    resolved = resolve(schema)
    resolved.pop("$defs", None)
    return resolved


class BaseAgent(abc.ABC, Generic[TIn, TOut]):
    """Base class for every AI agent."""

    #: Stable identifier used in logs and stored on records.
    name: str = "agent"
    #: Pydantic model the response must satisfy.
    output_model: type[TOut]
    #: ``True`` to prefer the cheaper model tier for this agent.
    prefers_cheap_model: bool = False
    temperature: float | None = None
    max_tokens: int | None = None
    max_attempts: int | None = None
    #: Seconds to wait before the first retry; doubles each attempt.
    retry_backoff: float = 1.5

    def __init__(self, llm: LLMProvider | None = None) -> None:
        self.llm = llm or get_llm_provider()

    # ------------------------------------------------------------ subclass API
    @abc.abstractmethod
    def build_system_prompt(self, payload: TIn) -> str:
        """Role, objective, criteria, constraints and failure behaviour."""

    @abc.abstractmethod
    def build_user_content(self, payload: TIn) -> str:
        """The data the agent reasons over. JSON, never prose instructions."""

    @abc.abstractmethod
    def fallback(self, payload: TIn) -> TOut:
        """Deterministic result used when no model is available.

        Must always return a valid output. Never raise for ordinary inputs.
        """

    def postprocess(self, output: TOut, payload: TIn) -> TOut:
        """Hook for clamping/deduplicating model output. Default: unchanged."""
        return output

    # --------------------------------------------------------------- execution
    @property
    def model(self) -> str:
        return settings.llm_model_cheap if self.prefers_cheap_model else settings.llm_model

    def json_schema(self) -> dict[str, Any]:
        return dereference_schema(self.output_model.model_json_schema())

    def run(self, payload: TIn, *, context: AgentContext | None = None) -> AgentResult[TOut]:
        started = time.perf_counter()
        context = context or AgentContext()

        if self.llm.is_mock:
            output = self.postprocess(self.fallback(payload), payload)
            message = f"{self.name}: no LLM provider configured, used deterministic fallback"
            context.warn(message)
            logger.info("agent used fallback", extra={"agent": self.name, "reason": "mock"})
            return AgentResult(
                output=output,
                used_fallback=True,
                attempts=0,
                model="deterministic",
                elapsed_seconds=time.perf_counter() - started,
                warnings=[message],
            )

        system = self.build_system_prompt(payload)
        messages = [LLMMessage(role="user", content=self.build_user_content(payload))]
        schema = self.json_schema()
        attempts = self.max_attempts or settings.llm_max_retries
        warnings: list[str] = []
        usage_total = LLMUsage()
        last_error: Exception | None = None

        for attempt in range(1, attempts + 1):
            try:
                response = self.llm.complete_json(
                    system=system,
                    messages=messages,
                    schema=schema,
                    schema_name=self.name,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    model=self.model,
                )
                usage_total = usage_total + response.usage
                context.record(self.llm, response.model, response.usage)
                if response.parsed is None:
                    raise LLMResponseError("Provider returned no parsed object.")
                output = self.output_model.model_validate(response.parsed)
                output = self.postprocess(output, payload)
                logger.info(
                    "agent completed",
                    extra={
                        "agent": self.name,
                        "attempt": attempt,
                        "input_tokens": response.usage.input_tokens,
                        "output_tokens": response.usage.output_tokens,
                    },
                )
                return AgentResult(
                    output=output,
                    used_fallback=False,
                    attempts=attempt,
                    model=response.model,
                    usage=usage_total,
                    cost_usd=self.llm.estimate_cost_usd(response.model, usage_total),
                    elapsed_seconds=time.perf_counter() - started,
                    warnings=warnings,
                )
            except ValidationError as exc:
                last_error = exc
                detail = _summarise_validation(exc)
                warnings.append(f"attempt {attempt}: schema validation failed")
                logger.warning(
                    "agent output failed validation",
                    extra={"agent": self.name, "attempt": attempt, "errors": detail},
                )
                # Repair: show the model exactly what was wrong, once.
                messages = messages[:1] + [
                    LLMMessage(
                        role="user",
                        content=(
                            "Your previous response did not satisfy the schema. "
                            f"Validation errors:\n{detail}\n"
                            "Return a corrected object. Change only what the "
                            "errors require."
                        ),
                    )
                ]
            except (LLMResponseError, ProviderError) as exc:
                last_error = exc
                warnings.append(f"attempt {attempt}: {exc.error_code}")
                logger.warning(
                    "agent provider call failed",
                    extra={"agent": self.name, "attempt": attempt, "error": str(exc)},
                )
            if attempt < attempts:
                time.sleep(self.retry_backoff * (2 ** (attempt - 1)))

        message = (
            f"{self.name}: {attempts} model attempt(s) failed "
            f"({type(last_error).__name__}); used deterministic fallback"
        )
        context.warn(message)
        warnings.append(message)
        logger.error(
            "agent exhausted retries, falling back",
            extra={"agent": self.name, "error": str(last_error)},
        )
        output = self.postprocess(self.fallback(payload), payload)
        return AgentResult(
            output=output,
            used_fallback=True,
            attempts=attempts,
            model="deterministic",
            usage=usage_total,
            cost_usd=self.llm.estimate_cost_usd(self.model, usage_total),
            elapsed_seconds=time.perf_counter() - started,
            warnings=warnings,
        )


def _summarise_validation(exc: ValidationError, limit: int = 12) -> str:
    lines = []
    for error in exc.errors()[:limit]:
        location = ".".join(str(part) for part in error.get("loc", ()))
        lines.append(f"- {location or '<root>'}: {error.get('msg')}")
    remaining = len(exc.errors()) - limit
    if remaining > 0:
        lines.append(f"- ... and {remaining} more")
    return "\n".join(lines)


def compact_json(value: Any) -> str:
    """Serialise payloads compactly -- whitespace in prompts costs tokens."""
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False, default=str)
