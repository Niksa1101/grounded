"""A scriptable ``LLMProvider`` for tests (Tech.md §9.1). No network, no randomness.

Each ``generate`` call consumes the next scripted step, in order:

- a ``BaseModel`` instance or a ``str``: the provider's "raw output". It goes through
  ``schema.model_validate_json`` exactly like a real adapter's would, so a ``str`` that isn't valid
  JSON (or fails a constraint) surfaces as ``ProviderBadOutput``, never as a half-parsed object;
- a ``ProviderError`` instance: raised as is (rate limit, timeout, ...).

Every call is recorded, including the ones that raise, so a test can assert what the retry sent.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

from pydantic import BaseModel, ValidationError

from grounded.generation.providers.base import GenerationResult, Usage
from grounded.infra.provider_errors import ProviderBadOutput, ProviderError

ScriptStep = BaseModel | str | ProviderError


@dataclass(frozen=True)
class RecordedCall:
    system: str
    user: str
    schema: type[BaseModel]
    temperature: float
    max_output_tokens: int
    timeout_s: float


class FakeLLMProvider:
    def __init__(
        self,
        script: Sequence[ScriptStep] = (),
        *,
        name: str = "fake",
        model: str = "fake-model",
        usage: Usage | None = None,
    ) -> None:
        self.name = name
        self.model = model
        self.calls: list[RecordedCall] = []
        self._steps: deque[ScriptStep] = deque(script)
        self._usage = usage or Usage(input_tokens=100, output_tokens=50)

    @property
    def remaining(self) -> int:
        """Scripted steps not yet consumed (a test can assert the script was used up)."""
        return len(self._steps)

    async def generate[T: BaseModel](
        self,
        *,
        system: str,
        user: str,
        schema: type[T],
        temperature: float,
        max_output_tokens: int,
        timeout_s: float,
    ) -> GenerationResult[T]:
        self.calls.append(
            RecordedCall(system, user, schema, temperature, max_output_tokens, timeout_s)
        )
        if not self._steps:
            # Not a ProviderError on purpose: an exhausted script is a bug in the test, and a
            # router that catches provider errors must not swallow it.
            raise AssertionError(f"FakeLLMProvider script exhausted after {len(self.calls)} calls")
        step = self._steps.popleft()
        if isinstance(step, ProviderError):
            raise step

        raw = step if isinstance(step, str) else step.model_dump_json()
        try:
            parsed = schema.model_validate_json(raw)
        except ValidationError as exc:
            raise ProviderBadOutput(
                f"output does not match {schema.__name__}",
                raw=raw,
                validation_error=str(exc),
            ) from exc
        return GenerationResult(
            parsed=parsed,
            raw_text=raw,
            usage=self._usage,
            provider=self.name,
            model=self.model,
            latency_ms=0,
        )
