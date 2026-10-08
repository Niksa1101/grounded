"""A scriptable ``LLMProvider`` for tests (Tech.md §9.1). No network, no randomness.

Each ``generate`` call consumes the next scripted step, in order:

- a ``BaseModel`` instance or a ``str``: the provider's "raw output". It goes through
  ``schema.model_validate_json`` exactly like a real adapter's would, so a ``str`` that isn't valid
  JSON (or fails a constraint) surfaces as ``ProviderBadOutput``, never as a half-parsed object.
  Like a real reply, a bad one carries its usage and the compact feedback of the real adapter;
- a ``ProviderError`` instance: raised as is (rate limit, timeout, ...).

Every call is recorded, including the ones that raise, so a test can assert what the retry sent.
"""

from __future__ import annotations

import json
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

from pydantic import BaseModel, ValidationError

from grounded.generation.providers.base import GenerationResult, Usage
from grounded.infra.provider_errors import (
    ProviderBadOutput,
    ProviderError,
    compact_validation_error,
)
from grounded.schemas.judge import FaithfulnessVerdict
from grounded.schemas.llm import LLMAnswer, LLMClaim

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
                validation_error=compact_validation_error(exc),
                input_tokens=self._usage.input_tokens,
                output_tokens=self._usage.output_tokens,
            ) from exc
        return GenerationResult(
            parsed=parsed,
            raw_text=raw,
            usage=self._usage,
            provider=self.name,
            model=self.model,
            latency_ms=0,
        )


class StubLLMProvider:
    """Dev-only: answers every request with the same canned answer, citing ``c1`` if it has sources.

    For ``grounded serve`` and ``grounded ask`` before a real adapter exists (``GENERATOR_PROVIDERS=
    fake``, refused in prod). Unlike ``FakeLLMProvider`` it never runs out, and it says plainly
    that the text is a stand-in, so nobody mistakes it for a model answer.
    """

    name = "fake"
    model = "stub"

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
        # No ``<source`` block means a ``no_rag`` request: there is nothing to cite.
        cited = ["c1"] if "<source " in user else []
        answer = LLMAnswer(
            status="answered",
            answer_markdown="Stub answer from the fake provider, not a model output."
            + (" [c1]" if cited else ""),
            claims=[
                LLMClaim(text="This is a stub claim.", citation_ids=cited, self_confidence=0.5)
            ],
        )
        raw = answer.model_dump_json()
        return GenerationResult(
            parsed=schema.model_validate_json(raw),
            raw_text=raw,
            usage=Usage(input_tokens=0, output_tokens=0),
            provider=self.name,
            model=self.model,
            latency_ms=0,
        )


class StubJudgeProvider:
    """Dev-only judge for ``GENERATOR_PROVIDERS=fake`` runs (Tech §15.3): the promptfoo harness then
    runs with no network, judge included. Every claim is ``SUPPORTED`` and every answer
    ``PARTIALLY_CORRECT``, with a reason that says it is a stand-in, so nobody reads the numbers of
    such a run as a judgment.

    Its name differs from the stub generator's (``fake``) on purpose: the judge must not be the
    generator's provider (AGENTS.md §6.5) and ``check_judge_provider`` compares names. Like the stub
    generator it is never put behind the eval LLM cache: canned verdicts must not be replayed from
    a file after the stub changes.
    """

    name = "fake-judge"
    model = "stub-judge"

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
        verdict = (
            {"verdict": "SUPPORTED", "reason": "Stub verdict from the fake judge, not a judgment."}
            if schema is FaithfulnessVerdict
            else {
                "verdict": "PARTIALLY_CORRECT",
                "reason": "Stub verdict from the fake judge, not a judgment.",
            }
        )
        raw = json.dumps(verdict)
        return GenerationResult(
            parsed=schema.model_validate_json(raw),
            raw_text=raw,
            usage=Usage(input_tokens=0, output_tokens=0),
            provider=self.name,
            model=self.model,
            latency_ms=0,
        )
