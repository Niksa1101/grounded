"""The LLM judge: faithfulness per claim, correctness per answer (Tech §15.3-15.4, FR-22, D50).

``Judge`` sends the Author's rubrics (``backend/prompts/judge_*_v1.md``) to the judge provider
through the adapter layer. In an eval run that provider is the Groq adapter inside ``EvalLLM.wrap``
(``runtime.open_judge``), so every verdict goes through the typed errors, the bounded 429 backoff
and the eval LLM cache. Nothing here knows an SDK.

**What is one call.** Faithfulness: one call per claim, with the claim and the text of the sources
*that claim* cites, rendered as ``<source id="c1">...</source>`` blocks like the answer prompt's
context. Correctness: one call per answer, with the question, the reference answer and the
candidate answer. The model names a label (``SUPPORTED``, ``CORRECT``...) and a reason; the code
maps a label to a number (``CORRECTNESS_SCORES``), so a score is never something the model wrote.

**A claim with no source is not sent.** Rubric rule 6 says "if no source is given, the verdict is
``NOT_SUPPORTED``". Asking the model would only spend quota on a question whose answer is fixed,
and would let it judge from its own knowledge. So the verdict is given here, with a fixed reason,
and the judgment says ``decided_locally`` and ``attempts=0`` so a report can tell it from a model
verdict. This is the verdict the rubric produces, not a different rule.

**Escaping.** Chunk text and the claim go through ``context.escape_content``, which neutralizes
``<source`` and ``</source``, so a documentation page or a claim cannot close a block or open a
fake one. ``{{...}}`` in any value is inserted as text (``Prompt.render_user`` substitutes in one
pass). The correctness template has no ``<source>`` framing, so its values are inserted as they
are; the rubric's first paragraph tells the model they are data.

**Invalid output: one retry, then errored.** A reply that fails Pydantic (``ProviderBadOutput``)
gets exactly one more call on the same provider with the prompt's ``# Retry feedback`` section
(AGENTS.md §6.4, the same rule as ``AskPipeline``). If that fails too, the judgment is *errored*:
it has no verdict and no score, never a fabricated 0 (Tech §15.5). The provider-side failures
(``ProviderRateLimited``, ``BackoffExhaustedError``, ``ProviderUnavailable``,
``ProviderTimeout``) are errored judgments as well, with the exception's class name as
``error.kind``; whether a tag counts toward ``inconclusive`` is 4.08's decision.
``ProviderRequestRejected`` (a bad key) is raised: no judgment can be made and the run must stop.
``judge_claims`` also stops asking after the first provider-side failure of an answer, so a daily
quota is not asked again for each remaining claim.

**Cost and cache.** ``usage`` is the sum over the calls of one judgment, the failed first attempt
included. ``cache_hit`` is true only when no live call was made (a cached reply carries the
original usage, Tech §11). Temperature is 0 by design (FR-22), not a setting.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from grounded.generation.context import escape_attribute, escape_content
from grounded.generation.pipeline import MAX_VALIDATION_RETRIES
from grounded.generation.prompts import (
    Prompt,
    load_judge_correctness_prompt,
    load_judge_faithfulness_prompt,
)
from grounded.generation.providers.base import LLMProvider, Usage
from grounded.infra.provider_errors import (
    ProviderBadOutput,
    ProviderError,
    ProviderRateLimited,
    ProviderTimeout,
    ProviderUnavailable,
)
from grounded.schemas.judge import (
    CorrectnessJudgment,
    CorrectnessVerdict,
    FaithfulnessJudgment,
    FaithfulnessVerdict,
    JudgeError,
)
from grounded.settings import Settings

logger = logging.getLogger(__name__)

# FR-22: the judge runs at temperature 0. It is a property of the judge, not a tunable, so it is not
# a setting (and not ``LLM_TEMPERATURE``, which is the generator's).
JUDGE_TEMPERATURE = 0.0

# The verdict for a claim with no usable source (rubric rule 6), decided without a call.
NO_SOURCE_REASON = "The claim cites no source, so there is nothing that could support it."

_MAX_DETAIL_CHARS = 300
_NO_USAGE = Usage(input_tokens=0, output_tokens=0)


@dataclass(frozen=True, slots=True)
class JudgeConfig:
    max_output_tokens: int
    timeout_s: float

    @classmethod
    def from_settings(cls, settings: Settings) -> JudgeConfig:
        return cls(
            max_output_tokens=settings.judge_max_output_tokens, timeout_s=settings.call_timeout_s
        )


@dataclass(frozen=True, slots=True)
class CitedSource:
    """One source a claim cited: the label the judge may quote in its reason, and the chunk text."""

    label: str  # "c1": the label the claim used
    text: str


@dataclass(frozen=True, slots=True)
class ClaimToJudge:
    text: str
    sources: Sequence[CitedSource]


def cited_sources(citation_ids: Sequence[str], chunks: Mapping[str, str]) -> list[CitedSource]:
    """The sources a claim cites, in citation order, from ``chunks`` (label -> chunk text).

    A label that is not in ``chunks`` and a label cited twice are skipped, so a claim whose labels
    are all unknown ends up with no source (and is decided without a call).
    """
    sources: list[CitedSource] = []
    for label in citation_ids:
        text = chunks.get(label)
        if text is not None and all(source.label != label for source in sources):
            sources.append(CitedSource(label, text))
    return sources


def render_sources(sources: Sequence[CitedSource]) -> str:
    """The ``{{sources}}`` value: the blocks of ``generation/context.py`` without its ``section``
    and ``url`` attributes (the rubric shows ``<source id="c1">``), separated by a blank line."""
    return "\n\n".join(
        f'<source id="{escape_attribute(source.label)}">\n'
        f"{escape_content(source.text.strip('\n'))}\n</source>"
        for source in sources
    )


@dataclass(frozen=True, slots=True)
class _Reply[T: BaseModel]:
    """What ``Judge._ask`` got: a parsed verdict or an error, and what it cost."""

    parsed: T | None
    error: JudgeError | None
    usage: Usage
    cache_hit: bool
    attempts: int


class Judge:
    def __init__(
        self,
        provider: LLMProvider,
        *,
        faithfulness_prompt: Prompt,
        correctness_prompt: Prompt,
        config: JudgeConfig,
    ) -> None:
        self._provider = provider
        self._faithfulness_prompt = faithfulness_prompt
        self._correctness_prompt = correctness_prompt
        self._config = config

    @classmethod
    def create(cls, provider: LLMProvider, config: JudgeConfig) -> Judge:
        """A judge with the committed rubrics. ``provider`` is the wrapped judge adapter."""
        return cls(
            provider,
            faithfulness_prompt=load_judge_faithfulness_prompt(),
            correctness_prompt=load_judge_correctness_prompt(),
            config=config,
        )

    async def judge_claim(
        self, claim_index: int, claim: str, sources: Sequence[CitedSource]
    ) -> FaithfulnessJudgment:
        """Is ``claim`` supported by ``sources`` (the ones it cites)? One call, or none without a
        usable source."""
        prompt = self._faithfulness_prompt
        usable = _usable(sources)
        shown: dict[str, Any] = {
            "claim_index": claim_index,
            "claim": claim,
            "cited_labels": tuple(source.label for source in usable),
        }
        if not usable:
            return FaithfulnessJudgment(
                **shown,
                **self._record(prompt, _NO_USAGE, cache_hit=False, attempts=0),
                verdict="NOT_SUPPORTED",
                reason=NO_SOURCE_REASON,
                decided_locally=True,
            )
        user = prompt.render_user(claim=escape_content(claim), sources=render_sources(usable))
        reply = await self._ask(prompt, user, FaithfulnessVerdict)
        verdict = reply.parsed
        return FaithfulnessJudgment(
            **shown,
            **self._record(prompt, reply.usage, cache_hit=reply.cache_hit, attempts=reply.attempts),
            error=reply.error,
            verdict=verdict.verdict if verdict else None,
            reason=verdict.reason if verdict else None,
        )

    async def judge_claims(self, claims: Sequence[ClaimToJudge]) -> list[FaithfulnessJudgment]:
        """Every claim of one answer, in order, one at a time (concurrency 1).

        After the first provider-side failure (a quota, a 429 the backoff gave up on, a 5xx, a
        timeout) the claims that still need a call are not asked: they come back errored with the
        same kind and ``is_quota``, so they count as unscored for the same reason. A claim the judge
        cannot grade (bad output twice) does not stop the others.
        """
        judgments: list[FaithfulnessJudgment] = []
        stopped: JudgeError | None = None
        for index, claim in enumerate(claims):
            if stopped is not None and _usable(claim.sources):
                judgments.append(self._not_asked(index, claim, stopped))
                continue
            judgment = await self.judge_claim(index, claim.text, claim.sources)
            if judgment.error is not None and judgment.error.provider_side:
                stopped = judgment.error
            judgments.append(judgment)
        return judgments

    async def judge_correctness(
        self, *, question: str, reference_answer: str, answer: str
    ) -> CorrectnessJudgment:
        """Does ``answer`` say what ``reference_answer`` says? One call."""
        prompt = self._correctness_prompt
        user = prompt.render_user(
            question=question, reference_answer=reference_answer, answer=answer
        )
        reply = await self._ask(prompt, user, CorrectnessVerdict)
        verdict = reply.parsed
        return CorrectnessJudgment(
            **self._record(prompt, reply.usage, cache_hit=reply.cache_hit, attempts=reply.attempts),
            error=reply.error,
            verdict=verdict.verdict if verdict else None,
            reason=verdict.reason if verdict else None,
        )

    def _record(
        self, prompt: Prompt, usage: Usage, *, cache_hit: bool, attempts: int
    ) -> dict[str, Any]:
        """The fields every judgment has: which rubric and judge produced it, and what it took."""
        return {
            "prompt_version": prompt.version,
            "judge_provider": self._provider.name,
            "judge_model": self._provider.model,
            "usage": usage,
            "cache_hit": cache_hit,
            "attempts": attempts,
        }

    def _not_asked(
        self, index: int, claim: ClaimToJudge, cause: JudgeError
    ) -> FaithfulnessJudgment:
        error = cause.model_copy(
            update={"detail": f"not asked: {cause.kind} on an earlier claim of this answer"}
        )
        return FaithfulnessJudgment(
            claim_index=index,
            claim=claim.text,
            cited_labels=tuple(source.label for source in _usable(claim.sources)),
            **self._record(self._faithfulness_prompt, _NO_USAGE, cache_hit=False, attempts=0),
            error=error,
        )

    async def _ask[T: BaseModel](self, prompt: Prompt, user: str, schema: type[T]) -> _Reply[T]:
        """The call, with the one retry on invalid output (``AskPipeline._generate_validated`` has
        the same rule, with the citation checks between its attempts)."""
        usage = _NO_USAGE
        all_cached = True
        attempt_user = user
        attempts = 0
        while True:
            attempts += 1
            try:
                result = await self._provider.generate(
                    system=prompt.system,
                    user=attempt_user,
                    schema=schema,
                    temperature=JUDGE_TEMPERATURE,
                    max_output_tokens=self._config.max_output_tokens,
                    timeout_s=self._config.timeout_s,
                )
            except ProviderBadOutput as exc:
                # Billed but unusable: the adapter reports the tokens, and nothing is cached.
                usage = _plus(
                    usage, Usage(input_tokens=exc.input_tokens, output_tokens=exc.output_tokens)
                )
                if not exc.retryable or attempts > MAX_VALIDATION_RETRIES:
                    return _Reply(None, self._error(prompt, exc, attempts), usage, False, attempts)
                all_cached = False
                attempt_user = prompt.render_retry(user, error=exc.validation_error or str(exc))
            except (ProviderRateLimited, ProviderUnavailable, ProviderTimeout) as exc:
                return _Reply(None, self._error(prompt, exc, attempts), usage, False, attempts)
            else:
                return _Reply(
                    result.parsed,
                    None,
                    _plus(usage, result.usage),
                    all_cached and result.cache_hit,
                    attempts,
                )

    def _error(self, prompt: Prompt, exc: ProviderError, attempts: int) -> JudgeError:
        error = _error_of(exc)
        # The kind and the prompt, never the model's output or the text being judged.
        logger.warning(
            "judge call failed",
            extra={
                "error": error.kind,
                "prompt_version": prompt.version,
                "judge_model": self._provider.model,
                "attempts": attempts,
            },
        )
        return error


def _usable(sources: Sequence[CitedSource]) -> list[CitedSource]:
    """The sources with any text: a blank chunk supports nothing, so it counts as no source."""
    return [source for source in sources if source.text.strip()]


def _plus(a: Usage, b: Usage) -> Usage:
    return Usage(
        input_tokens=a.input_tokens + b.input_tokens,
        output_tokens=a.output_tokens + b.output_tokens,
        thinking_tokens=a.thinking_tokens + b.thinking_tokens,
    )


def _error_of(exc: ProviderError) -> JudgeError:
    detail = " ".join(str(exc).split())
    if isinstance(exc, ProviderBadOutput) and exc.validation_error:
        detail = f"{detail}: {exc.validation_error}"  # field paths and rules, no input values
    return JudgeError(
        kind=type(exc).__name__,
        is_quota=isinstance(exc, ProviderRateLimited) and exc.is_quota,
        provider_side=isinstance(exc, ProviderRateLimited | ProviderUnavailable | ProviderTimeout),
        detail=detail[:_MAX_DETAIL_CHARS],
    )
