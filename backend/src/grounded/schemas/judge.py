"""What the judge model must return, and what the judge module reports (Tech.md §15.4).

``FaithfulnessVerdict`` and ``CorrectnessVerdict`` are the structured outputs of the two rubrics in
``backend/prompts/judge_*_v1.md``: a label and a short reason. The labels are the words of the
rubric files, so the schema and the prompt cannot drift apart without a test noticing. As with
``LLMAnswer`` (``schemas/llm.py``), the bound on ``reason`` is something Groq's strict mode does not
enforce while decoding (``to_groq_schema`` drops it), so Pydantic enforces it after the call and a
reply that breaks it is bad output, which gets the one retry.

The ``*Judgment`` models are the judge module's results: a verdict **or** an error, never both and
never neither, plus everything a later ticket persists or counts (the prompt version, the judge
model, the tokens, whether the cache answered). An errored judgment carries no verdict and no score:
a case the judge could not grade is left out of the metric, never scored as a failure (Tech §15.5).
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from grounded.generation.providers.base import Usage

FaithfulnessLabel = Literal["SUPPORTED", "NOT_SUPPORTED"]
CorrectnessLabel = Literal["CORRECT", "PARTIALLY_CORRECT", "INCORRECT"]

# The rubric asks for "one or two sentences"; the bound is generous so that a wordy but valid
# verdict is not thrown away (a retry costs judge quota), and short enough to keep a stored reason
# readable.
MAX_REASON_CHARS = 600

# The Author's design choice (4.04): the judge names a grade, the code maps it to the number. The
# model is never asked for a score.
CORRECTNESS_SCORES: Mapping[CorrectnessLabel, float] = MappingProxyType(
    {"CORRECT": 1.0, "PARTIALLY_CORRECT": 0.5, "INCORRECT": 0.0}
)

_FROZEN = ConfigDict(frozen=True, extra="forbid")


class FaithfulnessVerdict(BaseModel):
    """``judge_faithfulness_v1``: is one claim supported by the sources it cited?"""

    model_config = _FROZEN

    verdict: FaithfulnessLabel
    reason: str = Field(min_length=1, max_length=MAX_REASON_CHARS)


class CorrectnessVerdict(BaseModel):
    """``judge_correctness_v1``: does the answer say what the reference answer says?"""

    model_config = _FROZEN

    verdict: CorrectnessLabel
    reason: str = Field(min_length=1, max_length=MAX_REASON_CHARS)


class JudgeError(BaseModel):
    """Why a case has no verdict. ``kind`` is the exception's class name, the tag that
    ``inconclusive`` counting reads (Tech §15.6). ``BackoffExhaustedError`` is a
    ``ProviderRateLimited`` that has already waited, and keeps its own name."""

    model_config = _FROZEN

    kind: str
    # A daily quota: waiting does not help (``ProviderRateLimited.is_quota``).
    is_quota: bool = False
    # The provider failed (429, 5xx, timeout), as opposed to a reply the model got wrong
    # (``ProviderBadOutput``). Tech §15.6 calls these the provider-side failures; 4.08 fixes which
    # of them count toward ``inconclusive``.
    provider_side: bool
    # The exception's short message, never the model's raw output.
    detail: str = Field(default="", max_length=300)


class _Judgment(BaseModel):
    model_config = _FROZEN

    prompt_version: str  # "<name>@<8 hex>" of the rubric that judged it
    judge_provider: str
    judge_model: str
    # Tokens of every provider call this judgment took, a failed first attempt included (it was
    # billed). Zero when decided locally. ``thinking_tokens`` are inside ``output_tokens`` on Groq.
    usage: Usage
    # True when no live call was made: the eval LLM cache answered every call. A cached reply
    # carries the original ``usage``, so tokens and cost add up on a cached run (Tech §11).
    cache_hit: bool = False
    # Provider calls made: 0 (decided locally), 1, or 2 (the one retry after invalid output).
    attempts: int = Field(ge=0, le=2)
    error: JudgeError | None = None

    @property
    def errored(self) -> bool:
        return self.error is not None


def _check_verdict_xor_error(
    verdict: str | None, reason: str | None, error: JudgeError | None
) -> None:
    if (verdict is None) == (error is None):
        raise ValueError("a judgment has either a verdict or an error, not both and not neither")
    if (verdict is None) != (reason is None):
        raise ValueError("a verdict comes with its reason, and an error has none")


class FaithfulnessJudgment(_Judgment):
    """One claim, judged against the sources it cited."""

    claim_index: int = Field(ge=0)  # position in the answer's claims
    claim: str
    # Labels of the sources that reached the judge (``c1``...), in the order they were shown.
    cited_labels: tuple[str, ...]
    verdict: FaithfulnessLabel | None = None
    reason: str | None = None
    # True when the claim had no usable source and the judge was not called: rubric rule 6 gives
    # NOT_SUPPORTED, so the verdict is the same one the model would have been told to give.
    decided_locally: bool = False

    @model_validator(mode="after")
    def _check(self) -> Self:
        _check_verdict_xor_error(self.verdict, self.reason, self.error)
        return self

    @property
    def supported(self) -> bool | None:
        """``None`` when there is no verdict, so a caller cannot mistake it for ``False``."""
        return None if self.verdict is None else self.verdict == "SUPPORTED"


class CorrectnessJudgment(_Judgment):
    """One answer, graded against the reference answer."""

    verdict: CorrectnessLabel | None = None
    reason: str | None = None

    @model_validator(mode="after")
    def _check(self) -> Self:
        _check_verdict_xor_error(self.verdict, self.reason, self.error)
        return self

    @property
    def score(self) -> float | None:
        """1.0 / 0.5 / 0.0, or ``None`` for an errored judgment (never 0.0: Tech §15.5)."""
        return None if self.verdict is None else CORRECTNESS_SCORES[self.verdict]
