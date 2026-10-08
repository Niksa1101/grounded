"""The judge side of the promptfoo assertions: what ``faithfulness`` and ``correctness`` do once a
case needs a judge call (Tech §15.3, 4.06).

``promptfoo_asserts`` decides what is not applicable and parses the inputs with the standard
library only, and imports this module just for a case that needs the judge, because only here the
judge is built (Groq adapter, eval LLM cache: about a second of imports, mostly the Gemini SDK that
``grounded.runtime`` pulls in). promptfoo starts a new Python process for every assertion call, and
two things follow from that. Both are handled here.

**One judge session at a time.** promptfoo runs up to three assertions of a row at once, and the
free Groq plan has 8K tokens per minute (Tech §15.4). Two processes asking at once would only
produce more 429s, so a judge session holds an exclusive SQLite lock (``judge_run.sqlite`` in
``CACHE_DIR``) from before its first call to after its last. The lock goes away with the process.

**A stop is shared.** A daily quota, a rejected key or a missing judge configuration cannot recover
inside the run (the provider treats them the same way, ``promptfoo_provider.cannot_recover``). The
first process to meet one writes it to the same file under the run's id (``metadata.run_id``,
``promptfoo_tests.build_tests``) and every later judge assertion of that run answers "skipped"
without calling anything. A per-minute 429 that the eval backoff gave up on is not a stop.

**What an assertion returns** is the ``{pass, score, reason, not_applicable}`` of the deterministic
ones (N/A is decided before this module is imported) plus ``errored`` and ``judge``:

- scored: ``errored`` false, ``score`` the metric, ``judge`` the details;
- errored: the judge could not grade the case (a ``JudgeError``, Tech §15.4). ``errored`` is true,
  ``score`` is a placeholder 0.0 and ``pass`` false so promptfoo shows it as a failure, and
  ``judge.error`` has the tag (``kind``, ``is_quota``, ``provider_side``). **An aggregation must
  skip it**: it is neither a score nor a not-applicable case.

For faithfulness a case is errored when any of its claims is: a partly judged answer has no
faithfulness. The claims that were judged keep their verdicts in ``judge.claims``. The component's
``judge.error`` is the first provider-side error if there is one, else the first.
"""

from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import Awaitable, Callable, Generator, Iterable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, cast

from grounded.evals.judge import CitedSource, ClaimToJudge, Judge
from grounded.evals.promptfoo_asserts import judge_detail, judge_errored, judge_scored
from grounded.generation.providers.base import Usage
from grounded.infra.event_loop import loop_factory
from grounded.infra.logging import configure_logging
from grounded.infra.provider_errors import ProviderRequestRejected
from grounded.runtime import ProviderConfigError, open_judge
from grounded.schemas.judge import CorrectnessJudgment, FaithfulnessJudgment, JudgeError
from grounded.settings import Settings, get_settings

STATE_FILE: Final = "judge_run.sqlite"
# A safety net, not a tuning knob: the longest a judge session waits for the one before it. A
# session ends in minutes (the backoff bounds its waits, Tech §15.6); an hour means a hung holder.
LOCK_TIMEOUT_S: Final = 3600.0

type Result = dict[str, Any]


@dataclass(frozen=True, slots=True)
class ClaimInput:
    """One claim of the answer, as the assertion read it from the provider's metadata."""

    text: str
    confidence: float  # server-computed (AGENTS.md §6.6), kept next to the verdict for 4.11 / 8.04
    sources: Sequence[CitedSource]  # the chunks the claim cites, in citation order


# --- Entry points --------------------------------------------------------------------------------


def judge_faithfulness(claims: Sequence[ClaimInput], *, run_id: str) -> Result:
    """Judge every claim of one answer against the sources it cites; score = supported / claims."""

    async def work(judge: Judge) -> list[FaithfulnessJudgment]:
        return await judge.judge_claims([ClaimToJudge(c.text, c.sources) for c in claims])

    def quota(judgments: list[FaithfulnessJudgment]) -> JudgeError | None:
        return next((j.error for j in judgments if j.error and j.error.is_quota), None)

    outcome = _in_turn(run_id, work, quota)
    if isinstance(outcome, JudgeError):
        return _unjudged("faithfulness", outcome)
    return faithfulness_result(claims, outcome)


def judge_correctness(*, question: str, reference_answer: str, answer: str, run_id: str) -> Result:
    """Grade one answer against the reference answer: 1.0 / 0.5 / 0.0 (``CORRECTNESS_SCORES``)."""

    async def work(judge: Judge) -> CorrectnessJudgment:
        return await judge.judge_correctness(
            question=question, reference_answer=reference_answer, answer=answer
        )

    def quota(judgment: CorrectnessJudgment) -> JudgeError | None:
        return judgment.error if judgment.error and judgment.error.is_quota else None

    outcome = _in_turn(run_id, work, quota)
    if isinstance(outcome, JudgeError):
        return _unjudged("correctness", outcome)
    return correctness_result(outcome)


# --- Results (pure) ------------------------------------------------------------------------------


def faithfulness_result(
    claims: Sequence[ClaimInput], judgments: Sequence[FaithfulnessJudgment]
) -> Result:
    """The component result of one answer: scored, or errored if any claim could not be judged."""
    supported = sum(j.verdict == "SUPPORTED" for j in judgments)
    errors = [j.error for j in judgments if j.error is not None]
    first = judgments[0]
    detail = judge_detail(
        "faithfulness",
        prompt_version=first.prompt_version,
        judge_provider=first.judge_provider,
        judge_model=first.judge_model,
    )
    detail.update(
        n_claims=len(judgments),
        n_supported=supported,
        n_errored=len(errors),
        usage=_usage_sum(j.usage for j in judgments),
        claims=[_claim_entry(c, j) for c, j in zip(claims, judgments, strict=True)],
    )
    error = _first_error(errors)
    if error is not None:
        return judge_errored(detail, error.model_dump(mode="json"))
    unsupported = [str(j.claim_index + 1) for j in judgments if j.verdict != "SUPPORTED"]
    reason = f"{supported} of {len(judgments)} claim(s) supported" + (
        f"; not supported: claim {', '.join(unsupported)}" if unsupported else ""
    )
    return judge_scored(supported == len(judgments), supported / len(judgments), reason, detail)


def correctness_result(judgment: CorrectnessJudgment) -> Result:
    detail = judge_detail(
        "correctness",
        prompt_version=judgment.prompt_version,
        judge_provider=judgment.judge_provider,
        judge_model=judgment.judge_model,
    )
    detail.update(
        verdict=judgment.verdict,
        reason=judgment.reason,
        attempts=judgment.attempts,
        cache_hit=judgment.cache_hit,
        usage=_usage(judgment.usage),
    )
    score = judgment.score
    if judgment.error is not None or score is None:
        error = judgment.error or JudgeError(kind="NoVerdict", provider_side=False)
        return judge_errored(detail, error.model_dump(mode="json"))
    return judge_scored(score == 1.0, score, f"{judgment.verdict}: {judgment.reason}", detail)


def _unjudged(metric: str, error: JudgeError) -> Result:
    """Nothing was asked: the run had stopped, or the judge could not be built or was rejected."""
    return judge_errored(judge_detail(metric), error.model_dump(mode="json"))


def _claim_entry(claim: ClaimInput, judgment: FaithfulnessJudgment) -> dict[str, Any]:
    return {
        "claim_index": judgment.claim_index,
        "claim": judgment.claim,
        "cited_labels": list(judgment.cited_labels),
        "confidence": claim.confidence,
        "verdict": judgment.verdict,
        "reason": judgment.reason,
        "decided_locally": judgment.decided_locally,
        "cache_hit": judgment.cache_hit,
        "attempts": judgment.attempts,
        "usage": _usage(judgment.usage),
        "error": judgment.error.model_dump(mode="json") if judgment.error else None,
    }


def _first_error(errors: Sequence[JudgeError]) -> JudgeError | None:
    """The error that names the case: a provider-side one counts toward ``inconclusive`` (Tech
    §15.5), so it wins over a reply the model got wrong."""
    return next((e for e in errors if e.provider_side), errors[0] if errors else None)


def _usage(usage: Usage) -> dict[str, int]:
    return usage.model_dump(mode="json")


def _usage_sum(usages: Iterable[Usage]) -> dict[str, int]:
    total = Usage(input_tokens=0, output_tokens=0)
    for usage in usages:
        total = Usage(
            input_tokens=total.input_tokens + usage.input_tokens,
            output_tokens=total.output_tokens + usage.output_tokens,
            thinking_tokens=total.thinking_tokens + usage.thinking_tokens,
        )
    return _usage(total)


# --- One judge session at a time, and a stop that all of them see --------------------------------


def _in_turn[T](
    run_id: str,
    work: Callable[[Judge], Awaitable[T]],
    quota_error: Callable[[T], JudgeError | None],
) -> T | JudgeError:
    """``work`` on a judge, alone, unless this run has stopped.

    Returns what ``work`` returned, or the ``JudgeError`` that says nothing could be judged: the run
    had already stopped (the error is the stop's, "skipped"), or the judge could not be built or the
    key was refused (the error is recorded as the stop). When ``work`` ended on a daily quota
    (``quota_error``), the stop is recorded too and its result is still returned.
    """
    settings = _eval_settings()
    with _turn(settings.cache_dir / STATE_FILE, run_id) as turn:
        if turn.stopped is not None:
            cause = turn.stopped
            return cause.model_copy(
                update={"detail": f"skipped: not asked, the run stopped on an earlier {cause.kind}"}
            )
        try:
            value = asyncio.run(_session(settings, work), loop_factory=loop_factory())
        except (ProviderRequestRejected, ProviderConfigError) as exc:
            error = JudgeError(
                kind=type(exc).__name__,
                provider_side=False,
                detail=" ".join(str(exc).split())[:300],
            )
            turn.stop(error)
            return error
        quota = quota_error(value)
        if quota is not None:
            turn.stop(quota)
        return value


async def _session[T](settings: Settings, work: Callable[[Judge], Awaitable[T]]) -> T:
    async with open_judge(settings) as judge:
        return await work(judge)


def _eval_settings() -> Settings:
    settings = get_settings()
    if settings.app_env != "eval":
        raise ProviderConfigError(
            f"the promptfoo judge assertions run in eval mode, got APP_ENV={settings.app_env}"
        )
    configure_logging(settings.log_level)
    return settings


class _Turn:
    def __init__(self, conn: sqlite3.Connection, run_id: str, stopped: JudgeError | None) -> None:
        self._conn = conn
        self._run_id = run_id
        self.stopped = stopped

    def stop(self, error: JudgeError) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO stops (run_id, error) VALUES (?, ?)",
            (self._run_id, error.model_dump_json()),
        )


@contextmanager
def _turn(path: Path, run_id: str) -> Generator[_Turn]:
    """Wait for the exclusive lock on ``path``, read this run's stop, and keep the lock until the
    block ends. The lock is SQLite's own, so it is released when the process dies, too."""
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=LOCK_TIMEOUT_S, isolation_level=None)
    try:
        conn.execute("BEGIN EXCLUSIVE")
        conn.execute(
            "CREATE TABLE IF NOT EXISTS stops (run_id TEXT PRIMARY KEY, error TEXT NOT NULL)"
        )
        row = cast(
            "tuple[str] | None",
            conn.execute("SELECT error FROM stops WHERE run_id = ?", (run_id,)).fetchone(),
        )
        yield _Turn(conn, run_id, None if row is None else JudgeError.model_validate_json(row[0]))
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()
