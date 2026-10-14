"""Spec tests of the provider router and its circuit breaker (Tech.md §10, PRD D57, ticket 7.02).

They are the scenarios of the contract in ``grounded/generation/router.py`` (its module docstring
is the rulebook): one test per row of the Tech §10 event table, plus the D57 and 7.01 points
(timings from ``Settings``, the quota reset, the cap, a ``Retry-After`` of zero, logging, the
deadline). They run against scripted fake providers and a hand-moved clock, never against the wall
clock (AGENTS.md §8). Written in 7.02 as strict ``xfail`` specs, they all pass since the router
was implemented in 7.03 (the markers are gone).

How the Author adds a case
--------------------------
1. Script the providers: ``Probe("gemini", [step, ...])``. A step is what ``FakeLLMProvider``
   takes: an ``LLMAnswer`` (``GOOD``), a string that is not valid JSON (``BAD``: the fake turns it
   into a ``ProviderBadOutput`` that billed 100 + 50 tokens), or a ``ProviderError``
   (``limited(30.0)``, ``timeout()``, ``down()``, a ``ProviderRequestRejected``...). A call with no
   step left raises ``AssertionError``, so a router that calls a provider more often than the test
   expects fails loudly instead of being swallowed. ``Probe`` also takes ``clock=`` and
   ``takes_s=`` (each call "takes" that long on the hand-moved clock), ``log=`` (a shared list of
   events) and ``hold()`` (calls wait at a gate until you set it).
2. Build the router: ``rig = Rig(p, q)`` (first provider = primary). ``rig.clock`` and ``rig.wall``
   are the monotonic and the wall clock; move them with ``advance``. ``Rig(...,
   config=replace(CONFIG, ...))`` changes a timing.
3. Call ``await rig.ask(**hooks)`` (the hooks of ``generate``; ``on_attempt`` defaults to filling
   ``rig.seen``) and assert on the returned ``RoutedGeneration``, on the ``Probe`` (``calls``,
   ``entries``), on ``rig.router.breaker_status("gemini")`` and on ``trail(...)``, the report as
   ``(provider, outcome, is_retry)`` tuples.
"""

from __future__ import annotations

import asyncio
import logging
import math
from collections import deque
from collections.abc import Sequence
from dataclasses import FrozenInstanceError, replace
from datetime import UTC, datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pytest
from pydantic import BaseModel, ValidationError

from grounded.generation.providers.base import GenerationResult, Usage
from grounded.generation.providers.fake import FakeLLMProvider, RecordedCall, ScriptStep
from grounded.generation.router import (
    GEMINI_QUOTA_ZONE,
    LOG_BREAKER_OPENED,
    LOG_PROVIDER_SKIPPED,
    LOG_REQUEST_REJECTED,
    LOG_ZONE_UNAVAILABLE,
    SYSTEM_CLOCK,
    Admission,
    AllProvidersUnavailable,
    Attempt,
    AttemptOutcome,
    BreakerConfig,
    BreakerState,
    BreakerStatus,
    CircuitBreaker,
    Derivation,
    OpenPlan,
    OpenReason,
    ProviderRouter,
    RoutedGeneration,
    RouterClock,
    RouterConfigError,
    RouterDeadlineError,
    UnknownProviderError,
    ZoneLoader,
    open_plan,
    quota_reset_s,
)
from grounded.infra.provider_errors import (
    ProviderBadOutput,
    ProviderError,
    ProviderRateLimited,
    ProviderRequestRejected,
    ProviderTimeout,
    ProviderUnavailable,
    compact_validation_error,
)
from grounded.schemas.llm import LLMAnswer
from tests.support import FakeClock, WallClock, make_settings

HOUR = 3600.0
OUTCOME = AttemptOutcome

# The documented defaults (Tech §4, PRD D57b). One plain test pins that ``Settings`` yields them, so
# the scenarios below do not move when a default does.
CONFIG = BreakerConfig(
    rate_limit_open_s=60.0,
    quota_open_s=900.0,
    failure_threshold=3,
    failure_window_s=60.0,
    failure_open_s=30.0,
    max_retry_after_s=3600.0,
)

# 05:00 in Los Angeles (PDT), so the next midnight there is 19 h away.
WALL_START = datetime(2026, 10, 14, 12, 0, tzinfo=UTC)

GOOD = LLMAnswer(status="insufficient_context", answer_markdown="Not covered.", claims=[])
BAD = "not json at all"  # the fake raises ProviderBadOutput for it, with the compact feedback


def _reason_of(raw: str) -> str:
    try:
        LLMAnswer.model_validate_json(raw)
    except ValidationError as exc:
        return compact_validation_error(exc)
    raise AssertionError("the text was meant to be invalid")


BAD_REASON = _reason_of(BAD)


def limited(retry_after: float | None = None, *, quota: bool = False) -> ProviderRateLimited:
    return ProviderRateLimited("429", retry_after_s=retry_after, is_quota=quota)


def timeout() -> ProviderTimeout:
    return ProviderTimeout("late")


def down() -> ProviderUnavailable:
    return ProviderUnavailable("503")


def feedback(user: str, reason: str) -> str:
    """A stand-in for ``Prompt.render_retry``: the original message plus the reason."""
    return f"{user}\n\nYour previous output was invalid because {reason}"


class Probe(FakeLLMProvider):
    """A scripted provider that can also take time, wait at a gate and log what it is asked."""

    def __init__(
        self,
        name: str,
        script: Sequence[ScriptStep] = (),
        *,
        clock: FakeClock | None = None,
        takes_s: float = 0.0,
        log: list[str] | None = None,
        usage: Usage | None = None,
        cache_hit: bool = False,
    ) -> None:
        super().__init__(script, name=name, model=f"{name}-model", usage=usage)
        self.entries = 0  # calls that started, also those still waiting at the gate
        self.entered = asyncio.Event()  # set when a call starts
        self.gate: asyncio.Event | None = None
        self.crash_next: Exception | None = None
        self._clock = clock
        self._takes_s = takes_s
        self._log = log
        self._cache_hit = cache_hit

    def hold(self) -> asyncio.Event:
        """From now on every call waits until the returned event is set."""
        self.entered.clear()
        self.gate = asyncio.Event()
        return self.gate

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
        self.entries += 1
        if self._log is not None:
            self._log.append(f"call:{self.name}")
        self.entered.set()
        if self.gate is not None:
            await self.gate.wait()
        if self._clock is not None:
            self._clock.advance(self._takes_s)
        if self.crash_next is not None:
            crash, self.crash_next = self.crash_next, None
            raise crash
        result = await super().generate(
            system=system,
            user=user,
            schema=schema,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            timeout_s=timeout_s,
        )
        return result.model_copy(update={"cache_hit": self._cache_hit})


class Rig:
    """A router over the given providers with hand-moved clocks, and ``ask`` with test defaults."""

    def __init__(
        self,
        *providers: Probe,
        config: BreakerConfig = CONFIG,
        clock: FakeClock | None = None,
        load_zone: ZoneLoader = ZoneInfo,
    ) -> None:
        self.clock = clock or FakeClock()
        self.wall = WallClock(WALL_START)
        self.seen: list[Attempt] = []
        self.router = ProviderRouter(
            providers,
            config=config,
            clock=RouterClock(monotonic=self.clock, wall=self.wall, load_zone=load_zone),
        )

    async def _record(self, attempt: Attempt) -> None:
        self.seen.append(attempt)

    async def ask(self, **overrides: Any) -> RoutedGeneration[LLMAnswer]:
        kwargs: dict[str, Any] = {
            "system": "sys",
            "user": "question",
            "schema": LLMAnswer,
            "temperature": 0.0,
            "max_output_tokens": 256,
            "timeout_s": 12.0,
            "retry_user": feedback,
            "on_attempt": self._record,
            **overrides,
        }
        return await self.router.generate(**kwargs)

    def status(self, provider: str) -> BreakerStatus:
        return self.router.breaker_status(provider)

    def time_left(self, deadline_s: float = 25.0) -> Any:
        """A ``time_left`` hook for a request whose deadline is ``deadline_s`` after now."""
        deadline = self.clock() + deadline_s
        return lambda: deadline - self.clock()


def trail(attempts: Sequence[Attempt]) -> list[tuple[str, AttemptOutcome, bool]]:
    return [(a.provider, a.outcome, a.is_retry) for a in attempts]


def opened_logs(caplog: pytest.LogCaptureFixture) -> list[dict[str, Any]]:
    return [r.__dict__ for r in caplog.records if r.getMessage() == LOG_BREAKER_OPENED]


# --- Plain tests: the value types and BreakerConfig ----------------------------------------------


def test_breaker_config_is_built_from_the_settings_fields() -> None:
    settings = make_settings(
        breaker_rate_limit_open_s=11.0,
        breaker_quota_open_s=22.0,
        breaker_failure_threshold=4,
        breaker_failure_window_s=33.0,
        breaker_failure_open_s=44.0,
        breaker_max_retry_after_s=55.0,
    )
    assert BreakerConfig.from_settings(settings) == BreakerConfig(
        rate_limit_open_s=11.0,
        quota_open_s=22.0,
        failure_threshold=4,
        failure_window_s=33.0,
        failure_open_s=44.0,
        max_retry_after_s=55.0,
    )


def test_the_default_settings_give_the_documented_timings() -> None:
    """D57b: 60 s, 900 s, 3 failures in 60 s open for 30 s, a provider's wait capped at 3600 s."""
    assert BreakerConfig.from_settings(make_settings()) == CONFIG


@pytest.mark.parametrize("field", ["rate_limit_open_s", "quota_open_s", "failure_window_s"])
@pytest.mark.parametrize("bad", [0.0, -1.0, math.inf, math.nan])
def test_breaker_config_rejects_a_duration_that_cannot_work(field: str, bad: float) -> None:
    with pytest.raises(RouterConfigError, match=field):
        replace(CONFIG, **{field: bad})


@pytest.mark.parametrize("field", ["failure_open_s", "max_retry_after_s"])
@pytest.mark.parametrize("bad", [0.0, -1.0, math.inf, math.nan])
def test_breaker_config_rejects_the_other_durations_too(field: str, bad: float) -> None:
    with pytest.raises(RouterConfigError, match=field):
        replace(CONFIG, **{field: bad})


def test_breaker_config_rejects_a_threshold_below_one() -> None:
    with pytest.raises(RouterConfigError, match="failure_threshold"):
        replace(CONFIG, failure_threshold=0)


def test_the_value_types_are_frozen() -> None:
    result = GenerationResult[LLMAnswer](
        parsed=GOOD,
        raw_text=GOOD.model_dump_json(),
        usage=Usage(input_tokens=1, output_tokens=1),
        provider="gemini",
        model="m",
        latency_ms=0,
    )
    attempt = Attempt("gemini", "m", OUTCOME.SUCCESS)
    routed = RoutedGeneration(result, "gemini", "m", False, (attempt,))
    status = BreakerStatus(BreakerState.CLOSED, None, False)
    plan = OpenPlan(1.0, OpenReason.QUOTA, Derivation.DEFAULT)
    for value, name in ((attempt, "outcome"), (routed, "provider"), (status, "state")):
        with pytest.raises(FrozenInstanceError):
            setattr(value, name, None)
    for value, name in ((plan, "seconds"), (CONFIG, "quota_open_s"), (SYSTEM_CLOCK, "wall")):
        with pytest.raises(FrozenInstanceError):
            setattr(value, name, None)


@pytest.mark.parametrize(
    ("outcome", "called"),
    [
        (OUTCOME.SUCCESS, True),
        (OUTCOME.BAD_OUTPUT, True),
        (OUTCOME.RATE_LIMITED, True),
        (OUTCOME.QUOTA, True),
        (OUTCOME.TIMEOUT, True),
        (OUTCOME.UNAVAILABLE, True),
        (OUTCOME.REJECTED, True),
        (OUTCOME.SKIPPED_OPEN, False),
        (OUTCOME.SKIPPED_TRIAL, False),
        (OUTCOME.SKIPPED_DEADLINE, False),
    ],
)
def test_an_attempt_is_a_call_unless_it_is_a_skip(outcome: AttemptOutcome, called: bool) -> None:
    assert Attempt("gemini", "m", outcome).called is called


def test_an_attempt_defaults_to_no_usage_and_no_retry() -> None:
    attempt = Attempt("gemini", "m", OUTCOME.SKIPPED_OPEN)
    assert attempt.usage == Usage(input_tokens=0, output_tokens=0)
    assert (attempt.latency_ms, attempt.is_retry, attempt.cache_hit) == (0, False, False)


def test_all_providers_unavailable_is_a_provider_error_with_retry_after_and_attempts() -> None:
    attempts = (Attempt("gemini", "m", OUTCOME.SKIPPED_OPEN),)
    error = AllProvidersUnavailable("none left", retry_after_s=12.5, attempts=attempts)
    assert isinstance(error, ProviderError)
    assert (error.retry_after_s, error.attempts) == (12.5, attempts)
    assert AllProvidersUnavailable("x", retry_after_s=None).attempts == ()


def test_the_deadline_error_is_not_a_provider_error() -> None:
    """Nothing failed: the request ran out of time, so 7.04 answers 504, not 503."""
    error = RouterDeadlineError("late", time_left_s=1.0, timeout_s=12.0)
    assert not isinstance(error, ProviderError)
    assert not isinstance(error, AllProvidersUnavailable)
    assert (error.time_left_s, error.timeout_s, error.attempts) == (1.0, 12.0, ())


def test_the_system_clock_reads_real_time() -> None:
    assert isinstance(SYSTEM_CLOCK.monotonic(), float)
    assert SYSTEM_CLOCK.wall().utcoffset() == timedelta(0)


# --- Pure rules: the quota reset and the open plan ----------------------------------------------

# Each case: the wall-clock instant (UTC) and the seconds until the next midnight in Los Angeles.
# The expected numbers are worked out between UTC instants by hand (PDT = UTC-7 until the first
# Sunday of November, PST = UTC-8 after it, back to PDT on the second Sunday of March).
GEMINI_RESETS = [
    # 05:00 PDT, midnight is 00:00 PDT Oct 15 = 07:00 UTC.
    pytest.param(datetime(2026, 10, 14, 12, 0, tzinfo=UTC), 19 * HOUR, id="ordinary-day"),
    # The UTC date is already the 14th, the Los Angeles date is still the 13th (20:00 PDT).
    pytest.param(datetime(2026, 10, 14, 3, 0, tzinfo=UTC), 4 * HOUR, id="utc-date-ahead"),
    # Exactly 00:00 PDT: the next midnight is a full day away, never 0.
    pytest.param(datetime(2026, 10, 15, 7, 0, tzinfo=UTC), 24 * HOUR, id="exactly-midnight"),
    # Spring forward, Sunday 2026-03-08 (02:00 PST becomes 03:00 PDT): the day is 23 h long.
    # 00:30 PST = 08:30 UTC; the next midnight is 00:00 PDT Mar 9 = 07:00 UTC Mar 9.
    pytest.param(datetime(2026, 3, 8, 8, 30, tzinfo=UTC), 22.5 * HOUR, id="spring-forward-start"),
    # 01:00 PST = 09:00 UTC, one hour before the change.
    pytest.param(datetime(2026, 3, 8, 9, 0, tzinfo=UTC), 22 * HOUR, id="spring-forward-before"),
    # Fall back, Sunday 2026-11-01 (02:00 PDT becomes 01:00 PST): the day is 25 h long.
    # 00:30 PDT = 07:30 UTC; the next midnight is 00:00 PST Nov 2 = 08:00 UTC Nov 2.
    pytest.param(datetime(2026, 11, 1, 7, 30, tzinfo=UTC), 24.5 * HOUR, id="fall-back-start"),
    # 04:00 PST = 12:00 UTC, after the change.
    pytest.param(datetime(2026, 11, 1, 12, 0, tzinfo=UTC), 20 * HOUR, id="fall-back-after"),
]


@pytest.mark.parametrize(("now", "expected_s"), GEMINI_RESETS)
def test_gemini_quota_resets_at_the_next_midnight_in_los_angeles(
    now: datetime, expected_s: float
) -> None:
    """D57b / 7.01: a DST day is 23 h or 25 h long, and the cap does not apply to the reset."""
    assert quota_reset_s(
        "gemini", now, None, max_retry_after_s=CONFIG.max_retry_after_s
    ) == pytest.approx(expected_s)


def test_gemini_quota_reset_ignores_a_retry_after_and_the_offset_of_now() -> None:
    plus_nine = timezone(timedelta(hours=9))
    now = datetime(2026, 10, 14, 21, 0, tzinfo=plus_nine)  # the same instant as 12:00 UTC
    assert quota_reset_s("gemini", now, 5.0, max_retry_after_s=3600.0) == pytest.approx(19 * HOUR)


def test_gemini_quota_reset_needs_an_aware_now() -> None:
    with pytest.raises(ValueError, match="aware"):
        quota_reset_s("gemini", datetime(2026, 10, 14, 12, 0), None, max_retry_after_s=3600.0)


@pytest.mark.parametrize(
    "failure", [ZoneInfoNotFoundError("America/Los_Angeles"), OSError("corrupt"), ValueError("bad")]
)
def test_a_zone_that_cannot_be_loaded_makes_the_reset_not_derivable(
    failure: Exception, caplog: pytest.LogCaptureFixture
) -> None:
    """7.01: ``None`` and one WARNING naming the provider and the zone."""

    def broken(key: str) -> Any:
        raise failure

    assert (
        quota_reset_s("gemini", WALL_START, None, max_retry_after_s=3600.0, load_zone=broken)
        is None
    )
    [record] = [r for r in caplog.records if r.getMessage() == LOG_ZONE_UNAVAILABLE]
    assert record.levelno == logging.WARNING
    assert (record.__dict__["provider"], record.__dict__["zone"]) == ("gemini", GEMINI_QUOTA_ZONE)


@pytest.mark.parametrize(
    ("retry_after", "expected"),
    [
        (30.0, 30.0),
        (7200.0, 3600.0),  # cut to the cap
        (math.inf, 3600.0),
        (None, None),
        (0.0, None),  # zero or less counts as absent (7.01)
        (-5.0, None),
        (math.nan, None),
    ],
)
def test_groq_quota_resets_after_its_own_retry_after_cut_to_the_cap(
    retry_after: float | None, expected: float | None
) -> None:
    reset = quota_reset_s("groq", WALL_START, retry_after, max_retry_after_s=3600.0)
    assert reset == (None if expected is None else pytest.approx(expected))


@pytest.mark.parametrize("provider", ["fake", "other", ""])
def test_any_other_provider_has_no_derivable_reset(provider: str) -> None:
    assert quota_reset_s(provider, WALL_START, 30.0, max_retry_after_s=3600.0) is None


def assert_plan(plan: OpenPlan, seconds: float, reason: OpenReason, derived: Derivation) -> None:
    assert plan.seconds == pytest.approx(seconds)
    assert (plan.reason, plan.derived) == (reason, derived)


@pytest.mark.parametrize(
    ("provider", "retry_after", "seconds", "derived"),
    [
        ("gemini", 30.0, 30.0, Derivation.RETRY_AFTER),
        ("groq", 0.5, 0.5, Derivation.RETRY_AFTER),
        ("gemini", 7200.0, 3600.0, Derivation.RETRY_AFTER_CAPPED),
        ("gemini", None, 60.0, Derivation.DEFAULT),
        ("groq", 0.0, 60.0, Derivation.DEFAULT),  # zero or less counts as absent
        ("groq", -5.0, 60.0, Derivation.DEFAULT),
        ("groq", math.nan, 60.0, Derivation.DEFAULT),
    ],
)
def test_a_per_minute_429_opens_for_the_retry_after_or_the_default(
    provider: str, retry_after: float | None, seconds: float, derived: Derivation
) -> None:
    """Tech §10 'RateLimited' row; D57b: capped by BREAKER_MAX_RETRY_AFTER_S, 60 s when absent."""
    plan = open_plan(provider, limited(retry_after), WALL_START, CONFIG)
    assert_plan(plan, seconds, OpenReason.RATE_LIMIT, derived)


@pytest.mark.parametrize(
    ("provider", "retry_after", "seconds", "derived"),
    [
        ("gemini", None, 19 * HOUR, Derivation.QUOTA_RESET),
        ("gemini", 5.0, 19 * HOUR, Derivation.QUOTA_RESET),  # a daily 429's retryDelay is ignored
        ("groq", 120.0, 120.0, Derivation.RETRY_AFTER),
        ("groq", 7200.0, 3600.0, Derivation.RETRY_AFTER_CAPPED),
        ("groq", None, 900.0, Derivation.DEFAULT),
        ("fake", 30.0, 900.0, Derivation.DEFAULT),
    ],
)
def test_a_daily_quota_429_opens_until_the_providers_reset_or_for_the_default(
    provider: str, retry_after: float | None, seconds: float, derived: Derivation
) -> None:
    """Tech §10 'is_quota' row; D57b: Gemini midnight Pacific, Groq its own wait, else 900 s."""
    plan = open_plan(provider, limited(retry_after, quota=True), WALL_START, CONFIG)
    assert_plan(plan, seconds, OpenReason.QUOTA, derived)


def test_a_quota_429_with_an_unloadable_zone_falls_back_to_the_default() -> None:
    def broken(key: str) -> Any:
        raise ZoneInfoNotFoundError(key)

    plan = open_plan("gemini", limited(None, quota=True), WALL_START, CONFIG, load_zone=broken)
    assert_plan(plan, 900.0, OpenReason.QUOTA, Derivation.DEFAULT)


# --- The breaker state machine, on its own --------------------------------------------------------


def new_breaker() -> CircuitBreaker:
    return CircuitBreaker("gemini", CONFIG)


def fail(breaker: CircuitBreaker, now: float) -> float | None:
    """One timeout or 5xx of a call admitted at ``now``; the seconds it opened for, if it did."""
    admission = breaker.admit(now)
    assert admission is Admission.ALLOWED
    return breaker.record_failure(admission, now)


def open_and_wait(breaker: CircuitBreaker, seconds: float = 60.0) -> float:
    """Open at t=0 for ``seconds`` and return a moment when it is half-open."""
    assert breaker.open_for(breaker.admit(0.0), 0.0, seconds) is True
    return seconds + 1.0


def test_a_new_breaker_is_closed_and_lets_calls_through() -> None:
    breaker = new_breaker()
    assert breaker.status(0.0) == BreakerStatus(BreakerState.CLOSED, None, False)
    assert breaker.admit(0.0) is Admission.ALLOWED


def test_the_threshold_of_failures_within_the_window_opens_it() -> None:
    """Tech §10 'Unavailable / Timeout' row: 3 consecutive within 60 s open it for 30 s."""
    breaker = new_breaker()
    assert fail(breaker, 0.0) is None
    assert fail(breaker, 20.0) is None
    assert fail(breaker, 40.0) == pytest.approx(30.0)
    status = breaker.status(40.0)
    assert status.state is BreakerState.OPEN
    assert status.remaining_s == pytest.approx(30.0)


def test_the_breaker_ignores_failures_older_than_the_window() -> None:
    breaker = new_breaker()
    assert [fail(breaker, t) for t in (0.0, 40.0, 100.0, 110.0)] == [None, None, None, None]
    assert breaker.status(110.0).state is BreakerState.CLOSED
    assert fail(breaker, 120.0) == pytest.approx(30.0)  # 100, 110, 120: three inside one window


def test_a_reply_clears_the_failure_run() -> None:
    breaker = new_breaker()
    assert (fail(breaker, 0.0), fail(breaker, 10.0)) == (None, None)
    breaker.record_reply(breaker.admit(20.0), 20.0)
    assert (fail(breaker, 30.0), fail(breaker, 40.0)) == (None, None)
    assert breaker.status(40.0).state is BreakerState.CLOSED
    assert fail(breaker, 50.0) == pytest.approx(30.0)


def test_an_open_breaker_skips_until_the_open_time_has_elapsed() -> None:
    """Tech §10 'breaker open' and 'time elapsed' rows."""
    breaker = new_breaker()
    open_and_wait(breaker, 60.0)
    assert breaker.admit(59.0) is Admission.SKIP_OPEN
    assert breaker.status(59.0).remaining_s == pytest.approx(1.0)
    assert breaker.admit(61.0) is Admission.TRIAL
    assert breaker.status(61.0) == BreakerStatus(BreakerState.HALF_OPEN, None, True)


def test_half_open_admits_exactly_one_trial() -> None:
    breaker = new_breaker()
    later = open_and_wait(breaker)
    assert breaker.status(later) == BreakerStatus(BreakerState.HALF_OPEN, None, False)
    assert breaker.admit(later) is Admission.TRIAL
    assert [breaker.admit(later), breaker.admit(later + 5.0)] == [Admission.SKIP_TRIAL] * 2


def test_a_trial_reply_closes_the_breaker() -> None:
    breaker = new_breaker()
    later = open_and_wait(breaker)
    breaker.record_reply(breaker.admit(later), later)
    assert breaker.status(later).state is BreakerState.CLOSED
    assert breaker.admit(later) is Admission.ALLOWED


def test_a_trial_failure_opens_it_again_with_a_fresh_timer() -> None:
    breaker = new_breaker()
    later = open_and_wait(breaker)
    assert breaker.record_failure(breaker.admit(later), later) == pytest.approx(30.0)
    assert breaker.status(later).remaining_s == pytest.approx(30.0)
    assert breaker.admit(later + 29.0) is Admission.SKIP_OPEN
    assert breaker.admit(later + 31.0) is Admission.TRIAL


def test_a_trial_that_gets_a_429_opens_for_the_given_seconds() -> None:
    breaker = new_breaker()
    later = open_and_wait(breaker)
    assert breaker.open_for(breaker.admit(later), later, 45.0) is True
    assert breaker.status(later).remaining_s == pytest.approx(45.0)


def test_a_released_trial_is_given_back_without_a_verdict() -> None:
    breaker = new_breaker()
    later = open_and_wait(breaker)
    breaker.release(breaker.admit(later))
    assert breaker.status(later) == BreakerStatus(BreakerState.HALF_OPEN, None, False)
    assert breaker.admit(later) is Admission.TRIAL


def test_a_call_that_finishes_after_the_breaker_opened_changes_nothing() -> None:
    """A straggler (admitted while closed) must not close, extend or re-time an open breaker, and
    its failures must not pile up into a new opening (three of them reach the threshold)."""
    breaker = new_breaker()
    opener, replier, *failers, late = (breaker.admit(0.0) for _ in range(7))
    assert breaker.open_for(opener, 1.0, 60.0) is True  # opens until t=61
    breaker.record_reply(replier, 2.0)
    assert breaker.status(2.0).state is BreakerState.OPEN
    assert [breaker.record_failure(failer, 3.0) for failer in failers[:3]] == [None] * 3
    assert breaker.open_for(failers[3], 3.0, 600.0) is False
    assert breaker.status(3.0).remaining_s == pytest.approx(58.0)  # unchanged: still until t=61
    trial = breaker.admit(70.0)
    assert trial is Admission.TRIAL
    breaker.record_reply(late, 71.0)  # not the trial: it does not close the breaker
    assert breaker.status(71.0) == BreakerStatus(BreakerState.HALF_OPEN, None, True)
    breaker.record_reply(trial, 72.0)
    assert breaker.status(72.0).state is BreakerState.CLOSED


# --- The router: one test per row of the Tech §10 table ------------------------------------------


async def test_success_returns_the_answer_and_the_report() -> None:
    """Tech §10 'Success' row, and the report: provider, model, fallback_used, attempts."""
    p, q = Probe("gemini", [GOOD]), Probe("groq")
    rig = Rig(p, q)
    out = await rig.ask()
    assert out.result.parsed == GOOD
    assert (out.provider, out.model, out.fallback_used) == ("gemini", "gemini-model", False)
    assert trail(out.attempts) == [("gemini", OUTCOME.SUCCESS, False)]
    assert out.attempts[0].model == "gemini-model"
    assert rig.seen == list(out.attempts)  # on_attempt got the very record that is in the report
    assert q.entries == 0
    # Everything but ``user`` reaches the provider as given.
    assert p.calls == [RecordedCall("sys", "question", LLMAnswer, 0.0, 256, 12.0)]


async def test_success_resets_the_failure_count() -> None:
    """Tech §10 'Success' row: two timeouts, a success, two more timeouts open nothing."""
    p = Probe("gemini", [timeout(), timeout(), GOOD, timeout(), timeout(), GOOD])
    q = Probe("groq", [GOOD] * 4)
    rig = Rig(p, q)
    for _ in range(5):
        await rig.ask()
    assert rig.status("gemini").state is BreakerState.CLOSED
    assert (await rig.ask()).provider == "gemini"  # still called: never skipped
    assert len(p.calls) == 6


async def test_bad_output_gets_one_retry_on_the_same_provider_with_the_feedback() -> None:
    """Tech §10 'BadOutput' row, §9.5 step 3: the retry's user message carries the reason."""
    p, q = Probe("gemini", [BAD, GOOD]), Probe("groq")
    out = await Rig(p, q).ask()
    first, retry = p.calls
    assert first.user == "question"
    assert retry.user == feedback("question", BAD_REASON)
    assert (out.provider, out.fallback_used) == ("gemini", False)
    assert trail(out.attempts) == [
        ("gemini", OUTCOME.BAD_OUTPUT, False),
        ("gemini", OUTCOME.SUCCESS, True),
    ]
    assert q.entries == 0


async def test_a_validate_rejection_is_bad_output_with_the_hooks_reason() -> None:
    """§9.5 step 2: the citation checks run after each call; a failure is bad output."""
    p = Probe("gemini", [GOOD, GOOD])
    verdicts = iter(["no valid citation", None])
    out = await Rig(p).ask(validate=lambda result: next(verdicts))
    assert p.calls[1].user == feedback("question", "no valid citation")
    assert trail(out.attempts) == [
        ("gemini", OUTCOME.BAD_OUTPUT, False),
        ("gemini", OUTCOME.SUCCESS, True),
    ]


async def test_a_second_bad_output_goes_to_the_next_provider() -> None:
    """Tech §10 'BadOutput' row: exactly one retry, then the next provider (a fallback)."""
    p, q = Probe("gemini", [BAD, BAD]), Probe("groq", [GOOD])
    out = await Rig(p, q).ask()  # a third call on gemini would exhaust its script and fail loudly
    assert (out.provider, out.fallback_used) == ("groq", True)
    assert q.calls[0].user == "question"  # the fallback gets the original message, not the feedback
    assert trail(out.attempts) == [
        ("gemini", OUTCOME.BAD_OUTPUT, False),
        ("gemini", OUTCOME.BAD_OUTPUT, True),
        ("groq", OUTCOME.SUCCESS, False),
    ]


async def test_bad_output_that_is_not_retryable_skips_the_retry() -> None:
    """§9.5 step 3: a content-filter block would be hit again, so the next provider is tried."""
    blocked = ProviderBadOutput("blocked by the content filter", retryable=False)
    p, q = Probe("gemini", [blocked]), Probe("groq", [GOOD])
    out = await Rig(p, q).ask()
    assert len(p.calls) == 1
    assert trail(out.attempts) == [
        ("gemini", OUTCOME.BAD_OUTPUT, False),
        ("groq", OUTCOME.SUCCESS, False),
    ]
    assert out.fallback_used


async def test_bad_output_is_proof_of_life_and_resets_the_failure_count() -> None:
    """A reply of any kind clears the run, so T, T, bad output, T, T opens nothing."""
    p = Probe("gemini", [timeout(), timeout(), BAD, BAD, timeout(), timeout()])
    q = Probe("groq", [GOOD] * 5)
    rig = Rig(p, q)
    for _ in range(5):
        await rig.ask()
    assert rig.status("gemini").state is BreakerState.CLOSED


@pytest.mark.parametrize(
    ("retry_after", "open_s"),
    [(30.0, 30.0), (None, 60.0), (0.0, 60.0), (-1.0, 60.0), (7200.0, 3600.0)],
)
async def test_a_rate_limit_opens_the_breaker_for_the_planned_time(
    retry_after: float | None, open_s: float
) -> None:
    """Tech §10 'RateLimited' row, D57b: the wait, 60 s without one (zero is none), capped."""
    p, q = Probe("gemini", [limited(retry_after)]), Probe("groq", [GOOD])
    rig = Rig(p, q)
    out = await rig.ask()
    assert (out.provider, out.fallback_used) == ("groq", True)  # immediately, in the same call
    assert trail(out.attempts) == [
        ("gemini", OUTCOME.RATE_LIMITED, False),
        ("groq", OUTCOME.SUCCESS, False),
    ]
    status = rig.status("gemini")
    assert status.state is BreakerState.OPEN
    assert status.remaining_s == pytest.approx(open_s)


async def test_a_429_breaker_skips_the_provider_then_goes_half_open_and_recovers() -> None:
    """Tech §10 'RateLimited', 'breaker open' and 'time elapsed' rows, end to end."""
    p = Probe("gemini", [limited(30.0), GOOD])
    q = Probe("groq", [GOOD, GOOD])
    rig = Rig(p, q)
    await rig.ask()
    rig.clock.advance(29.0)
    skipped = await rig.ask()
    assert trail(skipped.attempts)[0] == ("gemini", OUTCOME.SKIPPED_OPEN, False)
    assert len(p.calls) == 1  # no call while it is open
    rig.clock.advance(2.0)
    assert rig.status("gemini").state is BreakerState.HALF_OPEN
    recovered = await rig.ask()
    assert (recovered.provider, recovered.fallback_used) == ("gemini", False)
    assert rig.status("gemini") == BreakerStatus(BreakerState.CLOSED, None, False)


async def test_gemini_quota_opens_until_the_next_midnight_pacific_then_half_open() -> None:
    """Tech §10 'is_quota' row, D57b: midnight America/Los_Angeles on the wall clock."""
    p = Probe("gemini", [limited(None, quota=True), GOOD])
    q = Probe("groq", [GOOD])
    rig = Rig(p, q)  # the wall clock reads 05:00 in Los Angeles
    out = await rig.ask()
    assert trail(out.attempts)[0] == ("gemini", OUTCOME.QUOTA, False)
    assert rig.status("gemini").remaining_s == pytest.approx(19 * HOUR)
    rig.clock.advance(19 * HOUR - 1.0)
    assert rig.status("gemini").state is BreakerState.OPEN
    rig.clock.advance(2.0)
    assert rig.status("gemini").state is BreakerState.HALF_OPEN
    assert (await rig.ask()).provider == "gemini"  # the trial succeeds and closes it
    assert rig.status("gemini").state is BreakerState.CLOSED


@pytest.mark.parametrize(
    ("retry_after", "open_s"), [(120.0, 120.0), (7200.0, 3600.0), (None, 900.0)]
)
async def test_groq_quota_opens_for_its_own_wait_capped_else_the_default(
    retry_after: float | None, open_s: float
) -> None:
    """D57b: Groq's daily limit refills continuously, so its reset is its own retry-after."""
    p, q = Probe("groq", [limited(retry_after, quota=True)]), Probe("gemini", [GOOD])
    rig = Rig(p, q)
    await rig.ask()
    assert rig.status("groq").remaining_s == pytest.approx(open_s)


async def test_a_quota_with_an_unloadable_zone_opens_for_the_quota_default_and_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """7.01: no zone database means the reset is not derivable: 900 s and a logged warning."""

    def broken(key: str) -> Any:
        raise ZoneInfoNotFoundError(key)

    p, q = Probe("gemini", [limited(None, quota=True)]), Probe("groq", [GOOD])
    rig = Rig(p, q, load_zone=broken)
    await rig.ask()
    assert rig.status("gemini").remaining_s == pytest.approx(900.0)
    assert [r for r in caplog.records if r.getMessage() == LOG_ZONE_UNAVAILABLE]


@pytest.mark.parametrize("failure", [timeout, down], ids=["timeout", "unavailable"])
async def test_three_consecutive_failures_open_the_breaker_and_each_fails_over_at_once(
    failure: Any,
) -> None:
    """Tech §10 'Unavailable / Timeout' row: counted; 3 in a row within 60 s open it for 30 s."""
    p, q = Probe("gemini", [failure(), failure(), failure()]), Probe("groq", [GOOD] * 4)
    rig = Rig(p, q)
    for _ in range(3):
        out = await rig.ask()
        assert (out.provider, out.fallback_used) == ("groq", True)
    status = rig.status("gemini")
    assert status.state is BreakerState.OPEN
    assert status.remaining_s == pytest.approx(30.0)
    fourth = await rig.ask()  # skipped without a call: a fourth call would exhaust the script
    assert len(p.calls) == 3
    assert trail(fourth.attempts) == [
        ("gemini", OUTCOME.SKIPPED_OPEN, False),
        ("groq", OUTCOME.SUCCESS, False),
    ]


async def test_failures_outside_the_window_do_not_accumulate() -> None:
    """Tech §10 'Unavailable / Timeout' row: the 60 s window."""
    p, q = Probe("gemini", [timeout(), timeout(), timeout()]), Probe("groq", [GOOD] * 3)
    rig = Rig(p, q)
    await rig.ask()
    rig.clock.advance(40.0)
    await rig.ask()
    rig.clock.advance(61.0)  # both earlier failures are now older than the window
    await rig.ask()
    assert rig.status("gemini").state is BreakerState.CLOSED
    assert len(p.calls) == 3


async def test_an_open_breaker_is_skipped_without_a_call() -> None:
    """Tech §10 'P's breaker open' row: zero calls on the provider, recorded as a skip."""
    p, q = Probe("gemini", [limited(30.0)]), Probe("groq", [GOOD, GOOD])
    rig = Rig(p, q)
    await rig.ask()
    out = await rig.ask()
    assert (p.entries, len(p.calls)) == (1, 1)
    skip = out.attempts[0]
    assert (skip.provider, skip.outcome, skip.called) == ("gemini", OUTCOME.SKIPPED_OPEN, False)
    assert skip.usage == Usage(input_tokens=0, output_tokens=0)
    assert [a.provider for a in rig.seen] == ["gemini", "groq", "groq"]  # never for a skip


async def test_reading_the_status_never_claims_the_trial() -> None:
    p, q = Probe("gemini", [limited(30.0), GOOD]), Probe("groq", [GOOD])
    rig = Rig(p, q)
    await rig.ask()
    rig.clock.advance(31.0)
    for _ in range(2):
        assert rig.status("gemini") == BreakerStatus(BreakerState.HALF_OPEN, None, False)
    assert (await rig.ask()).provider == "gemini"


async def test_half_open_lets_exactly_one_trial_through_under_concurrent_requests(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Tech §10 'Breaker time elapsed' row: one trial; a concurrent request skips, not waits."""
    caplog.set_level(logging.INFO, logger="grounded.generation.router")
    p = Probe("gemini", [limited(30.0), GOOD])
    q = Probe("groq", [GOOD, GOOD, GOOD])
    rig = Rig(p, q)
    await rig.ask()  # opens gemini for 30 s
    rig.clock.advance(31.0)
    release = p.hold()
    trial = asyncio.create_task(rig.ask())
    await p.entered.wait()  # the trial is now inside gemini's call
    assert rig.status("gemini") == BreakerStatus(BreakerState.HALF_OPEN, None, True)

    # Two more requests arrive while the trial is in flight; gather them under a guard that turns
    # "queued behind the trial" into a failure instead of a hang (it is not a timing assertion).
    others = await asyncio.wait_for(asyncio.gather(rig.ask(), rig.ask()), timeout=5.0)
    for out in others:
        assert out.provider == "groq"
        assert trail(out.attempts) == [
            ("gemini", OUTCOME.SKIPPED_TRIAL, False),
            ("groq", OUTCOME.SUCCESS, False),
        ]
    assert p.entries == 2  # the 429 call and the one trial: nobody else got in
    assert not trial.done()
    skips = [r.__dict__ for r in caplog.records if r.getMessage() == LOG_PROVIDER_SKIPPED]
    assert [(s["provider"], s["reason"]) for s in skips] == [("gemini", "trial_in_flight")] * 2

    release.set()
    out = await trial
    assert (out.provider, out.fallback_used) == ("gemini", False)
    assert rig.status("gemini").state is BreakerState.CLOSED


async def test_a_failed_trial_opens_again_with_a_fresh_timer_and_a_good_one_closes() -> None:
    """Tech §10 'Breaker time elapsed' row: success -> closed, failure -> open again."""
    p = Probe("gemini", [limited(30.0), timeout(), GOOD])
    q = Probe("groq", [GOOD, GOOD, GOOD])
    rig = Rig(p, q)
    await rig.ask()
    rig.clock.advance(31.0)
    failed_trial = await rig.ask()
    assert trail(failed_trial.attempts)[0] == ("gemini", OUTCOME.TIMEOUT, False)
    assert rig.status("gemini").remaining_s == pytest.approx(30.0)  # a fresh timer, not a leftover
    rig.clock.advance(29.0)
    still_open = await rig.ask()
    assert trail(still_open.attempts)[0] == ("gemini", OUTCOME.SKIPPED_OPEN, False)
    rig.clock.advance(2.0)
    recovered = await rig.ask()
    assert (recovered.provider, rig.status("gemini").state) == ("gemini", BreakerState.CLOSED)


async def test_all_providers_open_raises_with_the_minimum_remaining_time() -> None:
    """Tech §10 'All providers open/failed' row: ``retry_after_s`` is the smallest time left."""
    p, q = Probe("gemini", [limited(60.0)]), Probe("groq", [GOOD, limited(20.0)])
    rig = Rig(p, q)
    await rig.ask()  # gemini opens for 60 s; groq answers
    rig.clock.advance(10.0)
    with pytest.raises(AllProvidersUnavailable) as info:
        await rig.ask()  # gemini is skipped (50 s left), groq answers 429 and opens for 20 s
    assert isinstance(info.value, ProviderError)
    assert info.value.retry_after_s == pytest.approx(
        20.0
    )  # the breaker this very call opened counts
    assert trail(info.value.attempts) == [
        ("gemini", OUTCOME.SKIPPED_OPEN, False),
        ("groq", OUTCOME.RATE_LIMITED, False),
    ]
    rig.clock.advance(5.0)
    with pytest.raises(AllProvidersUnavailable) as again:
        await rig.ask()  # both skipped without a call
    assert again.value.retry_after_s == pytest.approx(15.0)
    assert (len(p.calls), len(q.calls)) == (1, 2)


async def test_every_provider_failing_this_call_with_nothing_open_has_no_retry_after() -> None:
    """The case the table leaves open: nothing to wait for, so ``retry_after_s`` is ``None``."""
    p, q = Probe("gemini", [timeout()]), Probe("groq", [down()])
    with pytest.raises(AllProvidersUnavailable) as info:
        await Rig(p, q).ask()
    assert info.value.retry_after_s is None
    assert trail(info.value.attempts) == [
        ("gemini", OUTCOME.TIMEOUT, False),
        ("groq", OUTCOME.UNAVAILABLE, False),
    ]


async def test_a_half_open_provider_with_its_trial_taken_is_not_a_time_to_wait_for() -> None:
    """A trial in flight has no remaining time: with nothing else open there is no wait."""
    p, q = Probe("gemini", [limited(30.0), GOOD]), Probe("groq", [GOOD, timeout()])
    rig = Rig(p, q)
    await rig.ask()
    rig.clock.advance(31.0)
    release = p.hold()
    trial = asyncio.create_task(rig.ask())
    await p.entered.wait()
    with pytest.raises(AllProvidersUnavailable) as info:
        await asyncio.wait_for(rig.ask(), timeout=5.0)  # gemini: trial taken, groq: times out
    assert info.value.retry_after_s is None
    release.set()
    await trial


async def test_when_nothing_answered_only_bad_output_raises_the_last_provider_bad_output() -> None:
    """§9.5 step 5: 502, not 503, when the providers answered but never validly."""
    p, q = Probe("gemini", [BAD, BAD]), Probe("groq", [BAD, BAD])
    with pytest.raises(ProviderBadOutput) as info:
        await Rig(p, q).ask()
    assert info.value.validation_error == BAD_REASON
    assert (len(p.calls), len(q.calls)) == (2, 2)  # one retry on each, no more


async def test_a_validate_rejection_that_survives_the_retry_raises_provider_bad_output() -> None:
    p = Probe("gemini", [GOOD, GOOD])
    with pytest.raises(ProviderBadOutput) as info:
        await Rig(p).ask(validate=lambda result: "no valid citation")
    assert info.value.validation_error == "no valid citation"
    assert info.value.raw == GOOD.model_dump_json()


async def test_bad_output_next_to_a_transient_failure_is_unavailable_not_bad_output() -> None:
    p, q = Probe("gemini", [BAD, BAD]), Probe("groq", [timeout()])
    with pytest.raises(AllProvidersUnavailable) as info:
        await Rig(p, q).ask()
    assert info.value.retry_after_s is None


async def test_a_single_entry_list_is_the_primary_only_with_no_fallback() -> None:
    """Tech §10 'Eval mode': ``Settings`` allows one provider there; the router routes over what it
    is given, so one entry means no fallback and ``fallback_used`` stays false."""
    p = Probe("gemini", [GOOD, limited(30.0)])
    rig = Rig(p)
    out = await rig.ask()
    assert (out.provider, out.fallback_used) == ("gemini", False)
    with pytest.raises(AllProvidersUnavailable) as info:
        await rig.ask()
    assert info.value.retry_after_s == pytest.approx(30.0)
    assert trail(info.value.attempts) == [("gemini", OUTCOME.RATE_LIMITED, False)]


async def test_fallback_used_is_true_when_the_answering_provider_is_not_the_first() -> None:
    """Tech §10 requirements: the router reports ``fallback_used``, provider, model, attempts."""
    p, q = Probe("gemini", [timeout()]), Probe("groq", [GOOD])
    out = await Rig(p, q).ask()
    assert (out.provider, out.model, out.fallback_used) == ("groq", "groq-model", True)
    assert out.result.provider == "groq"
    assert [a.outcome for a in out.attempts] == [OUTCOME.TIMEOUT, OUTCOME.SUCCESS]


async def test_fallback_used_is_also_true_when_the_first_provider_was_only_skipped() -> None:
    p, q = Probe("gemini", [limited(30.0)]), Probe("groq", [GOOD, GOOD])
    rig = Rig(p, q)
    await rig.ask()
    out = await rig.ask()  # gemini is skipped without a call, groq answers
    assert (out.provider, out.fallback_used) == ("groq", True)


async def test_attempts_carry_latency_usage_and_the_retry_flag() -> None:
    """The record the pipeline accounts from: usage also of a bad output that billed tokens."""
    clock = FakeClock()
    usage = Usage(input_tokens=11, output_tokens=7, thinking_tokens=2)
    cut = ProviderBadOutput(
        "cut off", validation_error="write less", input_tokens=5, output_tokens=3
    )
    p = Probe("gemini", [cut, GOOD], clock=clock, takes_s=2.5, usage=usage)
    rig = Rig(p, clock=clock)
    out = await rig.ask()
    first, second = out.attempts
    assert (first.latency_ms, first.usage, first.is_retry) == (
        2500,
        Usage(input_tokens=5, output_tokens=3),
        False,
    )
    assert (second.latency_ms, second.usage, second.is_retry) == (2500, usage, True)
    assert rig.seen == list(out.attempts)
    assert p.calls[1].user == feedback("question", "write less")


async def test_a_rejected_reply_still_bills_and_keeps_the_cache_flag() -> None:
    """A reply ``validate`` rejects carries its own usage and ``cache_hit`` (the eval cache)."""
    usage = Usage(input_tokens=11, output_tokens=7, thinking_tokens=2)
    p = Probe("gemini", [GOOD, GOOD], usage=usage, cache_hit=True)
    verdicts = iter(["no valid citation", None])
    out = await Rig(p).ask(validate=lambda result: next(verdicts))
    first = out.attempts[0]
    assert (first.outcome, first.usage, first.cache_hit) == (OUTCOME.BAD_OUTPUT, usage, True)


async def test_the_router_never_sleeps_and_never_waits_out_a_retry_after(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tech §10, AGENTS.md §6.4: fallback is immediate; a Retry-After is only remembered."""

    async def no_sleep(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("the router must not sleep")

    monkeypatch.setattr(asyncio, "sleep", no_sleep)
    p, q = Probe("gemini", [limited(30.0)]), Probe("groq", [GOOD])
    rig = Rig(p, q)
    started = rig.clock.now
    out = await asyncio.wait_for(rig.ask(), timeout=5.0)
    assert out.provider == "groq"
    assert rig.clock.now == started  # nothing moved the clock: the 30 s was not waited out
    alone = Rig(Probe("gemini", [limited(30.0)]))
    with pytest.raises(AllProvidersUnavailable):  # with nobody left it raises at once
        await asyncio.wait_for(alone.ask(), timeout=5.0)


# --- Requests the provider rejected (a 4xx other than 429) --------------------------------------


async def test_a_rejected_request_fails_over_and_never_opens_the_breaker() -> None:
    """Router docstring: our key or request, not the provider's health, so a bad key on the primary
    does not take the service down while a fallback exists."""
    rejected = ProviderRequestRejected("bad key", status_code=401)
    p = Probe("gemini", [rejected, rejected, rejected, GOOD])
    q = Probe("groq", [GOOD] * 3)
    rig = Rig(p, q)
    for _ in range(3):
        out = await rig.ask()
        assert (out.provider, out.fallback_used) == ("groq", True)
        assert trail(out.attempts) == [
            ("gemini", OUTCOME.REJECTED, False),
            ("groq", OUTCOME.SUCCESS, False),
        ]
    assert rig.status("gemini").state is BreakerState.CLOSED
    assert (await rig.ask()).provider == "gemini"  # a fourth time it is still called, never skipped


async def test_a_rejected_request_is_logged_with_provider_and_status_code_only(
    caplog: pytest.LogCaptureFixture,
) -> None:
    rejected = ProviderRequestRejected("API key SECRET-KEY-TEXT is not valid", status_code=401)
    p, q = Probe("gemini", [rejected]), Probe("groq", [GOOD])
    await Rig(p, q).ask()
    [record] = [r for r in caplog.records if r.getMessage() == LOG_REQUEST_REJECTED]
    assert record.levelno == logging.WARNING
    assert (record.__dict__["provider"], record.__dict__["status_code"]) == ("gemini", 401)
    assert "SECRET-KEY-TEXT" not in str(record.__dict__) + caplog.text


async def test_when_every_provider_rejects_the_first_rejection_propagates_unchanged() -> None:
    """Loud on purpose (a 500 in 7.04): a Retry-After would invite retries that cannot work."""
    first = ProviderRequestRejected("bad key", status_code=401)
    second = ProviderRequestRejected("bad model", status_code=404)
    rig = Rig(Probe("gemini", [first]), Probe("groq", [second]))
    with pytest.raises(ProviderRequestRejected) as info:
        await rig.ask()
    assert info.value is first
    assert [rig.status(name).state for name in ("gemini", "groq")] == [BreakerState.CLOSED] * 2


async def test_a_rejected_request_next_to_a_transient_failure_is_unavailable() -> None:
    rejected = ProviderRequestRejected("bad key", status_code=401)
    with pytest.raises(AllProvidersUnavailable):
        await Rig(Probe("gemini", [rejected]), Probe("groq", [timeout()])).ask()


async def test_a_rejected_trial_leaves_the_breaker_half_open_with_the_trial_free() -> None:
    rejected = ProviderRequestRejected("bad key", status_code=401)
    p, q = Probe("gemini", [limited(30.0), rejected]), Probe("groq", [GOOD, GOOD])
    rig = Rig(p, q)
    await rig.ask()
    rig.clock.advance(31.0)
    await rig.ask()  # the trial is rejected: no verdict either way
    assert rig.status("gemini") == BreakerStatus(BreakerState.HALF_OPEN, None, False)


class OtherProviderError(ProviderError):
    """A provider error the router has no rule for."""


async def test_a_provider_error_the_router_has_no_rule_for_propagates() -> None:
    """Only the five typed errors are routed; a new kind stays loud, not a quiet fallback."""
    odd = OtherProviderError("a new kind of failure")
    p, q = Probe("gemini", [odd]), Probe("groq", [GOOD])
    with pytest.raises(OtherProviderError) as info:
        await Rig(p, q).ask()
    assert info.value is odd
    assert q.entries == 0


async def test_the_open_time_counts_from_when_the_failure_is_seen() -> None:
    """The call took 5 s before the 429 arrived; the 30 s start there, not when the call began."""
    clock = FakeClock()
    p = Probe("gemini", [limited(30.0)], clock=clock, takes_s=5.0)
    rig = Rig(p, Probe("groq", [GOOD]), clock=clock)
    await rig.ask()
    assert rig.status("gemini").remaining_s == pytest.approx(30.0)


# --- Hooks and errors that must not be mistaken for a provider failure ---------------------------


async def test_the_hooks_run_in_the_documented_order() -> None:
    """before_call, the call, validate (only when there was a reply), on_attempt; per call."""
    log: list[str] = []
    p = Probe("gemini", [BAD, GOOD], log=log)

    async def before_call() -> None:
        log.append("before")

    def validate(result: GenerationResult[LLMAnswer]) -> str | None:
        log.append("validate")
        return None

    async def on_attempt(attempt: Attempt) -> None:
        log.append(f"attempt:{attempt.outcome.value}")

    await Rig(p).ask(before_call=before_call, validate=validate, on_attempt=on_attempt)
    assert log == [
        "before",
        "call:gemini",
        "attempt:bad_output",
        "before",
        "call:gemini",
        "validate",
        "attempt:success",
    ]


async def test_before_call_runs_for_every_call_but_not_for_a_skip() -> None:
    """The budget reserve is per call: retries and fallbacks included (5.05), skips excluded."""
    reserved = 0

    async def before_call() -> None:
        nonlocal reserved
        reserved += 1

    p, q = Probe("gemini", [BAD, BAD]), Probe("groq", [GOOD])
    rig = Rig(p, q)
    await rig.ask(before_call=before_call)
    assert reserved == len(p.calls) + len(q.calls) == 3
    opened = Rig(Probe("gemini", [limited(30.0)]), Probe("groq", [GOOD, GOOD]))
    await opened.ask()
    reserved = 0
    await opened.ask(before_call=before_call)  # gemini is skipped: only groq's call reserves
    assert reserved == 1


class BudgetGoneError(Exception):
    """Stands for the budget's ``BudgetExhaustedError``: a hook error, not a provider's."""


class BugError(Exception):
    """A bug in a hook or an adapter. Not a ``RuntimeError`` on purpose: ``NotImplementedError``
    is one, so ``pytest.raises(RuntimeError)`` would catch the stub's and fake a pass."""


async def test_a_before_call_error_propagates_and_is_not_a_provider_failure() -> None:
    """The reserve refuses the fallback's call: the error passes through as it is, the earlier
    call's attempt was already reported (its usage is not lost), and no provider is blamed."""
    allowed = deque([True, False])

    async def before_call() -> None:
        if not allowed.popleft():
            raise BudgetGoneError

    p, q = Probe("gemini", [timeout()]), Probe("groq", [GOOD])
    rig = Rig(p, q)
    with pytest.raises(BudgetGoneError):
        await rig.ask(before_call=before_call)
    assert q.entries == 0
    assert [a.outcome for a in rig.seen] == [OUTCOME.TIMEOUT]
    assert rig.status("groq").state is BreakerState.CLOSED


async def test_a_before_call_error_makes_no_call_and_opens_nothing() -> None:
    async def before_call() -> None:
        raise BudgetGoneError

    p = Probe("gemini", [GOOD])
    rig = Rig(p)
    for _ in range(5):
        with pytest.raises(BudgetGoneError):
            await rig.ask(before_call=before_call)
    assert p.entries == 0
    assert rig.status("gemini").state is BreakerState.CLOSED


async def test_a_before_call_error_releases_a_held_trial() -> None:
    async def before_call() -> None:
        raise BudgetGoneError

    p, q = Probe("gemini", [limited(30.0), GOOD]), Probe("groq", [GOOD])
    rig = Rig(p, q)
    await rig.ask()
    rig.clock.advance(31.0)
    with pytest.raises(BudgetGoneError):
        await rig.ask(before_call=before_call)
    assert rig.status("gemini") == BreakerStatus(BreakerState.HALF_OPEN, None, False)
    assert (await rig.ask()).provider == "gemini"


async def test_a_validate_error_propagates_and_is_never_counted_against_the_provider() -> None:
    """Four hook errors on a threshold of three: the breaker must still be closed."""

    def validate(result: GenerationResult[LLMAnswer]) -> str | None:
        raise BugError("bug in the checks")

    p = Probe("gemini", [GOOD] * 4)
    rig = Rig(p)
    for _ in range(4):
        with pytest.raises(BugError, match="bug in the checks"):
            await rig.ask(validate=validate)
    assert (len(p.calls), rig.status("gemini").state) == (4, BreakerState.CLOSED)


async def test_the_breaker_hears_of_the_reply_before_validate_runs() -> None:
    """A hook error after a trial's reply must not leave the trial claimed: the reply closed it."""

    def validate(result: GenerationResult[LLMAnswer]) -> str | None:
        raise BugError("bug in the checks")

    p, q = Probe("gemini", [limited(30.0), GOOD]), Probe("groq", [GOOD])
    rig = Rig(p, q)
    await rig.ask()
    rig.clock.advance(31.0)
    with pytest.raises(BugError):
        await rig.ask(validate=validate)
    assert rig.status("gemini") == BreakerStatus(BreakerState.CLOSED, None, False)


async def test_an_on_attempt_error_propagates_and_no_further_provider_is_tried() -> None:
    async def on_attempt(attempt: Attempt) -> None:
        raise BugError("could not record the usage")

    p, q = Probe("gemini", [timeout()]), Probe("groq", [GOOD])
    with pytest.raises(BugError, match="could not record"):
        await Rig(p, q).ask(on_attempt=on_attempt)
    assert q.entries == 0


async def test_an_unexpected_exception_propagates_and_releases_the_trial() -> None:
    """Anything that is not one of the five typed provider errors is a bug and must stay loud."""
    p, q = Probe("gemini", [limited(30.0), GOOD]), Probe("groq", [GOOD])
    rig = Rig(p, q)
    await rig.ask()
    rig.clock.advance(31.0)
    p.crash_next = BugError("bug in an adapter")
    with pytest.raises(BugError, match="bug in an adapter"):
        await rig.ask()
    assert rig.status("gemini") == BreakerStatus(BreakerState.HALF_OPEN, None, False)
    assert (await rig.ask()).provider == "gemini"  # the next request takes the trial


async def test_a_cancelled_trial_is_released() -> None:
    """The request deadline cancels a call in flight: the trial must not stay claimed forever."""
    p, q = Probe("gemini", [limited(30.0), GOOD]), Probe("groq", [GOOD])
    rig = Rig(p, q)
    await rig.ask()
    rig.clock.advance(31.0)
    release = p.hold()
    task = asyncio.create_task(rig.ask())
    await p.entered.wait()
    assert rig.status("gemini").trial_in_flight is True
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert rig.status("gemini") == BreakerStatus(BreakerState.HALF_OPEN, None, False)
    release.set()
    assert (await rig.ask()).provider == "gemini"
    assert p.entries == 3  # the 429, the cancelled trial and the new one


# --- The deadline ---------------------------------------------------------------------------------


async def test_the_router_does_not_start_an_attempt_the_deadline_cannot_hold(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """7.01 note 3: a bad output and its retry at 12 s each leave 1 s of 25: groq is not tried."""
    caplog.set_level(logging.INFO, logger="grounded.generation.router")
    clock = FakeClock()
    p = Probe("gemini", [BAD, BAD], clock=clock, takes_s=12.0)
    q = Probe("groq", [GOOD], clock=clock, takes_s=3.0)
    rig = Rig(p, q, clock=clock)
    with pytest.raises(RouterDeadlineError) as info:
        await rig.ask(time_left=rig.time_left(25.0))
    assert len(p.calls) == 2  # the retry started with 13 s left, which holds a 12 s attempt
    assert (q.entries, q.calls) == (0, [])
    assert trail(info.value.attempts) == [
        ("gemini", OUTCOME.BAD_OUTPUT, False),
        ("gemini", OUTCOME.BAD_OUTPUT, True),
        ("groq", OUTCOME.SKIPPED_DEADLINE, False),
    ]
    assert info.value.time_left_s == pytest.approx(1.0)
    assert info.value.timeout_s == 12.0
    assert not isinstance(info.value, ProviderError)
    assert [a.outcome for a in rig.seen] == [OUTCOME.BAD_OUTPUT, OUTCOME.BAD_OUTPUT]
    [skip] = [r for r in caplog.records if r.getMessage() == LOG_PROVIDER_SKIPPED]
    assert (skip.__dict__["provider"], skip.__dict__["reason"]) == ("groq", "deadline")


async def test_a_timeout_followed_by_a_fallback_still_fits_the_deadline() -> None:
    """After a 12 s timeout 13 s are left, enough for a 12 s attempt, which is not shortened."""
    clock = FakeClock()
    p = Probe("gemini", [timeout()], clock=clock, takes_s=12.0)
    q = Probe("groq", [GOOD], clock=clock, takes_s=3.0)
    rig = Rig(p, q, clock=clock)
    out = await rig.ask(time_left=rig.time_left(25.0))
    assert (out.provider, out.fallback_used) == ("groq", True)
    assert q.calls[0].timeout_s == 12.0


async def test_the_retry_is_gated_by_the_deadline_too() -> None:
    clock = FakeClock()
    p = Probe("gemini", [BAD, GOOD], clock=clock, takes_s=20.0)
    rig = Rig(p, Probe("groq", [GOOD]), clock=clock)
    with pytest.raises(RouterDeadlineError) as info:
        await rig.ask(time_left=rig.time_left(25.0))
    assert len(p.calls) == 1
    assert trail(info.value.attempts) == [
        ("gemini", OUTCOME.BAD_OUTPUT, False),
        ("gemini", OUTCOME.SKIPPED_DEADLINE, True),
    ]


async def test_without_a_deadline_slow_attempts_are_never_refused() -> None:
    """``time_left`` is optional (eval mode, the CLI): no deadline, no refusal."""
    clock = FakeClock()
    p = Probe("gemini", [BAD, BAD], clock=clock, takes_s=12.0)
    q = Probe("groq", [GOOD], clock=clock, takes_s=12.0)
    out = await Rig(p, q, clock=clock).ask()
    assert (out.provider, out.fallback_used) == ("groq", True)


async def test_a_deadline_refusal_does_not_consume_the_half_open_trial() -> None:
    p, q = Probe("gemini", [limited(30.0), GOOD]), Probe("groq", [GOOD])
    rig = Rig(p, q)
    await rig.ask()
    rig.clock.advance(31.0)
    with pytest.raises(RouterDeadlineError):
        await rig.ask(time_left=lambda: 1.0)
    assert p.entries == 1  # no call was made for the refused request
    assert rig.status("gemini") == BreakerStatus(BreakerState.HALF_OPEN, None, False)
    out = await rig.ask()  # the trial is still there
    assert (out.provider, rig.status("gemini").state) == ("gemini", BreakerState.CLOSED)


# --- Logging --------------------------------------------------------------------------------------


async def test_every_open_and_every_skip_is_logged_without_the_question(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """7.01 note 1: provider, reason, seconds and how derived; never the question."""
    caplog.set_level(logging.INFO, logger="grounded.generation.router")
    p, q = Probe("gemini", [limited(30.0)]), Probe("groq", [GOOD, GOOD])
    rig = Rig(p, q)
    await rig.ask(user="SECRET-QUESTION")
    await rig.ask(user="SECRET-QUESTION")
    [opened] = opened_logs(caplog)
    assert opened["levelname"] == "WARNING"
    assert (opened["provider"], opened["reason"], opened["derived"]) == (
        "gemini",
        "rate_limit",
        "retry_after",
    )
    assert opened["open_s"] == pytest.approx(30.0)
    [skipped] = [r for r in caplog.records if r.getMessage() == LOG_PROVIDER_SKIPPED]
    assert skipped.levelno == logging.INFO
    assert (skipped.__dict__["provider"], skipped.__dict__["reason"]) == ("gemini", "breaker_open")
    assert skipped.__dict__["remaining_s"] == pytest.approx(30.0)
    assert all("SECRET-QUESTION" not in str(r.__dict__) for r in caplog.records)


async def test_opening_after_repeated_failures_is_logged_with_its_reason(
    caplog: pytest.LogCaptureFixture,
) -> None:
    p, q = Probe("gemini", [timeout()] * 3), Probe("groq", [GOOD] * 3)
    rig = Rig(p, q)
    for _ in range(3):
        await rig.ask()
    [opened] = opened_logs(caplog)
    assert (opened["provider"], opened["reason"], opened["derived"]) == (
        "gemini",
        "failures",
        "default",
    )
    assert opened["open_s"] == pytest.approx(30.0)


async def test_a_quota_open_is_logged_with_the_derived_reset(
    caplog: pytest.LogCaptureFixture,
) -> None:
    p, q = Probe("gemini", [limited(None, quota=True)]), Probe("groq", [GOOD])
    await Rig(p, q).ask()
    [opened] = opened_logs(caplog)
    assert (opened["reason"], opened["derived"]) == ("quota", "quota_reset")
    assert opened["open_s"] == pytest.approx(19 * HOUR)


# --- Construction ---------------------------------------------------------------------------------


def test_a_router_needs_at_least_one_provider() -> None:
    with pytest.raises(RouterConfigError):
        ProviderRouter([], config=CONFIG)


def test_provider_names_must_be_unique() -> None:
    """Breakers are keyed by name, so two providers of one name would share (and corrupt) one."""
    with pytest.raises(RouterConfigError, match="gemini"):
        ProviderRouter([Probe("gemini"), Probe("gemini")], config=CONFIG)


def test_the_status_of_an_unknown_provider_is_a_typed_error() -> None:
    rig = Rig(Probe("gemini"))
    with pytest.raises(UnknownProviderError):
        rig.status("nope")
