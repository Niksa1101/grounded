"""Tests added in the review round of the router (ticket 7.03, PR #126): the typed errors kept in
``AllProvidersUnavailable.causes``, the router with its breakers off (eval mode), the narrower
``retry_after_s``, the ``log_extra`` that reaches every log line, and the one INFO line per failed
call. Nothing here changes or relaxes a spec test; the helpers (``Probe``, ``Rig``, ``trail``...)
are the spec file's.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pytest

from grounded.generation.router import (
    LOG_ATTEMPT_FAILED,
    LOG_BREAKER_OPENED,
    LOG_PROVIDER_SKIPPED,
    LOG_REQUEST_REJECTED,
    LOG_ZONE_UNAVAILABLE,
    AllProvidersUnavailable,
    AttemptOutcome,
    BreakerState,
    BreakerStatus,
    LogExtraError,
    ProviderRouter,
    RouterClock,
    RouterDeadlineError,
    ZoneLoader,
)
from grounded.infra.provider_errors import (
    ProviderBadOutput,
    ProviderRateLimited,
    ProviderRequestRejected,
)
from tests.support import FakeClock
from tests.unit.test_router import (
    BAD,
    BAD_REASON,
    CONFIG,
    GOOD,
    Probe,
    Rig,
    down,
    feedback,
    limited,
    timeout,
    trail,
)

OUTCOME = AttemptOutcome
ROUTER_LOGGER = "grounded.generation.router"
HOUR = 3600.0

# What every log record has by itself: whatever else is on a record came from ``extra=``.
_STANDARD = frozenset(logging.LogRecord("", 0, "", 0, "", None, None).__dict__) | {
    "message",
    "asctime",
}


def extras(record: logging.LogRecord) -> dict[str, Any]:
    return {key: value for key, value in record.__dict__.items() if key not in _STANDARD}


def router_records(caplog: pytest.LogCaptureFixture, message: str) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.name == ROUTER_LOGGER and r.getMessage() == message]


class EvalRig(Rig):
    """A ``Rig`` whose router has its breakers off, as the eval runtime will build it (7.04)."""

    def __init__(self, *providers: Probe, load_zone: ZoneLoader = ZoneInfo) -> None:
        super().__init__(*providers, load_zone=load_zone)
        self.router = ProviderRouter(
            providers,
            config=CONFIG,
            clock=RouterClock(monotonic=self.clock, wall=self.wall, load_zone=load_zone),
            breakers_enabled=False,
        )


# --- AllProvidersUnavailable.causes ---------------------------------------------------------------


def test_the_causes_default_to_none() -> None:
    assert AllProvidersUnavailable("x", retry_after_s=None).causes == ()


async def test_the_typed_error_of_each_transient_entry_is_kept_in_order_and_chained() -> None:
    first, second = limited(30.0), timeout()
    with pytest.raises(AllProvidersUnavailable) as info:
        await Rig(Probe("gemini", [first]), Probe("groq", [second])).ask()
    error = info.value
    assert len(error.causes) == 2
    assert error.causes[0] is first  # the very objects the providers raised
    assert error.causes[1] is second
    assert error.__cause__ is second  # the last one


async def test_an_entry_skipped_without_a_call_has_no_cause() -> None:
    late = timeout()
    rig = Rig(Probe("gemini", [limited(30.0)]), Probe("groq", [GOOD, late]))
    await rig.ask()  # opens gemini's breaker; groq answers
    with pytest.raises(AllProvidersUnavailable) as info:
        await rig.ask()  # gemini is skipped, groq times out
    assert info.value.causes == (late,)
    assert trail(info.value.attempts)[0] == ("gemini", OUTCOME.SKIPPED_OPEN, False)


async def test_when_everything_is_skipped_there_is_no_cause_to_chain() -> None:
    rig = Rig(Probe("gemini", [limited(30.0)]))
    with pytest.raises(AllProvidersUnavailable):
        await rig.ask()
    with pytest.raises(AllProvidersUnavailable) as info:
        await rig.ask()
    assert info.value.causes == ()
    assert info.value.__cause__ is None


async def test_bad_output_and_a_rejection_are_not_causes() -> None:
    rejected = ProviderRequestRejected("bad key", status_code=401)
    gone = timeout()
    rig = Rig(Probe("gemini", [BAD, BAD]), Probe("groq", [rejected]), Probe("fake", [gone]))
    with pytest.raises(AllProvidersUnavailable) as info:
        await rig.ask()
    assert info.value.causes == (gone,)


async def test_a_retry_that_fails_transiently_leaves_that_error_as_the_cause() -> None:
    gone = timeout()
    with pytest.raises(AllProvidersUnavailable) as info:
        await Rig(Probe("gemini", [BAD, gone])).ask()
    assert info.value.causes == (gone,)
    assert trail(info.value.attempts) == [
        ("gemini", OUTCOME.BAD_OUTPUT, False),
        ("gemini", OUTCOME.TIMEOUT, True),
    ]


# --- breakers_enabled=False (eval mode) -----------------------------------------------------------


class BackoffExhausted(ProviderRateLimited):
    """Stands for the eval backoff's error: a ``ProviderRateLimited`` with more on it."""

    def __init__(self) -> None:
        super().__init__("gave up", retry_after_s=None, is_quota=True)
        self.waited_s = 120.0


@pytest.mark.parametrize(
    "steps",
    [
        pytest.param([limited(30.0)], id="rate-limit"),
        pytest.param([limited(None, quota=True)], id="daily-quota"),
        pytest.param([BackoffExhausted()], id="backoff-exhausted"),
        pytest.param([timeout(), timeout(), timeout(), timeout()], id="timeouts"),
        pytest.param([down(), down(), down(), down()], id="outage"),
    ],
)
async def test_without_breakers_a_failure_never_makes_the_next_request_skip(
    steps: list[Any], caplog: pytest.LogCaptureFixture
) -> None:
    """The eval must see every real error: after any failure the next question is really asked."""
    caplog.set_level(logging.INFO, logger=ROUTER_LOGGER)
    p = Probe("gemini", [*steps, GOOD])
    rig = EvalRig(p)
    for step in steps:
        with pytest.raises(AllProvidersUnavailable) as info:
            await rig.ask()
        assert info.value.causes == (step,)
        assert info.value.causes[-1] is step  # the harness re-raises the provider's own error
        assert info.value.retry_after_s is None  # nothing to wait for: nothing is open
        assert rig.status("gemini") == BreakerStatus(BreakerState.CLOSED, None, False)
    out = await rig.ask()
    assert trail(out.attempts) == [("gemini", OUTCOME.SUCCESS, False)]
    assert len(p.calls) == len(steps) + 1  # every question reached the provider
    breaker_lines = (LOG_BREAKER_OPENED, LOG_PROVIDER_SKIPPED)
    assert [r for r in caplog.records if r.getMessage() in breaker_lines] == []


async def test_without_breakers_bad_output_is_still_retried_with_feedback_and_falls_back() -> None:
    p, q = Probe("gemini", [BAD, BAD]), Probe("groq", [GOOD])
    out = await EvalRig(p, q).ask()
    assert (out.provider, out.fallback_used) == ("groq", True)
    assert p.calls[1].user == feedback("question", BAD_REASON)
    assert trail(out.attempts) == [
        ("gemini", OUTCOME.BAD_OUTPUT, False),
        ("gemini", OUTCOME.BAD_OUTPUT, True),
        ("groq", OUTCOME.SUCCESS, False),
    ]


async def test_without_breakers_a_rejected_request_still_fails_over(
    caplog: pytest.LogCaptureFixture,
) -> None:
    rejected = ProviderRequestRejected("bad key", status_code=401)
    out = await EvalRig(Probe("gemini", [rejected]), Probe("groq", [GOOD])).ask()
    assert out.provider == "groq"
    assert len(router_records(caplog, LOG_REQUEST_REJECTED)) == 1


async def test_without_breakers_the_deadline_gate_still_refuses() -> None:
    p = Probe("gemini", [GOOD])
    with pytest.raises(RouterDeadlineError) as info:
        await EvalRig(p).ask(time_left=lambda: 1.0)
    assert p.entries == 0
    assert trail(info.value.attempts) == [("gemini", OUTCOME.SKIPPED_DEADLINE, False)]


async def test_without_breakers_a_quota_429_derives_no_reset_and_logs_no_zone_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    def broken(key: str) -> Any:
        raise ZoneInfoNotFoundError(key)

    rig = EvalRig(Probe("gemini", [limited(None, quota=True)]), load_zone=broken)
    with pytest.raises(AllProvidersUnavailable):
        await rig.ask()
    assert router_records(caplog, LOG_ZONE_UNAVAILABLE) == []


# --- retry_after_s: only when no entry is admissible now ------------------------------------------


async def test_retry_after_is_none_when_another_entry_can_be_tried_now() -> None:
    """Gemini is out for hours, but Groq's breaker is closed: a retry may succeed at once."""
    p, q = Probe("gemini", [limited(None, quota=True)]), Probe("groq", [GOOD, timeout()])
    rig = Rig(p, q)
    await rig.ask()
    assert rig.status("gemini").remaining_s == pytest.approx(19 * HOUR)
    with pytest.raises(AllProvidersUnavailable) as info:
        await rig.ask()  # gemini is skipped, groq times out once and stays closed
    assert info.value.retry_after_s is None


async def test_retry_after_is_none_when_a_half_open_provider_has_its_trial_free() -> None:
    rejected = ProviderRequestRejected("bad key", status_code=401)
    p, q = Probe("gemini", [limited(30.0), rejected]), Probe("groq", [limited(600.0)])
    rig = Rig(p, q)
    with pytest.raises(AllProvidersUnavailable) as both_open:
        await rig.ask()
    assert both_open.value.retry_after_s == pytest.approx(30.0)  # nothing admissible: the minimum
    rig.clock.advance(31.0)
    with pytest.raises(AllProvidersUnavailable) as info:
        await rig.ask()  # gemini's trial is rejected (given back), groq is skipped with 569 s left
    assert rig.status("gemini") == BreakerStatus(BreakerState.HALF_OPEN, None, False)
    assert info.value.retry_after_s is None  # gemini can be tried now


async def test_retry_after_is_the_open_minimum_when_the_only_other_entry_is_a_trial_in_flight() -> (
    None
):
    p, q = Probe("gemini", [limited(60.0)]), Probe("groq", [limited(30.0), GOOD])
    rig = Rig(p, q)
    with pytest.raises(AllProvidersUnavailable):
        await rig.ask()  # both open: gemini 60 s, groq 30 s
    rig.clock.advance(31.0)  # gemini has 29 s left, groq is half-open
    release = q.hold()
    trial = asyncio.create_task(rig.ask())  # groq's trial; gemini is skipped
    await q.entered.wait()
    with pytest.raises(AllProvidersUnavailable) as info:
        await asyncio.wait_for(rig.ask(), timeout=5.0)  # gemini open, groq's trial taken
    assert info.value.retry_after_s == pytest.approx(29.0)
    release.set()
    await trial


# --- log_extra ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "key",
    [
        # the router's own fields
        *("provider", "reason", "open_s", "derived", "remaining_s", "status_code", "zone"),
        *("outcome", "is_retry", "latency_ms"),
        # attributes every log record has, and the two ``logging`` fills in
        *("message", "asctime", "levelname", "name", "msg", "args", "exc_info", "taskName"),
    ],
)
async def test_a_log_extra_key_that_clashes_is_refused_before_any_call(key: str) -> None:
    p = Probe("gemini", [GOOD])
    with pytest.raises(LogExtraError, match=key):
        await Rig(p).ask(log_extra={"request_id": "r-1", key: "x"})
    assert p.entries == 0


async def test_log_extra_may_carry_other_keys() -> None:
    out = await Rig(Probe("gemini", [GOOD])).ask(log_extra={"request_id": "r-1", "route": "ask"})
    assert out.provider == "gemini"


async def test_log_extra_reaches_every_line_and_nothing_else_is_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """One scenario that makes the router emit each kind of line: the zone warning, an open, an
    attempt that failed, a skip by the breaker, a rejected request and a skip for the deadline.
    Every one carries ``request_id``, and none carries what the question, the prompt or a
    provider's reply said."""
    caplog.set_level(logging.INFO, logger=ROUTER_LOGGER)

    def broken(key: str) -> Any:
        raise ZoneInfoNotFoundError(key)

    daily = ProviderRateLimited("SECRET-429-BODY", retry_after_s=None, is_quota=True)
    messy = ProviderBadOutput("SECRET-BAD-MESSAGE", raw="SECRET-RAW", validation_error="SECRET-WHY")
    rejected = ProviderRequestRejected("SECRET-KEY-TEXT", status_code=401)
    rig = Rig(Probe("gemini", [daily]), Probe("groq", [messy, GOOD, rejected]), load_zone=broken)
    extra = {"request_id": "req-1"}
    common = {"system": "SECRET-PROMPT", "user": "SECRET-QUESTION", "log_extra": extra}

    await rig.ask(**common)  # gemini: quota 429 (zone warning, failed, opened); groq: bad, then ok
    with pytest.raises(AllProvidersUnavailable):
        await rig.ask(**common)  # gemini skipped; groq rejects the request
    with pytest.raises(RouterDeadlineError):
        await rig.ask(time_left=lambda: 1.0, **common)  # gemini skipped; groq refused: no time

    ours = [r for r in caplog.records if r.name == ROUTER_LOGGER]
    assert {r.getMessage() for r in ours} == {
        LOG_ZONE_UNAVAILABLE,
        LOG_BREAKER_OPENED,
        LOG_ATTEMPT_FAILED,
        LOG_PROVIDER_SKIPPED,
        LOG_REQUEST_REJECTED,
    }
    assert {r.__dict__["reason"] for r in router_records(caplog, LOG_PROVIDER_SKIPPED)} == {
        "breaker_open",
        "deadline",
    }
    for record in ours:
        assert record.__dict__["request_id"] == "req-1", record.getMessage()
        text = record.getMessage() + str(record.__dict__)
        assert "SECRET" not in text, record.getMessage()


async def test_without_log_extra_the_lines_carry_no_request_id(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger=ROUTER_LOGGER)
    await Rig(Probe("gemini", [timeout()]), Probe("groq", [GOOD])).ask()
    [record] = router_records(caplog, LOG_ATTEMPT_FAILED)
    assert "request_id" not in record.__dict__


# --- One INFO line per failed call ---------------------------------------------------------


@pytest.mark.parametrize(
    ("step", "outcome"),
    [
        pytest.param(timeout(), "timeout", id="timeout"),
        pytest.param(down(), "unavailable", id="unavailable"),
        pytest.param(limited(30.0), "rate_limited", id="rate-limited"),
        pytest.param(limited(None, quota=True), "quota", id="quota"),
        pytest.param(ProviderRequestRejected("x", status_code=400), "rejected", id="rejected"),
    ],
)
async def test_a_failed_call_leaves_one_info_line_with_four_fields(
    step: Any, outcome: str, caplog: pytest.LogCaptureFixture
) -> None:
    """The trace of a fallback: a timeout, an outage or a 429 that the next provider covers."""
    caplog.set_level(logging.INFO, logger=ROUTER_LOGGER)
    clock = FakeClock()
    rig = Rig(Probe("gemini", [step], clock=clock, takes_s=1.5), Probe("groq", [GOOD]), clock=clock)
    out = await rig.ask(log_extra={"request_id": "r-1"})
    assert out.fallback_used
    [record] = router_records(caplog, LOG_ATTEMPT_FAILED)  # groq succeeded: no line for it
    assert record.levelno == logging.INFO
    assert extras(record) == {
        "request_id": "r-1",
        "provider": "gemini",
        "outcome": outcome,
        "is_retry": False,
        "latency_ms": 1500,
    }


async def test_bad_output_and_its_retry_each_leave_a_line(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger=ROUTER_LOGGER)
    await Rig(Probe("gemini", [BAD, BAD]), Probe("groq", [GOOD])).ask()
    lines = router_records(caplog, LOG_ATTEMPT_FAILED)
    assert [(r.__dict__["outcome"], r.__dict__["is_retry"]) for r in lines] == [
        ("bad_output", False),
        ("bad_output", True),
    ]


async def test_a_reply_the_validation_rejects_leaves_a_bad_output_line(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger=ROUTER_LOGGER)
    verdicts = iter(["no valid citation", None])
    await Rig(Probe("gemini", [GOOD, GOOD])).ask(validate=lambda result: next(verdicts))
    [record] = router_records(caplog, LOG_ATTEMPT_FAILED)
    assert (record.__dict__["outcome"], record.__dict__["is_retry"]) == ("bad_output", False)


async def test_a_successful_call_leaves_no_failure_line(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger=ROUTER_LOGGER)
    await Rig(Probe("gemini", [GOOD])).ask()
    assert router_records(caplog, LOG_ATTEMPT_FAILED) == []


async def test_the_failure_line_is_logged_before_on_attempt_runs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A hook that fails must not hide the trace of the call that was just made."""
    caplog.set_level(logging.INFO, logger=ROUTER_LOGGER)

    async def on_attempt(attempt: Any) -> None:
        raise RuntimeError("the hook failed")

    with pytest.raises(RuntimeError, match="the hook failed"):
        await Rig(Probe("gemini", [timeout()]), Probe("groq", [GOOD])).ask(on_attempt=on_attempt)
    assert len(router_records(caplog, LOG_ATTEMPT_FAILED)) == 1
