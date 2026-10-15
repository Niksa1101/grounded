"""Tests added with the router's implementation (ticket 7.03) for what the 7.02 scenarios in
``test_router.py`` leave open. Nothing here changes or relaxes a spec test; the helpers (``Probe``,
``Rig``, ``trail``...) are the spec file's.

They cover what the review of the implementation looks at: a half-open trial is given back on every
exit path (a cancellation at the other ``await``s, a hook that raises), the time checks sit exactly
on their boundaries (``>`` against ``>=``), the daylight-saving arithmetic does not depend on the
zone ``now`` arrives in, the order of what is raised and what is logged for a call that ran out of
road (a refusal, a skip, a straggler), and the documented limit of the straggler rule. The last test
throws seeded random traffic (random outcomes, cancellations, hooks that raise, a clock that jumps)
at two providers, audits every trial the breakers hand out and checks that every breaker recovers.
"""

from __future__ import annotations

import asyncio
import contextlib
import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import pytest
from pydantic import BaseModel

from grounded.generation import router as router_module
from grounded.generation.providers.base import GenerationResult, Usage
from grounded.generation.providers.fake import FakeLLMProvider
from grounded.generation.router import (
    Admission,
    AllProvidersUnavailable,
    Attempt,
    AttemptOutcome,
    BreakerConfig,
    BreakerState,
    BreakerStatus,
    CircuitBreaker,
    RouterDeadlineError,
    quota_reset_s,
)
from grounded.infra.provider_errors import (
    ProviderBadOutput,
    ProviderError,
    ProviderRequestRejected,
)
from tests.unit.test_router import (
    BAD,
    CONFIG,
    GOOD,
    BugError,
    Probe,
    Rig,
    down,
    limited,
    opened_logs,
    timeout,
    trail,
)

HOUR = 3600.0
OUTCOME = AttemptOutcome
LOS_ANGELES = ZoneInfo("America/Los_Angeles")


async def half_open_rig() -> tuple[Rig, Probe, Probe]:
    """Gemini was rate limited for 30 s and 31 s have passed: its trial is free, nobody holds it."""
    p, q = Probe("gemini", [limited(30.0), GOOD, GOOD]), Probe("groq", [GOOD] * 3)
    rig = Rig(p, q)
    await rig.ask()
    rig.clock.advance(31.0)
    assert rig.status("gemini") == BreakerStatus(BreakerState.HALF_OPEN, None, False)
    return rig, p, q


# --- A trial is given back on every exit path ------------------------------------------------


async def test_a_trial_cancelled_inside_before_call_is_released() -> None:
    """The spec cancels a trial inside the provider call; the budget reserve is an ``await`` too."""
    rig, p, _ = await half_open_rig()
    entered, gate = asyncio.Event(), asyncio.Event()

    async def before_call() -> None:
        entered.set()
        await gate.wait()

    task = asyncio.create_task(rig.ask(before_call=before_call))
    await entered.wait()
    assert rig.status("gemini").trial_in_flight is True  # the request claimed the trial
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert rig.status("gemini") == BreakerStatus(BreakerState.HALF_OPEN, None, False)
    assert p.entries == 1  # the 429 only: the cancelled request never reached the provider
    assert (await rig.ask()).provider == "gemini"


async def test_a_time_left_error_propagates_and_releases_a_held_trial() -> None:
    rig, p, _ = await half_open_rig()

    def broken() -> float:
        raise BugError("bug in the deadline hook")

    with pytest.raises(BugError, match="deadline hook"):
        await rig.ask(time_left=broken)
    assert p.entries == 1
    assert rig.status("gemini") == BreakerStatus(BreakerState.HALF_OPEN, None, False)


async def test_cancelling_on_attempt_after_a_trial_reply_leaves_the_breaker_closed() -> None:
    """The reply already closed the breaker: a cancellation in the hook has nothing to give back."""
    rig, _, _ = await half_open_rig()
    entered, gate = asyncio.Event(), asyncio.Event()

    async def on_attempt(attempt: Attempt) -> None:
        entered.set()
        await gate.wait()

    task = asyncio.create_task(rig.ask(on_attempt=on_attempt))
    await entered.wait()
    assert rig.status("gemini") == BreakerStatus(BreakerState.CLOSED, None, False)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert rig.status("gemini") == BreakerStatus(BreakerState.CLOSED, None, False)


async def test_a_retry_user_error_after_a_trials_bad_output_leaves_nothing_claimed() -> None:
    """Bad output is a reply, so it closed the breaker; the wording hook then fails."""
    p, q = Probe("gemini", [limited(30.0), BAD, GOOD]), Probe("groq", [GOOD])
    rig = Rig(p, q)
    await rig.ask()
    rig.clock.advance(31.0)

    def broken(user: str, reason: str) -> str:
        raise BugError("bug in the retry wording")

    with pytest.raises(BugError, match="retry wording"):
        await rig.ask(retry_user=broken)
    assert rig.status("gemini") == BreakerStatus(BreakerState.CLOSED, None, False)
    assert rig.seen[-1].outcome is OUTCOME.BAD_OUTPUT  # the first call's usage was still reported


async def test_a_retry_is_skipped_when_the_breaker_opened_after_the_first_call() -> None:
    """The retry goes through the gate like any call: another request's 429, arriving between the
    first call and its retry, opens the breaker, and the retry is then a skip, not a call."""
    p, q = Probe("gemini", [BAD, limited(30.0)]), Probe("groq", [GOOD, GOOD])
    rig = Rig(p, q)
    fired = False

    async def concurrent_429(attempt: Attempt) -> None:
        nonlocal fired
        if not fired:
            fired = True
            await rig.ask()  # a whole other request: its gemini call gets the 429

    out = await rig.ask(on_attempt=concurrent_429)
    assert trail(out.attempts) == [
        ("gemini", OUTCOME.BAD_OUTPUT, False),
        ("gemini", OUTCOME.SKIPPED_OPEN, True),
        ("groq", OUTCOME.SUCCESS, False),
    ]
    assert len(p.calls) == 2  # the first call and the other request's 429: no retry call


async def test_two_routers_share_no_breaker_state() -> None:
    first, second = Rig(Probe("gemini", [limited(30.0)])), Rig(Probe("gemini", [GOOD]))
    with pytest.raises(ProviderError):
        await first.ask()
    assert first.status("gemini").state is BreakerState.OPEN
    assert second.status("gemini").state is BreakerState.CLOSED
    assert (await second.ask()).provider == "gemini"


class SequencedProbe(Probe):
    """A provider whose calls each wait at their own gate, so a test decides in which order
    concurrent calls finish. A call takes its script step when it finishes, not when it starts."""

    def __init__(self, name: str, script: Any = ()) -> None:
        super().__init__(name, script)
        self._gates: dict[int, asyncio.Event] = {}
        self._arrived: dict[int, asyncio.Event] = {}

    def gate_for(self, index: int) -> asyncio.Event:
        return self._gates.setdefault(index, asyncio.Event())

    def arrival_of(self, index: int) -> asyncio.Event:
        return self._arrived.setdefault(index, asyncio.Event())

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
        index = self.entries
        self.entries += 1
        self.arrival_of(index).set()
        await self.gate_for(index).wait()
        return await FakeLLMProvider.generate(
            self,
            system=system,
            user=user,
            schema=schema,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            timeout_s=timeout_s,
        )


async def test_a_validate_error_loses_that_calls_report_but_not_the_trial() -> None:
    """``validate`` must not raise (``generate`` says so). If it does anyway: the error propagates
    unchanged, ``on_attempt`` has not run for that call (its usage is not reported), and the trial
    is not lost, because the breaker heard the reply first and the reply closed it."""
    rig, p, _ = await half_open_rig()
    reported = len(rig.seen)

    def validate(result: GenerationResult[Any]) -> str | None:
        raise BugError("bug in the citation checks")

    with pytest.raises(BugError, match="citation checks"):
        await rig.ask(validate=validate)
    assert p.entries == 2  # the trial was a real call...
    assert len(rig.seen) == reported  # ...whose attempt was never reported
    assert rig.status("gemini") == BreakerStatus(BreakerState.CLOSED, None, False)


# --- A request that runs out of road: the order of what is raised and logged ---------------


async def test_a_refused_attempt_never_calls_before_call() -> None:
    """The deadline gate comes first: a refused call must not spend a unit of budget."""
    reserved = 0

    async def before_call() -> None:
        nonlocal reserved
        reserved += 1

    p = Probe("gemini", [GOOD])
    with pytest.raises(RouterDeadlineError):
        await Rig(p).ask(before_call=before_call, time_left=lambda: 1.0)
    assert (reserved, p.entries) == (0, 0)


async def test_a_refused_retry_has_reserved_nothing_for_itself() -> None:
    reserved = 0
    left = iter([25.0, 1.0])  # enough for the first call, not for the retry

    async def before_call() -> None:
        nonlocal reserved
        reserved += 1

    p = Probe("gemini", [BAD, GOOD])
    with pytest.raises(RouterDeadlineError):
        await Rig(p).ask(before_call=before_call, time_left=lambda: next(left))
    assert (reserved, len(p.calls)) == (1, 1)


async def test_with_every_breaker_open_the_deadline_is_never_consulted() -> None:
    """Only an entry whose breaker admits a call is looked at by the gate, so with nothing left
    to call the answer is "unavailable", not "out of time", whatever ``time_left`` says."""
    rig = Rig(Probe("gemini", [limited(60.0)]), Probe("groq", [limited(60.0)]))
    with pytest.raises(AllProvidersUnavailable):
        await rig.ask()  # both breakers open
    reads = 0

    def time_left() -> float:
        nonlocal reads
        reads += 1
        return 0.0

    with pytest.raises(AllProvidersUnavailable) as info:
        await rig.ask(time_left=time_left)
    assert reads == 0
    assert trail(info.value.attempts) == [
        ("gemini", OUTCOME.SKIPPED_OPEN, False),
        ("groq", OUTCOME.SKIPPED_OPEN, False),
    ]


async def test_when_entries_end_in_different_bad_outputs_the_last_one_is_raised() -> None:
    reasons = iter(["first: call", "first: retry", "second: call", "second: retry"])
    with pytest.raises(ProviderBadOutput) as info:
        await Rig(Probe("gemini", [GOOD, GOOD]), Probe("groq", [GOOD, GOOD])).ask(
            validate=lambda result: next(reasons)
        )
    assert info.value.validation_error == "second: retry"


async def test_a_stragglers_429_is_not_logged_as_a_breaker_open(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Two calls are in flight while the breaker is closed; the first one's 429 opens it. The
    second one's 429 arrives later: it must neither be logged as an open nor re-time the breaker."""
    p, q = SequencedProbe("gemini", [limited(30.0), limited(30.0)]), Probe("groq", [GOOD, GOOD])
    rig = Rig(p, q)
    first, second = (asyncio.create_task(rig.ask()) for _ in range(2))
    await p.arrival_of(0).wait()
    await p.arrival_of(1).wait()
    p.gate_for(0).set()
    await first
    rig.clock.advance(10.0)
    p.gate_for(1).set()
    await second
    assert len(opened_logs(caplog)) == 1
    assert rig.status("gemini").remaining_s == pytest.approx(20.0)  # not re-timed to 30 s


async def test_the_late_failures_of_calls_started_before_an_opening_still_count() -> None:
    """The documented limit of the straggler rule (Tech §10): a call admitted while the breaker
    was closed is ignored only while the breaker is not closed. Here four calls are in flight;
    one gets a 429 (open for 2 s), a trial then succeeds and closes the breaker, and only then do
    the other three time out. The breaker cannot tell them from fresh calls (an ``Admission``
    carries no epoch), so they count, reach the threshold and open it for ``failure_open_s``."""
    p = SequencedProbe("gemini", [limited(2.0), GOOD, timeout(), timeout(), timeout()])
    q = Probe("groq", [GOOD] * 4)
    rig = Rig(p, q)
    in_flight = [asyncio.create_task(rig.ask()) for _ in range(4)]
    for index in range(4):
        await p.arrival_of(index).wait()
    p.gate_for(0).set()  # script step 0: the 429
    await in_flight[0]
    assert rig.status("gemini").remaining_s == pytest.approx(2.0)
    rig.clock.advance(3.0)
    trial = asyncio.create_task(rig.ask())
    await p.arrival_of(4).wait()
    p.gate_for(4).set()  # script step 1: the trial answers
    assert (await trial).provider == "gemini"
    assert rig.status("gemini") == BreakerStatus(BreakerState.CLOSED, None, False)
    for index in (1, 2, 3):
        p.gate_for(index).set()  # script steps 2 to 4: the three late timeouts
    assert [out.provider for out in await asyncio.gather(*in_flight[1:])] == ["groq"] * 3
    status = rig.status("gemini")
    assert status.state is BreakerState.OPEN
    assert status.remaining_s == pytest.approx(CONFIG.failure_open_s)


# --- The boundaries --------------------------------------------------------------------------


def test_a_failure_exactly_one_window_old_still_counts() -> None:
    """Tech §10 says "within 60 s": the window is inclusive, only strictly older drops out."""
    breaker = CircuitBreaker("gemini", CONFIG)
    assert breaker.record_failure(breaker.admit(0.0), 0.0) is None
    assert breaker.record_failure(breaker.admit(30.0), 30.0) is None
    assert breaker.record_failure(breaker.admit(60.0), 60.0) == pytest.approx(30.0)


def test_a_failure_just_past_the_window_drops_out() -> None:
    breaker = CircuitBreaker("gemini", CONFIG)
    assert breaker.record_failure(breaker.admit(0.0), 0.0) is None
    assert breaker.record_failure(breaker.admit(30.0), 30.0) is None
    assert breaker.record_failure(breaker.admit(60.001), 60.001) is None


def test_the_open_time_has_elapsed_exactly_at_its_end() -> None:
    breaker = CircuitBreaker("gemini", CONFIG)
    assert breaker.open_for(breaker.admit(0.0), 0.0, 10.0) is True
    assert breaker.admit(9.999) is Admission.SKIP_OPEN
    assert breaker.status(10.0) == BreakerStatus(BreakerState.HALF_OPEN, None, False)
    assert breaker.admit(10.0) is Admission.TRIAL


async def test_the_deadline_gate_starts_a_call_with_exactly_the_timeout_left() -> None:
    """``time_left() >= timeout_s`` starts the call; anything below, or NaN, refuses it."""
    p = Probe("gemini", [GOOD, GOOD])
    rig = Rig(p)
    await rig.ask(time_left=lambda: 12.0)  # the timeout is 12 s
    for left in (11.999, float("nan"), -1.0):
        with pytest.raises(RouterDeadlineError):
            await rig.ask(time_left=lambda left=left: left)
    assert len(p.calls) == 1


@pytest.mark.parametrize(
    ("now", "expected_s"),
    [
        # Handed in as Los Angeles times, so ``now`` and the midnight share one ``tzinfo`` object:
        # subtracting them naively would compare wall clocks and miss the lost or repeated hour.
        pytest.param(datetime(2026, 3, 8, 0, 30, tzinfo=LOS_ANGELES), 22.5 * HOUR, id="spring"),
        pytest.param(datetime(2026, 11, 1, 0, 30, tzinfo=LOS_ANGELES), 24.5 * HOUR, id="fall"),
        # 01:30 happens twice on the fall-back day: first in PDT (fold=0), then in PST (fold=1).
        pytest.param(
            datetime(2026, 11, 1, 1, 30, tzinfo=LOS_ANGELES, fold=0), 23.5 * HOUR, id="fold-0"
        ),
        pytest.param(
            datetime(2026, 11, 1, 1, 30, tzinfo=LOS_ANGELES, fold=1), 22.5 * HOUR, id="fold-1"
        ),
    ],
)
def test_gemini_quota_reset_does_not_depend_on_the_zone_now_arrives_in(
    now: datetime, expected_s: float
) -> None:
    assert quota_reset_s("gemini", now, None, max_retry_after_s=3600.0) == pytest.approx(expected_s)


# --- Seeded random traffic -------------------------------------------------------------------


@dataclass
class TrialAudit:
    """What one breaker handed out. ``held`` is trials handed out and not yet resolved or released,
    so it must never go below zero (a trial resolved twice) or above one (two at once), and must be
    back to zero when the traffic stops (a trial lost)."""

    handed_out: int = 0
    held: int = 0
    most_held: int = 0
    violations: list[str] = field(default_factory=list)


def audited_breaker(audits: dict[str, TrialAudit]) -> type[CircuitBreaker]:
    """A ``CircuitBreaker`` that counts its trials through its public methods only."""

    class AuditedBreaker(CircuitBreaker):
        def __init__(self, name: str, config: BreakerConfig) -> None:
            super().__init__(name, config)
            self.audit = audits.setdefault(name, TrialAudit())

        def _resolved(self, admission: Admission) -> None:
            if admission is Admission.TRIAL:
                self.audit.held -= 1
                if self.audit.held < 0:
                    self.audit.violations.append("a trial was resolved twice")

        def admit(self, now: float) -> Admission:
            admission = super().admit(now)
            if admission is Admission.TRIAL:
                self.audit.handed_out += 1
                self.audit.held += 1
                self.audit.most_held = max(self.audit.most_held, self.audit.held)
            return admission

        def record_reply(self, admission: Admission, now: float) -> None:
            self._resolved(admission)
            super().record_reply(admission, now)

        def record_failure(self, admission: Admission, now: float) -> float | None:
            self._resolved(admission)
            return super().record_failure(admission, now)

        def open_for(self, admission: Admission, now: float, seconds: float) -> bool:
            self._resolved(admission)
            return super().open_for(admission, now, seconds)

        def release(self, admission: Admission) -> None:
            self._resolved(admission)
            super().release(admission)

    return AuditedBreaker


class ChaosProvider(Probe):
    """A provider that yields to the event loop a few times and then does something random."""

    def __init__(self, name: str, rng: random.Random) -> None:
        super().__init__(name)
        self._rng = rng

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
        for _ in range(self._rng.randint(0, 3)):
            await asyncio.sleep(0)
        roll = self._rng.random()
        if roll < 0.25:
            raise limited(self._rng.choice([None, 1.0, 5.0]), quota=self._rng.random() < 0.03)
        if roll < 0.40:
            raise timeout()
        if roll < 0.50:
            raise down()
        if roll < 0.55:
            raise ProviderBadOutput("bad", validation_error="not valid")
        if roll < 0.58:
            raise ProviderRequestRejected("bad key", status_code=401)
        if roll < 0.61:
            raise BugError("bug in an adapter")
        return GenerationResult(
            parsed=schema.model_validate_json(GOOD.model_dump_json()),
            raw_text=GOOD.model_dump_json(),
            usage=Usage(input_tokens=1, output_tokens=1),
            provider=self.name,
            model=self.model,
            latency_ms=0,
        )


@pytest.mark.parametrize("seed", range(12))
async def test_random_concurrent_traffic_never_loses_or_doubles_a_trial(
    seed: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """300 requests in flight over two providers, with random outcomes (rejected requests
    included), hooks that raise (``before_call``, ``validate``, ``retry_user``, ``time_left``),
    random cancellations and a clock that jumps past the open times. Whatever happens, no breaker
    ever holds two trials at once, resolves one twice, or still holds one when the traffic stops,
    and once every open time has passed every breaker admits a call again."""
    rng = random.Random(seed)
    audits: dict[str, TrialAudit] = {}
    monkeypatch.setattr(router_module, "CircuitBreaker", audited_breaker(audits))
    rig = Rig(ChaosProvider("gemini", rng), ChaosProvider("groq", rng))

    async def before_call() -> None:
        for _ in range(rng.randint(0, 2)):
            await asyncio.sleep(0)
        if rng.random() < 0.03:
            raise BugError("budget")

    def validate(result: GenerationResult[Any]) -> str | None:
        if rng.random() < 0.02:
            raise BugError("citation checks")
        return "no valid citation" if rng.random() < 0.1 else None

    def retry_user(user: str, reason: str) -> str:
        if rng.random() < 0.05:
            raise BugError("retry wording")
        return f"{user} ({reason})"

    def time_left() -> float:
        roll = rng.random()
        if roll < 0.03:
            raise BugError("deadline hook")
        return 1.0 if roll < 0.08 else 100.0  # now and then too little for a 12 s attempt

    async def one_request() -> None:
        # The outcomes under test are the breakers', not the answers': errors are expected.
        with contextlib.suppress(ProviderError, BugError, RouterDeadlineError):
            await rig.ask(
                before_call=before_call,
                validate=validate,
                retry_user=retry_user,
                time_left=time_left,
            )

    tasks: list[asyncio.Task[None]] = []
    for _ in range(300):
        tasks.append(asyncio.create_task(one_request()))
        roll = rng.random()
        if roll < 0.30:
            await asyncio.sleep(0)
        elif roll < 0.45:
            rig.clock.advance(rng.choice([1.0, 10.0, 31.0, 61.0, 1000.0]))
        elif roll < 0.50:
            rng.choice(tasks).cancel()
    await asyncio.gather(*tasks, return_exceptions=True)

    assert sum(audit.handed_out for audit in audits.values()) > 0  # trials were exercised
    for name, audit in audits.items():
        assert audit.violations == [], name
        assert audit.most_held <= 1, name
        assert audit.held == 0, name
        assert rig.status(name).trial_in_flight is False, name

    # Recoverability: the longest open time is a Gemini quota (up to a day), so two days later no
    # breaker may still be open or hold a trial, whatever state the traffic left it in.
    rig.clock.advance(2 * 24 * HOUR)
    for name in audits:
        status = rig.status(name)
        assert status.state in (BreakerState.CLOSED, BreakerState.HALF_OPEN), name
        assert status.remaining_s is None, name
        assert status.trial_in_flight is False, name
