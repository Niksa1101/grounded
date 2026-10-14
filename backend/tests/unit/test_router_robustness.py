"""Tests added with the router's implementation (ticket 7.03) for what the 7.02 scenarios in
``test_router.py`` leave open. Nothing here changes or relaxes a spec test; the helpers (``Probe``,
``Rig``, ``trail``...) are the spec file's.

They cover three things the review of the implementation looks at: a half-open trial is given back
on every exit path (a cancellation at the other ``await``s, a hook that raises), the time checks sit
exactly on their boundaries (``>`` against ``>=``), and the daylight-saving arithmetic does not
depend on the zone ``now`` arrives in. The last test throws seeded random traffic (random outcomes,
cancellations, a clock that jumps) at two providers and audits every trial the breakers hand out.
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
from grounded.generation.router import (
    Admission,
    Attempt,
    AttemptOutcome,
    BreakerConfig,
    BreakerState,
    BreakerStatus,
    CircuitBreaker,
    RouterDeadlineError,
    quota_reset_s,
)
from grounded.infra.provider_errors import ProviderBadOutput, ProviderError
from tests.unit.test_router import (
    BAD,
    CONFIG,
    GOOD,
    BugError,
    Probe,
    Rig,
    down,
    limited,
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
        if roll < 0.60:
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
    """300 requests in flight over two providers, with random outcomes, hooks that raise, random
    cancellations and a clock that jumps past the open times. Whatever happens, no breaker ever
    holds two trials at once, resolves one twice, or still holds one when the traffic stops."""
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

    async def one_request() -> None:
        # The outcomes under test are the breakers', not the answers': errors are expected.
        with contextlib.suppress(ProviderError, BugError):
            await rig.ask(before_call=before_call, validate=validate)

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
