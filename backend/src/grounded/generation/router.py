"""The provider router: fallback over an ordered provider list, one circuit breaker per provider
(Tech.md §10, §9.5, PRD D12 and D57, AGENTS.md §6.4 and §6.5).

**Author-owned** (AGENTS.md §3). This file is the contract from ticket 7.02: types, signatures and
the rules below, with ``raise NotImplementedError`` where the logic goes. The Author implements it
in 7.03. The scenarios are ``tests/unit/test_router.py``; those that need the logic are strict
``xfail`` until then. 7.04 wires the router into the pipeline.

**Why it exists.** When Gemini is rate limited, down or slow, Groq answers, and a provider that
keeps failing is left alone for a while instead of being hit by every request. The router **never
sleeps and never waits out a ``Retry-After``**: a failure is a fact to remember (the breaker) and a
reason to try the next provider now.

**What it is given.** ``providers``, the ordered list (``GENERATOR_PROVIDERS``), first entry =
primary. Names must be unique: a breaker is keyed by ``provider.name``, and the quota rules
recognize ``"gemini"`` and ``"groq"`` by it. One entry means no fallback (eval mode: ``Settings``
refuses any other list at startup, the router just routes over what it is given). Also a
``BreakerConfig`` (from ``Settings``) and a ``RouterClock``. Construction builds one
``CircuitBreaker`` per provider and raises ``RouterConfigError`` for an empty list or a duplicate
name (the message names it).

**Decoupled from the pipeline.** The router imports neither the pipeline, the budget, the citation
checks nor the prompts. 7.04 passes small closures to ``generate`` instead: ``before_call`` is the
budget reserve (every call, retries and fallbacks included), ``validate`` runs ``map_citations``
and returns its ``bad_output`` reason, ``retry_user`` is ``Prompt.render_retry``, ``on_attempt``
records the usage of every call, and ``time_left`` is the request deadline. Each hook's contract
is on ``generate``. 7.04 also registers the HTTP handlers of the two errors below.

**One attempt** is one call to one provider. For each provider in list order ``generate`` repeats
this until the entry is done (at most two calls per entry: the call and one retry):

1. *Gate, before every call, a retry included.* (a) ``CircuitBreaker.admit``: open, or half-open
   with the trial held by another call, means skip the entry without a call (an ``Attempt`` with a
   ``SKIPPED_*`` outcome and one INFO log) and go on to the next. (b) The deadline gate (below):
   too little time left ends the whole call with ``RouterDeadlineError``. It is looked at only for
   an entry whose breaker admits a call, and a refusal **never consumes** the half-open trial
   (it is released unused).
2. ``await before_call()``. Whatever it raises propagates, and a held trial is released.
3. Call ``provider.generate`` with the arguments of ``generate`` unchanged, except ``user`` on a
   retry. The adapter enforces ``timeout_s`` and raises ``ProviderTimeout``: the router adds no
   timeout of its own.
4. Tell the breaker what happened **before** ``validate`` or ``on_attempt`` run (so a hook error
   never leaves a trial claimed and is never a provider failure), then act (next section).

**Events and actions** (Tech §10; "fail over" = go on to the next entry at once). The
``AttemptOutcome`` of an event is the one of the same name (``QUOTA`` for ``is_quota=True``):

- *A reply* that passes ``validate`` (or there is no ``validate``): success, return a
  ``RoutedGeneration``.
- *A reply* that ``validate`` rejects, or ``ProviderBadOutput`` from the adapter: **bad output**.
  The reason is ``validate``'s text, or ``error.validation_error or str(error)``. If the error is
  ``retryable`` (a ``validate`` rejection always is) and this call was not already the retry:
  retry once on the same provider with ``user = retry_user(user, reason)``. Otherwise fail over.
  A ``validate`` rejection that has to be raised becomes ``ProviderBadOutput("the answer failed
  the citation checks", raw=result.raw_text, validation_error=reason)``. Bad output is **proof of
  life**, not a failure: the provider answered, so it clears the failure run and closes a
  half-open breaker.
- ``ProviderRateLimited`` with ``is_quota=False``: open the breaker as ``open_plan`` says, fail
  over.
- ``ProviderRateLimited`` with ``is_quota=True`` (a daily quota): open it until the provider's
  reset (``quota_reset_s``), else for ``BREAKER_QUOTA_OPEN_S``; fail over. After any open time the
  breaker is half-open (one trial).
- ``ProviderTimeout`` or ``ProviderUnavailable``: ``CircuitBreaker.record_failure``; fail over
  (the failure itself is not retried on the same provider).
- ``ProviderRequestRejected`` (a 4xx other than 429: our key, model or request): **not a breaker
  event** (it releases a held trial with no verdict). One WARNING ``LOG_REQUEST_REJECTED`` with
  ``provider`` and ``status_code`` (never the provider's text), then fail over. *Why:* the
  provider's health says nothing about our request, a breaker opened by a bad key would hide a
  config error behind a quiet outage, and a bad key on the primary must not take the service down
  while a fallback exists. It is not retried: the same call would be rejected again.
- Anything else a provider raises (an unknown ``ProviderError`` subclass, a bug, a cancellation, a
  ``FakeLLMProvider`` with no script left) propagates unchanged, after a held trial is released
  with no verdict: only the five typed errors above are routed, nothing is swallowed.

**What ``generate`` raises when no provider answered.** An entry is *transient* when its last
attempt was skipped (open or trial in flight), rate limited, quota, timeout or unavailable. In
this order:

1. an attempt was refused for lack of time: ``RouterDeadlineError`` (the deadline rule);
2. some entry was transient: ``AllProvidersUnavailable`` (503 with ``Retry-After``, Tech §10);
3. else some entry ended in bad output: the **last** ``ProviderBadOutput`` (502, Tech §9.5 step 5,
   which is what today's one-provider pipeline does);
4. else every entry was rejected: the **first** ``ProviderRequestRejected``, unchanged. It stays a
   loud 500 (our bug or our key), and a ``Retry-After`` would send clients to retry what cannot
   work.

**The deadline rule.** ``time_left`` (optional) returns the seconds left before the request
deadline (``REQUEST_DEADLINE_S``, 25 s, minus what the request has used). It is read right before
each call, and a call is started only if ``time_left() >= timeout_s``: an attempt that cannot
finish inside the deadline is never started, so the route's own deadline does not cut a call off
mid-flight. A Gemini attempt, its retry and a Groq attempt at their timeouts (12 s each) do not fit
in 25 s, which is the case. The refusal is ``RouterDeadlineError``, not ``AllProvidersUnavailable``
(which tells the client to back off for a time it does not need): the request ran out of time, so
7.04 maps it to the route's 504 ``deadline_exceeded``, like ``RetrievalTimeoutError``. It is not a
``ProviderError`` (no provider failed). The timeout is **not shortened** to fit: a timeout the
router imposes would be counted by the breaker as the provider's failure. ``None`` (eval mode, the
CLI) means no deadline. The first refusal ends the call; later entries are not examined.

**The breaker** (``CircuitBreaker``, one per provider, in memory per function instance: PRD D57a).
CLOSED: calls flow and failures are counted. OPEN: no calls until the open time has elapsed.
HALF_OPEN (the open time has elapsed, derived lazily from the clock, no timer task): exactly one
trial call at a time, and a concurrent request skips the provider instead of waiting for the trial.
A trial that gets a reply or bad output closes it, one that fails opens it again with a fresh
timer. Transitions are synchronous methods (atomic in asyncio, which is what makes "exactly one
trial" hold); a lock, if the implementation wants one, is never held across ``before_call``, a
provider call or ``on_attempt``.

**Clocks.** ``RouterClock.monotonic`` times everything (the open time is ``monotonic() + seconds``,
counted from the moment the failure is seen, after the call). ``RouterClock.wall`` (aware UTC) is
read only to derive Gemini's quota reset. A zone that ``RouterClock.load_zone`` cannot load makes
that reset not derivable, so ``BREAKER_QUOTA_OPEN_S`` applies and a warning is logged; no
dependency is added for it (``tzdata`` is locked for Windows only, a Linux image may lack it:
PRD §12). How the monotonic clock behaves while a function instance is frozen is not accounted for.

**Logging** (AGENTS.md §6.13: never the question or any provider text). Every open: WARNING
``LOG_BREAKER_OPENED`` with ``provider``, ``reason`` (``OpenReason``), ``open_s`` and ``derived``
(``Derivation``; ``failures`` opens say ``default``). Every skip: INFO ``LOG_PROVIDER_SKIPPED``
with ``provider``, ``reason`` (``breaker_open``, ``trial_in_flight`` or ``deadline``) and
``remaining_s`` (the breaker's while it is open, else ``None``). This is how the first real Gemini
429 shows what was derived, and the exit criterion "fallback rate visible in logs" starts here.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, tzinfo
from enum import StrEnum
from typing import Final
from zoneinfo import ZoneInfo

from pydantic import BaseModel

from grounded.generation.providers.base import GenerationResult, LLMProvider, Usage
from grounded.infra.provider_errors import ProviderError, ProviderRateLimited
from grounded.infra.timing import Clock
from grounded.settings import Settings

logger = logging.getLogger(__name__)

GEMINI: Final = "gemini"
GROQ: Final = "groq"
# Google: quotas "reset at midnight Pacific time" (PRD D57b, ai.google.dev rate-limits page).
GEMINI_QUOTA_ZONE: Final = "America/Los_Angeles"

# The log messages are part of the contract: 7.04 and the dashboards look for them.
LOG_BREAKER_OPENED: Final = "breaker opened"
LOG_PROVIDER_SKIPPED: Final = "provider skipped"
LOG_REQUEST_REJECTED: Final = "provider rejected the request"
LOG_ZONE_UNAVAILABLE: Final = "time zone unavailable, quota reset not derivable"

type WallClock = Callable[[], datetime]  # timezone-aware, UTC
type ZoneLoader = Callable[[str], tzinfo]  # raises on a zone it cannot load (``ZoneInfo`` does)
type BeforeCall = Callable[[], Awaitable[None]]
type Validate[T: BaseModel] = Callable[[GenerationResult[T]], str | None]
type RetryMessage = Callable[[str, str], str]  # (the original user message, the reason) -> retry
type OnAttempt = Callable[[Attempt], Awaitable[None]]
type TimeLeft = Callable[[], float]


# --- Errors ---------------------------------------------------------------------------------------


class RouterConfigError(ValueError):
    """The router or the breaker was built with something that cannot work; the message names it."""


class UnknownProviderError(LookupError):
    """A provider name that is not in this router's list (``ProviderRouter.breaker_status``)."""


class AllProvidersUnavailable(ProviderError):  # noqa: N818 (the name is the contract, Tech §10)
    """No provider could answer and at least one was transient (rule 2 of the raised errors).

    ``retry_after_s`` is the smallest remaining open time over the breakers that are OPEN at the
    moment of raising, read from the injected clock (a breaker this very call opened counts). It is
    ``None`` only when no breaker is open: every provider failed this call without reaching a
    threshold, or the only gaps are trials in flight (a half-open breaker has no remaining time).
    Otherwise it is a float ``> 0``. ``attempts`` is the whole routing report of the call, skips
    included. The message names providers and outcomes, never user text or a provider's message.
    7.04 registers the HTTP handler (503 ``provider_unavailable`` with ``Retry-After``).
    """

    def __init__(
        self, message: str, *, retry_after_s: float | None, attempts: tuple[Attempt, ...] = ()
    ) -> None:
        super().__init__(message)
        self.retry_after_s = retry_after_s
        self.attempts = attempts


class RouterDeadlineError(Exception):
    """An attempt was refused because ``time_left() < timeout_s`` (the deadline rule): the request
    ran out of time. Not a ``ProviderError``. ``time_left_s`` is what was left at the refusal,
    ``timeout_s`` what the attempt needed, ``attempts`` the report (ending with the
    ``SKIPPED_DEADLINE`` record). 7.04 maps it to the route's 504 ``deadline_exceeded``.
    """

    def __init__(
        self,
        message: str,
        *,
        time_left_s: float,
        timeout_s: float,
        attempts: tuple[Attempt, ...] = (),
    ) -> None:
        super().__init__(message)
        self.time_left_s = time_left_s
        self.timeout_s = timeout_s
        self.attempts = attempts


# --- Value types ----------------------------------------------------------------------------------


class BreakerState(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class Admission(StrEnum):
    """What ``CircuitBreaker.admit`` decided. The first two let a call through."""

    ALLOWED = "allowed"  # closed
    TRIAL = "trial"  # half-open, and this caller now holds the single trial
    SKIP_OPEN = "skip_open"
    SKIP_TRIAL = "skip_trial"  # half-open, the trial is held by another call


class OpenReason(StrEnum):
    RATE_LIMIT = "rate_limit"
    QUOTA = "quota"
    FAILURES = "failures"


class Derivation(StrEnum):
    """How an open time was obtained (for the log: the first real 429 shows which one applied)."""

    RETRY_AFTER = "retry_after"  # the provider's own wait
    RETRY_AFTER_CAPPED = "retry_after_capped"  # ... cut to ``BREAKER_MAX_RETRY_AFTER_S``
    QUOTA_RESET = "quota_reset"  # derived by the router (Gemini's midnight Pacific)
    DEFAULT = "default"  # a ``BREAKER_*`` setting stood in


class AttemptOutcome(StrEnum):
    SUCCESS = "success"
    BAD_OUTPUT = "bad_output"
    RATE_LIMITED = "rate_limited"
    QUOTA = "quota"
    TIMEOUT = "timeout"
    UNAVAILABLE = "unavailable"
    REJECTED = "rejected"
    SKIPPED_OPEN = "skipped_open"  # no call: the breaker is open
    SKIPPED_TRIAL = "skipped_trial"  # no call: half-open, the trial is held by another call
    SKIPPED_DEADLINE = "skipped_deadline"  # no call: ``time_left() < timeout_s``


SKIPPED_OUTCOMES: Final = frozenset(
    {
        AttemptOutcome.SKIPPED_OPEN,
        AttemptOutcome.SKIPPED_TRIAL,
        AttemptOutcome.SKIPPED_DEADLINE,
    }
)


@dataclass(frozen=True, slots=True)
class Attempt:
    """One step of the routing report: a call, or an entry skipped without one.

    Everything the pipeline needs to account for a call is here. ``usage`` is what the call billed:
    the reply's usage (also when ``validate`` rejected it), the ``input_tokens`` and
    ``output_tokens`` of a ``ProviderBadOutput``, zero when nothing was reported or no call was
    made. ``cache_hit`` is the reply's flag (the eval cache). ``is_retry`` marks the second call on
    one provider after bad output. ``latency_ms`` is measured on the injected monotonic clock
    around the call (0 for a skip). ``model`` is ``provider.model``.
    """

    provider: str
    model: str
    outcome: AttemptOutcome
    latency_ms: int = 0
    is_retry: bool = False
    usage: Usage = field(default_factory=lambda: Usage(input_tokens=0, output_tokens=0))
    cache_hit: bool = False

    @property
    def called(self) -> bool:
        return self.outcome not in SKIPPED_OUTCOMES


@dataclass(frozen=True, slots=True)
class RoutedGeneration[T: BaseModel]:
    """The answer and the routing report. ``result`` is the answering provider's result (the last
    ``validate`` call was on it). ``provider`` and ``model`` are the answering entry's ``name`` and
    ``model``. ``fallback_used`` is true iff that entry is not the first of the list (also when the
    first was only skipped). ``attempts`` is every step in order, skips included, the last one the
    ``SUCCESS``; the records are the very ones ``on_attempt`` was given."""

    result: GenerationResult[T]
    provider: str
    model: str
    fallback_used: bool
    attempts: tuple[Attempt, ...]


@dataclass(frozen=True, slots=True)
class BreakerStatus:
    """A read-only view of one breaker. ``remaining_s`` is set only while OPEN (else ``None``);
    ``trial_in_flight`` is true only while HALF_OPEN with the single trial held by a call."""

    state: BreakerState
    remaining_s: float | None
    trial_in_flight: bool


@dataclass(frozen=True, slots=True)
class OpenPlan:
    """How long to open a breaker for a 429, why, and how that was derived."""

    seconds: float
    reason: OpenReason
    derived: Derivation


def _utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True, slots=True)
class RouterClock:
    """The router's time sources, injected so tests never sleep (AGENTS.md §8): ``monotonic`` times
    everything (seconds), ``wall`` is read only to derive a daily quota reset, ``load_zone`` loads
    the quota's time zone (``zoneinfo`` only, see ``quota_reset_s``)."""

    monotonic: Clock = time.monotonic
    wall: WallClock = _utc_now
    load_zone: ZoneLoader = ZoneInfo


SYSTEM_CLOCK: Final = RouterClock()


@dataclass(frozen=True, slots=True)
class BreakerConfig:
    """The breaker's timings, taken from ``Settings`` (AGENTS.md §6.7) so the state machine never
    imports it, like ``ConfidenceConfig``. Durations are finite seconds ``> 0`` and the threshold
    is ``>= 1``, else ``RouterConfigError`` naming the field (``Settings`` enforces the same)."""

    rate_limit_open_s: float  # a per-minute 429 without a usable Retry-After
    quota_open_s: float  # a daily-quota 429 whose reset cannot be derived
    failure_threshold: int  # consecutive timeouts / 5xx ...
    failure_window_s: float  # ... within this many seconds ...
    failure_open_s: float  # ... open the breaker this long
    max_retry_after_s: float  # the longest provider-supplied wait that is honored

    def __post_init__(self) -> None:
        durations = {
            "rate_limit_open_s": self.rate_limit_open_s,
            "quota_open_s": self.quota_open_s,
            "failure_window_s": self.failure_window_s,
            "failure_open_s": self.failure_open_s,
            "max_retry_after_s": self.max_retry_after_s,
        }
        for name, value in durations.items():
            if not 0 < value < float("inf"):
                raise RouterConfigError(f"{name} must be a finite number of seconds > 0: {value}")
        if self.failure_threshold < 1:
            raise RouterConfigError(f"failure_threshold must be >= 1: {self.failure_threshold}")

    @classmethod
    def from_settings(cls, settings: Settings) -> BreakerConfig:
        return cls(
            rate_limit_open_s=settings.breaker_rate_limit_open_s,
            quota_open_s=settings.breaker_quota_open_s,
            failure_threshold=settings.breaker_failure_threshold,
            failure_window_s=settings.breaker_failure_window_s,
            failure_open_s=settings.breaker_failure_open_s,
            max_retry_after_s=settings.breaker_max_retry_after_s,
        )


# --- Pure rules -----------------------------------------------------------------------------------


def quota_reset_s(
    provider: str,
    now: datetime,
    retry_after_s: float | None,
    *,
    max_retry_after_s: float,
    load_zone: ZoneLoader = ZoneInfo,
) -> float | None:
    """Seconds until the daily quota of ``provider`` resets, or ``None`` when not derivable. Pure
    apart from the one warning below; ``now`` is the injected wall clock.

    - ``"gemini"``: until the **next** midnight in ``GEMINI_QUOTA_ZONE`` strictly after ``now``
      (exactly at midnight it is the following one, a full day away), counted between instants, so
      the 23 h and 25 h days of the daylight-saving changes come out right. ``retry_after_s`` is
      ignored (Gemini's ``retryDelay`` on a daily 429 may be a few seconds). ``now`` must be
      timezone-aware, any zone, else ``ValueError`` (the message says "aware"). If ``load_zone``
      raises (``ZoneInfoNotFoundError``, ``OSError`` or ``ValueError``) the answer is ``None`` and
      one WARNING ``LOG_ZONE_UNAVAILABLE`` is logged with ``provider`` and ``zone`` fields.
    - ``"groq"``: its own ``retry_after_s`` cut to ``max_retry_after_s`` (its docs state no reset
      time and its one observed daily 429 refills continuously, PRD D57b). ``None`` when that is
      absent: ``None``, zero, negative or NaN. ``+inf`` is cut to the cap.
    - any other provider: ``None``.
    """
    raise NotImplementedError("Author implements the quota reset in ticket 7.03")


def open_plan(
    provider: str,
    error: ProviderRateLimited,
    now: datetime,
    config: BreakerConfig,
    *,
    load_zone: ZoneLoader = ZoneInfo,
) -> OpenPlan:
    """How long a 429 opens ``provider``'s breaker (Tech §10, PRD D57b). Pure, like
    ``quota_reset_s``; ``now`` is the injected wall clock.

    A provider-supplied wait counts only if it is not ``None``, zero, negative or NaN. It is cut to
    ``config.max_retry_after_s`` (``Derivation.RETRY_AFTER_CAPPED`` when it was cut).

    - ``is_quota=False``: the usable wait (``RETRY_AFTER``), else ``rate_limit_open_s``
      (``DEFAULT``); reason ``RATE_LIMIT``.
    - ``is_quota=True``: ``quota_reset_s`` (Gemini ``QUOTA_RESET``, Groq ``RETRY_AFTER`` or
      ``RETRY_AFTER_CAPPED``), else ``quota_open_s`` (``DEFAULT``); reason ``QUOTA``.

    The cap does not apply to a reset the router derives itself, nor to the two defaults.
    """
    raise NotImplementedError("Author implements the open plan in ticket 7.03")


# --- The breaker ----------------------------------------------------------------------------------


class CircuitBreaker:
    """One provider's breaker: a state machine whose every method takes ``now`` (monotonic seconds),
    so it needs no clock and no ``Settings``.

    ``admit`` decides and returns an ``Admission``. The caller reports what the call it allowed
    came to, passing the same ``Admission`` back. An outcome applies **only** if the call's
    circumstances still hold: an ``ALLOWED`` call (admitted while closed) that finishes after the
    breaker opened since, a straggler, changes nothing whatever it reports (it must not close,
    extend or re-time an open breaker, and its failures do not pile up). Only the ``TRIAL`` decides
    a half-open breaker, and the caller resolves it exactly once, with ``record_reply``,
    ``record_failure``, ``open_for`` or ``release``.
    """

    def __init__(self, name: str, config: BreakerConfig) -> None:
        raise NotImplementedError("Author implements the breaker in ticket 7.03")

    def status(self, now: float) -> BreakerStatus:
        """A read-only view, never a transition. OPEN while ``now`` is before the end of the open
        time (``remaining_s`` is what is left), HALF_OPEN from then on until a trial closes it."""
        raise NotImplementedError

    def admit(self, now: float) -> Admission:
        """CLOSED: ``ALLOWED``. OPEN: ``SKIP_OPEN``. HALF_OPEN: ``TRIAL`` for the first caller (it
        now holds the trial), ``SKIP_TRIAL`` for everyone else until the trial is resolved."""
        raise NotImplementedError

    def record_reply(self, admission: Admission, now: float) -> None:
        """The provider answered, valid or not: clear the failure run, and a ``TRIAL`` closes the
        breaker."""
        raise NotImplementedError

    def record_failure(self, admission: Admission, now: float) -> float | None:
        """A timeout or an unavailable provider. ``ALLOWED`` while closed: add ``now`` to the run,
        drop failures older than ``failure_window_s`` (a sliding window ending at ``now``), and when
        the run reaches ``failure_threshold`` open for ``failure_open_s`` and clear it. ``TRIAL``:
        open again for ``failure_open_s`` with a fresh timer. Returns the seconds it opened for,
        else ``None`` (also for an ignored straggler)."""
        raise NotImplementedError

    def open_for(self, admission: Admission, now: float, seconds: float) -> bool:
        """A 429, already turned into ``seconds`` (``open_plan``): open now for that long, clear the
        failure run, release a held trial. Returns whether it opened (``False`` for an ignored
        straggler)."""
        raise NotImplementedError

    def release(self, admission: Admission) -> None:
        """Give a ``TRIAL`` back with no verdict (a deadline refusal, a hook error, a cancellation,
        a rejected request): HALF_OPEN stays and the next caller may take the trial. Nothing for an
        ``ALLOWED``."""
        raise NotImplementedError


# --- The router -----------------------------------------------------------------------------------


class ProviderRouter:
    def __init__(
        self,
        providers: Sequence[LLMProvider],
        *,
        config: BreakerConfig,
        clock: RouterClock = SYSTEM_CLOCK,
    ) -> None:
        raise NotImplementedError("Author implements the router in ticket 7.03")

    def breaker_status(self, provider: str) -> BreakerStatus:
        """The named provider's breaker as of now (the injected clock). ``UnknownProviderError`` for
        a name not in the list. Read-only: it never moves a breaker."""
        raise NotImplementedError

    async def generate[T: BaseModel](
        self,
        *,
        system: str,
        user: str,
        schema: type[T],
        temperature: float,
        max_output_tokens: int,
        timeout_s: float,
        retry_user: RetryMessage,
        before_call: BeforeCall | None = None,
        validate: Validate[T] | None = None,
        on_attempt: OnAttempt | None = None,
        time_left: TimeLeft | None = None,
    ) -> RoutedGeneration[T]:
        """Route one request over the providers (the module docstring has the rules).

        ``system``, ``schema``, ``temperature``, ``max_output_tokens`` and ``timeout_s`` go to every
        provider call as given. The hooks, all optional but ``retry_user``:

        - ``retry_user(user, reason)``: the complete user message of the one retry (the original
          plus the feedback, ``Prompt.render_retry``). The router never words feedback itself: the
          wording is a versioned prompt (AGENTS.md §5). If it raises, that propagates.
        - ``before_call()``: awaited before **every** call, retries and fallbacks included (the
          budget reserve, AGENTS.md §6.15). If it raises, the exception propagates as it is, no
          call is made, a held trial is released and nothing counts against the provider.
        - ``validate(result)``: sync, called once per reply, in call order, right after the breaker
          was told. Returns the bad-output reason, or ``None`` to accept (the citation checks). On
          success the last call was on the returned result, so a closure may keep what it
          computed. If it raises, that propagates and is not a provider failure.
        - ``on_attempt(attempt)``: awaited once per **call** (never for a skip) as soon as its
          outcome is known, the verdict of ``validate`` included, and before a retry, the next
          entry, the return or the raise. Usage reaches the pipeline here, so a request that later
          fails (a hook error, the request deadline's cancellation) has still recorded what its
          earlier calls billed. If it raises, that propagates. A cancelled call has no record.
        - ``time_left()``: seconds before the request deadline (the deadline rule). If it raises,
          that propagates and a held trial is released.
        """
        raise NotImplementedError
