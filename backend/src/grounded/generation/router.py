"""The provider router: fallback over an ordered provider list, one circuit breaker per provider
(Tech.md §10, §9.5, PRD D12 and D57, AGENTS.md §6.4 and §6.5).

**Author-owned** (AGENTS.md §3). The contract below was written in ticket 7.02 and implemented in
7.03; the Author said "write it" for this module on 2026-10-14, so the Agent wrote the code (its
line-by-line explanation is in the 7.03 PR). The scenarios are ``tests/unit/test_router.py``.
7.04 wires the router into the pipeline.

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
name (the message names it). ``breakers_enabled=False`` (keyword, default true) is for eval mode
only: see **Breakers off** below.

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
2. some entry was transient: ``AllProvidersUnavailable`` (503 with ``Retry-After``, Tech §10),
   carrying the typed error of every transient entry as ``causes`` and chained ``from`` the last;
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

**Breakers off** (``breakers_enabled=False``, eval mode, ``ProviderRouter`` docstring). ``admit``
always allows, every report is ignored, ``breaker_status`` is always CLOSED: no skip for a breaker,
no state, no ``LOG_BREAKER_OPENED`` and no breaker skip. The rest is unchanged: the retry with
feedback, the fallback order, the attempts and hooks, the deadline gate, the error mapping.

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
Every call that did not succeed leaves one INFO ``LOG_ATTEMPT_FAILED`` with ``provider``,
``outcome``, ``is_retry`` and ``latency_ms`` (nothing else), so a fallback after a timeout, an
outage or a second bad output is never silent. ``generate(log_extra=...)`` (``request_id``) is
merged into the ``extra`` of **every** line the router emits; keys that a log record or the router's
own fields already use are refused with ``LogExtraError``.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta, tzinfo
from enum import StrEnum
from typing import Any, Final
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel

from grounded.generation.providers.base import GenerationResult, LLMProvider, Usage
from grounded.infra.provider_errors import (
    ProviderBadOutput,
    ProviderError,
    ProviderRateLimited,
    ProviderRequestRejected,
    ProviderTimeout,
    ProviderUnavailable,
)
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
LOG_ATTEMPT_FAILED: Final = "provider attempt failed"

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


class LogExtraError(ValueError):
    """``generate(log_extra=...)`` used a key that a log record, or one of the router's own log
    fields, already has; the message names the keys. Raised before any provider is called."""


class AllProvidersUnavailable(ProviderError):  # noqa: N818 (the name is the contract, Tech §10)
    """No provider could answer and at least one was transient (rule 2 of the raised errors).

    ``retry_after_s`` says when a retry could succeed: the smallest remaining open time over the
    breakers that are OPEN at the moment of raising, read from the injected clock (a breaker this
    very call opened counts), **but only when no entry is admissible now**. An entry is admissible
    when its breaker is closed, or half-open with the trial free: a caller may retry at once, and a
    wait taken from another entry's breaker would overstate it. So it is ``None`` when any entry is
    admissible, when no breaker is open (every provider failed this call without reaching a
    threshold, or breakers are off) and when the only gaps are trials in flight. Otherwise it is a
    float ``> 0``. ``attempts`` is the whole routing report of the call, skips included.

    ``causes`` is the typed error each transient entry ended with (a rate limit, quota, timeout or
    outage, also after a bad-output retry), in list order, the very objects the providers raised;
    an entry skipped without a call has none. The last one is also ``__cause__``. It lets a caller
    reach the provider's own error through the wrapper: in eval mode (``breakers_enabled=False``,
    one provider) 7.04 re-raises ``causes[-1]`` so the eval harness sees the same ``is_quota``,
    ``retry_after_s`` or ``BackoffExhaustedError`` as before the router.

    The message names providers and outcomes, never user text or a provider's message. 7.04
    registers the HTTP handler (503 ``provider_unavailable`` with ``Retry-After``).
    """

    def __init__(
        self,
        message: str,
        *,
        retry_after_s: float | None,
        attempts: tuple[Attempt, ...] = (),
        causes: tuple[ProviderError, ...] = (),
    ) -> None:
        super().__init__(message)
        self.retry_after_s = retry_after_s
        self.attempts = attempts
        self.causes = causes


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


# --- Log fields -----------------------------------------------------------------------------------

# The fields the router puts on its own lines. A ``log_extra`` may not use them (it would be
# silently overwritten), nor any attribute a log record has (``logging`` raises ``KeyError`` for
# those), so a clash is a ``LogExtraError`` at the call, not a surprise in a log line.
_RESERVED_LOG_KEYS: Final = (
    frozenset(logging.LogRecord("", 0, "", 0, "", None, None).__dict__)
    | {"message", "asctime"}
    | {"provider", "reason", "open_s", "derived", "remaining_s", "status_code", "zone"}
    | {"outcome", "is_retry", "latency_ms"}
)


def _checked_log_extra(log_extra: Mapping[str, object] | None) -> dict[str, object]:
    """A copy of ``log_extra`` (so a caller mutating its dict later changes nothing), or
    ``LogExtraError`` naming the keys that clash."""
    extra = dict(log_extra or {})
    clashing = sorted(_RESERVED_LOG_KEYS.intersection(extra))
    if clashing:
        raise LogExtraError(
            f"log_extra may not use {', '.join(clashing)}: the log already has them"
        )
    return extra


def _fields(log_extra: Mapping[str, object] | None, **fields: object) -> dict[str, object]:
    """The ``extra`` of one log line: the caller's ``log_extra`` plus the line's own fields."""
    return {**(log_extra or {}), **fields}


# --- Pure rules -----------------------------------------------------------------------------------

# A wait in seconds and how it was obtained. ``quota_reset_s`` and ``open_plan`` share the provider
# rules through ``_quota_wait`` below, so the number and the ``Derivation`` that explains it can
# never disagree.
type _Wait = tuple[float, Derivation]


def _usable_wait(retry_after_s: float | None, max_retry_after_s: float) -> _Wait | None:
    """A provider-supplied wait, if it is one: positive (so ``None``, zero, negative and NaN are
    absent; ``not x > 0`` is true for NaN, ``x <= 0`` would not be), cut to the cap."""
    if retry_after_s is None or not retry_after_s > 0:
        return None
    if retry_after_s > max_retry_after_s:  # also +inf; a wait equal to the cap was not cut
        return max_retry_after_s, Derivation.RETRY_AFTER_CAPPED
    return retry_after_s, Derivation.RETRY_AFTER


def _seconds_to_next_midnight(
    now: datetime, load_zone: ZoneLoader, log_extra: Mapping[str, object] | None
) -> float | None:
    """Seconds from ``now`` to the next 00:00 in ``GEMINI_QUOTA_ZONE``; ``None`` (and one warning)
    when the zone cannot be loaded."""
    if now.utcoffset() is None:  # naive: ``tzinfo`` is None, or its ``utcoffset`` says so
        raise ValueError(f"now must be timezone-aware, got {now!r}")
    try:
        zone = load_zone(GEMINI_QUOTA_ZONE)
    except (ZoneInfoNotFoundError, OSError, ValueError):
        logger.warning(
            LOG_ZONE_UNAVAILABLE,
            extra=_fields(log_extra, provider=GEMINI, zone=GEMINI_QUOTA_ZONE),
        )
        return None
    # The local calendar date of ``now`` plus one day, at 00:00 local time. Adding the day to the
    # *date* (not 24 h to the instant) is what makes a 23 h or 25 h daylight-saving day come out
    # right, and the next midnight is always strictly after ``now``: at exactly 00:00 it is the
    # following one, a full day away. Midnight is never skipped or repeated in this zone (its
    # changes happen at 02:00), so the constructor needs no ``fold`` care.
    tomorrow = now.astimezone(zone).date() + timedelta(days=1)
    next_midnight = datetime(tomorrow.year, tomorrow.month, tomorrow.day, tzinfo=zone)
    # Both sides go to UTC first: subtracting two aware datetimes that share one ``tzinfo`` object
    # compares their wall clocks and ignores the offsets, which is wrong across a change-over (and
    # ``now`` may arrive in this very zone).
    return (next_midnight.astimezone(UTC) - now.astimezone(UTC)).total_seconds()


def _quota_wait(
    provider: str,
    now: datetime,
    retry_after_s: float | None,
    max_retry_after_s: float,
    load_zone: ZoneLoader,
    log_extra: Mapping[str, object] | None,
) -> _Wait | None:
    """The per-provider rules of D57b for when a daily quota is back, or ``None``."""
    if provider == GEMINI:
        seconds = _seconds_to_next_midnight(now, load_zone, log_extra)
        return None if seconds is None else (seconds, Derivation.QUOTA_RESET)
    if provider == GROQ:
        return _usable_wait(retry_after_s, max_retry_after_s)
    return None


def quota_reset_s(
    provider: str,
    now: datetime,
    retry_after_s: float | None,
    *,
    max_retry_after_s: float,
    load_zone: ZoneLoader = ZoneInfo,
    log_extra: Mapping[str, object] | None = None,
) -> float | None:
    """Seconds until the daily quota of ``provider`` resets, or ``None`` when not derivable. Pure
    apart from the one warning below; ``now`` is the injected wall clock.

    - ``"gemini"``: until the **next** midnight in ``GEMINI_QUOTA_ZONE`` strictly after ``now``
      (exactly at midnight it is the following one, a full day away), counted between instants, so
      the 23 h and 25 h days of the daylight-saving changes come out right. ``retry_after_s`` is
      ignored (Gemini's ``retryDelay`` on a daily 429 may be a few seconds). ``now`` must be
      timezone-aware, any zone, else ``ValueError`` (the message says "aware"). If ``load_zone``
      raises (``ZoneInfoNotFoundError``, ``OSError`` or ``ValueError``) the answer is ``None`` and
      one WARNING ``LOG_ZONE_UNAVAILABLE`` is logged with ``provider`` and ``zone`` fields
      (plus ``log_extra``, which the caller has already checked: ``generate`` does).
    - ``"groq"``: its own ``retry_after_s`` cut to ``max_retry_after_s`` (its docs state no reset
      time and its one observed daily 429 refills continuously, PRD D57b). ``None`` when that is
      absent: ``None``, zero, negative or NaN. ``+inf`` is cut to the cap.
    - any other provider: ``None``.
    """
    wait = _quota_wait(provider, now, retry_after_s, max_retry_after_s, load_zone, log_extra)
    return None if wait is None else wait[0]


def open_plan(
    provider: str,
    error: ProviderRateLimited,
    now: datetime,
    config: BreakerConfig,
    *,
    load_zone: ZoneLoader = ZoneInfo,
    log_extra: Mapping[str, object] | None = None,
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
    ``log_extra`` only decorates the one warning ``quota_reset_s`` may log.
    """
    if error.is_quota:
        wait = _quota_wait(
            provider, now, error.retry_after_s, config.max_retry_after_s, load_zone, log_extra
        )
        if wait is None:
            return OpenPlan(config.quota_open_s, OpenReason.QUOTA, Derivation.DEFAULT)
        return OpenPlan(wait[0], OpenReason.QUOTA, wait[1])
    wait = _usable_wait(error.retry_after_s, config.max_retry_after_s)
    if wait is None:
        return OpenPlan(config.rate_limit_open_s, OpenReason.RATE_LIMIT, Derivation.DEFAULT)
    return OpenPlan(wait[0], OpenReason.RATE_LIMIT, wait[1])


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
        self.name = name  # for debugging only: the state machine itself never needs it
        self._config = config
        # The whole state is these three fields. ``_open_until`` is the monotonic instant the open
        # time ends, ``None`` while CLOSED; OPEN and HALF_OPEN are derived from it and the clock
        # when asked (``status``), so no timer task is needed and nothing can fire late.
        self._open_until: float | None = None
        self._trial_in_flight = False
        self._failures: deque[float] = deque()  # when each failure of the current run was seen

    def _applies(self, admission: Admission) -> bool:
        """Whether a reported outcome still counts. A call admitted while closed is a straggler,
        and ignored, once the breaker is no longer closed. The trial counts while it is held."""
        if admission is Admission.ALLOWED:
            return self._open_until is None
        return admission is Admission.TRIAL and self._trial_in_flight

    def _open(self, now: float, seconds: float) -> None:
        self._open_until = now + seconds
        self._trial_in_flight = False  # an open breaker has no trial; it gets one when half-open
        self._failures.clear()

    def _close(self) -> None:
        self._open_until = None
        self._trial_in_flight = False
        self._failures.clear()

    def status(self, now: float) -> BreakerStatus:
        """A read-only view, never a transition. OPEN while ``now`` is before the end of the open
        time (``remaining_s`` is what is left), HALF_OPEN from then on until a trial closes it."""
        if self._open_until is None:
            return BreakerStatus(BreakerState.CLOSED, None, False)
        if now < self._open_until:
            return BreakerStatus(BreakerState.OPEN, self._open_until - now, False)
        # ``now == _open_until`` counts as elapsed: "the open time has elapsed" means half-open.
        return BreakerStatus(BreakerState.HALF_OPEN, None, self._trial_in_flight)

    def admit(self, now: float) -> Admission:
        """CLOSED: ``ALLOWED``. OPEN: ``SKIP_OPEN``. HALF_OPEN: ``TRIAL`` for the first caller (it
        now holds the trial), ``SKIP_TRIAL`` for everyone else until the trial is resolved."""
        state = self.status(now).state
        if state is BreakerState.CLOSED:
            return Admission.ALLOWED
        if state is BreakerState.OPEN:
            return Admission.SKIP_OPEN
        if self._trial_in_flight:
            return Admission.SKIP_TRIAL
        # No ``await`` anywhere in this method: the check above and this assignment cannot be
        # interleaved with another request, which is what makes "exactly one trial" true.
        self._trial_in_flight = True
        return Admission.TRIAL

    def record_reply(self, admission: Admission, now: float) -> None:
        """The provider answered, valid or not: clear the failure run, and a ``TRIAL`` closes the
        breaker."""
        if self._applies(admission):
            # While closed this only clears the run; for the trial it is the close. ``now`` is
            # unused: the method keeps the shape of the other reports.
            self._close()

    def record_failure(self, admission: Admission, now: float) -> float | None:
        """A timeout or an unavailable provider. ``ALLOWED`` while closed: add ``now`` to the run,
        drop failures older than ``failure_window_s`` (a sliding window ending at ``now``), and when
        the run reaches ``failure_threshold`` open for ``failure_open_s`` and clear it. ``TRIAL``:
        open again for ``failure_open_s`` with a fresh timer. Returns the seconds it opened for,
        else ``None`` (also for an ignored straggler)."""
        if not self._applies(admission):
            return None
        if admission is Admission.ALLOWED:
            self._failures.append(now)
            # Strictly older than the window drops out: a failure exactly ``failure_window_s`` old
            # still counts ("within 60 s"). The loop cannot empty the run: the failure just added
            # is 0 s old and the window is > 0 (``BreakerConfig`` checks it).
            while now - self._failures[0] > self._config.failure_window_s:
                self._failures.popleft()
            if len(self._failures) < self._config.failure_threshold:
                return None
        self._open(now, self._config.failure_open_s)  # the threshold, or a failed trial
        return self._config.failure_open_s

    def open_for(self, admission: Admission, now: float, seconds: float) -> bool:
        """A 429, already turned into ``seconds`` (``open_plan``): open now for that long, clear the
        failure run, release a held trial. Returns whether it opened (``False`` for an ignored
        straggler)."""
        if not self._applies(admission):
            return False
        self._open(now, seconds)
        return True

    def release(self, admission: Admission) -> None:
        """Give a ``TRIAL`` back with no verdict (a deadline refusal, a hook error, a cancellation,
        a rejected request): HALF_OPEN stays and the next caller may take the trial. Nothing for an
        ``ALLOWED``."""
        if admission is Admission.TRIAL:
            self._trial_in_flight = False


class _DisabledBreaker(CircuitBreaker):
    """A breaker that never decides anything: always CLOSED, every report ignored, no state. The
    router builds these when it is given ``breakers_enabled=False``, so the call path has no
    special cases for eval mode."""

    def status(self, now: float) -> BreakerStatus:
        return BreakerStatus(BreakerState.CLOSED, None, False)

    def admit(self, now: float) -> Admission:
        return Admission.ALLOWED

    def record_reply(self, admission: Admission, now: float) -> None:
        return None

    def record_failure(self, admission: Admission, now: float) -> float | None:
        return None

    def open_for(self, admission: Admission, now: float, seconds: float) -> bool:
        return False

    def release(self, admission: Admission) -> None:
        return None


# --- The router -----------------------------------------------------------------------------------

# The five errors the router has a rule for (the table of Tech §10). Anything else a provider
# raises, an unknown ``ProviderError`` subclass included, is not caught: it propagates.
type _RoutedError = (
    ProviderBadOutput
    | ProviderRateLimited
    | ProviderTimeout
    | ProviderUnavailable
    | ProviderRequestRejected
)

# What one call, or one provider entry, came to: the accepted reply, the typed error that ended it,
# or ``None`` when the breaker skipped it without a call.
type _Ended[T: BaseModel] = GenerationResult[T] | ProviderError | None


@dataclass(frozen=True, slots=True)
class _Request[T: BaseModel]:
    """The arguments and hooks of one ``generate`` call, bundled so the private steps take one
    parameter instead of eleven. ``user`` is the original message; a retry's is passed beside it.
    ``log_extra`` is the caller's, already checked and copied."""

    system: str
    user: str
    schema: type[T]
    temperature: float
    max_output_tokens: int
    timeout_s: float
    retry_user: RetryMessage
    before_call: BeforeCall | None
    validate: Validate[T] | None
    on_attempt: OnAttempt | None
    time_left: TimeLeft | None
    log_extra: Mapping[str, object]


def _billed(answer: GenerationResult[Any] | ProviderError) -> tuple[Usage, bool]:
    """The usage and the cache flag a call is accounted with: a reply's own, the tokens a bad
    output reports, else nothing."""
    if isinstance(answer, GenerationResult):
        return answer.usage, answer.cache_hit
    if isinstance(answer, ProviderBadOutput):
        return Usage(input_tokens=answer.input_tokens, output_tokens=answer.output_tokens), False
    return Usage(input_tokens=0, output_tokens=0), False


def _skip(
    report: list[Attempt],
    provider: LLMProvider,
    outcome: AttemptOutcome,
    reason: str,
    remaining_s: float | None,
    is_retry: bool,
    log_extra: Mapping[str, object],
) -> None:
    """Record an entry skipped without a call, and log it (INFO, no user text)."""
    report.append(Attempt(provider.name, provider.model, outcome, is_retry=is_retry))
    logger.info(
        LOG_PROVIDER_SKIPPED,
        extra=_fields(log_extra, provider=provider.name, reason=reason, remaining_s=remaining_s),
    )


def _log_opened(
    provider: str,
    reason: OpenReason,
    open_s: float,
    derived: Derivation,
    log_extra: Mapping[str, object],
) -> None:
    logger.warning(
        LOG_BREAKER_OPENED,
        extra=_fields(
            log_extra, provider=provider, reason=reason.value, open_s=open_s, derived=derived.value
        ),
    )


class ProviderRouter:
    """Fallback over ``providers``, one circuit breaker each (the module docstring has the rules).

    ``breakers_enabled=False`` is for eval mode (``APP_ENV=eval``) and nothing else. The router then
    never skips an entry for a breaker and never changes breaker state: ``admit`` always allows,
    the reports are ignored, ``breaker_status`` is always CLOSED, and there is no ``breaker opened``
    or breaker ``provider skipped`` log. Everything else is unchanged: the retry on bad output with
    feedback, the fallback order, the attempts and their hooks, the deadline gate (its refusal is
    still logged as a skip) and the error mapping.

    *Why a breaker is wrong in an eval.* The eval has its own backoff and a "daily quota stop"
    (``ask_batch``, ``promptfoo_provider``) that must see every real provider error. With a breaker,
    one 429 would turn every later golden question into a skip with no call at all: the eval's
    backoff would never get to wait or give up, and its quota logic would never see the quota. The
    wrapper still exists for a single failing provider, so the eval harness reads the provider's own
    error from ``AllProvidersUnavailable.causes`` (7.04 re-raises ``causes[-1]``). Eval mode has
    exactly one provider, and ``Settings`` already enforces that at startup; the router does not
    check it.
    """

    def __init__(
        self,
        providers: Sequence[LLMProvider],
        *,
        config: BreakerConfig,
        clock: RouterClock = SYSTEM_CLOCK,
        breakers_enabled: bool = True,
    ) -> None:
        if not providers:
            raise RouterConfigError("a router needs at least one provider")
        names = [provider.name for provider in providers]
        for name in names:
            if names.count(name) > 1:  # a handful of entries: no need for a Counter
                raise RouterConfigError(f"provider names must be unique, {name!r} is used twice")
        self._providers = tuple(providers)  # a copy: the caller's list may change later
        self._config = config
        self._clock = clock
        self._breakers_enabled = breakers_enabled
        # Per instance, never at module level: two routers (two tests, two apps) share nothing.
        breaker_type = CircuitBreaker if breakers_enabled else _DisabledBreaker
        self._breakers = {name: breaker_type(name, config) for name in names}

    def breaker_status(self, provider: str) -> BreakerStatus:
        """The named provider's breaker as of now (the injected clock). ``UnknownProviderError`` for
        a name not in the list. Read-only: it never moves a breaker."""
        breaker = self._breakers.get(provider)
        if breaker is None:
            raise UnknownProviderError(f"no provider named {provider!r} in this router")
        return breaker.status(self._clock.monotonic())

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
        log_extra: Mapping[str, object] | None = None,
    ) -> RoutedGeneration[T]:
        """Route one request over the providers (the module docstring has the rules).

        ``system``, ``schema``, ``temperature``, ``max_output_tokens`` and ``timeout_s`` go to every
        provider call as given. The hooks, all optional but ``retry_user``:

        - ``retry_user(user, reason)``: the complete user message of the one retry (the original
          plus the feedback, ``Prompt.render_retry``). The router never words feedback itself: the
          wording is a versioned prompt (AGENTS.md §5). It must be pure and must not raise. If it
          does, that propagates unchanged: the first call's attempt was already reported, nothing
          is held, and the request fails as an internal error (the retry is not made).
        - ``before_call()``: awaited before **every** call, retries and fallbacks included (the
          budget reserve, AGENTS.md §6.15). If it raises, the exception propagates as it is, no
          call is made, a held trial is released and nothing counts against the provider.
        - ``validate(result)``: sync, called once per reply, in call order, right after the breaker
          was told. Returns the bad-output reason, or ``None`` to accept (the citation checks). On
          success the last call was on the returned result, so a closure may keep what it
          computed. It must be pure and must not raise. If it does, that propagates unchanged and
          is not a provider failure: the breaker has already heard the reply (a trial is closed,
          not lost), but ``on_attempt`` has not run for that call, so its usage is not reported.
          A bug there is an internal error, not something to route around.
        - ``on_attempt(attempt)``: awaited once per **call** (never for a skip) as soon as its
          outcome is known, the verdict of ``validate`` included, and before a retry, the next
          entry, the return or the raise. Usage reaches the pipeline here, so a request that later
          fails (a hook error, the request deadline's cancellation) has still recorded what its
          earlier calls billed. If it raises, that propagates. A cancelled call has no record.
        - ``time_left()``: seconds before the request deadline (the deadline rule). If it raises,
          that propagates and a held trial is released.
        - ``log_extra``: fields (``request_id``) merged into the ``extra`` of every log line this
          call emits. A key that a log record or one of the router's own fields already has is
          ``LogExtraError``, raised before anything else happens.
        """
        request = _Request(
            system=system,
            user=user,
            schema=schema,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            timeout_s=timeout_s,
            retry_user=retry_user,
            before_call=before_call,
            validate=validate,
            on_attempt=on_attempt,
            time_left=time_left,
            log_extra=_checked_log_extra(log_extra),
        )
        report: list[Attempt] = []  # per request, so concurrent requests never share it
        transient = False  # some entry was skipped, rate limited, out of quota, timed out or down
        causes: list[ProviderError] = []  # the typed error of each transient entry that had one
        bad_outputs: list[ProviderBadOutput] = []  # one per entry that ended in bad output
        rejected: list[ProviderRequestRejected] = []  # one per entry that rejected the request

        for index, provider in enumerate(self._providers):
            ended = await self._route_entry(request, provider, report)
            if isinstance(ended, GenerationResult):
                return RoutedGeneration(
                    result=ended,
                    provider=provider.name,
                    model=provider.model,
                    fallback_used=index > 0,  # also when the first entry was only skipped
                    attempts=tuple(report),
                )
            if isinstance(ended, ProviderBadOutput):
                bad_outputs.append(ended)
            elif isinstance(ended, ProviderRequestRejected):
                rejected.append(ended)
            else:
                transient = True
                if ended is not None:  # None: skipped without a call, so no error to keep
                    causes.append(ended)

        # Nobody answered. The order is the contract: a deadline refusal has already raised above;
        # then any transient entry, then bad output (the last one), then rejection (the first).
        if transient:
            trail = ", ".join(f"{a.provider}: {a.outcome.value}" for a in report)
            unavailable = AllProvidersUnavailable(
                f"no provider could answer ({trail})",
                retry_after_s=self._retry_after_s(),
                attempts=tuple(report),
                causes=tuple(causes),
            )
            if causes:
                raise unavailable from causes[-1]
            raise unavailable
        if bad_outputs:
            raise bad_outputs[-1]
        # Every entry rejected the request (the router has at least one entry, and each entry
        # lands in exactly one of the four cases above).
        raise rejected[0]

    def _retry_after_s(self) -> float | None:
        """When a retry could succeed: the smallest time left over the OPEN breakers (a breaker
        this very call opened counts), but only if no entry is admissible now, meaning a closed
        breaker or a half-open one whose trial is free. Then a caller may retry at once, and a wait
        taken from another entry's breaker would overstate it. ``None`` otherwise."""
        now = self._clock.monotonic()
        statuses = [breaker.status(now) for breaker in self._breakers.values()]
        if any(
            status.state is BreakerState.CLOSED
            or (status.state is BreakerState.HALF_OPEN and not status.trial_in_flight)
            for status in statuses
        ):
            return None
        left = [status.remaining_s for status in statuses if status.remaining_s is not None]
        return min(left, default=None)

    async def _route_entry[T: BaseModel](
        self, req: _Request[T], provider: LLMProvider, report: list[Attempt]
    ) -> _Ended[T]:
        """The call on one provider and, after retryable bad output, its one retry. Written as two
        straight calls, not a loop: "at most one retry" holds by construction."""
        ended = await self._call(req, provider, req.user, False, report)
        if isinstance(ended, ProviderBadOutput) and ended.retryable:
            # The retry is built from the *original* message, and only after ``on_attempt`` has
            # recorded the first call (``_call`` awaited it), so a ``retry_user`` error loses no
            # usage.
            retry_user = req.retry_user(req.user, ended.validation_error or str(ended))
            ended = await self._call(req, provider, retry_user, True, report)
        return ended

    async def _call[T: BaseModel](
        self,
        req: _Request[T],
        provider: LLMProvider,
        user: str,
        is_retry: bool,
        report: list[Attempt],
    ) -> _Ended[T]:
        """One call to ``provider``: the gates, ``before_call``, the call, the breaker's verdict,
        ``validate``, the report and ``on_attempt``, in that order.

        The rule that keeps a half-open trial from ever being lost lives here. ``admit`` may hand
        this call the trial, and there is no ``await`` between it and the ``try``. From the ``try``
        on, every path either reaches ``settled = True`` (``_tell_breaker`` gave the breaker a
        verdict, or handed the trial back itself) or leaves through the ``finally``, which hands it
        back: a deadline refusal, a ``time_left`` or ``before_call`` error, a cancellation, a bug in
        an adapter. Everything after the ``finally`` (``validate``, ``on_attempt``) runs with the
        breaker already told, so an error or a cancellation there cannot leave it claimed.
        """
        breaker = self._breakers[provider.name]
        now = self._clock.monotonic()
        admission = breaker.admit(now)
        if admission is Admission.SKIP_OPEN:
            remaining_s = breaker.status(now).remaining_s
            _skip(
                report,
                provider,
                AttemptOutcome.SKIPPED_OPEN,
                "breaker_open",
                remaining_s,
                is_retry,
                req.log_extra,
            )
            return None
        if admission is Admission.SKIP_TRIAL:
            _skip(
                report,
                provider,
                AttemptOutcome.SKIPPED_TRIAL,
                "trial_in_flight",
                None,
                is_retry,
                req.log_extra,
            )
            return None

        settled = False
        try:
            # The deadline gate comes after the breaker (only an entry that would be called is
            # looked at) and before ``before_call`` (a refused call must not spend budget).
            self._check_deadline(req, provider, is_retry, report)
            if req.before_call is not None:
                await req.before_call()
            started = self._clock.monotonic()
            answer: GenerationResult[T] | _RoutedError
            try:
                answer = await provider.generate(
                    system=req.system,
                    user=user,
                    schema=req.schema,
                    temperature=req.temperature,
                    max_output_tokens=req.max_output_tokens,
                    timeout_s=req.timeout_s,  # the adapter enforces it; the router adds none
                )
            except (
                ProviderBadOutput,
                ProviderRateLimited,
                ProviderTimeout,
                ProviderUnavailable,
                ProviderRequestRejected,
            ) as error:
                answer = error
            # The moment the outcome is seen: the open time of a failure starts here, not when the
            # call began.
            finished = self._clock.monotonic()
            outcome = self._tell_breaker(
                provider.name, breaker, admission, answer, finished, req.log_extra
            )
            settled = True
        finally:
            if not settled:
                breaker.release(admission)

        usage, cache_hit = _billed(answer)
        if isinstance(answer, GenerationResult) and req.validate is not None:
            reason = req.validate(answer)
            if reason is not None:
                answer = ProviderBadOutput(
                    "the answer failed the citation checks",
                    raw=answer.raw_text,
                    validation_error=reason,
                )
                outcome = AttemptOutcome.BAD_OUTPUT
        attempt = Attempt(
            provider=provider.name,
            model=provider.model,
            outcome=outcome,
            latency_ms=round((finished - started) * 1000),
            is_retry=is_retry,
            usage=usage,
            cache_hit=cache_hit,
        )
        report.append(attempt)
        if outcome is not AttemptOutcome.SUCCESS:
            # The trace of a fallback: logged as soon as the outcome is final, before the awaited
            # hook, and with nothing the provider or the user wrote.
            logger.info(
                LOG_ATTEMPT_FAILED,
                extra=_fields(
                    req.log_extra,
                    provider=provider.name,
                    outcome=outcome.value,
                    is_retry=is_retry,
                    latency_ms=attempt.latency_ms,
                ),
            )
        if req.on_attempt is not None:
            await req.on_attempt(attempt)
        return answer

    def _check_deadline(
        self, req: _Request[Any], provider: LLMProvider, is_retry: bool, report: list[Attempt]
    ) -> None:
        """Raise ``RouterDeadlineError`` unless the attempt fits in the time left: it starts only if
        ``time_left() >= timeout_s``, and its timeout is never shortened to fit (a timeout the
        router imposed would be counted as the provider's failure)."""
        if req.time_left is None:
            return
        left = req.time_left()
        if left >= req.timeout_s:
            return
        # Also reached for NaN, the safe side: a call is started only if it is shown to fit.
        _skip(
            report,
            provider,
            AttemptOutcome.SKIPPED_DEADLINE,
            "deadline",
            None,
            is_retry,
            req.log_extra,
        )
        trail = ", ".join(f"{a.provider}: {a.outcome.value}" for a in report)
        raise RouterDeadlineError(
            f"{provider.name} not called: {left:.1f} s left, "
            f"{req.timeout_s:.1f} s needed ({trail})",
            time_left_s=left,
            timeout_s=req.timeout_s,
            attempts=tuple(report),
        )

    def _tell_breaker(
        self,
        provider: str,
        breaker: CircuitBreaker,
        admission: Admission,
        answer: GenerationResult[Any] | _RoutedError,
        now: float,
        log_extra: Mapping[str, object],
    ) -> AttemptOutcome:
        """The event table of Tech §10 as code: tell the breaker what came back and name the
        outcome. Synchronous on purpose: the verdict is in the breaker before anything else can
        run. (``validate`` has not run yet, so a reply is ``SUCCESS`` here; ``_call`` turns it into
        ``BAD_OUTPUT`` if the citation checks reject it.)"""
        if isinstance(answer, GenerationResult):
            breaker.record_reply(admission, now)
            return AttemptOutcome.SUCCESS
        if isinstance(answer, ProviderBadOutput):
            breaker.record_reply(admission, now)  # proof of life, not a failure
            return AttemptOutcome.BAD_OUTPUT
        if isinstance(answer, ProviderRateLimited):
            if self._breakers_enabled:  # a disabled breaker has nothing to open: no plan, no log
                plan = open_plan(
                    provider,
                    answer,
                    self._clock.wall(),
                    self._config,
                    load_zone=self._clock.load_zone,
                    log_extra=log_extra,
                )
                if breaker.open_for(admission, now, plan.seconds):
                    _log_opened(provider, plan.reason, plan.seconds, plan.derived, log_extra)
            return AttemptOutcome.QUOTA if answer.is_quota else AttemptOutcome.RATE_LIMITED
        if isinstance(answer, ProviderTimeout | ProviderUnavailable):
            opened_for = breaker.record_failure(admission, now)
            if opened_for is not None:
                _log_opened(
                    provider, OpenReason.FAILURES, opened_for, Derivation.DEFAULT, log_extra
                )
            if isinstance(answer, ProviderTimeout):
                return AttemptOutcome.TIMEOUT
            return AttemptOutcome.UNAVAILABLE
        # ProviderRequestRejected: our key, model or request, not the provider's health. No verdict
        # (a held trial goes back), and a loud log with the status code only, never the text.
        breaker.release(admission)
        logger.warning(
            LOG_REQUEST_REJECTED,
            extra=_fields(log_extra, provider=provider, status_code=answer.status_code),
        )
        return AttemptOutcome.REJECTED
