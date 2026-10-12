"""Timed requests against a live deployment: cold start and warm latency (ticket 5.10).

``grounded measure-cold-start`` drives this module. It answers one question with numbers instead
of guesses: does what happens *before* the backend handler's 25 s deadline starts (the function
start) fit in the 10 s between that deadline and the proxy's 35 s fetch timeout (Tech §18, D53)?
The Neon wake runs inside the deadline (the first database call), so including it, as the health
order does, makes the estimate stricter. When it does not fit, the UI's retry state covers it and
the numbers say so.

What it sends, and what it never sends:

* ``GET /healthz`` of the backend: public, no secret. It touches nothing but the function
  (function cold start).
* ``GET /readyz`` of the backend, with ``X-Proxy-Secret``: it queries the database (Neon wake), so
  it needs the proxy secret (PRD D54c). The secret comes from ``Settings`` (``PROXY_SHARED_SECRET``,
  the production value exported by the operator), goes on this one request only, over https
  (``_check_url`` allows plain http for localhost only), and is never printed, logged or put in an
  error message. ``Plan`` keeps it out of its ``repr``.
* ``POST /api/ask`` of the **frontend**: the public route a visitor uses, without any secret (the
  frontend adds its own).

Each request opens a new connection (``urllib`` keeps none alive), so every figure includes DNS,
TCP and TLS setup: tens of milliseconds, which does not matter next to a cold start of seconds.

The ask calls use real quota (one query embedding and one or two LLM calls each, a share of
``DAILY_LLM_BUDGET`` and of the per-IP limit), which is why their number is an explicit, capped
argument and a failed ask is never retried (AGENTS.md §6.15).
"""

from __future__ import annotations

import json
import statistics
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Literal, cast
from urllib.parse import urlsplit

# Most real questions one run may send. A cold run needs 1 and a useful warm sample 3 to 5; more is
# quota spent for no insight (PRD §12: the Gemini quota is shared with the CI evals).
MAX_ASKS = 10
MAX_PROBES = 30

# PRD NFR-1: warm p50 end to end. Printed next to the warm median as a reference, not a gate: a
# run of a few questions says nothing about p95.
WARM_P50_TARGET_S = 4.0

# Different questions, so the later ones are not answer-cache hits (Tech §11): a hit skips the LLM
# and would make the warm figure look better than the pipeline is. A `--first ask` run starts in the
# middle of the list (``question_offset``), so the runbook's second run does not ask what the first
# one already cached (as long as each run asks at most half of the list).
DEFAULT_QUESTIONS: tuple[str, ...] = (
    "How do I declare an optional query parameter with a default value?",
    "How do I add a dependency to a path operation?",
    "How do I return a custom status code from a path operation?",
    "How do I read form data instead of JSON?",
    "How do I raise an HTTP error with a custom response body?",
    "How do I validate a path parameter with constraints?",
    "How do I set the response model of an endpoint?",
    "How do I run code on application startup and shutdown?",
    "How do I add CORS to a FastAPI application?",
    "How do I declare a request body with nested models?",
)


class MeasureError(ValueError):
    """The plan cannot be run as given (bad URL, contradictory options)."""


class Step(StrEnum):
    HEALTHZ = "healthz"
    READYZ = "readyz"
    ASK = "ask"


class Phase(StrEnum):
    COLD = "cold"  # the first request of the run: the idle period came right before it
    WARM = "warm"
    AFTER_PROBES = "after probes"  # an ask whose function and database earlier probes woke


@dataclass(frozen=True)
class Plan:
    frontend_url: str
    backend_url: str | None
    idle_minutes: float
    first: Literal["health", "ask"] = "health"
    asks: int = 4  # the first one is the cold (or after-probes) one, the rest are warm
    warm_probes: int = 3
    questions: Sequence[str] = DEFAULT_QUESTIONS
    timeout_s: float = 45.0  # above the frontend route's maxDuration (40 s): see the platform's 504
    margin_s: float = 10.0  # proxy timeout 35 s minus backend deadline 25 s (Tech §18)
    waited: bool = False  # the run slept the idle period itself, rather than the operator having
    # Sent as X-Proxy-Secret on GET /readyz only (PRD D54c). Never in the repr, so never printed.
    proxy_secret: str | None = field(default=None, repr=False)
    # The backend's RETRIEVAL_TIMEOUT_S (PRD D54a): the cold ask's retrieval time is compared to it.
    retrieval_timeout_s: float = 2.0

    def validate(self) -> None:
        _check_url("--frontend-url", self.frontend_url)
        if self.backend_url is not None:
            _check_url("--backend-url", self.backend_url)
            if not self.proxy_secret:
                raise MeasureError(
                    "--backend-url probes /readyz, which needs the proxy secret: export "
                    "PROXY_SHARED_SECRET (the production value from Vercel) first"
                )
        if not 0 <= self.asks <= MAX_ASKS:
            raise MeasureError(f"--asks must be between 0 and {MAX_ASKS} (it spends real quota)")
        if not 0 <= self.warm_probes <= MAX_PROBES:
            raise MeasureError(f"--warm-probes must be between 0 and {MAX_PROBES}")
        if self.idle_minutes < 0 or not self.questions:
            raise MeasureError("--idle-minutes cannot be negative and a question is needed")
        if self.first == "health" and self.backend_url is None:
            raise MeasureError("--first health probes the backend: pass --backend-url")
        if self.first == "ask" and self.asks < 1:
            raise MeasureError("--first ask needs --asks of at least 1")
        if self.retrieval_timeout_s <= 0:
            raise MeasureError("RETRIEVAL_TIMEOUT_S must be positive")


def question_offset(plan: Plan) -> int:
    """Where the asks start in ``plan.questions``.

    The runbook runs ``--first health`` and then ``--first ask`` on the same deployment. With the
    default questions both would send the same first questions, so every ask of the second run would
    be an answer-cache hit of the first. A ``--first ask`` run with the default questions therefore
    starts halfway through the list; a list the operator chose starts at its beginning.
    """
    if plan.first == "ask" and tuple(plan.questions) == DEFAULT_QUESTIONS:
        return len(DEFAULT_QUESTIONS) // 2
    return 0


def _check_url(name: str, url: str) -> None:
    parts = urlsplit(url)
    local = parts.hostname in ("localhost", "127.0.0.1", "::1")
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise MeasureError(f"{name} must be an http(s) URL, got {url!r}")
    if parts.scheme == "http" and not local:
        raise MeasureError(f"{name} must be https (plain http is for localhost only)")
    if parts.username or parts.password or parts.query:
        raise MeasureError(f"{name} must be a plain origin: no credentials or query string")


@dataclass(frozen=True)
class Sample:
    step: Step
    phase: Phase
    status: int | None  # None: no HTTP response at all (timeout, refused, DNS)
    elapsed_ms: float
    error: str | None = None  # the API's error code, or the exception class name
    total_ms: int | None = None  # ask only: meta.latency_ms.total, the server's own view
    cache_hit: bool | None = None  # ask only: meta.cache_hit
    vercel_id: str | None = None  # x-vercel-id: the regions the request went through
    retrieval_ms: int | None = None  # ask only: meta.latency_ms.retrieval (none on a cache hit)
    request_id: str | None = None  # ask only: the backend's request_id (its request_logs row)

    @property
    def ok(self) -> bool:
        return self.status == 200


@dataclass(frozen=True)
class HttpResult:
    status: int | None
    body: bytes
    headers: dict[str, str]
    error: str | None


def http_call(
    method: str,
    url: str,
    *,
    body: bytes | None,
    timeout_s: float,
    extra_headers: Mapping[str, str] | None = None,
) -> HttpResult:
    """One request, one connection. Never raises: a failure is a result with ``status`` None."""
    headers = {"user-agent": "grounded-measure-cold-start/1", "accept": "application/json"}
    headers.update(extra_headers or {})
    if body is not None:
        headers["content-type"] = "application/json"
    request = urllib.request.Request(url, data=body, method=method, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            status, payload, got = response.status, response.read(), response.headers
    except urllib.error.HTTPError as exc:  # a 4xx/5xx still has a body and headers worth reading
        status, got = exc.code, exc.headers
        try:
            payload = exc.read()
        except OSError:
            payload = b""
    except urllib.error.URLError as exc:
        return HttpResult(None, b"", {}, type(exc.reason).__name__)
    except OSError as exc:  # includes TimeoutError while reading the body
        return HttpResult(None, b"", {}, type(exc).__name__)
    return HttpResult(status, payload, {k.lower(): v for k, v in got.items()}, None)


def _dict(value: object) -> dict[str, object]:
    return cast("dict[str, object]", value) if isinstance(value, dict) else {}


@dataclass(frozen=True)
class Details:
    """What one response says about itself, beyond its status and time."""

    error: str | None = None
    total_ms: int | None = None
    cache_hit: bool | None = None
    retrieval_ms: int | None = None
    request_id: str | None = None


def _details(step: Step, result: HttpResult) -> Details:
    """The error label and, for an ask, the server-reported timings, cache flag and request id."""
    if result.status is None:
        return Details(error=result.error)
    if step is not Step.ASK:
        if result.status == 401:  # /readyz refused X-Proxy-Secret: not the backend's value
            return Details(error="unauthorized")
        return Details(error=None if result.status == 200 else "http_error")
    try:
        payload = _dict(json.loads(result.body))
    except ValueError:
        payload = {}
    code = _dict(payload.get("error")).get("code")
    meta = _dict(payload.get("meta"))
    latency = _dict(meta.get("latency_ms"))
    total, retrieval, hit = latency.get("total"), latency.get("retrieval"), meta.get("cache_hit")
    request_id = meta.get("request_id", payload.get("request_id"))
    return Details(
        error=None if result.status == 200 else (code if isinstance(code, str) else "http_error"),
        total_ms=total if isinstance(total, int) else None,
        cache_hit=hit if isinstance(hit, bool) else None,
        retrieval_ms=retrieval if isinstance(retrieval, int) else None,
        request_id=request_id if isinstance(request_id, str) else None,
    )


@dataclass(frozen=True)
class Run:
    plan: Plan
    started_at: datetime
    samples: tuple[Sample, ...]
    notes: tuple[str, ...]


def run_measurement(
    plan: Plan,
    *,
    sleep: Callable[[float], None] = time.sleep,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
    progress: Callable[[str], None] = lambda _message: None,
) -> Run:
    """Wait out the idle period (if asked to), then send the requests in the order of the plan.

    ``first="health"``: ``/healthz`` (function start), ``/readyz`` (database wake), the warm
    probes, then the asks. The first ask then runs against a function and a database that are
    already awake, so it is labelled ``after probes``: it is not what a visitor sees cold.
    ``first="ask"``: the first request after the idle period is the ask itself, the visitor's cold
    start end to end; the probes and the other asks follow, warm. Use two idle cycles for both.
    """
    plan.validate()
    if plan.waited:
        remaining = plan.idle_minutes * 60
        while remaining > 0:
            progress(f"idle: {remaining / 60:.1f} min left (send nothing to the site meanwhile)")
            chunk = min(60.0, remaining)
            sleep(chunk)
            remaining -= chunk
    started = now()
    samples: list[Sample] = []
    notes: list[str] = []
    asked = 0
    offset = question_offset(plan)

    def send(step: Step, phase: Phase) -> bool:
        """Send one request; False when the asks must stop (rate limit or budget reached)."""
        nonlocal asked
        if step is Step.ASK:
            question = plan.questions[(offset + asked) % len(plan.questions)]
            asked += 1
            url, method = f"{plan.frontend_url.rstrip('/')}/api/ask", "POST"
            body = json.dumps({"question": question}).encode()
        else:
            assert plan.backend_url is not None
            url, method, body = f"{plan.backend_url.rstrip('/')}/{step.value}", "GET", None
        # The proxy secret goes to /readyz and nowhere else (not /healthz, not the frontend).
        secret = plan.proxy_secret if step is Step.READYZ else None
        headers = {"X-Proxy-Secret": secret} if secret else None
        began = time.perf_counter()
        result = http_call(method, url, body=body, timeout_s=plan.timeout_s, extra_headers=headers)
        elapsed = (time.perf_counter() - began) * 1000
        details = _details(step, result)
        error = details.error
        samples.append(
            Sample(
                step,
                phase,
                result.status,
                elapsed,
                error,
                details.total_ms,
                details.cache_hit,
                result.headers.get("x-vercel-id"),
                details.retrieval_ms,
                details.request_id,
            )
        )
        outcome = "no response" if result.status is None else f"HTTP {result.status}"
        progress(f"{step.value} {phase.value}: {outcome} {error or ''}, {elapsed / 1000:.2f} s")
        if step is Step.ASK and result.status in (429, 503):
            # Rate limited or out of budget: a retry would only spend more of what ran out.
            notes.append(f"Asks stopped after ask {asked}: HTTP {result.status} ({error}).")
            return False
        return True

    def warm_probes() -> None:
        for _ in range(plan.warm_probes if plan.backend_url else 0):
            send(Step.HEALTHZ, Phase.WARM)
            send(Step.READYZ, Phase.WARM)

    def asks(first_phase: Phase) -> None:
        phase = first_phase
        while asked < plan.asks and send(Step.ASK, phase):
            phase = Phase.WARM

    if plan.first == "health":
        send(Step.HEALTHZ, Phase.COLD)
        send(Step.READYZ, Phase.COLD)
        warm_probes()
        asks(Phase.AFTER_PROBES)
    else:
        go_on = send(Step.ASK, Phase.COLD)
        warm_probes()
        if go_on:
            asks(Phase.WARM)
    return Run(plan, started, tuple(samples), tuple(notes))


# --- the report --------------------------------------------------------------------------------


def _seconds(ms: float) -> str:
    return f"{ms / 1000:.2f} s"


def _pick(
    samples: Sequence[Sample], step: Step, phase: Phase, *, hit: bool | None = False
) -> list[Sample]:
    """The samples of one step and phase; ``hit=None`` keeps answer-cache hits and misses alike."""
    return [
        s
        for s in samples
        if s.step is step and s.phase is phase and (hit is None or (s.cache_hit is True) == hit)
    ]


def _median(samples: Sequence[Sample]) -> float | None:
    good = [s.elapsed_ms for s in samples if s.ok]
    return statistics.median(good) if good else None


def judge_margin(run: Run) -> tuple[bool | None, str]:
    """Whether the work before the handler's deadline fits in the margin: (fits, explanation).

    In ``health`` order the estimate is (cold ``/healthz`` - warm median) + (cold ``/readyz`` - warm
    median): the function start plus the database wake, round trip subtracted. The pool opens its
    first connection in the background at start (Tech §18), so the wake may already run during the
    cold ``/healthz``: the sum is an estimate, not a decomposition. It is also conservative: on a
    real ask the Neon wake happens at the first database call, which is already inside the
    handler's deadline, so only part of it competes for the margin. In ``ask`` order it is the cold
    ask's end-to-end time minus the server-reported total, which includes the frontend function's
    own start (outside the 35 s timeout), so it over-states the backend part. A cold ask that was an
    answer-cache hit still counts: end to end minus the server total does not depend on what the
    server did inside that total.
    """
    plan, samples, margin = run.plan, run.samples, run.plan.margin_s
    cold_ask = _pick(samples, Step.ASK, Phase.COLD, hit=None)
    if cold_ask and cold_ask[0].error == "deadline_exceeded":
        where = f" (request_id `{cold_ask[0].request_id}`)" if cold_ask[0].request_id else ""
        return None, (
            f"Not a margin verdict: the cold ask failed with `deadline_exceeded` after "
            f"{_seconds(cold_ask[0].elapsed_ms)}. The backend's deadline or retrieval timeout "
            f"fired; look up the request_id in request_logs{where} to see which stage ran out."
        )
    if cold_ask and cold_ask[0].error in ("upstream_timeout", "upstream_unavailable"):
        return False, (
            f"**The {margin:g} s margin is NOT enough**: the cold ask failed with "
            f"`{cold_ask[0].error}` after {_seconds(cold_ask[0].elapsed_ms)}. A visitor sees the "
            "UI's retry state; record the numbers as they are."
        )
    overhead: float | None = None
    parts = ""
    if plan.first == "health":
        cold_h, cold_r = (_pick(samples, s, Phase.COLD) for s in (Step.HEALTHZ, Step.READYZ))
        warm_h, warm_r = (
            _median(_pick(samples, s, Phase.WARM)) for s in (Step.HEALTHZ, Step.READYZ)
        )
        if cold_h and cold_h[0].ok and cold_r and cold_r[0].ok and warm_h and warm_r:
            start = max(0.0, cold_h[0].elapsed_ms - warm_h)
            wake = max(0.0, cold_r[0].elapsed_ms - warm_r)
            overhead = (start + wake) / 1000
            parts = (
                f"function start about {_seconds(start)} (cold `/healthz` minus the warm median), "
                f"database wake about {_seconds(wake)} (cold `/readyz` minus the warm median)"
            )
    elif cold_ask and cold_ask[0].ok and cold_ask[0].total_ms is not None:
        overhead = max(0.0, cold_ask[0].elapsed_ms - cold_ask[0].total_ms) / 1000
        parts = (
            f"cold ask {_seconds(cold_ask[0].elapsed_ms)} end to end minus the server-reported "
            f"total {_seconds(cold_ask[0].total_ms)}"
        )
    if overhead is None:
        return None, "The margin could not be judged: the cold requests did not all succeed."
    if overhead > margin:
        return False, (
            f"**The {margin:g} s margin is NOT enough**: estimated cold overhead {overhead:.2f} s; "
            f"{parts}. The UI's retry state covers it (Tech §19); record the numbers as they are."
        )
    return (
        True,
        f"The {margin:g} s margin is enough: estimated cold overhead {overhead:.2f} s; {parts}.",
    )


# A first-ask retrieval time this close to RETRIEVAL_TIMEOUT_S leaves no room for a slow day.
RETRIEVAL_HEADROOM = 0.25


def retrieval_line(run: Run) -> str:
    """The first ask's ``meta.latency_ms.retrieval`` against ``RETRIEVAL_TIMEOUT_S`` (D54a, V17).

    The first ask is the cold one in ``ask`` order; in ``health`` order it follows the probes, so
    its database is already awake and the figure is a lower bound of a cold retrieval.
    """
    limit_ms = run.plan.retrieval_timeout_s * 1000
    limit = f"`RETRIEVAL_TIMEOUT_S` = {run.plan.retrieval_timeout_s:g} s"
    first = [s for s in run.samples if s.step is Step.ASK][:1]
    if not first:
        return f"No ask was sent, so the retrieval time is not measured ({limit})."
    ask = first[0]
    which = "cold ask" if ask.phase is Phase.COLD else f"first ask ({ask.phase.value})"
    if ask.cache_hit:
        return f"The {which} was an answer-cache hit: no retrieval ran ({limit})."
    if ask.retrieval_ms is None:
        return f"The {which} reported no retrieval time ({ask.error or 'no meta'}); {limit}."
    share = ask.retrieval_ms / limit_ms
    line = (
        f"Retrieval of the {which}: {_seconds(ask.retrieval_ms)} against {limit} "
        f"({share:.0%} of it)."
    )
    if share >= 1 - RETRIEVAL_HEADROOM:
        line += (
            f" **Within {RETRIEVAL_HEADROOM:.0%} of the timeout:** raise `RETRIEVAL_TIMEOUT_S` in "
            "the backend's Vercel variables (a setting, not code) and record it in Tech §18."
        )
    return line


def render_report(run: Run) -> str:
    """The Markdown the operator pastes into the runbook's results section."""
    plan, samples = run.plan, run.samples
    date = run.started_at.strftime("%Y-%m-%d %H:%M")
    idle = f"{plan.idle_minutes:g} min"
    ask_method, probe_method = "`POST {frontend}/api/ask`", "`GET {backend}/%s`"
    out = [
        "## Cold start and warm latency",
        "",
        f"- Date (UTC): {date}; idle before the first request: {idle} "
        + ("(waited by the tool)" if plan.waited else "(declared by the operator)"),
        f"- Order: `--first {plan.first}`; frontend `{plan.frontend_url}`"
        + (f", backend `{plan.backend_url}`" if plan.backend_url else ""),
        "",
        "| Measurement | Method | Date (UTC) | Idle | n | min | median | max |",
        "|---|---|---|---|---|---|---|---|",
    ]
    rows = [
        ("function start, cold", probe_method % "healthz", Step.HEALTHZ, Phase.COLD, False),
        ("database wake, cold", probe_method % "readyz", Step.READYZ, Phase.COLD, False),
        ("ask end to end, cold", ask_method, Step.ASK, Phase.COLD, None),
        ("ask end to end, after probes", ask_method, Step.ASK, Phase.AFTER_PROBES, False),
        ("`/healthz`, warm", probe_method % "healthz", Step.HEALTHZ, Phase.WARM, False),
        ("`/readyz`, warm", probe_method % "readyz", Step.READYZ, Phase.WARM, False),
        ("ask end to end, warm", ask_method, Step.ASK, Phase.WARM, False),
        ("ask end to end, warm, answer-cache hit", ask_method, Step.ASK, Phase.WARM, True),
    ]
    for label, method, step, phase, hit in rows:
        group = _pick(samples, step, phase, hit=hit)
        if not group:
            continue
        if hit is None and any(s.cache_hit for s in group):
            label += " (cache hit)"
        times = [s.elapsed_ms for s in group if s.ok]
        n = f"{len(times)}" if len(times) == len(group) else f"{len(times)} of {len(group)}"
        low, mid, high = (
            (_seconds(min(times)), _seconds(statistics.median(times)), _seconds(max(times)))
            if times
            else ("-", "-", "-")
        )
        out.append(f"| {label} | {method} | {date} | {idle} | {n} | {low} | {mid} | {high} |")

    out += ["", "### Margin", "", judge_margin(run)[1]]
    out += ["", "### Retrieval", "", retrieval_line(run)]
    warm_asks = _pick(samples, Step.ASK, Phase.WARM)
    if (warm := _median(warm_asks)) is not None:
        out += [
            "",
            f"Warm ask median {_seconds(warm)} (n = {len([s for s in warm_asks if s.ok])}) against "
            f"the NFR-1 target of p50 <= {WARM_P50_TARGET_S:g} s. This n says nothing about p95.",
        ]
    ids = sorted(
        {s.vercel_id.rsplit("::", 1)[0] for s in samples if s.vercel_id and "::" in s.vercel_id}
    )
    out += [
        "",
        "Regions in `x-vercel-id` (expected `cle1`, D53): "
        + (", ".join(f"`{i}`" for i in ids) if ids else "none returned"),
        *(f"\n> {note}" for note in run.notes),
        "",
        "### Raw samples",
        "",
        "| # | Request | Phase | Result | Time | Server total | Cache hit |",
        "|---|---|---|---|---|---|---|",
    ]
    for index, s in enumerate(samples, start=1):
        result = "no response" if s.status is None else f"HTTP {s.status}"
        result += f" ({s.error})" if s.error else ""
        total = "" if s.total_ms is None else _seconds(s.total_ms)
        hit = "" if s.cache_hit is None else ("yes" if s.cache_hit else "no")
        out.append(
            f"| {index} | {s.step.value} | {s.phase.value} | {result} | "
            f"{_seconds(s.elapsed_ms)} | {total} | {hit} |"
        )
    return "\n".join(out) + "\n"
