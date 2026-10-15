"""``grounded demo-fallback``: Phase 7 working in front of you, with fakes only (ticket 7.04b).

Three requests go through the **real** app (the ASGI app of ``create_app``, in process: validation,
the answer cache, the pipeline, the provider router, the budget code, the ``request_logs`` insert,
the error handlers) over the **local** database, with the generator replaced by two scripted fakes
named like the real providers, so no real API is ever called and no key is needed:

1. the fake "gemini" answers the first call with a 429 (``Retry-After`` 30 s): the fake "groq"
   answers, ``fallback_used`` is true and the breaker of Gemini opens for 30 s;
2. five simulated seconds later the breaker is still open: Gemini is **not called** (its call count
   stays at 1), Groq answers again;
3. past the open time one trial goes to Gemini, it answers, the breaker closes, ``fallback_used``
   is false.

Time is a clock this module moves by hand (``RouterClock``), so nothing sleeps. The query is
embedded by ``FakeEmbedder`` under the settings' embedding model, so retrieval ranks by a stand-in
vector (the full-text half still matches): the demo shows routing, not answer quality. It needs the
database and an active index (``grounded migrate``, then ``grounded ingest --activate``, README
quickstart) and writes what any request writes: three ``request_logs`` rows and the answer of the
third request in the answer cache. Each run asks questions of its own (a nonce in the text), so a
re-run is not answered from the cache.

**Dev only.** ``APP_ENV=prod`` is refused (a production database must not get demo rows) and so is
``eval`` (eval mode runs one provider). The demo takes the names of its two fakes from
``GEMINI_MODEL`` and ``GROQ_MODEL`` (pinned IDs that ``pricing.toml`` prices, so the shadow cost is
a real calculation); it needs no key. The copy of the settings it runs on has rerank off, no budget,
no client IP hash (so no rate limit) and direct API access, none of which touches your ``.env``.

It prints one line per request and the router's and the request's own log lines (the very JSON
lines the service writes: ``breaker opened``, ``provider attempt failed``, ``provider skipped``,
``request completed`` with ``fallback_used``), then ``PASS`` or ``FAIL``. Never a question.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any, cast
from uuid import uuid4

import httpx
from psycopg_pool import AsyncConnectionPool

from grounded.generation.providers.base import Usage
from grounded.generation.providers.fake import FakeLLMProvider
from grounded.generation.router import RouterClock
from grounded.infra.logging import JsonFormatter
from grounded.infra.provider_errors import ProviderRateLimited
from grounded.ingest.embed import FakeEmbedder
from grounded.main import create_app
from grounded.schemas.llm import LLMAnswer, LLMClaim
from grounded.settings import Settings

# How long the fake Gemini's 429 tells the breaker to stay open (a ``Retry-After``), and the
# simulated seconds between the requests. The third request is past the open time.
RETRY_AFTER_S = 30.0
GAP_AFTER_FIRST_S = 5.0
GAP_AFTER_SECOND_S = 30.0

# What the demo prints of the logs: the router's lines and the request's summary.
_ROUTER_LOGGER = "grounded.generation.router"
_SUMMARY = "request completed"

_ROW = (
    "SELECT outcome, provider, model, fallback_used, validation_retries, input_tokens, "
    "output_tokens FROM request_logs WHERE id = %s"
)


class DemoRefusedError(Exception):
    """The demo must not run here (production, eval mode, or a missing model name); the message
    says why."""


class _HandClock:
    """The breaker's monotonic clock, moved by the demo (``RouterClock.monotonic``)."""

    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class _Collector(logging.Handler):
    """Keeps the log records of the demo's own interest, formatted as the service writes them."""

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.lines: list[str] = []
        self._formatter = JsonFormatter()

    def emit(self, record: logging.LogRecord) -> None:
        if record.name == _ROUTER_LOGGER or record.getMessage() == _SUMMARY:
            self.lines.append(self._formatter.format(record))

    def take(self) -> list[str]:
        taken, self.lines = self.lines, []
        return taken


@dataclass(frozen=True, slots=True)
class _Outcome:
    """What one demo request showed: the response and the row it left."""

    status: int
    provider: str
    model: str
    fallback_used: bool
    cache_hit: bool
    row: dict[str, Any]
    gemini_calls: int = 0  # how many calls the fake Gemini had received when the response came


def demo_settings(settings: Settings) -> Settings:
    """The settings the demo runs on, or ``DemoRefusedError``. A copy: ``.env`` is not touched."""
    if settings.app_env in ("prod", "eval"):
        raise DemoRefusedError(
            f"demo-fallback is for dev and test: refused with APP_ENV={settings.app_env} "
            "(it writes request_logs rows, and eval mode runs one provider)"
        )
    missing = [
        name
        for name, value in (
            ("GEMINI_MODEL", settings.gemini_model),
            ("GROQ_MODEL", settings.groq_model),
        )
        if not (value or "").strip()
    ]
    if missing:
        raise DemoRefusedError(
            f"set {' and '.join(missing)} (pinned model IDs that backend/pricing.toml prices): the "
            "demo's fake providers take their names. No API key is needed."
        )
    return settings.model_copy(
        update={
            "allow_direct_api": True,  # the in-process client sends no proxy secret
            "rerank_provider": "none",  # never a Cohere call, whatever .env says
            "daily_llm_budget": None,
            "ip_hash_secret": None,  # no client key: not rate limited, nothing counted
        }
    )


def _canned(provider: str) -> LLMAnswer:
    claim = LLMClaim(
        text="A canned claim from a fake provider.", citation_ids=["c1"], self_confidence=0.5
    )
    return LLMAnswer(
        status="answered",
        answer_markdown=f"Canned answer of the fake {provider} provider, not a model output. [c1]",
        claims=[claim],
    )


async def run_demo(settings: Settings, echo: Callable[[str], None]) -> bool:
    """Run the three requests, print what they show to ``echo``, and say whether every expectation
    held. Raises ``DemoRefusedError`` for a settings the demo refuses."""
    settings = demo_settings(settings)
    gemini = FakeLLMProvider(
        [
            ProviderRateLimited("demo 429", retry_after_s=RETRY_AFTER_S, is_quota=False),
            _canned("gemini"),
        ],
        name="gemini",
        model=cast(str, settings.gemini_model).strip(),
        usage=Usage(input_tokens=1_000, output_tokens=300),
    )
    groq = FakeLLMProvider(
        [_canned("groq"), _canned("groq")],
        name="groq",
        model=cast(str, settings.groq_model).strip(),
        usage=Usage(input_tokens=1_500, output_tokens=300),
    )
    hand = _HandClock()
    app = create_app(
        settings,
        embedder=FakeEmbedder(model=settings.embedding_model, dim=settings.embedding_dim),
        provider=gemini,
        fallback_providers=[groq],
        router_clock=RouterClock(monotonic=hand),
    )
    collector = _Collector()
    root = logging.getLogger()
    # ``create_app`` put the JSON handler on stderr; the demo prints the lines it wants itself.
    muted = [h for h in root.handlers if isinstance(h.formatter, JsonFormatter)]
    level = root.level
    for handler in muted:
        root.removeHandler(handler)
    root.addHandler(collector)
    root.setLevel(logging.INFO)
    nonce = uuid4().hex[:8]
    questions = [
        f"Demo {nonce}: how do I declare a path parameter?",
        f"Demo {nonce}: how do I declare a query parameter?",
        f"Demo {nonce}: how do I declare a request body?",
    ]
    outcomes: list[_Outcome] = []
    try:
        async with app.router.lifespan_context(app):
            pool = cast(AsyncConnectionPool, app.state.db_pool)
            transport = httpx.ASGITransport(
                app=app, raise_app_exceptions=False, client=("127.0.0.1", 123)
            )
            async with httpx.AsyncClient(transport=transport, base_url="http://demo") as client:
                echo("The fake gemini answers its first call with a 429 (Retry-After 30 s).")
                echo("Time is a clock the demo moves by hand: nothing sleeps.")
                for number, question in enumerate(questions, start=1):
                    if number == 2:
                        hand.advance(GAP_AFTER_FIRST_S)
                        echo(f"-- simulated clock: +{GAP_AFTER_FIRST_S:g} s --")
                    if number == 3:
                        hand.advance(GAP_AFTER_SECOND_S)
                        echo(
                            f"-- simulated clock: +{GAP_AFTER_SECOND_S:g} s more, "
                            f"{GAP_AFTER_FIRST_S + GAP_AFTER_SECOND_S:g} s after the 429 --"
                        )
                    response = await client.post("/v1/ask", json={"question": question})
                    outcome = replace(
                        await _outcome(pool, response), gemini_calls=len(gemini.calls)
                    )
                    outcomes.append(outcome)
                    echo(_result_line(number, outcome, gemini, groq))
                    echo(_row_line(outcome))
                    for line in collector.take():
                        echo(f"   log {line}")
    finally:  # leave the process logging as it was found
        root.removeHandler(collector)
        for handler in muted:
            root.addHandler(handler)
        root.setLevel(level)
    return _verdict(outcomes, gemini, groq, echo)


async def _outcome(pool: AsyncConnectionPool, response: httpx.Response) -> _Outcome:
    body = cast("dict[str, Any]", response.json())
    meta = cast("dict[str, Any]", body.get("meta") or {})
    row: dict[str, Any] = {}
    request_id = meta.get("request_id")
    if request_id is not None:
        async with pool.connection() as conn:
            cursor = await conn.execute(_ROW, (request_id,))
            found = await cursor.fetchone()
        if found is not None:
            names = (
                "outcome",
                "provider",
                "model",
                "fallback_used",
                "validation_retries",
                "input_tokens",
                "output_tokens",
            )
            row = dict(zip(names, found, strict=True))
    return _Outcome(
        status=response.status_code,
        provider=str(meta.get("provider")),
        model=str(meta.get("model")),
        fallback_used=bool(meta.get("fallback_used")),
        cache_hit=bool(meta.get("cache_hit")),
        row=row,
    )


def _result_line(
    number: int, outcome: _Outcome, gemini: FakeLLMProvider, groq: FakeLLMProvider
) -> str:
    return (
        f"request {number}: HTTP {outcome.status}  provider={outcome.provider}  "
        f"model={outcome.model}  fallback_used={_flag(outcome.fallback_used)}  "
        f"cache_hit={_flag(outcome.cache_hit)}  calls so far: gemini={len(gemini.calls)} "
        f"groq={len(groq.calls)}"
    )


def _row_line(outcome: _Outcome) -> str:
    if not outcome.row:
        return "   request_logs row: none found (the request did not reach the pipeline)"
    row = outcome.row
    return (
        f"   request_logs row: outcome={row['outcome']}  provider={row['provider']}  "
        f"fallback_used={_flag(bool(row['fallback_used']))}  "
        f"validation_retries={row['validation_retries']}  tokens={row['input_tokens']}/"
        f"{row['output_tokens']}"
    )


def _flag(value: bool) -> str:
    return "true" if value else "false"


def _verdict(
    outcomes: list[_Outcome],
    gemini: FakeLLMProvider,
    groq: FakeLLMProvider,
    echo: Callable[[str], None],
) -> bool:
    """The exit criteria of Phase 7, as the demo saw them. Every failed expectation is listed."""
    problems: list[str] = []
    if len(outcomes) != 3 or any(o.status != 200 for o in outcomes):
        statuses = [o.status for o in outcomes]
        problems.append(
            f"a request did not answer (statuses {statuses}); the database needs an active index "
            "(grounded ingest --activate)"
        )
    else:
        first, second, third = outcomes
        if not (first.provider == "groq" and first.fallback_used):
            problems.append("request 1 was not answered by the fallback")
        if not (second.provider == "groq" and second.fallback_used):
            problems.append("request 2 was not answered by the fallback")
        if second.gemini_calls != first.gemini_calls:
            problems.append("the breaker did not skip Gemini on request 2: it was called again")
        if len(gemini.calls) != 2:
            problems.append(
                f"Gemini was called {len(gemini.calls)} times, expected the 429 and one trial"
            )
        if not (third.provider == "gemini" and not third.fallback_used):
            problems.append("request 3 was not answered by Gemini after the trial")
        if [bool(o.row.get("fallback_used")) for o in outcomes] != [True, True, False]:
            problems.append("the request_logs rows do not say fallback_used true, true, false")
    if problems:
        echo("FAIL: " + "; ".join(problems))
        return False
    echo(
        "PASS: the 429 on gemini was answered by groq; the next request skipped gemini without "
        "a call; after the open time one trial went to gemini, which answered again; "
        "request_logs.fallback_used is true, true, false."
    )
    return True
