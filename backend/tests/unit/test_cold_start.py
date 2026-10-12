"""The cold-start measurement tool (ticket 5.10): order of requests, stop rules, report, CLI.

Real HTTP, but only to a throwaway server on 127.0.0.1 (the network guard allows nothing else).
Timings are never asserted: they come from the wall clock. What is pinned is what the tool sends,
in which order, what it refuses to send, and how the verdict is derived from given samples.
"""

from __future__ import annotations

import json
import socket
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest
from typer.testing import CliRunner

from grounded.cli import app
from grounded.infra.cold_start import (
    DEFAULT_QUESTIONS,
    MAX_ASKS,
    MeasureError,
    Phase,
    Plan,
    Run,
    Sample,
    Step,
    judge_margin,
    render_report,
    run_measurement,
)
from grounded.settings import get_settings

runner = CliRunner()

SECRET = "proxy-secret-for-tests"


@pytest.fixture(autouse=True)
def proxy_secret(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """The command reads PROXY_SHARED_SECRET through ``get_settings()``; give it a known value."""
    monkeypatch.setenv("PROXY_SHARED_SECRET", SECRET)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def answer(*, cache_hit: bool = False, total_ms: int = 1500) -> dict[str, Any]:
    return {
        "status": "answered",
        "meta": {"cache_hit": cache_hit, "latency_ms": {"total": total_ms, "llm": 900}},
    }


class FakeSite:
    """A frontend and a backend in one: /healthz, /readyz and POST /api/ask on localhost."""

    def __init__(self) -> None:
        self.requests: list[tuple[str, str, dict[str, str], bytes]] = []
        self.ask_replies: list[tuple[int, object]] = []  # popped in order, then the default
        self.url = ""

    def reply_for(self, method: str, path: str) -> tuple[int, object]:
        if method == "GET":
            return 200, {"status": "ok"}
        if self.ask_replies:
            return self.ask_replies.pop(0)
        return 200, answer()


@pytest.fixture
def site(monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeSite]:
    # An environment proxy (or none, and the Windows registry) must not take localhost traffic.
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    fake = FakeSite()

    class Handler(BaseHTTPRequestHandler):
        def _serve(self) -> None:
            length = int(self.headers.get("content-length") or 0)
            raw = self.rfile.read(length) if length else b""
            fake.requests.append(
                (self.command, self.path, {k.lower(): v for k, v in self.headers.items()}, raw)
            )
            status, payload = fake.reply_for(self.command, self.path)
            data = payload.encode() if isinstance(payload, str) else json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(data)))
            self.send_header("x-vercel-id", "cle1::cle1::abc-123")
            self.end_headers()
            self.wfile.write(data)

        do_GET = _serve  # noqa: N815
        do_POST = _serve  # noqa: N815

        def log_message(self, format: str, *args: Any) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    fake.url = f"http://127.0.0.1:{server.server_address[1]}"
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
    )
    thread.start()
    yield fake
    server.shutdown()
    server.server_close()


def paths(site: FakeSite) -> list[str]:
    return [f"{method} {path}" for method, path, _, _ in site.requests]


def closed_port_url() -> str:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    return f"http://127.0.0.1:{port}"  # nothing listens there any more


def plan(site: FakeSite, **overrides: Any) -> Plan:
    values: dict[str, Any] = {
        "frontend_url": site.url,
        "backend_url": site.url,
        "idle_minutes": 30,
        "warm_probes": 2,
        "asks": 3,
        "proxy_secret": SECRET,
    }
    return Plan(**{**values, **overrides})


# --- the order of the requests ---------------------------------------------------------------


def test_health_order_probes_first_then_asks_after_probes(site: FakeSite) -> None:
    run = run_measurement(plan(site))
    assert [(s.step, s.phase) for s in run.samples] == [
        (Step.HEALTHZ, Phase.COLD),
        (Step.READYZ, Phase.COLD),
        (Step.HEALTHZ, Phase.WARM),
        (Step.READYZ, Phase.WARM),
        (Step.HEALTHZ, Phase.WARM),
        (Step.READYZ, Phase.WARM),
        (Step.ASK, Phase.AFTER_PROBES),
        (Step.ASK, Phase.WARM),
        (Step.ASK, Phase.WARM),
    ]
    assert all(s.ok for s in run.samples)
    assert paths(site)[:2] == ["GET /healthz", "GET /readyz"]
    assert paths(site).count("POST /api/ask") == 3


def test_ask_order_sends_the_ask_first_and_labels_it_cold(site: FakeSite) -> None:
    run = run_measurement(plan(site, first="ask", asks=2, warm_probes=1))
    assert [(s.step, s.phase) for s in run.samples] == [
        (Step.ASK, Phase.COLD),
        (Step.HEALTHZ, Phase.WARM),
        (Step.READYZ, Phase.WARM),
        (Step.ASK, Phase.WARM),
    ]
    assert paths(site)[0] == "POST /api/ask"


def test_ask_order_works_without_a_backend_url(site: FakeSite) -> None:
    run = run_measurement(plan(site, first="ask", backend_url=None, asks=2))
    assert [s.step for s in run.samples] == [Step.ASK, Step.ASK]
    assert paths(site) == ["POST /api/ask", "POST /api/ask"]


def test_zero_asks_sends_no_question(site: FakeSite) -> None:
    run_measurement(plan(site, asks=0))
    assert "POST /api/ask" not in paths(site)


def test_every_ask_is_a_different_question_and_nothing_secret_is_sent(site: FakeSite) -> None:
    run_measurement(plan(site, asks=4))
    asks = [(headers, raw) for method, _, headers, raw in site.requests if method == "POST"]
    questions = [json.loads(raw)["question"] for _, raw in asks]
    assert questions == list(DEFAULT_QUESTIONS[:4])
    assert len(set(questions)) == 4
    for method, path, headers, _ in site.requests:
        if path != "/readyz":
            assert "x-proxy-secret" not in headers, (method, path)
        assert "authorization" not in headers
        assert "cookie" not in headers


def test_the_proxy_secret_goes_to_readyz_and_nowhere_else(site: FakeSite) -> None:
    run_measurement(plan(site, asks=2))
    sent = {
        (method, path, headers.get("x-proxy-secret")) for method, path, headers, _ in site.requests
    }
    assert sent == {
        ("GET", "/healthz", None),
        ("GET", "/readyz", SECRET),
        ("POST", "/api/ask", None),
    }


@pytest.mark.parametrize("secret", [None, ""])
def test_a_backend_url_without_the_proxy_secret_is_refused(
    site: FakeSite, secret: str | None
) -> None:
    with pytest.raises(MeasureError, match="PROXY_SHARED_SECRET"):
        run_measurement(plan(site, proxy_secret=secret))
    assert site.requests == []


def test_ask_order_without_a_backend_needs_no_secret(site: FakeSite) -> None:
    run = run_measurement(plan(site, backend_url=None, first="ask", asks=1, proxy_secret=None))
    assert [s.step for s in run.samples] == [Step.ASK]


def test_the_secret_is_not_in_the_plan_repr() -> None:
    assert SECRET not in repr(
        Plan("https://f.example", "https://b.example", 30, proxy_secret=SECRET)
    )


# --- stop rules and failures -----------------------------------------------------------------


@pytest.mark.parametrize("status", [429, 503])
def test_an_ask_that_hits_the_limit_or_the_budget_stops_the_asks(
    site: FakeSite, status: int
) -> None:
    code = "rate_limited" if status == 429 else "budget_exhausted"
    site.ask_replies = [(200, answer()), (status, {"error": {"code": code}})]
    run = run_measurement(plan(site, asks=5))
    assert paths(site).count("POST /api/ask") == 2  # nothing was retried, nothing more was sent
    assert run.samples[-1].error == code
    assert f"HTTP {status}" in run.notes[0]


def test_a_platform_504_page_is_recorded_as_a_failed_ask(site: FakeSite) -> None:
    site.ask_replies = [(504, "<html>FUNCTION_INVOCATION_TIMEOUT</html>")]
    run = run_measurement(plan(site, first="ask", asks=2, warm_probes=0))
    assert (run.samples[0].status, run.samples[0].error) == (504, "http_error")
    assert not run.samples[0].ok
    assert run.samples[1].ok  # a 504 is not a stop: the next ask is the first warm one


def test_a_proxy_error_code_is_kept_as_the_error(site: FakeSite) -> None:
    site.ask_replies = [(504, {"error": {"code": "upstream_timeout"}, "request_id": None})]
    run = run_measurement(plan(site, first="ask", asks=1, warm_probes=0))
    assert run.samples[0].error == "upstream_timeout"


def test_nothing_listening_is_a_sample_without_a_response_not_a_crash() -> None:
    url = closed_port_url()
    run = run_measurement(
        Plan(frontend_url=url, backend_url=None, idle_minutes=0, first="ask", asks=1)
    )
    assert [s.status for s in run.samples] == [None]
    assert all(s.error for s in run.samples)


def test_the_cache_flag_and_server_total_are_read_from_the_response(site: FakeSite) -> None:
    site.ask_replies = [(200, answer(cache_hit=True, total_ms=40))]
    run = run_measurement(plan(site, first="ask", asks=1, warm_probes=0))
    assert (run.samples[0].cache_hit, run.samples[0].total_ms) == (True, 40)


# --- the idle period -------------------------------------------------------------------------


def test_waiting_sleeps_the_whole_idle_period_in_chunks_before_the_first_request(
    site: FakeSite,
) -> None:
    slept: list[float] = []

    def sleep(seconds: float) -> None:
        assert site.requests == []  # nothing has been sent while idling
        slept.append(seconds)

    run_measurement(plan(site, idle_minutes=2.5, waited=True, asks=0), sleep=sleep)
    assert slept == [60.0, 60.0, 30.0]


def test_without_waiting_the_tool_does_not_sleep(site: FakeSite) -> None:
    def sleep(_seconds: float) -> None:
        raise AssertionError("slept")

    run_measurement(plan(site, asks=0), sleep=sleep)


# --- plan validation -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"asks": MAX_ASKS + 1}, "--asks"),
        ({"frontend_url": "http://grounded.example.com"}, "https"),
        ({"frontend_url": "ftp://grounded.example.com"}, r"http\(s\)"),
        ({"frontend_url": "https://user:pw@grounded.example.com"}, "credentials"),
        ({"frontend_url": "https://grounded.example.com/?token=x"}, "query"),
        ({"backend_url": None}, "--backend-url"),
        ({"first": "ask", "asks": 0}, "--asks"),
        ({"idle_minutes": -1}, "negative"),
        ({"questions": ()}, "question"),
    ],
)
def test_a_plan_that_cannot_be_run_is_refused_before_anything_is_sent(
    site: FakeSite, overrides: dict[str, Any], message: str
) -> None:
    bad = plan(site, **overrides)
    with pytest.raises(MeasureError, match=message):
        run_measurement(bad)
    assert site.requests == []


# --- the verdict -----------------------------------------------------------------------------


def sample(step: Step, phase: Phase, ms: float, *, status: int = 200, **kw: Any) -> Sample:
    return Sample(step, phase, status, ms, **kw)


def health_run(cold_h: float, cold_r: float, warm: float = 100.0, *, margin: float = 10.0) -> Run:
    from datetime import UTC, datetime

    samples = (
        sample(Step.HEALTHZ, Phase.COLD, cold_h),
        sample(Step.READYZ, Phase.COLD, cold_r),
        sample(Step.HEALTHZ, Phase.WARM, warm),
        sample(Step.READYZ, Phase.WARM, warm),
    )
    plan = Plan("https://f.example", "https://b.example", 30, margin_s=margin)
    return Run(plan, datetime(2026, 10, 12, tzinfo=UTC), samples, ())


def test_the_margin_fits_when_function_start_plus_wake_is_under_it() -> None:
    fits, text = judge_margin(health_run(cold_h=3100, cold_r=2100))  # 3.0 + 2.0 s
    assert fits is True
    assert "5.00 s" in text
    assert "enough" in text


def test_the_margin_is_not_enough_when_they_add_up_to_more_and_it_says_so_plainly() -> None:
    fits, text = judge_margin(health_run(cold_h=6100, cold_r=5100))  # 6.0 + 5.0 s
    assert fits is False
    assert "NOT enough" in text
    assert "11.00 s" in text


def test_a_failed_cold_ask_is_not_enough_whatever_the_probes_say() -> None:
    from datetime import UTC, datetime

    failed = Sample(Step.ASK, Phase.COLD, 504, 35_100, error="upstream_timeout")
    plan = Plan("https://f.example", None, 30, first="ask")
    fits, text = judge_margin(Run(plan, datetime(2026, 10, 12, tzinfo=UTC), (failed,), ()))
    assert fits is False
    assert "upstream_timeout" in text


def test_ask_order_subtracts_the_server_total_from_the_cold_ask() -> None:
    from datetime import UTC, datetime

    cold = Sample(Step.ASK, Phase.COLD, 200, 9_000, total_ms=2_000)  # 7 s outside the handler
    plan = Plan("https://f.example", None, 30, first="ask")
    fits, text = judge_margin(Run(plan, datetime(2026, 10, 12, tzinfo=UTC), (cold,), ()))
    assert fits is True
    assert "7.00 s" in text


def test_the_verdict_is_undetermined_when_a_cold_probe_failed() -> None:
    run = health_run(cold_h=3000, cold_r=2000)
    broken = (
        Sample(Step.READYZ, Phase.COLD, 503, 5100, error="http_error"),
        *[s for s in run.samples if not (s.step is Step.READYZ and s.phase is Phase.COLD)],
    )
    fits, text = judge_margin(Run(run.plan, run.started_at, broken, ()))
    assert fits is None
    assert "could not be judged" in text


# --- the report ------------------------------------------------------------------------------


def test_the_report_has_the_table_the_runbook_asks_for(site: FakeSite) -> None:
    site.ask_replies = [(200, answer()), (200, answer(cache_hit=True, total_ms=30))]
    report = render_report(run_measurement(plan(site, asks=3)))
    assert "| Measurement | Method | Date (UTC) | Idle | n | min | median | max |" in report
    assert "| function start, cold | `GET {backend}/healthz` |" in report
    assert "| ask end to end, after probes | `POST {frontend}/api/ask` |" in report
    assert "| ask end to end, warm, answer-cache hit |" in report
    assert "| 30 min |" in report
    assert "### Raw samples" in report
    assert "`cle1::cle1`" in report  # the regions line, from x-vercel-id
    assert "x-proxy-secret" not in report.lower()
    assert SECRET not in report


# --- the command -----------------------------------------------------------------------------


def invoke(*args: str, **kwargs: Any) -> Any:
    return runner.invoke(app, ["measure-cold-start", *args], **kwargs)


def test_the_command_prints_the_report_and_warns_about_quota(site: FakeSite) -> None:
    result = invoke(
        "--frontend-url",
        site.url,
        "--backend-url",
        site.url,
        "--idle-minutes",
        "30",
        "--asks",
        "2",
        "--warm-probes",
        "1",
        "--yes",
    )
    assert result.exit_code == 0, result.output
    assert "## Cold start and warm latency" in result.stdout
    assert "WARNING: this sends 2 real question(s)" in result.output
    assert paths(site).count("POST /api/ask") == 2
    assert SECRET not in result.output
    readyz = [headers for _, path, headers, _ in site.requests if path == "/readyz"]
    assert readyz
    assert all(h.get("x-proxy-secret") == SECRET for h in readyz)


def test_the_command_refuses_a_backend_url_without_the_secret(
    site: FakeSite, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PROXY_SHARED_SECRET", "")  # wins over any .env file
    get_settings.cache_clear()
    result = invoke(
        "--frontend-url", site.url, "--backend-url", site.url, "--idle-minutes", "30", "--yes"
    )
    assert result.exit_code == 1
    assert "PROXY_SHARED_SECRET" in result.output
    assert site.requests == []


def test_declining_the_confirmation_sends_nothing(site: FakeSite) -> None:
    result = invoke(
        "--frontend-url",
        site.url,
        "--backend-url",
        site.url,
        "--idle-minutes",
        "30",
        input="n\n",
    )
    assert result.exit_code != 0
    assert site.requests == []


def test_probes_only_needs_no_confirmation_and_sends_no_question(site: FakeSite) -> None:
    result = invoke(
        "--frontend-url",
        site.url,
        "--backend-url",
        site.url,
        "--idle-minutes",
        "30",
        "--asks",
        "0",
        "--warm-probes",
        "1",
    )
    assert result.exit_code == 0, result.output
    assert "WARNING" not in result.output
    assert "POST /api/ask" not in paths(site)


def test_the_command_refuses_a_bad_plan_with_a_message(site: FakeSite) -> None:
    result = invoke(
        "--frontend-url",
        "http://grounded.example.com",
        "--backend-url",
        site.url,
        "--idle-minutes",
        "30",
        "--yes",
    )
    assert result.exit_code == 1
    assert "Cannot measure" in result.output
    assert "https" in result.output


def test_the_command_caps_the_number_of_asks(site: FakeSite) -> None:
    result = invoke(
        "--frontend-url",
        site.url,
        "--backend-url",
        site.url,
        "--idle-minutes",
        "30",
        "--asks",
        str(MAX_ASKS + 1),
        "--yes",
    )
    assert result.exit_code == 2  # click's usage error: the cap is in the option itself
    assert site.requests == []


def test_the_command_fails_when_not_one_request_succeeded() -> None:
    url = closed_port_url()
    result = invoke(
        "--frontend-url",
        url,
        "--first",
        "ask",
        "--idle-minutes",
        "0",
        "--asks",
        "1",
        "--yes",
    )
    assert result.exit_code == 1
    assert "No request succeeded" in result.output


# --- R.23: distinct questions per order, cold cache hits, cold retrieval -------------------------


def asked_questions(site: FakeSite) -> list[str]:
    return [json.loads(raw)["question"] for method, _, _, raw in site.requests if method == "POST"]


def test_ask_order_starts_halfway_through_the_default_questions(site: FakeSite) -> None:
    # Run 2 of the runbook must not ask what run 1 (--first health) already put in the cache.
    run_measurement(plan(site, first="ask", asks=3))
    half = len(DEFAULT_QUESTIONS) // 2
    assert asked_questions(site) == list(DEFAULT_QUESTIONS[half : half + 3])


def test_health_order_starts_at_the_first_default_question(site: FakeSite) -> None:
    run_measurement(plan(site, first="health", asks=2))
    assert asked_questions(site) == list(DEFAULT_QUESTIONS[:2])


def test_a_question_list_of_the_operator_starts_at_its_beginning(site: FakeSite) -> None:
    run_measurement(plan(site, first="ask", asks=3, questions=("one?", "two?")))
    assert asked_questions(site) == ["one?", "two?", "one?"]


def test_a_cold_ask_that_hit_the_answer_cache_is_still_judged_and_marked(site: FakeSite) -> None:
    site.ask_replies = [(200, answer(cache_hit=True, total_ms=40))]
    run = run_measurement(plan(site, first="ask", asks=1, warm_probes=0))
    fits, text = judge_margin(run)
    assert fits is True  # end to end minus the server total does not depend on the cache
    assert "enough" in text
    report = render_report(run)
    assert "| ask end to end, cold (cache hit) |" in report
    assert "answer-cache hit: no retrieval ran" in report


def retrieval_answer(retrieval_ms: int) -> dict[str, Any]:
    reply = answer()
    reply["meta"]["latency_ms"]["retrieval"] = retrieval_ms
    return reply


def test_the_report_compares_the_cold_retrieval_with_the_timeout(site: FakeSite) -> None:
    site.ask_replies = [(200, retrieval_answer(300))]
    report = render_report(run_measurement(plan(site, first="ask", asks=1, warm_probes=0)))
    assert "### Retrieval" in report
    assert "Retrieval of the cold ask: 0.30 s against `RETRIEVAL_TIMEOUT_S` = 2 s (15% of it)." in (
        report
    )
    assert "raise `RETRIEVAL_TIMEOUT_S`" not in report


def test_a_cold_retrieval_within_a_quarter_of_the_timeout_recommends_raising_it(
    site: FakeSite,
) -> None:
    site.ask_replies = [(200, retrieval_answer(1_600))]  # 80% of 2 s
    report = render_report(run_measurement(plan(site, first="ask", asks=1, warm_probes=0)))
    assert "(80% of it)" in report
    assert "**Within 25% of the timeout:** raise `RETRIEVAL_TIMEOUT_S`" in report


def test_in_health_order_the_retrieval_line_names_the_after_probes_ask(site: FakeSite) -> None:
    site.ask_replies = [(200, retrieval_answer(250))]
    report = render_report(run_measurement(plan(site, asks=1, retrieval_timeout_s=5.0)))
    assert "Retrieval of the first ask (after probes): 0.25 s against" in report
    assert "= 5 s" in report


def test_a_cold_deadline_exceeded_points_at_request_logs_not_at_the_margin(
    site: FakeSite,
) -> None:
    error = {"code": "deadline_exceeded", "message": "too long", "retry_after_s": None}
    site.ask_replies = [(504, {"error": error, "request_id": "0b7c-req"})]
    run = run_measurement(plan(site, first="ask", asks=1, warm_probes=0))
    fits, text = judge_margin(run)
    assert fits is None
    assert "deadline or retrieval timeout fired" in text
    assert "request_logs" in text
    assert "`0b7c-req`" in text
