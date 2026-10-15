"""The pure parts of the two recording scripts and their shared helper (``recording_tools.py``,
``record_generate_fixture.py`` under ``tests/fixtures/gemini`` and ``tests/fixtures/groq``): the
scrubbing, the ``--burst`` argument, the counting and the write rules. The scripts record REAL
provider responses by hand. **Nothing here touches the network** (the socket guard of
``conftest.py`` would fail it), reads a key or writes a committed fixture: every write goes to
``tmp_path`` and the "calls" are fakes or an ``httpx.MockTransport``.

The scripts are imported by path, because ``tests/fixtures`` is data, not a package.
"""

from __future__ import annotations

import asyncio
import importlib
import importlib.util
import json
import re
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import httpx
import pytest
from google.genai import errors

from tests.support import make_settings

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
# Built from parts so no key-shaped text is committed (secret scanners match the shapes).
GROQ_KEY = "gsk" + "_" + "k" * 40
GOOGLE_KEY = "AI" + "za" + "K" * 35
ORG = "org_" + "01hxyz9abc3defg"
DAILY_429 = (
    f"Rate limit reached for model `openai/gpt-oss-120b` in organization `{ORG}` service tier "
    "`on_demand` on tokens per day (TPD): Limit 200000, Used 199219, Requested 1418. "
    "Please try again in 4m35.18s."
)


@pytest.fixture
def tools(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.setattr(sys, "path", [*sys.path, str(FIXTURES)])  # restored after the test
    return importlib.import_module("recording_tools")


def load_script(monkeypatch: pytest.MonkeyPatch, name: str, relative: str) -> ModuleType:
    monkeypatch.setattr(sys, "path", [*sys.path, str(FIXTURES)])
    path = FIXTURES / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def groq_script(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> ModuleType:
    module = load_script(monkeypatch, "record_groq", "groq/record_generate_fixture.py")
    monkeypatch.setattr(module, "HERE", tmp_path)  # never a committed fixture
    return module


@pytest.fixture
def gemini_script(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> ModuleType:
    module = load_script(monkeypatch, "record_gemini", "gemini/record_generate_fixture.py")
    monkeypatch.setattr(module, "HERE", tmp_path)
    return module


def refuse_settings(module: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ``Settings()`` fail: the argument checks must come before any setting or key is read."""

    def boom() -> Any:
        raise AssertionError("the script read Settings before it checked its arguments")

    monkeypatch.setattr(module, "Settings", boom)


# --- scrub and keep_headers ---------------------------------------------------------------------


def test_scrub_replaces_the_organization_id_of_a_daily_429_and_changes_nothing_else(
    tools: ModuleType,
) -> None:
    body = {"error": {"message": DAILY_429, "type": "tokens", "code": "rate_limit_exceeded"}}
    scrubbed = tools.scrub(body)
    assert scrubbed == {
        "error": {
            "message": DAILY_429.replace(ORG, "org_REDACTED"),
            "type": "tokens",
            "code": "rate_limit_exceeded",
        }
    }
    assert ORG not in json.dumps(scrubbed)
    assert "Limit 200000, Used 199219, Requested 1418" in scrubbed["error"]["message"]


@pytest.mark.parametrize(
    ("secret", "placeholder"),
    [
        ("req_" + "01k2m3n4p5q6r7s8", "req_REDACTED"),
        (GROQ_KEY, "gsk_REDACTED"),
        (GOOGLE_KEY, "AIza_REDACTED"),
        ("Bearer " + "t" * 30, "Bearer REDACTED"),
        ("123e4567-e89b-12d3-a456-426614174000", "00000000-0000-0000-0000-000000000000"),
    ],
    ids=["request-id", "groq-key", "google-key", "bearer", "uuid"],
)
def test_scrub_replaces_each_id_and_key_shape_with_a_fixed_placeholder(
    tools: ModuleType, secret: str, placeholder: str
) -> None:
    text = f"see {secret} for details"
    assert tools.scrub(text) == f"see {placeholder} for details"
    assert tools.scrub({"nested": [{"x": text}], text: 1}) == {
        "nested": [{"x": f"see {placeholder} for details"}],
        f"see {placeholder} for details": 1,
    }


def test_scrub_is_idempotent_and_keeps_numbers_and_plain_text(tools: ModuleType) -> None:
    body = {"code": 429, "retry": 1.5, "ok": True, "none": None, "message": DAILY_429}
    once = tools.scrub(body)
    assert tools.scrub(once) == once
    assert [once[k] for k in ("code", "retry", "ok", "none")] == [429, 1.5, True, None]
    assert tools.scrub("a normal sentence with no ids") == "a normal sentence with no ids"


def test_only_the_rate_limit_and_retry_after_headers_are_kept(tools: ModuleType) -> None:
    kept = tools.keep_headers(
        {
            "Retry-After": "275",
            "x-ratelimit-limit-tokens": "200000",
            "X-RateLimit-Reset-Tokens": "4m35.18s",
            "x-request-id": "req_abcdef123456",
            "set-cookie": "session=abc",
            "cf-ray": "8a1b2c3d",
            "date": "Wed, 15 Oct 2026 12:00:00 GMT",
        }
    )
    assert kept == {
        "retry-after": "275",
        "x-ratelimit-limit-tokens": "200000",
        "x-ratelimit-reset-tokens": "4m35.18s",
    }
    assert list(kept) == sorted(kept)


# --- the --burst argument -----------------------------------------------------------------------


def test_the_burst_cap_is_forty(tools: ModuleType) -> None:
    assert tools.BURST_CAP == 40


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (["429_per_minute"], (["429_per_minute"], 1)),
        (["429_per_minute", "--burst", "7"], (["429_per_minute"], 7)),
        (["--burst=7", "error", "429_per_minute"], (["error", "429_per_minute"], 7)),
        (["error", "--burst", "40", "429_per_minute"], (["error", "429_per_minute"], 40)),
        (["success", "--burst", "1"], (["success"], 1)),
    ],
)
def test_burst_is_taken_out_of_the_arguments_wherever_it_stands(
    tools: ModuleType, args: list[str], expected: tuple[list[str], int]
) -> None:
    assert tools.split_burst(args) == expected


def test_burst_above_the_cap_is_refused_with_a_clear_message(tools: ModuleType) -> None:
    for value in ("41", "1000"):
        with pytest.raises(SystemExit, match="above the hard cap of 40 requests"):
            tools.split_burst(["429_per_minute", "--burst", value])


@pytest.mark.parametrize("args", [["--burst", "0"], ["--burst", "-3"], ["--burst", "abc"]])
def test_a_burst_that_is_not_a_count_is_refused(tools: ModuleType, args: list[str]) -> None:
    with pytest.raises(SystemExit, match="--burst"):
        tools.split_burst(["429_per_minute", *args])


def test_a_burst_with_no_value_or_given_twice_is_refused(tools: ModuleType) -> None:
    with pytest.raises(SystemExit, match="whole number"):
        tools.split_burst(["429_per_minute", "--burst"])
    with pytest.raises(SystemExit, match="more than once"):
        tools.split_burst(["--burst", "2", "--burst=3"])


def test_the_notice_names_the_count_and_says_it_uses_quota_without_quota_numbers(
    tools: ModuleType,
) -> None:
    text = tools.burst_notice(12, "Groq (a-model)")
    assert "12 real requests" in text
    assert "uses quota" in text
    assert set(re.findall(r"\d+", text)) == {"12"}  # no quota figure is hardcoded
    assert "1 real request " in tools.burst_notice(1, "Groq (a-model)")


# --- run_burst, with fakes ----------------------------------------------------------------------


def scripted(tools: ModuleType, plan: list[tuple[Any, int]]) -> Any:
    """A ``send`` whose n-th call (in creation order) finishes after ``ticks`` loop turns with the
    status, or raises the exception. Calls start in order, so the plan is deterministic."""
    steps = iter(plan)

    async def send() -> Any:
        outcome, ticks = next(steps)
        for _ in range(ticks):
            await asyncio.sleep(0)
        if isinstance(outcome, Exception):
            raise outcome
        return tools.Reply(
            outcome, {"retry-after": str(ticks)}, {"error": {"message": f"b{ticks}"}}
        )

    return send


async def test_the_first_429_to_arrive_is_kept_and_every_status_is_counted(
    tools: ModuleType,
) -> None:
    # Task order: 200 (5 turns), 429 (3), 429 (1), 503 (2), 200 (0). The reply that arrives first
    # among the 429s is the third task's, not the second's.
    send = scripted(tools, [(200, 5), (429, 3), (429, 1), (503, 2), (200, 0)])
    result = await tools.run_burst(5, send)
    assert result.counts == {"200": 2, "429": 2, "503": 1}
    assert result.first_429.headers == {"retry-after": "1"}
    assert result.first_429.body == {"error": {"message": "b1"}}
    assert result.summary() == "sent 5: 200 x2, 429 x2, 503 x1"


async def test_a_burst_with_no_429_has_none_to_record(tools: ModuleType) -> None:
    result = await tools.run_burst(3, scripted(tools, [(200, 0), (200, 1), (503, 0)]))
    assert result.first_429 is None
    assert result.summary() == "sent 3: 200 x2, 503 x1"


async def test_a_failure_with_no_reply_is_counted_by_class_name_never_by_message(
    tools: ModuleType,
) -> None:
    boom = ConnectionError("key=SECRET-VALUE")
    result = await tools.run_burst(2, scripted(tools, [(boom, 0), (429, 1)]))
    assert result.counts == {"ConnectionError": 1, "429": 1}
    assert "SECRET" not in result.summary()
    assert result.first_429 is not None


async def test_all_the_requests_are_in_flight_at_once(tools: ModuleType) -> None:
    started = 0
    gate = asyncio.Event()

    async def send() -> Any:
        nonlocal started
        started += 1
        if started == 6:
            gate.set()
        await gate.wait()  # a sequential burst would wait here forever
        return tools.Reply(429, {}, {})

    result = await asyncio.wait_for(tools.run_burst(6, send), timeout=5.0)
    assert (started, result.counts) == (6, {"429": 6})


# --- the Groq script ------------------------------------------------------------------------------


def groq_settings() -> Any:
    return make_settings(groq_api_key=GROQ_KEY, groq_model="openai/gpt-oss-120b")


def a_synthetic_file(directory: Path, name: str) -> Path:
    path = directory / f"generate_{name}.json"
    path.write_text(json.dumps({"_synthetic": True, "_note": "hand-made", "status_code": 429}))
    return path


@pytest.mark.parametrize(
    "args", [["429_per_minute", "--burst", "41"], ["429_per_minute", "--burst=1000"]]
)
async def test_the_groq_script_refuses_a_burst_over_the_cap_before_reading_a_setting(
    groq_script: ModuleType, monkeypatch: pytest.MonkeyPatch, args: list[str]
) -> None:
    refuse_settings(groq_script, monkeypatch)
    with pytest.raises(SystemExit, match="above the hard cap of 40"):
        await groq_script.main(args)


@pytest.mark.parametrize("mode", ["429_daily_quota", "success", "400_schema_violation"])
async def test_the_groq_script_never_bursts_a_daily_quota_or_a_normal_mode(
    groq_script: ModuleType, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    refuse_settings(groq_script, monkeypatch)
    with pytest.raises(SystemExit, match="never provoked"):
        await groq_script.main([mode, "--burst", "2"])


async def test_the_groq_script_reads_a_429_through_the_adapter_and_keeps_only_what_a_fixture_holds(
    groq_script: ModuleType,
) -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            headers={
                "retry-after": "275",
                "x-ratelimit-remaining-tokens": "0",
                "x-request-id": "req_abcdef123456",
                "set-cookie": "session=abc",
            },
            json={"error": {"message": DAILY_429, "type": "tokens", "code": "rate_limit_exceeded"}},
        )

    reply = await groq_script.one_call(
        groq_settings(), "429_daily_quota", transport=httpx.MockTransport(respond)
    )
    assert reply.status_code == 429
    assert reply.headers == {"retry-after": "275", "x-ratelimit-remaining-tokens": "0"}
    assert reply.read == "ProviderRateLimited(is_quota=True, retry_after_s=275.0)"
    assert reply.body["error"]["message"] == DAILY_429  # the raw body: scrubbed when written


async def test_the_groq_script_sends_the_smallest_request_for_a_429(
    groq_script: ModuleType,
) -> None:
    seen: list[dict[str, Any]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(429, headers={"retry-after": "1"}, json={"error": {"message": "x"}})

    await groq_script.one_call(
        groq_settings(), "429_per_minute", transport=httpx.MockTransport(respond)
    )
    [request] = seen
    assert request["max_completion_tokens"] == 20  # the cap the invalid_output mode uses
    assert request["messages"][1]["content"] == groq_script.TINY_USER


def reply_429(tools: ModuleType, message: str, retry_after: str) -> Any:
    return tools.Reply(
        429,
        {"retry-after": retry_after},
        {"error": {"message": message, "type": "tokens", "code": "rate_limit_exceeded"}},
        "ProviderRateLimited(is_quota=False, retry_after_s=1.0)",
    )


async def test_a_groq_burst_writes_the_first_429_scrubbed_and_drops_the_synthetic_marker(
    groq_script: ModuleType,
    tools: ModuleType,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = a_synthetic_file(tmp_path, "429_per_minute")
    steps = iter(
        [
            (tools.Reply(200, {}, {}), 5),
            (reply_429(tools, "SECOND-429-BODY", "9"), 3),
            (reply_429(tools, f"first, in organization `{ORG}`", "4"), 1),
            (tools.Reply(200, {}, {}), 0),
            (tools.Reply(200, {}, {}), 2),
        ]
    )

    async def send() -> Any:
        reply, ticks = next(steps)
        for _ in range(ticks):
            await asyncio.sleep(0)
        return reply

    recorded = {"date": "2026-10-15", "model": "m"}
    await groq_script.record_429("429_per_minute", 5, groq_settings(), recorded, send)

    written = json.loads(path.read_text())
    assert "_synthetic" not in written
    assert "_note" not in written
    assert written["_recorded"] == recorded
    assert written["status_code"] == 429
    assert written["headers"] == {"retry-after": "4"}
    assert written["body"]["error"]["message"] == "first, in organization `org_REDACTED`"
    assert ORG not in path.read_text()
    assert "SECOND-429-BODY" not in path.read_text()
    out = capsys.readouterr().out
    assert "About to send 5 real requests" in out  # printed before anything was sent
    assert "sent 5: 200 x3, 429 x2" in out
    assert "ProviderRateLimited(is_quota=False" in out
    assert GROQ_KEY not in out
    assert "SECOND-429-BODY" not in out
    assert ORG not in out


async def test_a_groq_burst_with_no_429_writes_nothing_and_fails(
    groq_script: ModuleType,
    tools: ModuleType,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = a_synthetic_file(tmp_path, "429_per_minute")
    before = path.read_text()

    async def send() -> Any:
        return tools.Reply(200, {}, {})

    with pytest.raises(SystemExit) as info:
        await groq_script.record_429("429_per_minute", 4, groq_settings(), {}, send)
    assert info.value.code == 1
    assert path.read_text() == before  # the hand-made file is still there, marker included
    assert "no 429 came: nothing written" in capsys.readouterr().out


async def test_a_single_groq_call_that_gets_no_429_writes_nothing_and_says_so(
    groq_script: ModuleType,
    tools: ModuleType,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = a_synthetic_file(tmp_path, "429_daily_quota")
    before = path.read_text()

    async def send() -> Any:
        return tools.Reply(200, {}, {})

    with pytest.raises(SystemExit) as info:
        await groq_script.record_429("429_daily_quota", 1, groq_settings(), {}, send)
    assert info.value.code == 0
    assert path.read_text() == before
    out = capsys.readouterr().out
    assert "About to send 1 real request " in out
    assert "nothing written" in out


# --- the Gemini script ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "args",
    [["error", "429_per_minute", "--burst", "41"], ["error", "429_per_minute", "--burst=99"]],
)
async def test_the_gemini_script_refuses_a_burst_over_the_cap_before_reading_a_setting(
    gemini_script: ModuleType, monkeypatch: pytest.MonkeyPatch, args: list[str]
) -> None:
    refuse_settings(gemini_script, monkeypatch)
    with pytest.raises(SystemExit, match="above the hard cap of 40"):
        await gemini_script.main(args)


@pytest.mark.parametrize(
    "args",
    [
        ["error", "429_daily_quota", "--burst", "2"],
        ["error", "500", "--burst", "2"],
        ["success", "--burst", "2"],
        ["invalid_output", "--burst", "2"],
    ],
)
async def test_the_gemini_script_never_bursts_a_daily_quota_or_a_normal_mode(
    gemini_script: ModuleType, monkeypatch: pytest.MonkeyPatch, args: list[str]
) -> None:
    refuse_settings(gemini_script, monkeypatch)
    with pytest.raises(SystemExit, match="never provoked"):
        await gemini_script.main(args)


def per_minute_429(delay: str) -> errors.ClientError:
    return errors.ClientError(
        429,
        {
            "error": {
                "code": 429,
                "message": f"You exceeded your current quota. Key {GOOGLE_KEY}",
                "status": "RESOURCE_EXHAUSTED",
                "details": [
                    {
                        "@type": "type.googleapis.com/google.rpc.QuotaFailure",
                        "violations": [{"quotaId": "GenerateRequestsPerMinutePerProjectPerModel"}],
                    },
                    {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": delay},
                ],
            }
        },
    )


class FakeModels:
    """``client.aio.models`` with a script: each call finishes after some loop turns."""

    def __init__(self, plan: list[tuple[Exception | None, int]]) -> None:
        self._steps = iter(plan)
        self.calls: list[dict[str, Any]] = []

    async def generate_content(self, *, model: str, contents: str, config: Any) -> object:
        self.calls.append(
            {"model": model, "contents": contents, "max_output_tokens": config.max_output_tokens}
        )
        outcome, ticks = next(self._steps)
        for _ in range(ticks):
            await asyncio.sleep(0)
        if outcome is not None:
            raise outcome
        return object()


def gemini_settings() -> Any:
    return make_settings(gemini_api_key=GOOGLE_KEY, gemini_model="gemini-test")


async def test_a_gemini_burst_sends_minimal_requests_and_writes_the_first_429_scrubbed(
    gemini_script: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "generate_429_per_minute.json"
    path.write_text(json.dumps({"_synthetic": True}))
    models = FakeModels(
        [
            (None, 4),
            (per_minute_429("40s"), 3),
            (per_minute_429("53s"), 1),
            (None, 0),
        ]
    )
    client = SimpleNamespace(aio=SimpleNamespace(models=models))
    await gemini_script._record_burst(client, gemini_settings(), {"model": "gemini-test"}, 4)

    assert [c["contents"] for c in models.calls] == [gemini_script.TINY_USER] * 4
    assert {c["max_output_tokens"] for c in models.calls} == {20}
    written = json.loads(path.read_text())
    assert "_synthetic" not in written
    assert written["status_code"] == 429
    assert written["headers"] == {}
    details = written["body"]["error"]["details"]
    assert {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "53s"} in details
    assert GOOGLE_KEY not in path.read_text()
    assert "AIza_REDACTED" in written["body"]["error"]["message"]
    out = capsys.readouterr().out
    assert "About to send 4 real requests" in out
    assert "sent 4: 200 x2, 429 x2" in out
    assert "ProviderRateLimited(is_quota=False, retry_after_s=53.0)" in out
    assert GOOGLE_KEY not in out


async def test_a_gemini_burst_with_no_429_writes_nothing_and_fails(
    gemini_script: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "generate_429_per_minute.json"
    path.write_text(json.dumps({"_synthetic": True}))
    models = FakeModels([(None, 0), (errors.ServerError(503, {"error": {"message": "x"}}), 1)])
    client = SimpleNamespace(aio=SimpleNamespace(models=models))
    with pytest.raises(SystemExit) as info:
        await gemini_script._record_burst(client, gemini_settings(), {}, 2)
    assert info.value.code == 1
    assert json.loads(path.read_text()) == {"_synthetic": True}
    out = capsys.readouterr().out
    assert "sent 2: 200 x1, 503 x1" in out
    assert "no 429 came: nothing written" in out


@pytest.mark.parametrize("script", ["gemini_script", "groq_script"])
def test_both_docstrings_say_a_daily_429_is_recorded_after_it_happened_never_provoked(
    script: str, request: pytest.FixtureRequest
) -> None:
    doc = " ".join((request.getfixturevalue(script).__doc__ or "").split())
    assert "right after a real daily 429" in doc
    assert "never provoked" in doc
    assert "--burst N" in doc
    assert "hard cap 40" in doc
