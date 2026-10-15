"""Record the ``generate_*.json`` fixtures from REAL Groq calls. Run by hand, never by pytest.

    uv run python tests/fixtures/groq/record_generate_fixture.py success
    uv run python tests/fixtures/groq/record_generate_fixture.py invalid_output
    uv run python tests/fixtures/groq/record_generate_fixture.py 400_schema_violation
    uv run python tests/fixtures/groq/record_generate_fixture.py 429_per_minute --burst 12
    uv run python tests/fixtures/groq/record_generate_fixture.py 429_daily_quota

Reads GROQ_API_KEY and GROQ_MODEL through Settings (repo-root .env) and never prints the key. Each
run uses quota (one request, or ``--burst N``). The call goes through ``GroqProvider``, so the request
is the one the adapter really sends; an event hook on the HTTP client keeps the raw response (status,
rate-limit and retry-after headers, body), which is what the fixture stores, trimmed of ids.

- ``success``: one normal call, schema-valid output.
- ``invalid_output``: the same call with ``max_output_tokens=20``, which the reasoning alone uses up,
  so Groq answers ``400 json_validate_failed`` with an empty ``failed_generation``. This is how a
  cut-off reply arrives (a real "bad output" without the model misbehaving).
- ``400_schema_violation``: asks for a long ``reason`` under a model that caps it at 40 characters,
  so Groq's own schema check rejects the finished JSON (``failed_generation`` holds it).
- ``429_per_minute`` and ``429_daily_quota``: make one minimal call (a tiny prompt and
  ``max_output_tokens=20``) and store the 429 it gets, under ``generate_<name>.json``. A call that
  does not get a 429 writes nothing and says so. The stored file replaces the hand-made one, so its ``_synthetic``
  marker and note are gone. Before writing, ids and key shapes are replaced by fixed placeholders
  (``recording_tools.scrub``): a real daily 429 names the organization (``org_...``), and none of it
  may reach the repo. Read the body once before you commit it.

``--burst N`` (``429_per_minute`` only, default 1, hard cap 40) sends N such calls at once to
provoke a per-minute 429 on purpose. It first prints how many real requests it is about to make,
keeps the FIRST 429 that arrives, ignores the others, prints one line of status counts (no key, no
body), and exits non-zero without writing if no 429 came. Never in CI, never in pytest.

A **daily** 429 is never provoked (that would burn a daily quota on purpose): run
``429_daily_quota`` right after a real daily 429 happened, e.g. when the CI judge stopped on
``ProviderRateLimited (daily quota)``. The 5xx fixture can't be produced on demand and is hand-made
(see README.md).
"""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import date
from pathlib import Path
from collections.abc import Awaitable, Callable
from typing import Any, Literal

import httpx
from groq import AsyncGroq
from pydantic import BaseModel, Field

HERE = Path(__file__).parent
# The helper both recording scripts share lives in tests/fixtures. Appended, not prepended, so
# the directories next to it (groq/, cohere/) can never shadow an installed package.
sys.path.append(str(HERE.parent))

from recording_tools import (  # noqa: E402
    BURST_ONLY_FOR_PER_MINUTE,
    Reply,
    burst_notice,
    keep_headers,
    run_burst,
    scrub,
    split_burst,
)

from grounded.generation.providers.groq import GroqProvider  # noqa: E402
from grounded.infra.provider_errors import ProviderError, ProviderRateLimited  # noqa: E402
from grounded.schemas.llm import LLMAnswer  # noqa: E402
from grounded.settings import Settings  # noqa: E402

SYSTEM = "Answer using only the sources. Cite them as [c1]. Reply as JSON."
SOURCE = (
    '<source id="c1" section="Tutorial > Background Tasks" '
    'url="https://fastapi.tiangolo.com/tutorial/background-tasks/">\n'
    "Declare a parameter of type BackgroundTasks and call add_task() on it. "
    "Tasks run after the response is sent.\n</source>"
)
USER = f"Question: How do I add a background task?\n\n{SOURCE}"
# The smallest valid request, for the 429 modes: no source, and the cap the invalid_output mode uses.
TINY_USER = "Say ok."
MIN_OUTPUT_TOKENS = 20
SCENARIOS = ("success", "invalid_output", "400_schema_violation", "429_per_minute", "429_daily_quota")


class Verdict(BaseModel):
    verdict: Literal["supported", "partially_supported", "unsupported"]
    reason: str = Field(max_length=40)


VERDICT_USER = (
    f"Claim: Tasks run after the response is sent.\n\nSources:\n{SOURCE}\n\n"
    "Is the claim supported by the source? Explain your reasoning in at least three full sentences."
)


def trim_body(body: dict[str, Any]) -> dict[str, Any]:
    """A 200 reply without the ids and timings; ``id`` stays as a placeholder (the SDK model has it)."""
    usage = body["usage"]
    return {
        "id": "chatcmpl-REDACTED",
        "object": body["object"],
        "created": body["created"],
        "model": body["model"],
        "choices": [
            {
                "index": choice["index"],
                "message": {"role": "assistant", "content": choice["message"]["content"]},
                "finish_reason": choice["finish_reason"],
            }
            for choice in body["choices"]
        ],
        "usage": {
            "prompt_tokens": usage["prompt_tokens"],
            "completion_tokens": usage["completion_tokens"],
            "total_tokens": usage["total_tokens"],
            "completion_tokens_details": usage["completion_tokens_details"],
        },
        "service_tier": body["service_tier"],
    }


def write_fixture(name: str, *, recorded: dict[str, Any], reply: Reply) -> None:
    """Write ``generate_<name>.json`` fresh (an older file, its ``_synthetic`` marker included, is
    replaced), with ids and key shapes scrubbed."""
    body = trim_body(reply.body) if reply.status_code == 200 else reply.body
    payload = {
        "_recorded": recorded,
        "status_code": reply.status_code,
        "headers": reply.headers,
        "body": scrub(body),
    }
    path = HERE / f"generate_{name}.json"
    path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {path.name}")


async def one_call(
    settings: Settings, scenario: str, transport: httpx.AsyncBaseTransport | None = None
) -> Reply:
    """One call through ``GroqProvider``; the reply is what the event hook saw on the wire.

    ``transport`` is for the unit test (a mock transport); the script never passes one. A transport
    failure with no HTTP reply is raised.
    """
    assert settings.groq_api_key is not None and settings.groq_model
    captured: list[httpx.Response] = []

    async def keep(response: httpx.Response) -> None:
        await response.aread()
        captured.append(response)

    client = AsyncGroq(
        api_key=settings.groq_api_key.get_secret_value(),
        max_retries=0,
        http_client=httpx.AsyncClient(event_hooks={"response": [keep]}, transport=transport),
    )
    provider = GroqProvider(
        client, model=settings.groq_model, reasoning_effort=settings.groq_reasoning_effort
    )
    minimal = scenario.startswith("429_")
    schema: type[BaseModel] = Verdict if scenario == "400_schema_violation" else LLMAnswer
    read = ""
    try:
        await provider.generate(
            system=SYSTEM,
            user=TINY_USER if minimal else VERDICT_USER if schema is Verdict else USER,
            schema=schema,
            temperature=settings.llm_temperature,
            max_output_tokens=(
                MIN_OUTPUT_TOKENS
                if minimal or scenario == "invalid_output"
                else settings.llm_max_output_tokens
            ),
            timeout_s=settings.llm_timeout_s,
        )
    except ProviderError as exc:  # the point of the error scenarios; the response was captured
        read = type(exc).__name__
        if isinstance(exc, ProviderRateLimited):
            read += f"(is_quota={exc.is_quota}, retry_after_s={exc.retry_after_s})"
    finally:
        await provider.aclose()
    if not captured:
        raise RuntimeError(f"no HTTP reply was received ({read or 'no adapter error'})")
    response = captured[0]
    return Reply(response.status_code, keep_headers(response.headers), response.json(), read)


async def record_429(
    name: str,
    burst: int,
    settings: Settings,
    recorded: dict[str, Any],
    send: Callable[[], Awaitable[Reply]] | None = None,
) -> None:
    """Make ``burst`` minimal calls, write the first 429 as ``generate_<name>.json``. Nothing is
    written, and a burst exits non-zero, when no 429 came."""
    print(burst_notice(burst, f"Groq ({settings.groq_model})"))
    result = await run_burst(burst, send or (lambda: one_call(settings, name)))
    print(result.summary())
    reply = result.first_429
    if reply is None:
        print("no 429 came: nothing written")
        raise SystemExit(0 if burst == 1 else 1)
    print(f"the adapter read the 429 as {reply.read}")
    write_fixture(name, recorded=recorded, reply=reply)


async def main(args: list[str]) -> None:
    args, burst = split_burst(args)
    if len(args) != 1 or args[0] not in SCENARIOS:
        raise SystemExit(__doc__)
    scenario = args[0]
    if burst > 1 and scenario != "429_per_minute":
        raise SystemExit(BURST_ONLY_FOR_PER_MINUTE)
    settings = Settings()
    if settings.groq_api_key is None or not settings.groq_model:
        raise SystemExit("GROQ_API_KEY and GROQ_MODEL must be set")
    recorded = {
        "date": date.today().isoformat(),
        "model": settings.groq_model,
        "reasoning_effort": settings.groq_reasoning_effort,
        "temperature": settings.llm_temperature,
    }
    if scenario.startswith("429_"):
        await record_429(scenario, burst, settings, recorded)
        return
    reply = await one_call(settings, scenario)
    if reply.read:
        print(f"the adapter raised {reply.read}")
    write_fixture(scenario, recorded=recorded, reply=reply)


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:]))
