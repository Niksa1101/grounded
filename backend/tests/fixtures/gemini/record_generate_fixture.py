"""Record the ``generate_*.json`` fixtures from REAL Gemini calls. Run by hand, never by pytest.

    uv run python tests/fixtures/gemini/record_generate_fixture.py success
    uv run python tests/fixtures/gemini/record_generate_fixture.py invalid_output
    uv run python tests/fixtures/gemini/record_generate_fixture.py error 429_per_minute --burst 20
    uv run python tests/fixtures/gemini/record_generate_fixture.py error 429_daily_quota

Reads GEMINI_API_KEY and GEMINI_MODEL through Settings (repo-root .env) and never prints the key.
Each run uses quota (one request, or ``--burst N``). The fixtures keep only the parsed body (no HTTP
headers except ``retry-after`` on an error), so tests replay the real shape of the answer.

- ``success``: one normal call, schema-valid output.
- ``invalid_output``: the same call with a tiny ``max_output_tokens``, so the answer is cut off
  (``finish_reason=MAX_TOKENS``) and is not valid JSON. This is a real "bad output" without
  needing the model to misbehave.
- ``error <name>``: makes a call and stores the ``APIError`` it raises, under
  ``generate_<name>.json``. Errors can't be produced on demand: run it while you are actually
  rate limited (``429_per_minute``, ``429_daily_quota``) or when Google returns a 5xx (``500``).
  A call that succeeds writes nothing. A stored error has ids and key shapes replaced by fixed
  placeholders (``recording_tools.scrub``); read the body once before you commit it.

``--burst N`` (``error 429_per_minute`` only, default 1, hard cap 40) sends N minimal calls at once
(a tiny prompt, ``max_output_tokens=20``) to provoke a per-minute 429 on purpose. It first prints
how many real requests it is about to make, keeps the FIRST 429 that arrives, ignores the others,
prints one line of status counts (no key, no body), and exits non-zero without writing if no 429
came. Never in CI, never in pytest.

A **daily** 429 is never provoked (that would burn a daily quota on purpose): run
``error 429_daily_quota`` right after a real daily 429 happened (the demo or CI got one).

The checked-in 429 and 500 fixtures were first **hand-built** (marked ``"_synthetic": true``) from the
SDK's types and the google.rpc error model, because the Author had no key at the time. Re-recording
overwrites a file and drops the marker.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from google import genai
from google.genai import errors
from google.genai import types

HERE = Path(__file__).parent
# The helper both recording scripts share lives in tests/fixtures. Appended, not prepended, so
# the directories next to it (groq/, cohere/) can never shadow an installed package.
sys.path.append(str(HERE.parent))

from recording_tools import (  # noqa: E402
    BURST_ONLY_FOR_PER_MINUTE,
    Reply,
    burst_notice,
    run_burst,
    scrub,
    split_burst,
)

from grounded.generation.providers.gemini import to_gemini_schema  # noqa: E402
from grounded.infra.gemini_errors import map_api_error  # noqa: E402
from grounded.infra.provider_errors import ProviderRateLimited  # noqa: E402
from grounded.schemas.llm import LLMAnswer  # noqa: E402
from grounded.settings import Settings  # noqa: E402

SYSTEM = "Answer using only the sources. Cite them as [c1]. Reply as JSON."
USER = (
    "Question: How do I add a background task?\n\n"
    '<source id="c1" section="Tutorial > Background Tasks" '
    'url="https://fastapi.tiangolo.com/tutorial/background-tasks/">\n'
    "Declare a parameter of type BackgroundTasks and call add_task() on it. "
    "Tasks run after the response is sent.\n</source>"
)
# The smallest valid request, for --burst: no source, and the cap the invalid_output mode uses.
TINY_USER = "Say ok."
MIN_OUTPUT_TOKENS = 20


def _config(settings: Settings, max_output_tokens: int) -> types.GenerateContentConfig:
    return types.GenerateContentConfig(
        system_instruction=SYSTEM,
        temperature=settings.llm_temperature,
        max_output_tokens=max_output_tokens,
        response_mime_type="application/json",
        response_json_schema=to_gemini_schema(LLMAnswer),
        thinking_config=types.ThinkingConfig(
            thinking_level=types.ThinkingLevel(settings.gemini_thinking_level.upper())
        ),
    )


def _write(name: str, recorded: dict[str, Any], payload: dict[str, Any]) -> None:
    path = HERE / f"generate_{name}.json"
    path.write_text(json.dumps({"_recorded": recorded} | payload, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {path.name}")


def _error_reply(exc: errors.APIError) -> Reply:
    """The ``APIError`` as what a fixture keeps (``retry-after`` only, the scrubbed body) and how the
    adapter reads it, so the Author sees at once whether the file name matches the classification."""
    headers = getattr(exc.response, "headers", {})
    retry_after = headers.get("retry-after") if headers else None
    body = exc.details if isinstance(exc.details, dict) else {"error": {"message": exc.message}}
    mapped = map_api_error(exc)
    read = type(mapped).__name__
    if isinstance(mapped, ProviderRateLimited):
        read += f"(is_quota={mapped.is_quota}, retry_after_s={mapped.retry_after_s})"
    return Reply(
        exc.code or 0,
        {"retry-after": str(retry_after)} if retry_after else {},
        scrub(body),
        read,
    )


def _write_error(name: str, recorded: dict[str, Any], reply: Reply) -> None:
    print(f"the adapter reads it as {reply.read}")
    _write(
        name,
        recorded,
        {"status_code": reply.status_code, "headers": reply.headers, "body": reply.body},
    )


async def _record_burst(
    client: genai.Client, settings: Settings, recorded: dict[str, Any], count: int
) -> None:
    """``count`` minimal calls at once; the first 429 becomes ``generate_429_per_minute.json``."""
    model = settings.gemini_model
    assert model
    print(burst_notice(count, f"Gemini ({model})"))

    async def one() -> Reply:
        try:
            await client.aio.models.generate_content(
                model=model, contents=TINY_USER, config=_config(settings, MIN_OUTPUT_TOKENS)
            )
        except errors.APIError as exc:
            return _error_reply(exc)
        return Reply(200, {}, {})

    result = await run_burst(count, one)
    print(result.summary())
    if result.first_429 is None:
        print("no 429 came: nothing written")
        raise SystemExit(1)
    _write_error("429_per_minute", recorded, result.first_429)


async def main(args: list[str]) -> None:
    args, burst = split_burst(args)
    if not args or args[0] not in ("success", "invalid_output", "error"):
        raise SystemExit(__doc__)
    kind = args[0]
    if kind == "error" and len(args) != 2:
        raise SystemExit("usage: ... error <429_per_minute|429_daily_quota|500> [--burst N]")
    if burst > 1 and args != ["error", "429_per_minute"]:
        raise SystemExit(BURST_ONLY_FOR_PER_MINUTE)
    settings = Settings()
    key = settings.gemini_api_key
    if key is None or not settings.gemini_model:
        raise SystemExit("GEMINI_API_KEY and GEMINI_MODEL must be set")
    model = settings.gemini_model
    recorded = {
        "model": model,
        "thinking_level": settings.gemini_thinking_level,
        "temperature": settings.llm_temperature,
    }
    max_tokens = MIN_OUTPUT_TOKENS if kind == "invalid_output" else settings.llm_max_output_tokens
    client = genai.Client(api_key=key.get_secret_value())
    if burst > 1:
        try:
            await _record_burst(client, settings, recorded, burst)
        finally:
            await client.aio.aclose()
        return
    try:
        response = await client.aio.models.generate_content(
            model=model, contents=USER, config=_config(settings, max_tokens)
        )
    except errors.APIError as exc:
        if kind != "error":
            raise
        _write_error(args[1], recorded, _error_reply(exc))
        return
    finally:
        await client.aio.aclose()
    if kind == "error":
        print("the call succeeded; no error to record")
        return
    body = response.model_dump(mode="json", exclude_none=True, exclude={"sdk_http_response"})
    _write(kind, recorded, {"response": body})
    usage = response.usage_metadata
    print("usage:", usage.model_dump(exclude_none=True) if usage else None)


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:]))
