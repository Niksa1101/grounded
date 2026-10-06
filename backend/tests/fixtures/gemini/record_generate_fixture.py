"""Record the ``generate_*.json`` fixtures from REAL Gemini calls. Run by hand, never by pytest.

    uv run python tests/fixtures/gemini/record_generate_fixture.py success
    uv run python tests/fixtures/gemini/record_generate_fixture.py invalid_output
    uv run python tests/fixtures/gemini/record_generate_fixture.py error 429_per_minute

Reads GEMINI_API_KEY and GEMINI_MODEL through Settings (repo-root .env) and never prints the key.
Each run uses quota (one request). The fixtures keep only the parsed body (no HTTP headers except
``retry-after`` on an error), so tests replay the real shape of the answer.

- ``success``: one normal call, schema-valid output.
- ``invalid_output``: the same call with a tiny ``max_output_tokens``, so the answer is cut off
  (``finish_reason=MAX_TOKENS``) and is not valid JSON. This is a real "bad output" without
  needing the model to misbehave.
- ``error <name>``: makes a call and stores the ``APIError`` it raises, under
  ``generate_<name>.json``. Errors can't be produced on demand: run it while you are actually
  rate limited (``429_per_minute``, ``429_daily_quota``) or when Google returns a 5xx (``500``).
  A call that succeeds writes nothing.

The checked-in fixtures were first **hand-built** (marked ``"_synthetic": true``) from the SDK's
types and the google.rpc error model, because the Author had no key at the time. Re-recording
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

from grounded.generation.providers.gemini import to_gemini_schema
from grounded.schemas.llm import LLMAnswer
from grounded.settings import Settings

HERE = Path(__file__).parent
SYSTEM = "Answer using only the sources. Cite them as [c1]. Reply as JSON."
USER = (
    "Question: How do I add a background task?\n\n"
    '<source id="c1" section="Tutorial > Background Tasks" '
    'url="https://fastapi.tiangolo.com/tutorial/background-tasks/">\n'
    "Declare a parameter of type BackgroundTasks and call add_task() on it. "
    "Tasks run after the response is sent.\n</source>"
)


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


async def main(args: list[str]) -> None:
    settings = Settings()
    key = settings.gemini_api_key
    if key is None or not settings.gemini_model:
        raise SystemExit("GEMINI_API_KEY and GEMINI_MODEL must be set")
    if not args or args[0] not in ("success", "invalid_output", "error"):
        raise SystemExit(__doc__)
    kind = args[0]
    if kind == "error" and len(args) != 2:
        raise SystemExit("usage: ... error <429_per_minute|429_daily_quota|500>")
    model = settings.gemini_model
    recorded = {
        "model": model,
        "thinking_level": settings.gemini_thinking_level,
        "temperature": settings.llm_temperature,
    }
    max_tokens = 20 if kind == "invalid_output" else settings.llm_max_output_tokens
    client = genai.Client(api_key=key.get_secret_value())
    try:
        response = await client.aio.models.generate_content(
            model=model, contents=USER, config=_config(settings, max_tokens)
        )
    except errors.APIError as exc:
        if kind != "error":
            raise
        headers = getattr(exc.response, "headers", {})
        retry_after = headers.get("retry-after") if headers else None
        body = exc.details if isinstance(exc.details, dict) else {"error": {"message": exc.message}}
        _write(
            args[1],
            recorded,
            {
                "status_code": exc.code,
                "headers": {"retry-after": retry_after} if retry_after else {},
                "body": body,
            },
        )
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
