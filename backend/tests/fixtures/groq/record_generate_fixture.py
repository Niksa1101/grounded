"""Record the ``generate_*.json`` fixtures from REAL Groq calls. Run by hand, never by pytest.

    uv run python tests/fixtures/groq/record_generate_fixture.py success
    uv run python tests/fixtures/groq/record_generate_fixture.py invalid_output
    uv run python tests/fixtures/groq/record_generate_fixture.py 400_schema_violation

Reads GROQ_API_KEY and GROQ_MODEL through Settings (repo-root .env) and never prints the key. Each
run uses quota (one request). The call goes through ``GroqProvider``, so the request is the one the
adapter really sends; an event hook on the HTTP client keeps the raw response (status, rate-limit
headers, body), which is what the fixture stores, trimmed of ids.

- ``success``: one normal call, schema-valid output.
- ``invalid_output``: the same call with ``max_output_tokens=20``, which the reasoning alone uses up,
  so Groq answers ``400 json_validate_failed`` with an empty ``failed_generation``. This is how a
  cut-off reply arrives (a real "bad output" without the model misbehaving).
- ``400_schema_violation``: asks for a long ``reason`` under a model that caps it at 40 characters,
  so Groq's own schema check rejects the finished JSON (``failed_generation`` holds it).

The 429 and 5xx fixtures can't be produced on demand and are hand-made (see README.md).
"""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any, Literal

import httpx
from groq import AsyncGroq
from pydantic import BaseModel, Field

from grounded.generation.providers.groq import GroqProvider
from grounded.infra.provider_errors import ProviderError
from grounded.schemas.llm import LLMAnswer
from grounded.settings import Settings

HERE = Path(__file__).parent
SYSTEM = "Answer using only the sources. Cite them as [c1]. Reply as JSON."
SOURCE = (
    '<source id="c1" section="Tutorial > Background Tasks" '
    'url="https://fastapi.tiangolo.com/tutorial/background-tasks/">\n'
    "Declare a parameter of type BackgroundTasks and call add_task() on it. "
    "Tasks run after the response is sent.\n</source>"
)
USER = f"Question: How do I add a background task?\n\n{SOURCE}"


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


def write_fixture(
    name: str,
    *,
    recorded: dict[str, Any],
    status_code: int,
    headers: dict[str, str],
    body: dict[str, Any],
) -> None:
    trimmed = trim_body(body) if status_code == 200 else body
    payload = {
        "_recorded": recorded,
        "status_code": status_code,
        "headers": headers,
        "body": trimmed,
    }
    path = HERE / f"generate_{name}.json"
    path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {path.name}")


async def main(args: list[str]) -> None:
    scenarios = ("success", "invalid_output", "400_schema_violation")
    if len(args) != 1 or args[0] not in scenarios:
        raise SystemExit(__doc__)
    scenario = args[0]
    settings = Settings()
    if settings.groq_api_key is None or not settings.groq_model:
        raise SystemExit("GROQ_API_KEY and GROQ_MODEL must be set")

    captured: list[httpx.Response] = []

    async def keep(response: httpx.Response) -> None:
        await response.aread()
        captured.append(response)

    client = AsyncGroq(
        api_key=settings.groq_api_key.get_secret_value(),
        max_retries=0,
        http_client=httpx.AsyncClient(event_hooks={"response": [keep]}),
    )
    provider = GroqProvider(
        client, model=settings.groq_model, reasoning_effort=settings.groq_reasoning_effort
    )
    schema: type[BaseModel] = Verdict if scenario == "400_schema_violation" else LLMAnswer
    try:
        await provider.generate(
            system=SYSTEM,
            user=VERDICT_USER if schema is Verdict else USER,
            schema=schema,
            temperature=settings.llm_temperature,
            max_output_tokens=20 if scenario == "invalid_output" else settings.llm_max_output_tokens,
            timeout_s=settings.llm_timeout_s,
        )
    except ProviderError as exc:  # the point of the error scenarios; the response was captured
        print(f"the adapter raised {type(exc).__name__}")
    finally:
        await provider.aclose()
    [response] = captured
    write_fixture(
        scenario,
        recorded={
            "date": date.today().isoformat(),
            "model": settings.groq_model,
            "reasoning_effort": settings.groq_reasoning_effort,
            "temperature": settings.llm_temperature,
        },
        status_code=response.status_code,
        headers={k: v for k, v in response.headers.items() if k.startswith("x-ratelimit")},
        body=response.json(),
    )


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:]))
