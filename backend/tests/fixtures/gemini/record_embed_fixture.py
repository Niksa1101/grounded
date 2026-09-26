"""Record ``embed_3.json`` from ONE real Gemini call (3 short texts). Run by hand, never by pytest.

    uv run python tests/fixtures/gemini/record_embed_fixture.py

Reads GEMINI_API_KEY through Settings (repo-root .env) and never prints it. The fixture keeps
only the parsed response body (no HTTP headers), so tests replay the real shape of the answer.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from google import genai
from google.genai import types

from grounded.settings import Settings

MODEL = "gemini-embedding-001"
DIM = 768
TEXTS = [
    "Background tasks run after the response is sent.",
    "Use Depends to declare a dependency.",
    "HTTPException returns an HTTP error to the client.",
]
OUT = Path(__file__).with_name("embed_3.json")


async def main() -> None:
    key = Settings().gemini_api_key
    if key is None:
        raise SystemExit("GEMINI_API_KEY is not set")
    client = genai.Client(api_key=key.get_secret_value())
    response = await client.aio.models.embed_content(
        model=MODEL,
        contents=TEXTS,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT", output_dimensionality=DIM),
    )
    body = response.model_dump(mode="json", exclude_none=True, exclude={"sdk_http_response"})
    fixture = {
        "_recorded": {
            "model": MODEL,
            "dim": DIM,
            "task_type": "RETRIEVAL_DOCUMENT",
            "texts": TEXTS,
        },
        "response": body,
    }
    OUT.write_text(json.dumps(fixture, indent=1) + "\n", encoding="utf-8")
    sizes = [len(e.values or []) for e in response.embeddings or []]
    print(f"wrote {OUT.name}: {len(sizes)} embeddings, dims {sizes}")


if __name__ == "__main__":
    asyncio.run(main())
