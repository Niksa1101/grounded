"""A rig that puts the real ``CohereReranker`` and the real ``cohere`` SDK client under it in front
of a scripted fake backend (no network).

Like ``GroqRig`` in ``provider_rigs.py``: the SDK runs for real over an ``httpx.MockTransport``, so
the request it builds (URL, key, body), the error classes it raises and a retry it must not make
(a second HTTP request) are all visible. The fixtures are in ``tests/fixtures/cohere``: they follow
the documented response shape and are NOT recorded (see their README).

A step is a name (``success``, ``429_retry_after``, ``429_no_retry_after``, ``500``, ``401``,
``404``, ``hang``, ``timeout``, ``connect_error``), a ready-made ``httpx.Response`` or an exception
to raise from the transport. ``scores_response`` builds a success for any number of documents.
"""

from __future__ import annotations

import asyncio
import json
from collections import deque
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import httpx

from grounded.retrieval.rerank import CohereReranker
from grounded.retrieval.types import RetrievedChunk

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "cohere"
COHERE_MODEL = "rerank-v4.0-pro"
QUERY = "How do I declare a path parameter?"


def load_fixture(name: str) -> dict[str, Any]:
    return json.loads((FIXTURES / f"rerank_{name}.json").read_text(encoding="utf-8"))


def fixture_response(name: str) -> httpx.Response:
    fixture = load_fixture(name)
    return httpx.Response(fixture["status_code"], json=fixture["body"], headers=fixture["headers"])


def success_body() -> dict[str, Any]:
    """The documented-shape success for five documents (``rerank_success.json``), to mutate."""
    return load_fixture("success")["body"]


def scores_response(scores: Sequence[float]) -> httpx.Response:
    """A success in the documented shape: one result per document, best first, equal scores in a
    deliberately reversed index order (Cohere's order among ties is its own business)."""
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], -i))
    results = [{"index": i, "relevance_score": scores[i]} for i in order]
    return httpx.Response(200, json={"id": "test", "results": results})


def chunks(count: int) -> list[RetrievedChunk]:
    """``count`` chunks in RRF order; chunk ``i`` has id ``i``, hash ``h<i>`` and its own text."""
    return [
        RetrievedChunk(
            chunk_id=i,
            section_id=f"tutorial/page.md#section-{i}",
            anchor_path=(f"section-{i}",),
            breadcrumb_text=f"Tutorial > Page > Section {i}",
            url=f"https://fastapi.tiangolo.com/tutorial/page/#section-{i}",
            content=f"Content of section {i}.",
            token_count=5,
            content_hash=f"h{i}",
            rrf_score=1.0 / (i + 1),
        )
        for i in range(count)
    ]


type Step = str | httpx.Response | Exception


class CohereRig:
    def __init__(self, *steps: Step, timeout_s: float = 3.0, api_key: str = "test-key") -> None:
        self._script: deque[Step] = deque(steps)
        self.http_requests: list[httpx.Request] = []
        self.http_client = httpx.AsyncClient(transport=httpx.MockTransport(self._handle))
        self.reranker = CohereReranker.create(
            api_key, model=COHERE_MODEL, timeout_s=timeout_s, http_client=self.http_client
        )

    async def _handle(self, request: httpx.Request) -> httpx.Response:
        self.http_requests.append(request)
        step = self._script.popleft()
        if isinstance(step, Exception):
            raise step
        if isinstance(step, httpx.Response):
            return step
        match step:
            case "success" | "429_retry_after" | "429_no_retry_after" | "500" | "401" | "404":
                return fixture_response(step)
            case "hang":
                await asyncio.Event().wait()
            case "timeout":
                raise httpx.ReadTimeout("slow", request=request)
            case "connect_error":
                raise httpx.ConnectError("reset", request=request)
        raise AssertionError(f"unknown step {step!r}")

    @property
    def request_bodies(self) -> list[dict[str, Any]]:
        return [json.loads(request.content) for request in self.http_requests]

    @property
    def remaining(self) -> int:
        """Scripted steps the adapter did not consume."""
        return len(self._script)
