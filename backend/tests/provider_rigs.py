"""Rigs that put a real provider adapter in front of a scripted fake backend (no network).

The shared contract test (``tests/unit/test_provider_contract.py``) drives every adapter through the
same scenarios; a rig turns a scenario name into what its SDK would have seen:

- ``GeminiRig``: a fake ``google-genai`` client that returns the recorded response objects and
  raises the SDK's own error types;
- ``GroqRig``: the real ``groq`` SDK client over an ``httpx.MockTransport``, so the SDK's request
  building, response parsing and error classes run for real, and a retry would show up as a second
  HTTP request.

A step is a name (``success``, ``invalid_output``, ``success_without_usage``, ``429_per_minute``,
``429_daily_quota``, ``500``, ``400``, ``hang``, ``timeout``, ``connect_error``) or a ``Reply`` (a
normal completion carrying this text and the success recording's usage). The fixtures are in
``tests/fixtures/gemini`` and ``tests/fixtures/groq`` (recorded, or hand-made: see their READMEs).
"""

from __future__ import annotations

import asyncio
import json
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import httpx
from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types
from groq import AsyncGroq

from grounded.generation.providers.base import LLMProvider
from grounded.generation.providers.gemini import GeminiProvider
from grounded.generation.providers.groq import GroqProvider
from grounded.settings import GeminiThinkingLevel, GroqReasoningEffort

FIXTURES = Path(__file__).resolve().parent / "fixtures"
GEMINI_MODEL = "gemini-3.5-flash-lite"
GROQ_MODEL = "openai/gpt-oss-120b"


@dataclass(frozen=True)
class Reply:
    """A normal completion with this text (and the usage of the success recording)."""

    text: str


type Step = str | Reply


@dataclass(frozen=True)
class SeenRequest:
    """What reached the SDK, in the adapters' common vocabulary."""

    system: str
    user: str
    temperature: float
    max_output_tokens: int


class Rig(Protocol):
    @property
    def provider(self) -> LLMProvider: ...

    @property
    def requests(self) -> list[SeenRequest]: ...

    @property
    def remaining(self) -> int:
        """Scripted steps the adapter did not consume."""
        ...

    @property
    def closed(self) -> bool: ...

    async def aclose(self) -> None: ...


def load_fixture(directory: str, name: str) -> dict[str, Any]:
    path = FIXTURES / directory / f"generate_{name}.json"
    return json.loads(path.read_text(encoding="utf-8"))


class StepClock:
    """Each reading moves 0.25 s, so a call that reads it twice took exactly 250 ms."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        self.now += 0.25
        return self.now


class NeverReturns:
    """A script step that hangs until the provider's own timeout cancels it."""


# --- Gemini ---------------------------------------------------------------------------------------

type GeminiStep = BaseException | genai_types.GenerateContentResponse | NeverReturns


class FakeGeminiModels:
    def __init__(self, script: Sequence[GeminiStep]) -> None:
        self.script = list(script)
        self.calls: list[dict[str, Any]] = []

    async def generate_content(
        self, *, model: str, contents: str, config: genai_types.GenerateContentConfig
    ) -> genai_types.GenerateContentResponse:
        self.calls.append({"model": model, "contents": contents, "config": config})
        step = self.script.pop(0)
        if isinstance(step, BaseException):
            raise step
        if isinstance(step, NeverReturns):
            await asyncio.Event().wait()
        assert isinstance(step, genai_types.GenerateContentResponse)
        return step


class FakeGeminiClient:
    def __init__(self, models: FakeGeminiModels) -> None:
        self.closed = False
        client = self

        class Aio:
            def __init__(self) -> None:
                self.models = models

            async def aclose(self) -> None:
                client.closed = True

        self.aio = Aio()


def gemini_response(name: str) -> genai_types.GenerateContentResponse:
    return genai_types.GenerateContentResponse.model_validate(
        load_fixture("gemini", name)["response"]
    )


def gemini_error(name: str) -> genai_errors.APIError:
    fixture = load_fixture("gemini", name)
    code, body = fixture["status_code"], fixture["body"]
    response = httpx.Response(code, json=body, headers=fixture["headers"])
    cls = genai_errors.ServerError if code >= 500 else genai_errors.ClientError
    return cls(code, body, response)


def _gemini_step(step: Step | GeminiStep) -> GeminiStep:
    if isinstance(step, BaseException | genai_types.GenerateContentResponse | NeverReturns):
        return step  # ready-made, for tests that need a shape the fixtures do not have
    if isinstance(step, Reply):
        response = gemini_response("success")
        assert response.candidates
        parts = response.candidates[0].content.parts if response.candidates[0].content else []
        assert parts
        parts[0].text = step.text
        return response
    match step:
        case "success" | "invalid_output":
            return gemini_response(step)
        case "success_without_usage":
            response = gemini_response("success")
            response.usage_metadata = None
            return response
        case "429_per_minute" | "429_daily_quota" | "500":
            return gemini_error(step)
        case "400":
            body = {"error": {"code": 400, "message": "bad key", "status": "INVALID_ARGUMENT"}}
            return genai_errors.ClientError(400, body, httpx.Response(400, json=body))
        case "hang":
            return NeverReturns()
        case "timeout":
            return httpx.ReadTimeout("slow")
        case "connect_error":
            return httpx.ConnectError("reset")
    raise AssertionError(f"unknown step {step!r}")


class GeminiRig:
    """``steps`` may also hold ready-made SDK responses or exceptions."""

    def __init__(
        self, *steps: Step | GeminiStep, thinking_level: GeminiThinkingLevel = "minimal"
    ) -> None:
        self.models = FakeGeminiModels([_gemini_step(step) for step in steps])
        self.client = FakeGeminiClient(self.models)
        self.provider: LLMProvider = GeminiProvider(
            cast(genai.Client, self.client),
            model=GEMINI_MODEL,
            thinking_level=thinking_level,
            clock=StepClock(),
        )

    @property
    def requests(self) -> list[SeenRequest]:
        return [
            SeenRequest(
                system=str(call["config"].system_instruction),
                user=call["contents"],
                temperature=call["config"].temperature,
                max_output_tokens=call["config"].max_output_tokens,
            )
            for call in self.models.calls
        ]

    @property
    def remaining(self) -> int:
        return len(self.models.script)

    @property
    def closed(self) -> bool:
        return self.client.closed

    async def aclose(self) -> None:
        await cast(GeminiProvider, self.provider).aclose()


# --- Groq -----------------------------------------------------------------------------------------


def groq_response(name: str) -> httpx.Response:
    fixture = load_fixture("groq", name)
    return httpx.Response(fixture["status_code"], json=fixture["body"], headers=fixture["headers"])


def groq_success_body() -> dict[str, Any]:
    return load_fixture("groq", "success")["body"]


def groq_reply_response(text: str) -> httpx.Response:
    body = groq_success_body()
    body["choices"][0]["message"]["content"] = text
    return httpx.Response(200, json=body)


def groq_error_response(status_code: int, **error: Any) -> httpx.Response:
    return httpx.Response(status_code, json={"error": error})


class GroqRig:
    """``steps`` may also hold ready-made ``httpx.Response`` objects or exceptions to raise from
    the transport, for tests that need a shape the fixtures do not have."""

    def __init__(
        self,
        *steps: Step | httpx.Response | Exception,
        reasoning_effort: GroqReasoningEffort = "low",
    ) -> None:
        self._script: deque[Step | httpx.Response | Exception] = deque(steps)
        self.http_requests: list[httpx.Request] = []
        http_client = httpx.AsyncClient(transport=httpx.MockTransport(self._handle))
        self.client = AsyncGroq(api_key="test-key", max_retries=0, http_client=http_client)
        self.provider: LLMProvider = GroqProvider(
            self.client, model=GROQ_MODEL, reasoning_effort=reasoning_effort, clock=StepClock()
        )

    async def _handle(self, request: httpx.Request) -> httpx.Response:
        self.http_requests.append(request)
        step = self._script.popleft()
        if isinstance(step, Exception):
            raise step
        if isinstance(step, httpx.Response):
            return step
        if isinstance(step, Reply):
            return groq_reply_response(step.text)
        match step:
            case (
                "success"
                | "invalid_output"
                | "400_schema_violation"
                | "429_per_minute"
                | "429_daily_quota"
                | "500"
            ):
                return groq_response(step)
            case "success_without_usage":
                body = groq_success_body()
                del body["usage"]
                return httpx.Response(200, json=body)
            case "400":
                return groq_error_response(
                    400, message="Bad request.", type="invalid_request_error"
                )
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
    def requests(self) -> list[SeenRequest]:
        seen: list[SeenRequest] = []
        for body in self.request_bodies:
            system, user = body["messages"]
            assert (system["role"], user["role"]) == ("system", "user")
            seen.append(
                SeenRequest(
                    system=system["content"],
                    user=user["content"],
                    temperature=body["temperature"],
                    max_output_tokens=body["max_completion_tokens"],
                )
            )
        return seen

    @property
    def remaining(self) -> int:
        return len(self._script)

    @property
    def closed(self) -> bool:
        return self.client.is_closed()

    async def aclose(self) -> None:
        await cast(GroqProvider, self.provider).aclose()
