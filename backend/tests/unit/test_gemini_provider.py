"""``GeminiProvider`` against recorded-shape fixtures and a fake SDK client (no network).

The ``generate_*.json`` fixtures are SYNTHETIC until the Author re-records them
(``tests/fixtures/gemini/record_generate_fixture.py``); each carries ``"_synthetic": true``. The
shapes come from the SDK's own response and error types, so the tests keep their meaning when the
files are replaced by real recordings.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import httpx
import pytest
from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types
from pydantic import BaseModel

from grounded.generation.providers.base import LLMProvider
from grounded.generation.providers.gemini import GeminiProvider, to_gemini_schema
from grounded.infra.provider_errors import (
    ProviderBadOutput,
    ProviderRateLimited,
    ProviderRequestRejected,
    ProviderTimeout,
    ProviderUnavailable,
)
from grounded.schemas.llm import LLMAnswer
from grounded.settings import GeminiThinkingLevel

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "gemini"
MODEL = "gemini-3.5-flash-lite"


def _fixture(name: str) -> dict[str, Any]:
    return json.loads((FIXTURES / f"generate_{name}.json").read_text(encoding="utf-8"))


def _response(name: str) -> genai_types.GenerateContentResponse:
    return genai_types.GenerateContentResponse.model_validate(_fixture(name)["response"])


def _api_error(name: str) -> genai_errors.APIError:
    fixture = _fixture(name)
    code, body = fixture["status_code"], fixture["body"]
    response = httpx.Response(code, json=body, headers=fixture["headers"])
    cls = genai_errors.ServerError if code >= 500 else genai_errors.ClientError
    return cls(code, body, response)


# --- A fake SDK client ---------------------------------------------------------------------------


class _NeverReturns:
    """A script step that hangs until the provider's own timeout cancels it."""


type _Step = BaseException | genai_types.GenerateContentResponse | _NeverReturns


class _FakeModels:
    def __init__(self, script: Sequence[_Step]) -> None:
        self.script = list(script)
        self.calls: list[dict[str, Any]] = []

    async def generate_content(
        self, *, model: str, contents: str, config: genai_types.GenerateContentConfig
    ) -> genai_types.GenerateContentResponse:
        self.calls.append({"model": model, "contents": contents, "config": config})
        step = self.script.pop(0)
        if isinstance(step, BaseException):
            raise step
        if isinstance(step, _NeverReturns):
            await asyncio.Event().wait()
        assert isinstance(step, genai_types.GenerateContentResponse)
        return step


class _FakeClient:
    def __init__(self, models: _FakeModels) -> None:
        self.closed = False
        client = self

        class Aio:
            def __init__(self) -> None:
                self.models = models

            async def aclose(self) -> None:
                client.closed = True

        self.aio = Aio()


class _Clock:
    """Each reading moves 0.25 s, so a call that reads it twice took exactly 250 ms."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        self.now += 0.25
        return self.now


def _provider(
    *steps: _Step, level: GeminiThinkingLevel = "minimal"
) -> tuple[GeminiProvider, _FakeModels, _FakeClient]:
    models = _FakeModels(steps)
    client = _FakeClient(models)
    provider = GeminiProvider(
        cast(genai.Client, client), model=MODEL, thinking_level=level, clock=_Clock()
    )
    return provider, models, client


async def _generate(provider: LLMProvider, *, timeout_s: float = 12.0) -> Any:
    return await provider.generate(
        system="SYSTEM",
        user="USER",
        schema=LLMAnswer,
        temperature=0.0,
        max_output_tokens=800,
        timeout_s=timeout_s,
    )


# --- Success: parsing, usage, request -----------------------------------------------------------


async def test_a_valid_response_is_parsed_into_the_schema() -> None:
    provider, _, _ = _provider(_response("success"))
    result = await _generate(provider)
    assert isinstance(result.parsed, LLMAnswer)
    assert result.parsed.status == "answered"
    assert result.parsed.claims[0].citation_ids == ["c1"]
    assert LLMAnswer.model_validate_json(result.raw_text) == result.parsed
    assert (result.provider, result.model) == ("gemini", MODEL)
    assert result.latency_ms == 250


async def test_usage_counts_thinking_tokens_as_output() -> None:
    # Fixture: 1234 prompt, 210 candidates, 40 thoughts. Gemini keeps thoughts out of the
    # candidates count but bills them as output, so output_tokens is their sum.
    provider, _, _ = _provider(_response("success"))
    usage = (await _generate(provider)).usage
    assert (usage.input_tokens, usage.output_tokens, usage.thinking_tokens) == (1234, 250, 40)


async def test_missing_thoughts_count_means_no_thinking() -> None:
    response = _response("success")
    assert response.usage_metadata is not None
    response.usage_metadata.thoughts_token_count = None
    provider, _, _ = _provider(response)
    usage = (await _generate(provider)).usage
    assert (usage.output_tokens, usage.thinking_tokens) == (210, 0)


async def test_missing_usage_metadata_counts_zero_tokens() -> None:
    response = _response("success")
    response.usage_metadata = None
    provider, _, _ = _provider(response)
    usage = (await _generate(provider)).usage
    assert (usage.input_tokens, usage.output_tokens, usage.thinking_tokens) == (0, 0, 0)


async def test_the_request_carries_the_settings_and_the_schema() -> None:
    provider, models, _ = _provider(_response("success"), level="low")
    await _generate(provider)
    [call] = models.calls
    config = call["config"]
    assert (call["model"], call["contents"]) == (MODEL, "USER")
    assert config.system_instruction == "SYSTEM"
    assert (config.temperature, config.max_output_tokens) == (0.0, 800)
    assert config.response_mime_type == "application/json"
    assert config.response_json_schema == to_gemini_schema(LLMAnswer)
    assert config.thinking_config.thinking_level == genai_types.ThinkingLevel.LOW


@pytest.mark.parametrize(
    ("level", "expected"),
    [
        ("minimal", genai_types.ThinkingLevel.MINIMAL),
        ("low", genai_types.ThinkingLevel.LOW),
        ("medium", genai_types.ThinkingLevel.MEDIUM),
        ("high", genai_types.ThinkingLevel.HIGH),
    ],
)
async def test_every_configured_thinking_level_maps_to_the_sdk_enum(
    level: GeminiThinkingLevel, expected: genai_types.ThinkingLevel
) -> None:
    provider, models, _ = _provider(_response("success"), level=level)
    await _generate(provider)
    assert models.calls[0]["config"].thinking_config.thinking_level == expected


async def test_aclose_closes_the_sdk_client() -> None:
    provider, _, client = _provider()
    await provider.aclose()
    assert client.closed


# --- Bad output ---------------------------------------------------------------------------------


async def test_a_schema_valid_but_constraint_violating_answer_is_bad_output() -> None:
    # The citation label ``c12`` breaks a pattern Gemini never saw (it is not sent), so only the
    # Pydantic validation after the call can catch it (AGENTS.md §6.2).
    provider, _, _ = _provider(_response("invalid_output"))
    with pytest.raises(ProviderBadOutput) as caught:
        await _generate(provider)
    assert "c12" in caught.value.validation_error
    assert "citation_ids" in caught.value.validation_error
    assert caught.value.raw.startswith('{"status"')


async def test_a_truncated_answer_is_bad_output_and_names_the_finish_reason() -> None:
    response = _response("success")
    assert response.candidates
    response.candidates[0].finish_reason = genai_types.FinishReason.MAX_TOKENS
    parts = response.candidates[0].content.parts if response.candidates[0].content else []
    assert parts
    parts[0].text = '{"status": "answered", "answer_markdown": "Declare a par'
    provider, _, _ = _provider(response)
    with pytest.raises(ProviderBadOutput, match="MAX_TOKENS") as caught:
        await _generate(provider)
    assert caught.value.validation_error  # the retry quotes this back to the model


async def test_text_that_is_not_json_is_bad_output() -> None:
    response = genai_types.GenerateContentResponse(
        candidates=[
            genai_types.Candidate(
                content=genai_types.Content(parts=[genai_types.Part(text="Sorry, no JSON.")])
            )
        ]
    )
    provider, _, _ = _provider(response)
    with pytest.raises(ProviderBadOutput) as caught:
        await _generate(provider)
    assert caught.value.raw == "Sorry, no JSON."


async def test_thought_parts_are_not_part_of_the_answer() -> None:
    text = _response("success").candidates[0].content.parts[0].text  # type: ignore[index, union-attr]
    response = genai_types.GenerateContentResponse(
        candidates=[
            genai_types.Candidate(
                content=genai_types.Content(
                    parts=[
                        genai_types.Part(text="I should cite c1.", thought=True),
                        genai_types.Part(text=text),
                    ]
                )
            )
        ]
    )
    provider, _, _ = _provider(response)
    assert (await _generate(provider)).parsed.status == "answered"


async def test_a_blocked_prompt_with_no_candidate_is_bad_output() -> None:
    response = genai_types.GenerateContentResponse(
        prompt_feedback=genai_types.GenerateContentResponsePromptFeedback(
            block_reason=genai_types.BlockedReason.SAFETY
        )
    )
    provider, _, _ = _provider(response)
    with pytest.raises(ProviderBadOutput, match="SAFETY"):
        await _generate(provider)


# --- Error mapping ------------------------------------------------------------------------------


async def test_a_per_minute_429_is_rate_limited_with_the_server_delay() -> None:
    provider, _, _ = _provider(_api_error("429_per_minute"))
    with pytest.raises(ProviderRateLimited) as caught:
        await _generate(provider)
    assert (caught.value.retry_after_s, caught.value.is_quota) == (53.0, False)


async def test_a_daily_quota_429_is_flagged_as_quota() -> None:
    provider, _, _ = _provider(_api_error("429_daily_quota"))
    with pytest.raises(ProviderRateLimited) as caught:
        await _generate(provider)
    assert caught.value.is_quota is True


async def test_a_500_is_unavailable() -> None:
    provider, _, _ = _provider(_api_error("500"))
    with pytest.raises(ProviderUnavailable, match="500"):
        await _generate(provider)


async def test_a_bad_request_is_rejected_without_a_retry_hint() -> None:
    body = {"error": {"code": 400, "message": "bad key", "status": "INVALID_ARGUMENT"}}
    error = genai_errors.ClientError(400, body, httpx.Response(400, json=body))
    provider, _, _ = _provider(error)
    with pytest.raises(ProviderRequestRejected) as caught:
        await _generate(provider)
    assert caught.value.status_code == 400


async def test_a_call_that_hangs_times_out() -> None:
    provider, _, _ = _provider(_NeverReturns())
    with pytest.raises(ProviderTimeout):
        await _generate(provider, timeout_s=0.01)


async def test_an_httpx_timeout_is_a_provider_timeout() -> None:
    provider, _, _ = _provider(httpx.ReadTimeout("slow"))
    with pytest.raises(ProviderTimeout):
        await _generate(provider)


async def test_a_transport_failure_is_unavailable() -> None:
    provider, _, _ = _provider(httpx.ConnectError("reset"))
    with pytest.raises(ProviderUnavailable):
        await _generate(provider)


# --- Schema conversion --------------------------------------------------------------------------


def _keys(node: Any) -> set[str]:
    """Every dict key that is a schema keyword (property names are skipped)."""
    found: set[str] = set()
    if isinstance(node, dict):
        for key, value in cast(dict[str, Any], node).items():
            found.add(key)
            if key in ("properties", "$defs"):
                for sub in cast(dict[str, Any], value).values():
                    found |= _keys(sub)
            else:
                found |= _keys(value)
    elif isinstance(node, list):
        for item in cast(list[Any], node):
            found |= _keys(item)
    return found


def test_the_answer_schema_converts_to_the_documented_subset() -> None:
    schema = to_gemini_schema(LLMAnswer)
    json.dumps(schema)  # plain JSON
    assert _keys(schema).isdisjoint({"maxLength", "minLength", "pattern", "default"})
    # What Gemini documents stays, so the structure is still enforced natively.
    assert schema["required"] == ["status", "answer_markdown", "claims"]
    assert schema["properties"]["status"]["enum"] == ["answered", "partial", "insufficient_context"]
    assert schema["properties"]["claims"]["maxItems"] == 8
    assert schema["properties"]["follow_up_questions"]["maxItems"] == 3
    claim = schema["$defs"]["LLMClaim"]
    assert claim["properties"]["self_confidence"]["minimum"] == 0.0
    assert claim["properties"]["self_confidence"]["maximum"] == 1.0
    assert claim["properties"]["citation_ids"]["items"] == {"type": "string"}
    assert schema["properties"]["claims"]["items"] == {"$ref": "#/$defs/LLMClaim"}


def test_the_sdk_accepts_the_converted_schema() -> None:
    config = genai_types.GenerateContentConfig(
        response_mime_type="application/json", response_json_schema=to_gemini_schema(LLMAnswer)
    )
    assert config.response_json_schema is not None


def test_a_property_named_like_a_dropped_keyword_survives() -> None:
    class Odd(BaseModel):
        default: str
        pattern: int = 3

    schema = to_gemini_schema(Odd)
    assert set(schema["properties"]) == {"default", "pattern"}
    assert "default" not in schema["properties"]["pattern"]
