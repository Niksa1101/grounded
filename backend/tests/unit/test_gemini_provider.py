"""What is specific to ``GeminiProvider``: thinking tokens and levels, finish reasons, the content
filter, and the schema conversion.

The behavior it shares with ``GroqProvider`` (success, usage, bad output, error mapping, timeouts,
the retry) is in ``test_provider_contract.py``. ``generate_success.json`` and
``generate_invalid_output.json`` are REAL recordings (2026-10-07,
``tests/fixtures/gemini/record_generate_fixture.py``). The three error fixtures (429 per minute, 429
daily quota, 500) are still SYNTHETIC (``"_synthetic": true``): those errors can't be produced on
demand without hammering the quota. Their shapes come from the SDK's own error types.
"""

from __future__ import annotations

import json
from typing import Any, cast

import pytest
from google.genai import types as genai_types
from pydantic import BaseModel

from grounded.generation.providers.base import LLMProvider
from grounded.generation.providers.gemini import to_gemini_schema
from grounded.infra.provider_errors import TRUNCATED_FEEDBACK, ProviderBadOutput
from grounded.schemas.llm import LLMAnswer
from grounded.settings import GeminiThinkingLevel
from tests.provider_rigs import (
    GEMINI_MODEL,
    FakeGeminiClient,
    FakeGeminiModels,
    GeminiRig,
    gemini_response,
)


def _provider(
    *steps: genai_types.GenerateContentResponse, level: GeminiThinkingLevel = "minimal"
) -> tuple[LLMProvider, FakeGeminiModels, FakeGeminiClient]:
    rig = GeminiRig(*steps, thinking_level=level)
    return rig.provider, rig.models, rig.client


async def _generate(provider: LLMProvider) -> Any:
    return await provider.generate(
        system="SYSTEM",
        user="USER",
        schema=LLMAnswer,
        temperature=0.0,
        max_output_tokens=800,
        timeout_s=12.0,
    )


# --- Usage, request -----------------------------------------------------------------------------


async def test_usage_counts_thinking_tokens_as_output() -> None:
    # Recording: 91 prompt, 143 candidates. At level ``minimal`` Gemini reported no thoughts, so the
    # 40 here is set on the response to exercise the rule: thoughts are kept out of the candidates
    # count but billed as output, so output_tokens is their sum.
    response = gemini_response("success")
    assert response.usage_metadata is not None
    response.usage_metadata.thoughts_token_count = 40
    provider, _, _ = _provider(response)
    usage = (await _generate(provider)).usage
    assert (usage.input_tokens, usage.output_tokens, usage.thinking_tokens) == (91, 183, 40)


async def test_missing_thoughts_count_means_no_thinking() -> None:
    # The recording is exactly this case: at level ``minimal`` the usage has no thoughts count.
    response = gemini_response("success")
    assert response.usage_metadata is not None
    assert response.usage_metadata.thoughts_token_count is None
    provider, _, _ = _provider(response)
    usage = (await _generate(provider)).usage
    assert (usage.output_tokens, usage.thinking_tokens) == (143, 0)


async def test_the_request_carries_the_settings_and_the_schema() -> None:
    provider, models, _ = _provider(gemini_response("success"), level="low")
    await _generate(provider)
    [call] = models.calls
    config = call["config"]
    assert (call["model"], call["contents"]) == (GEMINI_MODEL, "USER")
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
    provider, models, _ = _provider(gemini_response("success"), level=level)
    await _generate(provider)
    assert models.calls[0]["config"].thinking_config.thinking_level == expected


# --- Bad output ---------------------------------------------------------------------------------


async def test_a_truncated_answer_is_bad_output_and_names_the_finish_reason() -> None:
    response = gemini_response("success")
    assert response.candidates
    response.candidates[0].finish_reason = genai_types.FinishReason.MAX_TOKENS
    parts = response.candidates[0].content.parts if response.candidates[0].content else []
    assert parts
    parts[0].text = '{"status": "answered", "answer_markdown": "Declare a par'
    provider, _, _ = _provider(response)
    with pytest.raises(ProviderBadOutput, match="MAX_TOKENS") as caught:
        await _generate(provider)
    assert caught.value.validation_error == TRUNCATED_FEEDBACK  # quoted back by the retry
    assert caught.value.retryable  # a shorter answer can fit


async def test_the_recorded_cut_off_reply_names_the_finish_reason() -> None:
    provider, _, _ = _provider(gemini_response("invalid_output"))
    with pytest.raises(ProviderBadOutput, match="MAX_TOKENS"):
        await _generate(provider)


async def test_thought_parts_are_not_part_of_the_answer() -> None:
    text = gemini_response("success").candidates[0].content.parts[0].text  # type: ignore[index, union-attr]
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
    response.usage_metadata = genai_types.GenerateContentResponseUsageMetadata(
        prompt_token_count=91
    )
    provider, _, _ = _provider(response)
    with pytest.raises(ProviderBadOutput, match="SAFETY") as caught:
        await _generate(provider)
    assert (caught.value.input_tokens, caught.value.output_tokens) == (91, 0)
    # The same prompt would be blocked again: a retry only spends a request of the daily quota.
    assert not caught.value.retryable


@pytest.mark.parametrize("reason", ["SAFETY", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII"])
async def test_an_answer_stopped_by_a_content_filter_is_bad_output_without_a_retry(
    reason: str,
) -> None:
    response = gemini_response("success")
    assert response.candidates
    response.candidates[0].finish_reason = genai_types.FinishReason[reason]
    response.candidates[0].content = genai_types.Content(parts=[])
    provider, _, _ = _provider(response)
    with pytest.raises(ProviderBadOutput, match=reason) as caught:
        await _generate(provider)
    assert not caught.value.retryable


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
