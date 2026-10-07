from __future__ import annotations

import pytest
from pydantic import ValidationError

from grounded.generation.providers.base import LLMProvider, Usage
from grounded.generation.providers.fake import FakeLLMProvider
from grounded.infra.provider_errors import (
    ProviderBadOutput,
    ProviderRateLimited,
    ProviderTimeout,
    compact_validation_error,
)
from grounded.schemas.llm import LLMAnswer

GOOD = LLMAnswer(status="insufficient_context", answer_markdown="Not covered.", claims=[])
OTHER = LLMAnswer(status="insufficient_context", answer_markdown="Still not.", claims=[])


async def _generate(provider: LLMProvider, user: str = "q") -> LLMAnswer:
    result = await provider.generate(
        system="sys",
        user=user,
        schema=LLMAnswer,
        temperature=0.0,
        max_output_tokens=256,
        timeout_s=5.0,
    )
    return result.parsed


async def test_returns_scripted_results_in_order() -> None:
    fake = FakeLLMProvider([GOOD, OTHER])
    assert (await _generate(fake)).answer_markdown == "Not covered."
    assert (await _generate(fake)).answer_markdown == "Still not."
    assert fake.remaining == 0


async def test_result_carries_provider_identity_and_raw_text() -> None:
    usage = Usage(input_tokens=7, output_tokens=3, thinking_tokens=1)
    fake = FakeLLMProvider([GOOD], name="gem", model="gem-1", usage=usage)
    result = await fake.generate(
        system="s", user="u", schema=LLMAnswer, temperature=0.0, max_output_tokens=1, timeout_s=1
    )
    assert (result.provider, result.model, result.usage) == ("gem", "gem-1", usage)
    assert LLMAnswer.model_validate_json(result.raw_text) == GOOD


async def test_raw_json_text_is_validated_into_the_schema() -> None:
    fake = FakeLLMProvider([GOOD.model_dump_json()])
    assert await _generate(fake) == GOOD


async def test_scripted_error_is_raised_then_the_script_continues() -> None:
    limited = ProviderRateLimited("slow down", retry_after_s=2.0, is_quota=False)
    fake = FakeLLMProvider([limited, ProviderTimeout("late"), GOOD])
    with pytest.raises(ProviderRateLimited) as info:
        await _generate(fake)
    assert info.value is limited
    with pytest.raises(ProviderTimeout):
        await _generate(fake)
    assert await _generate(fake) == GOOD


@pytest.mark.parametrize(
    "raw",
    [
        "not json at all",
        "{}",
        '{"status": "answered", "answer_markdown": "x", "claims": [',
        '{"status": "maybe", "answer_markdown": "x", "claims": []}',
        (
            '{"status": "answered", "answer_markdown": "x", "claims":'
            ' [{"text": "t", "citation_ids": ["source-1"], "self_confidence": 0.5}]}'
        ),
    ],
)
async def test_text_that_fails_validation_is_bad_output(raw: str) -> None:
    fake = FakeLLMProvider([raw])
    with pytest.raises(ProviderBadOutput) as info:
        await _generate(fake)
    assert info.value.raw == raw
    # The text the retry quotes back to the model: compact, like the real adapter's.
    assert info.value.validation_error
    assert "input_value" not in info.value.validation_error
    assert "errors.pydantic.dev" not in info.value.validation_error
    # A bad reply was billed like a good one (the fake's default usage).
    assert (info.value.input_tokens, info.value.output_tokens) == (100, 50)


def test_compact_validation_error_names_each_field_and_rule_without_the_input() -> None:
    raw = (
        '{"status": "maybe", "answer_markdown": "SECRET-INPUT", "claims":'
        ' [{"text": "t", "citation_ids": ["c12"], "self_confidence": 2}]}'
    )
    with pytest.raises(ValidationError) as info:
        LLMAnswer.model_validate_json(raw)
    assert compact_validation_error(info.value) == (
        "status: Input should be 'answered', 'partial' or 'insufficient_context'; "
        "claims.0.citation_ids.0: String should match pattern '^c[1-9]$'; "
        "claims.0.self_confidence: Input should be less than or equal to 1"
    )


def test_compact_validation_error_reports_broken_json_against_the_output() -> None:
    with pytest.raises(ValidationError) as info:
        LLMAnswer.model_validate_json('{"status": "answ')
    assert compact_validation_error(info.value).startswith("output: Invalid JSON: ")


async def test_bad_output_then_good_output_models_the_retry() -> None:
    fake = FakeLLMProvider(["oops", GOOD])
    with pytest.raises(ProviderBadOutput):
        await _generate(fake, user="first")
    assert await _generate(fake, user="first\n\nYour previous output was invalid") == GOOD


async def test_calls_are_recorded_including_failed_ones() -> None:
    fake = FakeLLMProvider([ProviderTimeout("late"), GOOD])
    with pytest.raises(ProviderTimeout):
        await _generate(fake, user="one")
    await _generate(fake, user="two")

    assert [c.user for c in fake.calls] == ["one", "two"]
    first = fake.calls[0]
    assert (first.system, first.schema, first.temperature) == ("sys", LLMAnswer, 0.0)
    assert (first.max_output_tokens, first.timeout_s) == (256, 5.0)


async def test_exhausted_script_is_a_test_bug_not_a_provider_error() -> None:
    fake = FakeLLMProvider([GOOD])
    await _generate(fake)
    with pytest.raises(AssertionError, match="exhausted"):
        await _generate(fake)
