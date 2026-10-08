"""What is specific to ``GroqProvider``: the strict schema conversion, the request it sends, the
reasoning and usage rules, Groq's ``json_validate_failed`` 400, and the 429 / 4xx / 5xx mapping.

The behavior it shares with ``GeminiProvider`` is in ``test_provider_contract.py``.
``generate_success.json``, ``generate_invalid_output.json`` and
``generate_400_schema_violation.json`` are REAL recordings (2026-10-08); the 429 and 500 fixtures
are hand-made (``tests/fixtures/groq/README.md``).
"""

from __future__ import annotations

import json
from typing import Any, Literal, cast

import httpx
import pytest
from groq import AsyncGroq
from pydantic import BaseModel, Field

from grounded.generation.providers.groq import GroqProvider, to_groq_schema
from grounded.infra.provider_errors import (
    TRUNCATED_FEEDBACK,
    ProviderBadOutput,
    ProviderRateLimited,
    ProviderRequestRejected,
    ProviderUnavailable,
)
from grounded.runtime import ProviderConfigError, build_groq_provider
from grounded.schemas.llm import LLMAnswer
from grounded.settings import GroqReasoningEffort
from tests.provider_rigs import (
    GROQ_MODEL,
    GroqRig,
    groq_error_response,
    groq_reply_response,
    groq_response,
    groq_success_body,
    load_fixture,
)
from tests.support import make_settings

CONSTRAINT_KEYWORDS = {
    "minLength",
    "maxLength",
    "minItems",
    "maxItems",
    "minimum",
    "maximum",
    "pattern",
}


class Verdict(BaseModel):
    """The shape of the future judge verdict (4.04), with a constraint to be dropped."""

    verdict: Literal["supported", "partially_supported", "unsupported"]
    reason: str = Field(max_length=40)


async def generate(rig: GroqRig, schema: type[BaseModel] = LLMAnswer) -> Any:
    return await rig.provider.generate(
        system="SYSTEM",
        user="USER",
        schema=schema,
        temperature=0.0,
        max_output_tokens=800,
        timeout_s=12.0,
    )


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


def test_the_answer_schema_converts_to_the_strict_shape() -> None:
    schema = to_groq_schema(LLMAnswer)
    json.dumps(schema)  # plain JSON
    # Strict mode: every property is required (the defaulted follow_up_questions included) and every
    # object is closed.
    assert schema["required"] == ["status", "answer_markdown", "claims", "follow_up_questions"]
    assert schema["additionalProperties"] is False
    claim = schema["$defs"]["LLMClaim"]
    assert claim["required"] == ["text", "citation_ids", "self_confidence"]
    assert claim["additionalProperties"] is False
    # What strict mode documents stays: types, enums, items and the $defs / $ref structure.
    assert schema["properties"]["status"]["enum"] == ["answered", "partial", "insufficient_context"]
    assert schema["properties"]["claims"] == {
        "type": "array",
        "items": {"$ref": "#/$defs/LLMClaim"},
    }
    assert claim["properties"]["citation_ids"] == {"type": "array", "items": {"type": "string"}}
    assert claim["properties"]["self_confidence"] == {"type": "number"}


def test_constraints_titles_and_defaults_are_left_to_pydantic() -> None:
    # Groq checks these on the finished JSON and answers a violation with a usage-less 400 (see the
    # recording), so they are not sent; Pydantic enforces them after the call.
    assert _keys(to_groq_schema(LLMAnswer)).isdisjoint(CONSTRAINT_KEYWORDS | {"title", "default"})
    assert _keys(to_groq_schema(Verdict)).isdisjoint(CONSTRAINT_KEYWORDS | {"title", "default"})


def test_the_verdict_shape_converts_to_the_strict_shape() -> None:
    assert to_groq_schema(Verdict) == {
        "type": "object",
        "properties": {
            "verdict": {
                "enum": ["supported", "partially_supported", "unsupported"],
                "type": "string",
            },
            "reason": {"type": "string"},
        },
        "required": ["verdict", "reason"],
        "additionalProperties": False,
    }


def test_an_optional_field_stays_a_union_with_null_and_becomes_required() -> None:
    class WithNote(BaseModel):
        note: str | None = None

    schema = to_groq_schema(WithNote)
    assert schema["required"] == ["note"]
    assert schema["properties"]["note"]["anyOf"] == [{"type": "string"}, {"type": "null"}]


def test_a_property_named_like_a_dropped_keyword_survives() -> None:
    class Odd(BaseModel):
        default: str
        pattern: int = 3

    schema = to_groq_schema(Odd)
    assert set(schema["properties"]) == {"default", "pattern"}
    assert schema["required"] == ["default", "pattern"]
    assert "default" not in schema["properties"]["pattern"]


# --- The request --------------------------------------------------------------------------------


async def test_the_request_is_strict_structured_output_with_small_reasoning_kept_out() -> None:
    rig = GroqRig(groq_response("success"), reasoning_effort="medium")
    await generate(rig)
    [request] = rig.http_requests
    [body] = rig.request_bodies
    assert request.url.path.endswith("/chat/completions")
    assert body["model"] == GROQ_MODEL
    assert body["messages"] == [
        {"role": "system", "content": "SYSTEM"},
        {"role": "user", "content": "USER"},
    ]
    assert (body["temperature"], body["max_completion_tokens"]) == (0.0, 800)
    assert body["reasoning_effort"] == "medium"
    assert body["include_reasoning"] is False
    assert body["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "LLMAnswer", "strict": True, "schema": to_groq_schema(LLMAnswer)},
    }
    assert "stream" not in body  # structured output cannot be streamed


@pytest.mark.parametrize("effort", ["low", "medium", "high"])
async def test_every_configured_reasoning_effort_is_sent(effort: GroqReasoningEffort) -> None:
    rig = GroqRig("success", reasoning_effort=effort)
    await generate(rig)
    assert rig.request_bodies[0]["reasoning_effort"] == effort


async def test_the_format_name_is_made_valid_for_a_generic_model() -> None:
    class Page[T](BaseModel):
        item: T

    rig = GroqRig(groq_reply_response('{"item": 1}'))
    await generate(rig, Page[int])
    name = rig.request_bodies[0]["response_format"]["json_schema"]["name"]
    assert name == "Page_int_"  # a-zA-Z0-9_- only, 64 characters at most


# --- Reasoning and usage ------------------------------------------------------------------------


async def test_the_reasoning_is_not_part_of_the_answer() -> None:
    body = groq_success_body()
    body["choices"][0]["message"]["reasoning"] = "I should cite c1."
    rig = GroqRig(httpx.Response(200, json=body))
    result = await generate(rig)
    assert result.parsed.status == "answered"
    assert "I should cite" not in result.raw_text


async def test_reasoning_tokens_are_inside_the_completion_tokens_not_added_to_them() -> None:
    # The recording: 323 completion tokens, 27 of them reasoning (a real reply of 34 content tokens
    # reported 114 with 68 reasoning). Adding 27 would bill them twice.
    usage = (await generate(GroqRig("success"))).usage
    assert (usage.output_tokens, usage.thinking_tokens) == (323, 27)


async def test_a_reply_without_the_reasoning_breakdown_counts_no_thinking() -> None:
    body = groq_success_body()
    del body["usage"]["completion_tokens_details"]
    usage = (await generate(GroqRig(httpx.Response(200, json=body)))).usage
    assert (usage.input_tokens, usage.output_tokens, usage.thinking_tokens) == (376, 323, 0)


async def test_a_200_cut_off_at_the_length_limit_is_bad_output_with_the_truncation_feedback() -> (
    None
):
    # Not seen from Groq (the cut arrives as the 400 below), but the finish reason is documented.
    body = groq_success_body()
    body["choices"][0]["message"]["content"] = '{"status": "answered", "answer_markdown": "Declare'
    body["choices"][0]["finish_reason"] = "length"
    with pytest.raises(ProviderBadOutput, match="finish_reason=length") as caught:
        await generate(GroqRig(httpx.Response(200, json=body)))
    assert caught.value.validation_error == TRUNCATED_FEEDBACK
    assert (caught.value.input_tokens, caught.value.output_tokens) == (376, 323)  # it was billed


async def test_a_reply_with_no_choice_is_bad_output() -> None:
    body = groq_success_body()
    body["choices"] = []
    with pytest.raises(ProviderBadOutput) as caught:
        await generate(GroqRig(httpx.Response(200, json=body)))
    assert caught.value.raw == ""


# --- json_validate_failed -----------------------------------------------------------------------


async def test_groqs_own_schema_check_failing_is_bad_output_not_a_rejected_request() -> None:
    # Recording: Groq validated the finished JSON against maxLength 40 and answered 400.
    fixture = load_fixture("groq", "400_schema_violation")
    with pytest.raises(ProviderBadOutput) as caught:
        await generate(GroqRig("400_schema_violation"), Verdict)
    error = caught.value
    failed = fixture["body"]["error"]["failed_generation"]
    assert error.raw == failed
    assert error.validation_error == "reason: String should have at most 40 characters"
    assert error.retryable
    assert (error.input_tokens, error.output_tokens) == (0, 0)  # the 400 carries no usage
    assert "got 237, want 40" in str(error)  # the server's reason is in the message


async def test_a_failed_generation_that_pydantic_accepts_quotes_the_servers_reason() -> None:
    # Groq is stricter than the model: nothing of ours to quote, so the feedback is its reason.
    good = LLMAnswer(status="insufficient_context", answer_markdown="No.", claims=[])
    response = groq_error_response(
        400,
        message="Generated JSON does not match the expected schema.",
        type="invalid_request_error",
        code="json_validate_failed",
        failed_generation=good.model_dump_json(),
    )
    with pytest.raises(ProviderBadOutput) as caught:
        await generate(GroqRig(response))
    assert caught.value.validation_error == str(caught.value)
    assert "does not match the expected schema" in caught.value.validation_error


# --- Error mapping ------------------------------------------------------------------------------


def _rate_limit_response(*, retry_after: str | None, message: str) -> httpx.Response:
    headers = {"retry-after": retry_after} if retry_after is not None else {}
    return httpx.Response(
        429,
        json={"error": {"message": message, "type": "tokens", "code": "rate_limit_exceeded"}},
        headers=headers,
    )


async def test_a_429_keeps_the_servers_wait_and_the_limit_it_names() -> None:
    with pytest.raises(ProviderRateLimited, match=r"\(TPM\)") as caught:
        await generate(GroqRig("429_per_minute"))
    assert (caught.value.retry_after_s, caught.value.is_quota) == (6.0, False)


@pytest.mark.parametrize(
    ("retry_after", "message", "is_quota"),
    [
        # The message names a per-day limit, whatever the wait.
        ("30", "Limit reached on tokens per day (TPD): Limit 200000.", True),
        ("30", "Limit reached on requests per day (RPD): Limit 1000.", True),
        # No marker, but the wait is longer than the 60 s window of any per-minute limit.
        ("3600", "Rate limit reached.", True),
        ("60", "Rate limit reached on requests per minute (RPM).", False),
        ("2", "Rate limit reached on tokens per minute (TPM).", False),
        # No wait given: nothing says it is a quota.
        (None, "Rate limit reached on tokens per minute (TPM).", False),
    ],
)
async def test_the_daily_quota_guess(retry_after: str | None, message: str, is_quota: bool) -> None:
    response = _rate_limit_response(retry_after=retry_after, message=message)
    with pytest.raises(ProviderRateLimited) as caught:
        await generate(GroqRig(response))
    assert caught.value.is_quota is is_quota
    assert caught.value.retry_after_s == (float(retry_after) if retry_after else None)


async def test_a_retry_after_that_is_not_seconds_is_no_wait() -> None:
    response = _rate_limit_response(retry_after="Wed, 21 Oct 2026 07:28:00 GMT", message="Limit.")
    with pytest.raises(ProviderRateLimited) as caught:
        await generate(GroqRig(response))
    assert caught.value.retry_after_s is None


@pytest.mark.parametrize("status_code", [400, 401, 403, 404, 413, 422])
async def test_any_other_4xx_is_rejected_with_its_status(status_code: int) -> None:
    response = groq_error_response(status_code, message="No.", type="invalid_request_error")
    with pytest.raises(ProviderRequestRejected, match=r"No\.") as caught:
        await generate(GroqRig(response))
    assert caught.value.status_code == status_code


@pytest.mark.parametrize("status_code", [500, 502, 503, 529])
async def test_every_5xx_is_unavailable(status_code: int) -> None:
    response = groq_error_response(status_code, message="Down.", type="internal_server_error")
    with pytest.raises(ProviderUnavailable, match=str(status_code)):
        await generate(GroqRig(response))


async def test_an_error_body_that_is_not_json_is_still_mapped_by_its_status() -> None:
    response = httpx.Response(401, text="<html>unauthorized</html>")
    with pytest.raises(ProviderRequestRejected) as caught:
        await generate(GroqRig(response))
    assert caught.value.status_code == 401


# --- The client and its builder -----------------------------------------------------------------


def test_a_client_that_retries_is_refused() -> None:
    # The SDK's default is two retries that sleep (and honor retry-after up to 60 s).
    with pytest.raises(ValueError, match="max_retries=0"):
        GroqProvider(AsyncGroq(api_key="k"), model=GROQ_MODEL, reasoning_effort="low")


def test_create_builds_a_client_without_retries() -> None:
    provider = GroqProvider.create("k", model=GROQ_MODEL, reasoning_effort="high")
    assert provider.model == GROQ_MODEL
    assert provider.name == "groq"
    assert provider._client.max_retries == 0  # pyright: ignore[reportPrivateUsage]


def test_the_builder_makes_a_provider_from_settings() -> None:
    settings = make_settings(groq_api_key="k-123", groq_reasoning_effort="medium")
    provider = build_groq_provider(settings, model="openai/gpt-oss-120b")
    assert isinstance(provider, GroqProvider)
    assert (provider.name, provider.model) == ("groq", "openai/gpt-oss-120b")
    assert provider._reasoning_effort == "medium"  # pyright: ignore[reportPrivateUsage]


@pytest.mark.parametrize(
    ("key", "model", "missing"),
    [
        (None, "openai/gpt-oss-120b", "GROQ_API_KEY"),
        ("  ", "openai/gpt-oss-120b", "GROQ_API_KEY"),
        ("k-123", None, "pinned model ID"),
        ("k-123", " ", "pinned model ID"),
    ],
)
def test_the_builder_without_a_key_or_a_pinned_model_is_a_clear_error(
    key: str | None, model: str | None, missing: str
) -> None:
    with pytest.raises(ProviderConfigError, match=missing):
        build_groq_provider(make_settings(groq_api_key=key), model=model)
