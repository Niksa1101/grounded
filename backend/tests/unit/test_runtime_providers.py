"""How ``runtime.py`` builds the generator providers for the router (7.04, Tech §4, §10, PRD D57).

No network and no database: the adapters are only constructed (an SDK client makes no call until it
is used). What is pinned here: the first entry of ``GENERATOR_PROVIDERS`` must be buildable and must
be ``gemini`` (or the dev stub), a later entry that cannot be built is skipped with a warning that
names the provider and the missing variable and never a value, a contradiction in the list stays
loud, and the Gemini client makes exactly one HTTP attempt.
"""

from __future__ import annotations

import logging
from contextlib import AsyncExitStack

import pytest
from google.genai import _api_client

from grounded.generation.providers.fake import StubLLMProvider
from grounded.generation.providers.gemini import GeminiProvider
from grounded.generation.providers.groq import GroqProvider
from grounded.runtime import (
    ProviderConfigError,
    build_fallback_providers,
    build_provider,
    gemini_http_options,
)
from grounded.settings import Settings
from tests.support import make_settings

GEMINI_MODEL = "gemini-3.5-flash-lite"
GROQ_MODEL = "openai/gpt-oss-120b"


def settings_for(*providers: str, **overrides: object) -> Settings:
    return make_settings(generator_providers=list(providers), **overrides)


# --- the Gemini client makes one attempt -----------------------------------------------


def test_the_gemini_client_makes_one_attempt() -> None:
    # google-genai 2.25.0 retries nothing by default, but Google's docs say its SDKs retry a 429 or
    # 5xx up to four times: a retry inside the SDK would sleep in the request path and hide the 429
    # from the router. The option is set explicitly, and the SDK's own retry arguments say "stop
    # after one attempt".
    provider = build_provider(
        settings_for("gemini", gemini_api_key="k-123", gemini_model=GEMINI_MODEL)
    )
    assert isinstance(provider, GeminiProvider)
    http_options = provider._client._api_client._http_options  # pyright: ignore[reportPrivateUsage]
    assert http_options.retry_options is not None
    assert http_options.retry_options.attempts == 1
    stop = _api_client.retry_args(http_options.retry_options)["stop"]
    assert stop.max_attempt_number == 1  # pyright: ignore[reportUnknownMemberType]


def test_the_pinned_options_are_the_ones_the_builder_uses() -> None:
    options = gemini_http_options()
    assert options.retry_options is not None
    assert options.retry_options.attempts == 1


# --- the first entry -----------------------------------------------


def test_groq_as_the_first_entry_is_refused_with_the_reason() -> None:
    settings = settings_for("groq", "gemini", groq_api_key="k", groq_model=GROQ_MODEL)
    with pytest.raises(ProviderConfigError) as caught:
        build_provider(settings)
    message = str(caught.value)
    assert "'groq' cannot be the first entry" in message
    assert "GROQ_REASONING_EFFORT" in message  # why: the cache key lacks it (PRD §12)
    assert "gemini,groq" in message  # what to set instead


def test_an_unbuildable_first_entry_is_not_skipped() -> None:
    with pytest.raises(ProviderConfigError, match="GEMINI_API_KEY") as caught:
        build_provider(settings_for("gemini", "groq", gemini_model=GEMINI_MODEL))
    # The class is the one the eval harness and the gate have always seen for a config error.
    assert type(caught.value) is ProviderConfigError
    assert caught.value.unconfigured


@pytest.mark.parametrize("names", [["gemini", "mistral"], ["mistral"], ["gemini", "gemini"]])
def test_an_unknown_or_repeated_name_is_a_loud_error(names: list[str]) -> None:
    settings = settings_for(*names, gemini_api_key="k", gemini_model=GEMINI_MODEL)
    with pytest.raises(ProviderConfigError, match=names[-1]):
        build_provider(settings)
    with pytest.raises(ProviderConfigError, match=names[-1]):
        # The later entries are checked too, before anything is built or skipped.
        build_fallback_providers(settings, AsyncExitStack())  # never entered: nothing is built


# --- later entries: built, or skipped with a warning ----------------------------------


async def test_a_later_entry_that_cannot_be_built_is_skipped_with_a_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    settings = settings_for("gemini", "groq", groq_api_key="groq-secret-key", groq_model=None)
    async with AsyncExitStack() as stack:
        built = build_fallback_providers(settings, stack)

    assert built == []
    [record] = [r for r in caplog.records if r.name == "grounded.runtime"]
    assert record.levelno == logging.WARNING
    assert record.__dict__["provider"] == "groq"
    assert "GROQ_MODEL" in record.__dict__["reason"]
    assert "groq-secret-key" not in caplog.text  # the variable's name, never a value


async def test_a_missing_groq_key_is_named(caplog: pytest.LogCaptureFixture) -> None:
    settings = settings_for("gemini", "groq", groq_model=GROQ_MODEL)
    async with AsyncExitStack() as stack:
        build_fallback_providers(settings, stack)

    [record] = [r for r in caplog.records if r.name == "grounded.runtime"]
    assert "GROQ_API_KEY" in record.__dict__["reason"]


async def test_a_configured_groq_entry_is_built_and_closed_with_the_stack() -> None:
    settings = settings_for("gemini", "groq", groq_api_key="k-123", groq_model=GROQ_MODEL)
    async with AsyncExitStack() as stack:
        [groq] = build_fallback_providers(settings, stack)
        assert isinstance(groq, GroqProvider)
        assert (groq.name, groq.model) == ("groq", GROQ_MODEL)
        client = groq._client  # pyright: ignore[reportPrivateUsage]
        assert not client.is_closed()
    assert client.is_closed()  # the stack's exit awaited ``aclose``


async def test_a_single_entry_list_has_no_fallback() -> None:
    settings = settings_for("gemini", gemini_api_key="k", gemini_model=GEMINI_MODEL)
    async with AsyncExitStack() as stack:
        assert build_fallback_providers(settings, stack) == []


async def test_the_stub_can_follow_the_primary_outside_prod_and_not_in_prod() -> None:
    async with AsyncExitStack() as stack:
        [stub] = build_fallback_providers(settings_for("gemini", "fake"), stack)
        assert isinstance(stub, StubLLMProvider)

    prod = settings_for(
        "gemini",
        "fake",
        app_env="prod",
        database_url="postgresql://app@db/grounded",
        proxy_shared_secret="x" * 32,
        ip_hash_secret="y" * 32,
        daily_llm_budget=300,
    )
    async with AsyncExitStack() as stack:
        with pytest.raises(ProviderConfigError, match="APP_ENV=prod"):
            build_fallback_providers(prod, stack)
