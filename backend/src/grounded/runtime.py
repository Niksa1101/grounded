"""Long-lived resources of the request path, built once and shared by the API and the CLI.

``open_runtime`` creates the connection pool, the query embedder, the generator provider and the
``AskPipeline`` over them, and closes what it opened. The API lifespan and ``grounded ask`` both go
through it, so they cannot drift apart. Tests pass their own ``embedder`` and ``provider``
(AGENTS.md §8: no real calls).
"""

from __future__ import annotations

import time
from collections.abc import AsyncGenerator
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass

from google import genai
from psycopg_pool import AsyncConnectionPool

from grounded.evals.judge import Judge, JudgeConfig
from grounded.generation.pipeline import AskPipeline
from grounded.generation.prompts import load_answer_prompt, load_no_rag_prompt
from grounded.generation.providers.base import LLMProvider
from grounded.generation.providers.eval_wrappers import EvalLLM
from grounded.generation.providers.fake import StubJudgeProvider, StubLLMProvider
from grounded.generation.providers.gemini import GeminiProvider
from grounded.generation.providers.groq import GroqProvider
from grounded.infra.db import create_pool
from grounded.infra.timing import Clock
from grounded.ingest.embed import Embedder
from grounded.observability.cost import Pricing, load_pricing
from grounded.observability.request_log import RequestLogger
from grounded.retrieval.index import ActiveIndexCache
from grounded.retrieval.query_embedding import build_query_embedder
from grounded.settings import Settings


class ProviderConfigError(Exception):
    """A provider this build cannot create or must not use: an unknown ``GENERATOR_PROVIDERS``
    entry, a missing key or model ID, a judge on the generator's provider."""


@dataclass(frozen=True, slots=True)
class Runtime:
    pool: AsyncConnectionPool
    pipeline: AskPipeline
    request_logger: RequestLogger
    # Eval mode only (Tech §15.6), and only for a provider built from the settings: the cache and
    # backoff around it, whose counters a run prints at its end. ``wrap`` takes the judge as well.
    eval_llm: EvalLLM | None = None


def build_provider(settings: Settings) -> LLMProvider:
    """The generator: the first entry of ``GENERATOR_PROVIDERS`` (no router before Phase 7).

    ``gemini`` is the real adapter; ``fake`` is a dev-only stand-in (refused in prod).
    """
    name = settings.generator_providers[0]
    if name == "gemini":
        return _build_gemini(settings)
    if name == "fake":
        if settings.app_env == "prod":
            raise ProviderConfigError("the fake provider must not be used with APP_ENV=prod")
        return StubLLMProvider()
    raise ProviderConfigError(
        f"provider {name!r} cannot be the generator yet (the router arrives in Phase 7); "
        "set GENERATOR_PROVIDERS=gemini (or fake for local runs)"
    )


def _build_gemini(settings: Settings) -> GeminiProvider:
    # A blank value counts as unset (an unfilled deploy secret arrives as ""), as in the embedder.
    key = settings.gemini_api_key.get_secret_value().strip() if settings.gemini_api_key else ""
    model = (settings.gemini_model or "").strip()
    if not key:
        raise ProviderConfigError("GEMINI_API_KEY must be set to use the gemini provider")
    if not model:
        raise ProviderConfigError(
            "GEMINI_MODEL must be set to a pinned model ID (PRD D46: gemini-3.5-flash-lite)"
        )
    return GeminiProvider(
        genai.Client(api_key=key), model=model, thinking_level=settings.gemini_thinking_level
    )


def build_groq_provider(settings: Settings, *, model: str | None) -> GroqProvider:
    """A Groq adapter for ``model``: ``GROQ_MODEL`` for the fallback generator (Phase 7),
    ``JUDGE_MODEL`` for the judge (4.04). ``GENERATOR_PROVIDERS`` cannot select it before the router
    exists. The caller owns the provider and closes it (``aclose``)."""
    key = settings.groq_api_key.get_secret_value().strip() if settings.groq_api_key else ""
    pinned = (model or "").strip()
    if not key:
        raise ProviderConfigError("GROQ_API_KEY must be set to use the groq provider")
    if not pinned:
        raise ProviderConfigError(
            "the Groq model must be set to a pinned model ID (GROQ_MODEL or JUDGE_MODEL; "
            "PRD D48: openai/gpt-oss-120b)"
        )
    return GroqProvider.create(key, model=pinned, reasoning_effort=settings.groq_reasoning_effort)


def build_judge_provider(settings: Settings) -> GroqProvider | StubJudgeProvider:
    """The judge's adapter: Groq with ``JUDGE_MODEL`` (PRD D48). A blank model is an error here,
    with the name of the variable, not a silently shared ``GROQ_MODEL``.

    ``GENERATOR_PROVIDERS=fake`` (the existing no-network switch, refused in prod like
    ``build_provider``) gives the canned stub judge instead, with no key or model needed."""
    if settings.generator_providers[0] == "fake":
        if settings.app_env == "prod":
            raise ProviderConfigError("the fake provider must not be used with APP_ENV=prod")
        return StubJudgeProvider()
    if not (settings.judge_model or "").strip():
        raise ProviderConfigError(
            "JUDGE_MODEL must be set to a pinned model ID (PRD D48: openai/gpt-oss-120b)"
        )
    return build_groq_provider(settings, model=settings.judge_model)


def check_judge_provider(judge: LLMProvider, generator: str) -> None:
    """The judge must not grade its own provider's answers (AGENTS.md §6.5, PRD FR-22).

    ``generator`` is the provider that wrote the answers, ``GENERATOR_PROVIDERS[0]``: eval mode
    allows no other (fallback is off). The comparison is by provider name; ``Settings`` refuses a
    ``groq`` generator in eval mode too, but this also holds for the other environments and for a
    provider that is handed in.
    """
    if judge.name == generator:
        raise ProviderConfigError(
            f"the judge ({judge.name}) is the same provider as the generator ({generator}): "
            "the judge must run on a different provider (AGENTS.md §6.5); "
            "set GENERATOR_PROVIDERS=gemini"
        )


@asynccontextmanager
async def open_judge(
    settings: Settings,
    *,
    eval_llm: EvalLLM | None = None,
    provider: LLMProvider | None = None,
) -> AsyncGenerator[Judge]:
    """The judge of an eval run (Tech §15.4): the Groq adapter behind the eval LLM cache and the
    429 backoff, on the committed rubrics.

    ``eval_llm`` is the kit the generator already uses (``Runtime.eval_llm``), so both share one
    cache file and one set of counters; without it a kit of its own is opened over the same file.
    ``provider`` is the inner adapter for tests, which never build a real one (AGENTS.md §8); it is
    wrapped like the real one. Raises ``ProviderConfigError`` for a missing ``JUDGE_MODEL`` or
    ``GROQ_API_KEY`` and for a judge on the generator's provider, before any call. The stub judge of
    a ``fake`` run is the one provider that is not wrapped: its canned verdicts must never be
    replayed from the cache file, nor written to it.
    """
    config = JudgeConfig.from_settings(settings)
    async with AsyncExitStack() as stack:
        if provider is None:
            provider = build_judge_provider(settings)
            if isinstance(provider, GroqProvider):
                stack.push_async_callback(provider.aclose)
        check_judge_provider(provider, settings.generator_providers[0])
        if isinstance(provider, StubJudgeProvider):
            yield Judge.create(provider, config)
            return
        kit = eval_llm or EvalLLM.open(settings, stack)
        yield Judge.create(kit.wrap(provider), config)


@asynccontextmanager
async def open_runtime(
    settings: Settings,
    *,
    embedder: Embedder | None = None,
    provider: LLMProvider | None = None,
    pricing: Pricing | None = None,
    clock: Clock = time.perf_counter,
) -> AsyncGenerator[Runtime]:
    async with AsyncExitStack() as stack:
        pool = create_pool(settings)
        # wait=False: start serving even if the database is unreachable; /readyz reports it.
        await pool.open(wait=False)
        stack.push_async_callback(pool.close)
        eval_llm: EvalLLM | None = None
        if provider is None:
            provider = build_provider(settings)
            if isinstance(provider, GeminiProvider):
                stack.push_async_callback(provider.aclose)
            # The stub is no quota-limited API, and its canned answers must never be replayed from a
            # file after the stub changes, so only a real adapter gets the eval wrappers.
            if settings.app_env == "eval" and not isinstance(provider, StubLLMProvider):
                eval_llm = EvalLLM.open(settings, stack)
                provider = eval_llm.wrap(provider)
        pricing = pricing or load_pricing()
        pipeline = AskPipeline(
            settings=settings,
            pool=pool,
            index_cache=ActiveIndexCache(pool, ttl_s=settings.active_index_ttl_s),
            embedder=embedder or build_query_embedder(settings, stack),
            provider=provider,
            prompt=load_answer_prompt(),
            no_rag_prompt=load_no_rag_prompt(),
            pricing=pricing,
            clock=clock,
        )
        # Eval runs are told apart from real traffic in the table (DB.md §4, CI database only).
        request_logger = RequestLogger(
            pool, pricing, clock=clock, source="eval" if settings.app_env == "eval" else "api"
        )
        yield Runtime(
            pool=pool, pipeline=pipeline, request_logger=request_logger, eval_llm=eval_llm
        )
