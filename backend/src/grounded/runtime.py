"""Long-lived resources of the request path, built once and shared by the API and the CLI.

``open_runtime`` creates the connection pool, the query embedder, the generator providers and the
``AskPipeline`` over them (which builds the provider router, Tech §10), and closes what it opened.
The API lifespan and ``grounded ask`` both go through it, so they cannot drift apart. Tests pass
their own ``embedder``, ``provider`` and ``fallback_providers`` (AGENTS.md §8: no real calls).
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncGenerator, Callable, Sequence
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from datetime import datetime

from google import genai
from google.genai import types as genai_types
from psycopg_pool import AsyncConnectionPool

from grounded.evals.judge import Judge, JudgeConfig
from grounded.generation.pipeline import AskPipeline
from grounded.generation.prompts import load_answer_prompt, load_no_rag_prompt
from grounded.generation.providers.base import LLMProvider
from grounded.generation.providers.eval_wrappers import EvalLLM
from grounded.generation.providers.fake import StubJudgeProvider, StubLLMProvider
from grounded.generation.providers.gemini import GeminiProvider
from grounded.generation.providers.groq import GroqProvider
from grounded.generation.router import SYSTEM_CLOCK, RouterClock
from grounded.infra.budget import DailyBudget, RerankCap
from grounded.infra.db import create_pool
from grounded.infra.kvcache import KVCache
from grounded.infra.ratelimit import RateLimiter
from grounded.infra.timing import Clock
from grounded.ingest.embed import Embedder
from grounded.observability.cost import Pricing, load_pricing
from grounded.observability.request_log import RequestLogger
from grounded.retrieval.index import ActiveIndexCache
from grounded.retrieval.query_embedding import build_query_embedder
from grounded.retrieval.rerank import (
    RERANK_CACHE_FILE,
    CachedReranker,
    CohereReranker,
    PacedReranker,
    Reranker,
)
from grounded.settings import Settings

# Vercel gives a function 500 ms after SIGTERM and shows no shutdown logs (Tech §18), so closing the
# pool must not use psycopg_pool's default of waiting up to 5 s for its workers. A platform limit,
# not a tunable: 0.4 s leaves the rest of the budget to the other closers.
POOL_CLOSE_TIMEOUT_S = 0.4

logger = logging.getLogger(__name__)

# The names ``GENERATOR_PROVIDERS`` may hold, and the ones that may come first (Tech §4, §10). The
# first entry names the answer-cache key and the adapter parameters in it (``thinking_level``;
# ``GROQ_REASONING_EFFORT`` is not there), so Groq cannot be first until the key moves (PRD §12).
KNOWN_PROVIDERS = ("gemini", "groq", "fake")
PRIMARY_PROVIDERS = ("gemini", "fake")


class ProviderConfigError(Exception):
    """A provider this build cannot create or must not use: an unknown ``GENERATOR_PROVIDERS``
    entry, a missing key or model ID, a judge on the generator's provider.

    ``unconfigured`` marks the one case a later entry of ``GENERATOR_PROVIDERS`` may be skipped
    for (PRD D57d, ``build_fallback_providers``): a missing key or model ID, whose message names the
    variable and never a value. It is a flag, not a subclass, because the eval harness records an
    error by its class name and the gate classifies it that way (``ProviderConfigError``).
    """

    def __init__(self, message: str, *, unconfigured: bool = False) -> None:
        super().__init__(message)
        self.unconfigured = unconfigured


@dataclass(frozen=True, slots=True)
class Runtime:
    pool: AsyncConnectionPool
    pipeline: AskPipeline
    request_logger: RequestLogger
    # None in eval mode: the limiter is not consulted at all (Tech §15.6), so a run of 30+
    # questions from one process is never answered with our own 429 (5.04).
    rate_limiter: RateLimiter | None = None
    # Eval mode only (Tech §15.6), and only for a provider built from the settings: the cache and
    # backoff around it, whose counters a run prints at its end. ``wrap`` takes the judge as well.
    eval_llm: EvalLLM | None = None


def build_provider(settings: Settings) -> LLMProvider:
    """The primary generator: the first entry of ``GENERATOR_PROVIDERS`` (Tech §4, §10).

    ``gemini`` is the real adapter; ``fake`` is a dev-only stand-in (refused in prod). Anything
    else is refused with an error that says why. **``groq`` cannot be first** (7.04, PRD §12): the
    answer-cache key names the first entry's model and ``thinking_level`` but not
    ``GROQ_REASONING_EFFORT``, so Groq answers stored under it could be served after the setting
    moved. Groq is the fallback (a later entry) and the judge.
    """
    _check_provider_names(settings.generator_providers)
    name = settings.generator_providers[0]
    if name == "gemini":
        return _build_gemini(settings)
    if name == "fake":
        _refuse_fake_in_prod(settings)
        return StubLLMProvider()
    raise ProviderConfigError(
        f"provider {name!r} cannot be the first entry of GENERATOR_PROVIDERS: only "
        f"{' or '.join(PRIMARY_PROVIDERS)} can (Groq answers would be cached under a key that "
        "lacks GROQ_REASONING_EFFORT, PRD §12); set GENERATOR_PROVIDERS=gemini,groq "
        "(or fake for local runs)"
    )


def build_fallback_providers(settings: Settings, stack: AsyncExitStack) -> list[LLMProvider]:
    """The entries of ``GENERATOR_PROVIDERS`` after the first, in order, for the router (D57d).

    An entry whose key or model is missing (``GROQ_API_KEY``, ``GROQ_MODEL``) is **skipped with a
    WARNING** that names the provider and the reason (a variable name, never a value): the router
    then runs on what is left, and a service without its fallback still answers from the primary.
    Only the first entry stops startup (``build_provider``). A contradiction in the configuration
    is not "cannot be built": an unknown name, a duplicate, or the stub in prod stay loud. Eval
    mode has no later entries (``Settings`` takes exactly one provider), so it gets none. The
    caller owns what is built and ``stack`` closes it.
    """
    _check_provider_names(settings.generator_providers)
    built: list[LLMProvider] = []
    for name in settings.generator_providers[1:]:
        try:
            built.append(_build_named(settings, name, stack))
        except ProviderConfigError as exc:
            if not exc.unconfigured:
                raise  # a contradiction in the list, not a missing variable: stays loud
            logger.warning(
                "fallback provider skipped, it cannot be built",
                extra={"provider": name, "reason": str(exc)},
            )
    return built


def _check_provider_names(names: Sequence[str]) -> None:
    for name in names:
        if name not in KNOWN_PROVIDERS:
            raise ProviderConfigError(
                f"unknown provider {name!r} in GENERATOR_PROVIDERS (known: "
                f"{', '.join(KNOWN_PROVIDERS)})"
            )
        if names.count(name) > 1:
            raise ProviderConfigError(f"GENERATOR_PROVIDERS names {name!r} more than once")


def _refuse_fake_in_prod(settings: Settings) -> None:
    if settings.app_env == "prod":
        raise ProviderConfigError("the fake provider must not be used with APP_ENV=prod")


def _build_named(settings: Settings, name: str, stack: AsyncExitStack) -> LLMProvider:
    """One later entry of ``GENERATOR_PROVIDERS``, with its client closed by ``stack``."""
    if name == "gemini":
        gemini = _build_gemini(settings)
        stack.push_async_callback(gemini.aclose)
        return gemini
    if name == "groq":
        groq = build_groq_provider(settings, model=settings.groq_model)
        stack.push_async_callback(groq.aclose)
        return groq
    _refuse_fake_in_prod(settings)
    return StubLLMProvider()


def gemini_http_options() -> genai_types.HttpOptions:
    """One HTTP attempt, set explicitly (7.04, Tech §10 "For 7.02 to 7.04", PRD §12).

    ``google-genai`` 2.25.0 retries nothing by default, but Google's troubleshooting page says its
    SDKs retry a 429 or 5xx up to four times, and a retry inside the SDK would sleep in the request
    path and turn a 429 into a timeout that the router never sees as a 429. ``attempts=1`` means
    no retries (the SDK's own words), so the failure reaches the adapter and the router decides.
    """
    return genai_types.HttpOptions(retry_options=genai_types.HttpRetryOptions(attempts=1))


def _build_gemini(settings: Settings) -> GeminiProvider:
    # A blank value counts as unset (an unfilled deploy secret arrives as ""), as in the embedder.
    key = settings.gemini_api_key.get_secret_value().strip() if settings.gemini_api_key else ""
    model = (settings.gemini_model or "").strip()
    if not key:
        raise ProviderConfigError(
            "GEMINI_API_KEY must be set to use the gemini provider", unconfigured=True
        )
    if not model:
        raise ProviderConfigError(
            "GEMINI_MODEL must be set to a pinned model ID (PRD D46: gemini-3.5-flash-lite)",
            unconfigured=True,
        )
    return GeminiProvider(
        genai.Client(api_key=key, http_options=gemini_http_options()),
        model=model,
        thinking_level=settings.gemini_thinking_level,
    )


def build_groq_provider(
    settings: Settings, *, model: str | None, model_variable: str = "GROQ_MODEL"
) -> GroqProvider:
    """A Groq adapter for ``model``: ``GROQ_MODEL`` for the fallback generator (a later entry of
    ``GENERATOR_PROVIDERS``), ``JUDGE_MODEL`` for the judge (4.04); ``model_variable`` is the name
    an error gives for it. The caller owns the provider and closes it (``aclose``)."""
    key = settings.groq_api_key.get_secret_value().strip() if settings.groq_api_key else ""
    pinned = (model or "").strip()
    if not key:
        raise ProviderConfigError(
            "GROQ_API_KEY must be set to use the groq provider", unconfigured=True
        )
    if not pinned:
        raise ProviderConfigError(
            f"the Groq model ({model_variable}) must be set to a pinned model ID "
            "(PRD D48: openai/gpt-oss-120b)",
            unconfigured=True,
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
    return build_groq_provider(settings, model=settings.judge_model, model_variable="JUDGE_MODEL")


def build_reranker(
    settings: Settings, stack: AsyncExitStack, *, paced: bool = False
) -> Reranker | None:
    """The reranker for ``RERANK_PROVIDER``, or ``None`` when it is ``none`` (Tech §8, PRD D55).

    ``cohere`` needs ``COHERE_API_KEY`` and ``RERANK_MODEL`` (blank counts as unset, like the
    other providers) and raises ``ProviderConfigError`` naming the variable before any call. Dev,
    eval and test put the SQLite cache in front of it (the manual evals re-run for free); prod does
    not, because the answer cache covers repeats there and a function's disk is ephemeral. The
    caller closes what this opens through ``stack``.

    ``paced`` is for the eval harness (6.04): the live calls are spaced by
    ``EVAL_RERANK_MIN_INTERVAL_S`` and metered by a ``PacedReranker`` that sits between the cache
    and the adapter, so a cache hit never waits. The result is ``CachedReranker(PacedReranker(
    CohereReranker))``. It is refused in prod, where nothing may sleep (AGENTS.md §6.12).
    """
    if settings.rerank_provider == "none":
        return None
    if paced and settings.app_env == "prod":
        raise ProviderConfigError(
            "rerank pacing is an eval tool: it must not be used with APP_ENV=prod"
        )
    key = settings.cohere_api_key.get_secret_value().strip() if settings.cohere_api_key else ""
    model = (settings.rerank_model or "").strip()
    if not key:
        raise ProviderConfigError("COHERE_API_KEY must be set to use the cohere reranker")
    if not model:
        raise ProviderConfigError(
            "RERANK_MODEL must be set to a pinned model ID (PRD D55: rerank-v4.0-pro)"
        )
    adapter = CohereReranker.create(key, model=model, timeout_s=settings.rerank_timeout_s)
    stack.push_async_callback(adapter.aclose)
    if settings.app_env == "prod":
        return adapter
    cache = stack.enter_context(KVCache(settings.cache_dir / RERANK_CACHE_FILE))
    if paced:
        return CachedReranker(
            PacedReranker(adapter, min_interval_s=settings.eval_rerank_min_interval_s), cache
        )
    return CachedReranker(adapter, cache)


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
    wall_clock: Callable[[], datetime] | None = None,
    reranker: Reranker | None = None,
    fallback_providers: Sequence[LLMProvider] | None = None,
    router_clock: RouterClock = SYSTEM_CLOCK,
    primary_only: bool = False,
) -> AsyncGenerator[Runtime]:
    """``wall_clock`` replaces the database's ``now()`` as the rate limiter's clock, and
    ``reranker`` the one ``build_reranker`` makes from the settings (tests only; it must go with a
    ``RERANK_PROVIDER`` other than ``none``, the pipeline checks).

    The router is built once here (inside the pipeline), over the primary and the later entries of
    ``GENERATOR_PROVIDERS`` (``build_fallback_providers``: one that cannot be built is skipped with
    a warning). An injected ``provider`` stands for the whole list unless the test also passes
    ``fallback_providers`` (no real fallback is ever built next to an injected primary), and
    ``router_clock`` is how a test moves the breaker's time (the real monotonic clock otherwise).

    ``primary_only`` is for the CLI (``grounded ask``) and is implied by eval mode: only the first
    entry is built, without a breaker, and the provider's own typed errors come out (the golden
    batch waits out a 429 and stops at a daily quota by reading them, Tech §15.8).
    """
    primary_only = primary_only or settings.app_env == "eval"
    async with AsyncExitStack() as stack:
        pool = create_pool(settings)
        # wait=False: start serving even if the database is unreachable; /readyz reports it.
        await pool.open(wait=False)
        stack.push_async_callback(pool.close, POOL_CLOSE_TIMEOUT_S)
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
            if fallback_providers is None and not primary_only:
                fallback_providers = build_fallback_providers(settings, stack)
        pricing = pricing or load_pricing()
        # No budget in eval mode (Tech §15.6: the LLM calls of a run must neither use up the demo's
        # day nor end in budget_exhausted) and none without a cap (dev and test; prod requires one).
        budget = (
            DailyBudget(pool, limit=settings.daily_llm_budget)
            if settings.daily_llm_budget is not None and settings.app_env != "eval"
            else None
        )
        # Built from the settings unless a test hands one in: the factory refuses a missing key or
        # model before the first request. ``None`` for RERANK_PROVIDER=none, the shipped default.
        if reranker is None:
            # The eval runtime paces its live calls to the Trial key's limit (6.04, Tech §15.6).
            reranker = build_reranker(settings, stack, paced=settings.app_env == "eval")
        # No cap in eval mode, like no budget (Tech §15.6, §8): the manual eval runs have the
        # Author's own call budget for the phase. Everywhere else a reranker is capped per UTC day.
        rerank_cap = (
            RerankCap(pool, limit=settings.rerank_daily_cap)
            if reranker is not None and settings.app_env != "eval"
            else None
        )
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
            budget=budget,
            reranker=reranker,
            rerank_cap=rerank_cap,
            fallback_providers=fallback_providers or (),
            router_clock=router_clock,
            primary_only=primary_only,
        )
        # Eval runs are told apart from real traffic in the table (DB.md §4, CI database only).
        request_logger = RequestLogger(
            pool, pricing, clock=clock, source="eval" if settings.app_env == "eval" else "api"
        )
        rate_limiter = (
            None
            if settings.app_env == "eval"
            else RateLimiter(
                pool,
                per_minute=settings.rate_limit_per_min,
                per_day=settings.rate_limit_per_day,
                clock=wall_clock,
            )
        )
        yield Runtime(
            pool=pool,
            pipeline=pipeline,
            request_logger=request_logger,
            rate_limiter=rate_limiter,
            eval_llm=eval_llm,
        )
