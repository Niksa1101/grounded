"""Application settings: the only place configuration is read from the environment.

Every tunable (K values, limits, model IDs, secrets) is a field here, never a literal in code and
never an ``os.environ`` read elsewhere (AGENTS.md §6.7). The variable list mirrors Tech.md §4, and
``.env.example`` must stay in sync with it.

The generator, judge and rerank model IDs have no defaults on purpose: they are pinned in ``.env``
after being verified against the provider docs in the phase that first uses them (AGENTS.md §6.8).
The embedding model is the exception: it was verified on 2026-09-26 and is tied to the index (a
different value is a new index version), so it and the corpus tag have pinned defaults.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

AppEnv = Literal["dev", "test", "prod", "eval"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR"]
RerankProvider = Literal["none", "cohere"]
# Gemini 3.x is controlled by a level, not a token budget (PRD D46, Tech §4). Which levels a model
# accepts differs (3.7/3.8 reject "minimal"), so the model is checked by the API, not here.
GeminiThinkingLevel = Literal["minimal", "low", "medium", "high"]

# Matches infra/docker-compose.yml (DB.md §2). Convenient for local dev; prod must set its own.
_LOCAL_DATABASE_URL = "postgresql://grounded:grounded@localhost:5433/grounded"
_LOCAL_TEST_DATABASE_URL = "postgresql://grounded:grounded@localhost:5433/grounded_test"

# The repo-root .env (this file is backend/src/grounded/settings.py). Anchored to the source tree,
# not the working directory, so no stray .env above the repo can leak in. In a deployed install the
# path doesn't exist and is ignored: Vercel and CI use real env vars.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_ENV_FILE = _REPO_ROOT / ".env"

# Confidence invariant 4 (Tech §9.8): a claim whose only strong signal is the model's self-report,
# backed by the weakest retrieval support, must not score above this. See ``_check_confidence``.
SELF_CARRY_CEILING = 0.6


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    app_env: AppEnv = "dev"
    log_level: LogLevel = "INFO"

    # --- Database ------------------------------------------------------------------------------
    database_url: SecretStr = SecretStr(_LOCAL_DATABASE_URL)
    # Owner connection for migrations/ingest/retention. Falls back to database_url when unset.
    database_url_direct: SecretStr | None = None
    test_database_url: SecretStr = SecretStr(_LOCAL_TEST_DATABASE_URL)
    db_pool_min_size: int = Field(default=1, ge=0)
    db_pool_max_size: int = Field(default=5, ge=1)
    db_pool_timeout_s: float = Field(default=5.0, gt=0)

    # --- Providers -----------------------------------------------------------------------------
    gemini_api_key: SecretStr | None = None
    groq_api_key: SecretStr | None = None
    cohere_api_key: SecretStr | None = None

    embedding_model: str = Field(default="gemini-embedding-001", min_length=1)
    embedding_dim: int = Field(default=768, gt=0)
    # Defaults = gemini-embedding-001 free tier (AI Studio, 2026-09-26) and API caps (Tech.md §5.6).
    embedding_batch_size: int = Field(default=100, gt=0, le=100)  # the API rejects > 100 per call
    embedding_rpm: int = Field(default=100, gt=0)
    embedding_tpm: int = Field(default=30_000, gt=0)
    embedding_max_input_tokens: int = Field(default=2048, gt=0)
    embedding_max_retries: int = Field(default=5, ge=0)
    # The longest server-given Retry-After that is waited out; a longer one stops the run.
    embedding_max_retry_wait_s: float = Field(default=60.0, gt=0)
    embedding_timeout_s: float = Field(default=30.0, gt=0)

    generator_providers: Annotated[list[str], NoDecode] = ["gemini", "groq"]
    gemini_model: str | None = None
    groq_model: str | None = None
    judge_model: str | None = None
    gemini_thinking_level: GeminiThinkingLevel = "minimal"
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    llm_max_output_tokens: int = Field(default=800, gt=0)
    llm_timeout_s: float = Field(default=12.0, gt=0)  # each generation attempt (Tech.md §6)

    rerank_provider: RerankProvider = "none"
    rerank_model: str | None = None
    rerank_daily_cap: int = Field(default=30, ge=0)

    # --- Chunking (grouped into ChunkingConfig, stored per index version) ----------------------
    chunk_max_tokens: int = Field(default=450, gt=0)
    chunk_overlap_tokens: int = Field(default=50, ge=0)
    chunk_min_tokens: int = Field(default=40, gt=0)
    tokenizer_encoding: str = "o200k_base"  # tiktoken; approximate counts for sizing only

    # --- Retrieval (grouped into a hashed RetrievalConfig, retrieval/config.py) ----------------
    k_dense: int = Field(default=20, gt=0)
    k_fts: int = Field(default=20, gt=0)
    k_fused: int = Field(default=40, gt=0)
    k_context: int = Field(default=5, gt=0, le=9)  # c1..c9: the citation grammar's limit
    rrf_k: int = Field(default=60, gt=0)
    # Not part of RetrievalConfig: they change how fast the request path reacts, not what a query
    # returns, so tuning them must not change retrieval_config_hash.
    active_index_ttl_s: float = Field(default=300.0, gt=0)  # DB.md §7.3
    query_embedding_cache_size: int = Field(default=256, gt=0)  # in-memory LRU, prod (Tech.md §11)
    # The embed stage timeout of /v1/ask (Tech.md §6): one attempt, no retry, no sleep (§10).
    query_embedding_timeout_s: float = Field(default=3.0, gt=0)

    # --- Confidence (generation/confidence.py, Tech §9.8) -------------------------------------
    # Relative importance of each signal. The defaults sum to 1.0 so they read as shares; the
    # heuristic divides by their sum, so any non-negative set works. ``w_rerank`` is 0 until
    # Phase 6 measures whether rerank scores help (it is also moot whenever rerank is off).
    confidence_w_retrieval: float = Field(default=0.40, ge=0.0, le=1.0)
    confidence_w_agreement: float = Field(default=0.25, ge=0.0, le=1.0)
    confidence_w_citations: float = Field(default=0.20, ge=0.0, le=1.0)
    # A first fence only; the real guard is the share check in ``_check_confidence``.
    confidence_w_self: float = Field(default=0.15, ge=0.0, le=0.6)
    confidence_w_rerank: float = Field(default=0.0, ge=0.0, le=1.0)
    # The most a claim with no valid citation can score. Tech §9.8 fixes 0.2 as the ceiling, so a
    # config can lower the cap but never raise it past the documented invariant.
    confidence_uncited_cap: float = Field(default=0.2, ge=0.0, le=0.2)

    # --- Protection ----------------------------------------------------------------------------
    rate_limit_per_min: int = Field(default=5, gt=0)
    rate_limit_per_day: int = Field(default=30, gt=0)
    # Set below the verified provider free-tier RPD in Phase 5; unset means "not configured yet".
    daily_llm_budget: int | None = Field(default=None, gt=0)
    answer_cache_ttl_days: int = Field(default=30, gt=0, le=30)  # must not exceed retention
    proxy_shared_secret: SecretStr | None = None
    ip_hash_secret: SecretStr | None = None
    allow_direct_api: bool = False
    request_deadline_s: float = Field(default=25.0, gt=0)

    # --- Paths and corpus ----------------------------------------------------------------------
    # A relative path is taken from the repo root, like .env: commands run from backend/ and CI
    # runs from the root, and both must find the same clone and SQLite caches.
    cache_dir: Path = Path(".cache")
    fastapi_ref: str = Field(default="0.141.1", min_length=1)

    @field_validator("generator_providers", mode="before")
    @classmethod
    def _split_providers(cls, value: object) -> object:
        # Env form is "gemini,groq" (Tech.md §4), not JSON.
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @field_validator("cache_dir", mode="after")
    @classmethod
    def _anchor_cache_dir(cls, value: Path) -> Path:
        return value if value.is_absolute() else _REPO_ROOT / value

    @model_validator(mode="after")
    def _check_consistency(self) -> Self:
        if self.db_pool_min_size > self.db_pool_max_size:
            raise ValueError("DB_POOL_MIN_SIZE must be <= DB_POOL_MAX_SIZE")
        if not self.generator_providers:
            raise ValueError("GENERATOR_PROVIDERS must name at least one provider")
        if self.embedding_max_input_tokens > self.embedding_tpm:
            # A single text over the per-minute budget could never be sent.
            raise ValueError("EMBEDDING_MAX_INPUT_TOKENS must be <= EMBEDDING_TPM")
        if self.embedding_batch_size > self.embedding_rpm:
            # Every text in a batch counts as one request toward RPM.
            raise ValueError("EMBEDDING_BATCH_SIZE must be <= EMBEDDING_RPM")
        if not self.chunk_min_tokens < self.chunk_max_tokens:
            raise ValueError("CHUNK_MIN_TOKENS must be < CHUNK_MAX_TOKENS")
        if not self.chunk_overlap_tokens < self.chunk_max_tokens:
            raise ValueError("CHUNK_OVERLAP_TOKENS must be < CHUNK_MAX_TOKENS")
        self._check_confidence()
        if self.app_env == "prod":
            self._check_prod()
        return self

    def self_carry_worst_case(self) -> float:
        """The highest confidence a claim carried by the self-report can reach with these weights.

        The heuristic is a weighted mean divided by the sum of the weights (``score_claims`` in
        ``generation/confidence.py``), so a cap on ``w_self`` alone guarantees nothing: what counts
        is its share. This is the worst case the docstring of ``score_claims`` proves the bound
        for: ``self_confidence = 1``; one valid citation, so ``citations = 1/2``; a chunk found by
        one list only, so ``agreement = 0`` and ``retrieval <= 2/3`` (the missing list's slot is
        0); rerank off, so ``rerank = 0``. The bound uses the best values that one list could
        have, so it covers every weaker support too, and it is reached (a list of one chunk, at
        rank 1 with a perfect score). ``test_confidence.py`` pins it against ``score_claims``, so
        this copy of the formula cannot drift from the heuristic unnoticed.
        """
        total = (
            self.confidence_w_retrieval
            + self.confidence_w_agreement
            + self.confidence_w_citations
            + self.confidence_w_self
            + self.confidence_w_rerank
        )
        return (
            self.confidence_w_self
            + self.confidence_w_retrieval * 2 / 3
            + self.confidence_w_citations / 2
        ) / total

    def _check_confidence(self) -> None:
        """Keep confidence invariant 4 true for *this* config, not only for the defaults."""
        weights = (
            self.confidence_w_retrieval,
            self.confidence_w_agreement,
            self.confidence_w_citations,
            self.confidence_w_self,
            self.confidence_w_rerank,
        )
        if sum(weights) <= 0.0:
            raise ValueError("CONFIDENCE_W_* must not all be zero: every claim would score 0")
        worst = self.self_carry_worst_case()
        if worst > SELF_CARRY_CEILING:
            raise ValueError(
                f"CONFIDENCE_W_* let the self-report carry a weakly supported claim to "
                f"{worst:.3f}, over the {SELF_CARRY_CEILING} ceiling (Tech §9.8 invariant 4): "
                "lower CONFIDENCE_W_SELF or raise CONFIDENCE_W_AGREEMENT"
            )

    def _check_prod(self) -> None:
        """Fail fast on a misconfigured deploy instead of serving with dev defaults."""
        missing: list[str] = []
        for name in ("database_url", "proxy_shared_secret", "ip_hash_secret"):
            value: SecretStr | None = getattr(self, name)
            # An env var that exists but is blank (an unfilled deploy secret) counts as missing.
            if (
                name not in self.model_fields_set
                or value is None
                or not value.get_secret_value().strip()
            ):
                missing.append(name.upper())
        if missing:
            raise ValueError(f"APP_ENV=prod requires explicit {', '.join(missing)}")
        if self.allow_direct_api:
            raise ValueError("ALLOW_DIRECT_API must be false in prod")

    @property
    def migration_database_url(self) -> SecretStr:
        """Owner URL for migrations/ingest; the runtime URL when no separate one is configured."""
        return self.database_url_direct or self.database_url


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
