"""Application settings: the only place configuration is read from the environment.

Every tunable (K values, limits, model IDs, secrets) is a field here, never a literal in code and
never an ``os.environ`` read elsewhere (AGENTS.md §6.7). The variable list mirrors Tech.md §4, and
``.env.example`` must stay in sync with it.

Model IDs have no defaults on purpose: they are pinned in ``.env`` after being verified against the
provider docs in the phase that first uses them (AGENTS.md §6.8).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

AppEnv = Literal["dev", "test", "prod", "eval"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR"]

# Matches infra/docker-compose.yml (DB.md §2). Convenient for local dev; prod must set its own.
_LOCAL_DATABASE_URL = "postgresql://grounded:grounded@localhost:5433/grounded"
_LOCAL_TEST_DATABASE_URL = "postgresql://grounded:grounded@localhost:5433/grounded_test"

# The repo-root .env (this file is backend/src/grounded/settings.py). Anchored to the source tree,
# not the working directory, so no stray .env above the repo can leak in. In a deployed install the
# path doesn't exist and is ignored: Vercel and CI use real env vars.
_ENV_FILE = Path(__file__).resolve().parents[3] / ".env"


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

    embedding_model: str | None = None
    embedding_dim: int = Field(default=768, gt=0)

    generator_providers: Annotated[list[str], NoDecode] = ["gemini", "groq"]
    gemini_model: str | None = None
    groq_model: str | None = None
    judge_model: str | None = None
    gemini_thinking_budget: int = Field(default=0, ge=0)
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    llm_max_output_tokens: int = Field(default=800, gt=0)

    rerank_provider: Literal["none", "cohere"] = "none"
    rerank_model: str | None = None
    rerank_daily_cap: int = Field(default=30, ge=0)

    # --- Retrieval (grouped into a hashed RetrievalConfig in Phase 2) --------------------------
    k_dense: int = Field(default=20, gt=0)
    k_fts: int = Field(default=20, gt=0)
    k_fused: int = Field(default=40, gt=0)
    k_context: int = Field(default=5, gt=0)
    rrf_k: int = Field(default=60, gt=0)

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
    cache_dir: Path = Path(".cache")
    fastapi_ref: str | None = None

    @field_validator("generator_providers", mode="before")
    @classmethod
    def _split_providers(cls, value: object) -> object:
        # Env form is "gemini,groq" (Tech.md §4), not JSON.
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @model_validator(mode="after")
    def _check_consistency(self) -> Self:
        if self.db_pool_min_size > self.db_pool_max_size:
            raise ValueError("DB_POOL_MIN_SIZE must be <= DB_POOL_MAX_SIZE")
        if not self.generator_providers:
            raise ValueError("GENERATOR_PROVIDERS must name at least one provider")
        if self.app_env == "prod":
            self._check_prod()
        return self

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
