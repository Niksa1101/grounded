from __future__ import annotations

import re
from pathlib import Path

import pytest
from pydantic import ValidationError

from grounded.settings import Settings
from tests.support import make_settings


def test_defaults_match_tech_md() -> None:
    s = make_settings()
    assert (s.k_dense, s.k_fts, s.k_fused, s.k_context, s.rrf_k) == (20, 20, 40, 5, 60)
    assert s.generator_providers == ["gemini", "groq"]
    assert s.rerank_provider == "none"
    assert s.allow_direct_api is False


def test_embedding_defaults_match_the_free_tier() -> None:
    s = make_settings()
    assert (s.embedding_batch_size, s.embedding_rpm, s.embedding_tpm) == (100, 100, 30_000)
    assert (s.embedding_max_input_tokens, s.embedding_max_retries) == (2048, 5)
    assert s.embedding_timeout_s == 30.0


def test_embedding_batch_size_is_capped_at_the_api_maximum() -> None:
    with pytest.raises(ValidationError):
        make_settings(embedding_batch_size=101)


def test_embedding_input_limit_cannot_exceed_tpm() -> None:
    with pytest.raises(ValidationError, match="EMBEDDING_MAX_INPUT_TOKENS"):
        make_settings(embedding_max_input_tokens=5000, embedding_tpm=4000)


def test_generator_providers_parse_comma_separated_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GENERATOR_PROVIDERS", " gemini , groq,")
    assert make_settings().generator_providers == ["gemini", "groq"]


def test_empty_generator_providers_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GENERATOR_PROVIDERS", "")
    with pytest.raises(ValidationError, match="at least one provider"):
        make_settings()


def test_migration_url_falls_back_to_runtime_url() -> None:
    s = make_settings(database_url="postgresql://app@db/grounded")
    assert s.migration_database_url.get_secret_value() == "postgresql://app@db/grounded"

    s = make_settings(
        database_url="postgresql://app@db/grounded",
        database_url_direct="postgresql://owner@db/grounded",
    )
    assert s.migration_database_url.get_secret_value() == "postgresql://owner@db/grounded"


def test_secrets_are_not_in_repr() -> None:
    s = make_settings(database_url="postgresql://u:hunter2@db/grounded", gemini_api_key="k-123")
    assert "hunter2" not in repr(s)
    assert "k-123" not in repr(s)


def test_answer_cache_ttl_cannot_exceed_retention() -> None:
    with pytest.raises(ValidationError):
        make_settings(answer_cache_ttl_days=31)


def test_pool_min_cannot_exceed_max() -> None:
    with pytest.raises(ValidationError, match="DB_POOL_MIN_SIZE"):
        make_settings(db_pool_min_size=6, db_pool_max_size=5)


class TestProdGuard:
    @pytest.fixture(autouse=True)
    def _unset_guarded_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # make_settings honors real env vars, and CI sets DATABASE_URL for the job; without this
        # the "missing" case would depend on where the tests run.
        for name in ("DATABASE_URL", "PROXY_SHARED_SECRET", "IP_HASH_SECRET"):
            monkeypatch.delenv(name, raising=False)

    def test_prod_requires_explicit_database_and_secrets(self) -> None:
        with pytest.raises(ValidationError) as excinfo:
            make_settings(app_env="prod")
        message = str(excinfo.value)
        for name in ("DATABASE_URL", "PROXY_SHARED_SECRET", "IP_HASH_SECRET"):
            assert name in message

    @pytest.mark.parametrize("name", ["database_url", "proxy_shared_secret", "ip_hash_secret"])
    @pytest.mark.parametrize("blank", ["", "   "])
    def test_prod_rejects_blank_values(self, name: str, blank: str) -> None:
        # An env var that exists but is empty (e.g. an unfilled deploy secret) is not "configured".
        values = {
            "database_url": "postgresql://app@db/grounded",
            "proxy_shared_secret": "x" * 32,
            "ip_hash_secret": "y" * 32,
        } | {name: blank}
        with pytest.raises(ValidationError, match=name.upper()):
            make_settings(app_env="prod", **values)

    def test_prod_rejects_direct_api(self) -> None:
        with pytest.raises(ValidationError, match="ALLOW_DIRECT_API"):
            make_settings(
                app_env="prod",
                database_url="postgresql://app@db/grounded",
                proxy_shared_secret="x" * 32,
                ip_hash_secret="y" * 32,
                allow_direct_api=True,
            )

    def test_prod_accepts_complete_config(self) -> None:
        s = make_settings(
            app_env="prod",
            database_url="postgresql://app@db/grounded",
            proxy_shared_secret="x" * 32,
            ip_hash_secret="y" * 32,
        )
        assert s.app_env == "prod"


def test_env_example_lists_every_setting() -> None:
    # .env.example is the documented config surface (Tech.md §4); it must not drift from Settings.
    env_example = Path(__file__).resolve().parents[3] / ".env.example"
    text = env_example.read_text(encoding="utf-8")
    listed = set(re.findall(r"^#? ?([A-Z][A-Z0-9_]*)=", text, re.MULTILINE))
    assert listed == {name.upper() for name in Settings.model_fields}


def test_env_file_is_the_repo_root_env_regardless_of_working_directory() -> None:
    # A CWD-relative "../.env" would pick up a stray .env above the repo when run from the root.
    env_file = Path(str(Settings.model_config.get("env_file")))
    assert env_file.is_absolute()
    assert env_file.name == ".env"
    assert (env_file.parent / ".env.example").is_file()
