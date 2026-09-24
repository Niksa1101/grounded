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
    def test_prod_requires_explicit_database_and_secrets(self) -> None:
        with pytest.raises(ValidationError) as excinfo:
            make_settings(app_env="prod")
        message = str(excinfo.value)
        for name in ("DATABASE_URL", "PROXY_SHARED_SECRET", "IP_HASH_SECRET"):
            assert name in message

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
