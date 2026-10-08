from __future__ import annotations

import re
from pathlib import Path

import pytest
from pydantic import ValidationError

from grounded.settings import SELF_CARRY_CEILING, Settings
from tests.support import EVAL_SETTINGS, make_settings


def test_defaults_match_tech_md() -> None:
    s = make_settings()
    assert (s.k_dense, s.k_fts, s.k_fused, s.k_context, s.rrf_k) == (20, 20, 40, 5, 60)
    assert s.generator_providers == ["gemini", "groq"]
    assert s.rerank_provider == "none"
    assert s.allow_direct_api is False


def test_gemini_thinking_level_defaults_to_minimal_and_is_typed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GEMINI_THINKING_LEVEL", raising=False)
    assert make_settings().gemini_thinking_level == "minimal"
    assert make_settings(gemini_thinking_level="high").gemini_thinking_level == "high"
    with pytest.raises(ValidationError):
        make_settings(gemini_thinking_level="0")


def test_groq_reasoning_effort_defaults_to_low_and_takes_what_gpt_oss_takes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GROQ_REASONING_EFFORT", raising=False)
    assert make_settings().groq_reasoning_effort == "low"
    assert make_settings(groq_reasoning_effort="high").groq_reasoning_effort == "high"
    # "none" and "default" are for Qwen on Groq; gpt-oss answers any other value with a 400.
    for value in ("none", "default", "minimal"):
        with pytest.raises(ValidationError):
            make_settings(groq_reasoning_effort=value)


def test_pinned_corpus_and_embedding_defaults_match_tech_md(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in ("EMBEDDING_MODEL", "FASTAPI_REF"):
        monkeypatch.delenv(name, raising=False)
    s = make_settings()
    assert (s.embedding_model, s.embedding_dim) == ("gemini-embedding-001", 768)
    assert s.fastapi_ref == "0.141.1"


@pytest.mark.parametrize("name", ["embedding_model", "fastapi_ref"])
def test_blank_embedding_model_and_corpus_tag_are_rejected(name: str) -> None:
    # An unfilled variable is an empty string, not "unset": it must not silently pick another index.
    with pytest.raises(ValidationError):
        make_settings(**{name: ""})


def test_embedding_defaults_match_the_free_tier() -> None:
    s = make_settings()
    assert (s.embedding_batch_size, s.embedding_rpm, s.embedding_tpm) == (100, 100, 30_000)
    assert (s.embedding_max_input_tokens, s.embedding_max_retries) == (2048, 5)
    assert s.embedding_max_retry_wait_s == 60.0
    assert s.embedding_timeout_s == 30.0


def test_embedding_max_retry_wait_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        make_settings(embedding_max_retry_wait_s=0)


def test_embedding_batch_size_is_capped_at_the_api_maximum() -> None:
    with pytest.raises(ValidationError):
        make_settings(embedding_batch_size=101)


def test_embedding_input_limit_cannot_exceed_tpm() -> None:
    with pytest.raises(ValidationError, match="EMBEDDING_MAX_INPUT_TOKENS"):
        make_settings(embedding_max_input_tokens=5000, embedding_tpm=4000)


def test_embedding_batch_cannot_exceed_rpm() -> None:
    # Every text in a batch counts as one request toward RPM.
    with pytest.raises(ValidationError, match="EMBEDDING_BATCH_SIZE"):
        make_settings(embedding_batch_size=50, embedding_rpm=40)


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


def test_chunking_defaults_match_tech_md() -> None:
    s = make_settings()
    assert (s.chunk_max_tokens, s.chunk_overlap_tokens, s.chunk_min_tokens) == (450, 50, 40)
    assert s.tokenizer_encoding == "o200k_base"


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"chunk_min_tokens": 450}, "CHUNK_MIN_TOKENS"),
        ({"chunk_overlap_tokens": 450}, "CHUNK_OVERLAP_TOKENS"),
    ],
)
def test_chunk_sizes_must_fit_under_max(overrides: dict[str, int], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        make_settings(**overrides)


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


class TestEvalGuard:
    """APP_ENV=eval refuses, at startup, a config that would make a run unreproducible (4.03)."""

    @pytest.fixture(autouse=True)
    def _unset_guarded_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # make_settings honors real env vars; a developer's .env-style shell must not decide these.
        for name in ("GENERATOR_PROVIDERS", "LLM_TEMPERATURE", "EVAL_MAX_TOTAL_WAIT_S"):
            monkeypatch.delenv(name, raising=False)

    def test_a_provider_list_with_a_fallback_is_refused_not_ignored(self) -> None:
        # The default list is "gemini,groq": eval needs the primary alone, and says so.
        with pytest.raises(ValidationError, match=r"exactly one provider.*got gemini,groq"):
            make_settings(app_env="eval")
        with pytest.raises(ValidationError, match="exactly one provider"):
            make_settings(app_env="eval", generator_providers=["gemini", "fake"])

    @pytest.mark.parametrize("providers", [["gemini"], ["fake"]])
    def test_a_single_provider_is_accepted(self, providers: list[str]) -> None:
        assert make_settings(app_env="eval", generator_providers=providers).app_env == "eval"

    def test_the_judge_provider_cannot_also_be_the_eval_generator(self) -> None:
        # The judge runs on Groq and must differ from the generator (AGENTS.md §6.5).
        with pytest.raises(ValidationError, match="judge runs on Groq"):
            make_settings(app_env="eval", generator_providers=["groq"])

    def test_temperature_must_be_zero(self) -> None:
        with pytest.raises(ValidationError, match="LLM_TEMPERATURE=0"):
            make_settings(app_env="eval", generator_providers=["gemini"], llm_temperature=0.3)

    @pytest.mark.parametrize("env", ["dev", "test"])
    def test_other_environments_keep_the_router_list_and_any_temperature(self, env: str) -> None:
        s = make_settings(app_env=env, llm_temperature=0.3)
        assert s.generator_providers == ["gemini", "groq"]

    def test_the_backoff_bound_is_a_positive_setting_with_a_default(self) -> None:
        assert make_settings().eval_max_total_wait_s == 120.0
        assert make_settings(eval_max_total_wait_s=30).eval_max_total_wait_s == 30.0
        with pytest.raises(ValidationError):
            make_settings(eval_max_total_wait_s=0)


def test_eval_mode_takes_its_own_llm_timeout_in_place_of_the_request_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 4.05: 2 of 6 real Gemini calls hit the 12 s request-path timeout (typical: 2-3 s), and a
    # timeout counts toward `inconclusive`. The eval has no user-facing deadline, so it has its own.
    for name in ("LLM_TIMEOUT_S", "EVAL_LLM_TIMEOUT_S", "GENERATOR_PROVIDERS", "LLM_TEMPERATURE"):
        monkeypatch.delenv(name, raising=False)
    assert make_settings().eval_llm_timeout_s == 40.0
    assert make_settings().call_timeout_s == 12.0  # test/dev/prod: LLM_TIMEOUT_S
    eval_settings = make_settings(**EVAL_SETTINGS)
    assert (eval_settings.call_timeout_s, eval_settings.llm_timeout_s) == (40.0, 12.0)
    assert make_settings(**EVAL_SETTINGS, eval_llm_timeout_s=90).call_timeout_s == 90.0
    assert make_settings(llm_timeout_s=5, eval_llm_timeout_s=90).call_timeout_s == 5.0
    with pytest.raises(ValidationError):
        make_settings(eval_llm_timeout_s=0)


def test_the_judge_output_cap_is_its_own_positive_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    # Not LLM_MAX_OUTPUT_TOKENS: the cap is part of the eval cache key, so sharing it would drop the
    # cached judge verdicts whenever the generator's answer cap moved (Tech §11).
    monkeypatch.delenv("JUDGE_MAX_OUTPUT_TOKENS", raising=False)
    assert make_settings().judge_max_output_tokens == 800
    assert make_settings(judge_max_output_tokens=1200).judge_max_output_tokens == 1200
    with pytest.raises(ValidationError):
        make_settings(judge_max_output_tokens=0)


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


def test_relative_cache_dir_is_anchored_at_the_repo_root(tmp_path: Path) -> None:
    # Commands run from backend/ and CI from the root; both must share one .cache/.
    repo_root = Path(__file__).resolve().parents[3]
    assert make_settings().cache_dir == repo_root / ".cache"
    assert make_settings(cache_dir="custom").cache_dir == repo_root / "custom"
    assert make_settings(cache_dir=tmp_path).cache_dir == tmp_path


def test_request_path_defaults_match_tech_md() -> None:
    s = make_settings()
    assert (s.llm_timeout_s, s.active_index_ttl_s, s.query_embedding_cache_size) == (
        12.0,
        300.0,
        256,
    )


def test_confidence_weights_that_let_the_self_report_carry_a_claim_are_rejected() -> None:
    # Phase 3 review #4: W_SELF passes its own field cap (0.6), but the weights are relative, so
    # its share is what counts: (0.6 + 0.1*2/3 + 0.1/2) / 0.9 = 0.796.
    with pytest.raises(ValidationError, match=r"0\.796, over the 0\.6 ceiling"):
        make_settings(
            confidence_w_self=0.6,
            confidence_w_retrieval=0.1,
            confidence_w_agreement=0.1,
            confidence_w_citations=0.1,
            confidence_w_rerank=0.0,
        )


def test_confidence_weights_cannot_all_be_zero() -> None:
    zero = {f"confidence_w_{name}": 0.0 for name in ("retrieval", "agreement", "citations")}
    with pytest.raises(ValidationError, match="must not all be zero"):
        make_settings(**zero, confidence_w_self=0.0, confidence_w_rerank=0.0)


@pytest.mark.parametrize(
    ("weights", "worst"),
    [
        ({}, (0.15 + 0.40 * 2 / 3 + 0.20 / 2) / 1.0),  # the defaults: 0.517
        (  # close to the ceiling: 0.594
            {
                "confidence_w_self": 0.3,
                "confidence_w_retrieval": 0.3,
                "confidence_w_agreement": 0.21,
                "confidence_w_citations": 0.2,
                "confidence_w_rerank": 0.0,
            },
            (0.3 + 0.3 * 2 / 3 + 0.2 / 2) / 1.01,
        ),
    ],
    ids=["defaults", "near-ceiling"],
)
def test_confidence_weights_under_the_ceiling_are_accepted(
    weights: dict[str, float], worst: float
) -> None:
    # Accepted (no ValidationError), with the worst case written out by hand. That the formula
    # matches the heuristic is pinned against ``score_claims`` in test_confidence.py.
    s = make_settings(**weights)
    assert s.self_carry_worst_case() == pytest.approx(worst)
    assert worst <= SELF_CARRY_CEILING


def test_k_context_is_limited_to_the_nine_labels_the_citation_grammar_has() -> None:
    assert make_settings(k_context=9).k_context == 9
    with pytest.raises(ValidationError):
        make_settings(k_context=10)
