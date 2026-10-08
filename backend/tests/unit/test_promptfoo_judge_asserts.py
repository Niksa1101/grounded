"""The judge-based promptfoo assertions (4.06): ``faithfulness`` and ``correctness`` called the way
promptfoo calls them, with a scripted judge provider behind the real ``open_judge`` (so the eval LLM
cache and the typed errors run for real). No promptfoo, no database, no network.

What is pinned here is the contract 4.07 and 4.08 read: N/A, score math, an errored component that
is never a score, the per-claim record, the shared stop. The recorded promptfoo output of the same
shapes is ``tests/fixtures/promptfoo/results_sample_judge.json``."""

from __future__ import annotations

import sqlite3
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from grounded.evals import promptfoo_asserts as asserts
from grounded.evals import promptfoo_judge
from grounded.evals.judge import NO_SOURCE_REASON
from grounded.generation.prompts import (
    load_judge_correctness_prompt,
    load_judge_faithfulness_prompt,
)
from grounded.generation.providers.base import Usage
from grounded.generation.providers.eval_wrappers import LLM_EVAL_CACHE_FILE
from grounded.generation.providers.fake import FakeLLMProvider
from grounded.infra.provider_errors import (
    ProviderRateLimited,
    ProviderRequestRejected,
    ProviderTimeout,
)
from grounded.runtime import ProviderConfigError, open_judge
from grounded.schemas.judge import CorrectnessVerdict, FaithfulnessVerdict
from grounded.settings import Settings
from tests.provider_rigs import GROQ_MODEL
from tests.support import EVAL_SETTINGS, make_settings

RUN = "run-1"
USAGE = Usage(input_tokens=1200, output_tokens=90, thinking_tokens=40)
SUPPORTED = FaithfulnessVerdict(verdict="SUPPORTED", reason="c1 says so.")
UNSUPPORTED = FaithfulnessVerdict(verdict="NOT_SUPPORTED", reason="The 422 part is missing.")

CONTEXT = [
    {"label": "c1", "chunk_id": 11, "content": "Use `status_code=201` to respond with 201."},
    {"label": "c2", "chunk_id": 12, "content": "A router can take a prefix."},
]


def claim(text: str, *chunk_ids: int, confidence: float = 0.7) -> dict[str, Any]:
    return {
        "text": text,
        "citations": list(range(1, len(chunk_ids) + 1)),
        "chunk_ids": list(chunk_ids),
        "confidence": confidence,
        "confidence_components": {"retrieval": 0.5},
    }


CLAIMS = [claim("Claim A.", 11, confidence=0.9), claim("Claim B.", 12, confidence=0.4)]


def case(
    *,
    mode: str = "hybrid",
    claims: list[dict[str, Any]] | None = None,
    run_id: str | None = RUN,
) -> dict[str, Any]:
    """The context promptfoo passes: the question, the golden payload, the provider's metadata."""
    metadata = {
        "mode": mode,
        "claims": CLAIMS if claims is None else claims,
        "context": [] if mode == "no_rag" else CONTEXT,
    }
    test_metadata: dict[str, Any] = {
        "golden": {"id": "q003", "answerable": True, "reference_answer": "Pass status_code=201."}
    }
    if run_id is not None:
        test_metadata["run_id"] = run_id
    return {
        "vars": {"question": "How do I return 201?"},
        "test": {"metadata": test_metadata},
        "providerResponse": {"output": {}, "metadata": metadata},
        "metadata": metadata,
    }


def answer(status: str = "answered") -> dict[str, Any]:
    return {"status": status, "answer_markdown": "Use status_code=201 in the decorator. [1]"}


Install = Callable[..., FakeLLMProvider]


@pytest.fixture
def install(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Install:
    """``install(*script, **settings)`` puts a scripted Groq-named provider behind the real
    ``open_judge`` (eval cache and backoff included) and returns it."""

    def build(*script: Any, **overrides: Any) -> FakeLLMProvider:
        settings: Settings = make_settings(
            **{**EVAL_SETTINGS, "cache_dir": tmp_path, "judge_model": GROQ_MODEL, **overrides}
        )
        provider = FakeLLMProvider(script, name="groq", model=GROQ_MODEL, usage=USAGE)
        monkeypatch.setattr(promptfoo_judge, "get_settings", lambda: settings)
        monkeypatch.setattr(promptfoo_judge, "configure_logging", lambda _level: None)
        monkeypatch.setattr(
            promptfoo_judge, "open_judge", lambda s: open_judge(s, provider=provider)
        )
        return provider

    return build


# --- Faithfulness: N/A ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("output", "context", "why"),
    [
        (answer(), case(mode="no_rag"), "no_rag"),
        (answer("insufficient_context"), case(), "refused"),
        (answer(), case(claims=[]), "no claims"),
    ],
)
def test_faithfulness_is_not_applicable_without_sources_or_claims(
    install: Install, output: dict[str, Any], context: dict[str, Any], why: str
) -> None:
    provider = install()  # an empty script: a judge call would fail the test

    result = asserts.faithfulness(output, context)

    assert (result["pass"], result["score"]) == (True, 1.0)  # promptfoo's placeholder, as in 4.05
    assert (result["not_applicable"], result["errored"]) == (True, False)
    assert result["reason"].startswith("N/A: ")
    assert why in result["reason"]
    assert "judge" not in result
    assert provider.calls == []


# --- Faithfulness: scored ------------------------------------------------------------------------


def test_the_score_is_supported_claims_over_claims(install: Install) -> None:
    claims = [claim("A.", 11), claim("B.", 12), claim("C.", 11, 12)]
    install(SUPPORTED, UNSUPPORTED, SUPPORTED)

    result = asserts.faithfulness(answer(), case(claims=claims))

    assert result["score"] == pytest.approx(2 / 3)
    assert (result["pass"], result["not_applicable"], result["errored"]) == (False, False, False)
    assert result["reason"] == "2 of 3 claim(s) supported; not supported: claim 2"
    judge = result["judge"]
    assert (judge["n_claims"], judge["n_supported"], judge["n_errored"]) == (3, 2, 0)
    assert judge["error"] is None


def test_every_claim_supported_is_a_perfect_score(install: Install) -> None:
    install(SUPPORTED, SUPPORTED)

    result = asserts.faithfulness(answer(), case())

    assert (result["pass"], result["score"]) == (True, 1.0)
    assert result["reason"] == "2 of 2 claim(s) supported"


def test_no_claim_supported_is_zero_not_not_applicable(install: Install) -> None:
    install(UNSUPPORTED, UNSUPPORTED)

    result = asserts.faithfulness(answer(), case())

    assert (result["pass"], result["score"]) == (False, 0.0)
    assert (result["not_applicable"], result["errored"]) == (False, False)


def test_each_claim_is_judged_against_the_chunks_it_cites_and_nothing_else(
    install: Install,
) -> None:
    provider = install(SUPPORTED, SUPPORTED)

    asserts.faithfulness(answer(), case(claims=[claim("A.", 11), claim("B.", 12, 11)]))

    first, second = provider.calls
    assert 'id="c1"' in first.user
    assert 'id="c2"' not in first.user
    assert [first.user.count("<source "), second.user.count("<source ")] == [1, 2]
    assert second.user.index('id="c2"') < second.user.index('id="c1"')  # citation order
    assert first.system == load_judge_faithfulness_prompt().system


def test_the_per_claim_verdicts_are_kept_with_the_servers_confidence(install: Install) -> None:
    install(SUPPORTED, UNSUPPORTED)

    judge = asserts.faithfulness(answer(), case())["judge"]

    prompt_version = load_judge_faithfulness_prompt().version
    assert (judge["metric"], judge["prompt_version"]) == ("faithfulness", prompt_version)
    assert (judge["judge_provider"], judge["judge_model"]) == ("groq", GROQ_MODEL)
    assert judge["usage"] == {"input_tokens": 2400, "output_tokens": 180, "thinking_tokens": 80}
    assert judge["claims"] == [
        {
            "claim_index": 0,
            "claim": "Claim A.",
            "cited_labels": ["c1"],
            "confidence": 0.9,
            "verdict": "SUPPORTED",
            "reason": "c1 says so.",
            "decided_locally": False,
            "cache_hit": False,
            "attempts": 1,
            "usage": {"input_tokens": 1200, "output_tokens": 90, "thinking_tokens": 40},
            "error": None,
        },
        {
            "claim_index": 1,
            "claim": "Claim B.",
            "cited_labels": ["c2"],
            "confidence": 0.4,
            "verdict": "NOT_SUPPORTED",
            "reason": "The 422 part is missing.",
            "decided_locally": False,
            "cache_hit": False,
            "attempts": 1,
            "usage": {"input_tokens": 1200, "output_tokens": 90, "thinking_tokens": 40},
            "error": None,
        },
    ]


def test_a_claim_without_a_valid_source_is_not_supported_and_costs_no_call(
    install: Install,
) -> None:
    provider = install(SUPPORTED)

    result = asserts.faithfulness(answer(), case(claims=[claim("A.", 11), claim("Orphan.")]))

    assert len(provider.calls) == 1
    orphan = result["judge"]["claims"][1]
    assert (orphan["verdict"], orphan["reason"]) == ("NOT_SUPPORTED", NO_SOURCE_REASON)
    assert (orphan["decided_locally"], orphan["attempts"], orphan["cited_labels"]) == (True, 0, [])
    assert result["score"] == 0.5  # a local verdict is a verdict: it counts in the denominator


def test_a_replayed_verdict_is_marked_and_counts_the_original_usage(install: Install) -> None:
    install(SUPPORTED, SUPPORTED)
    asserts.faithfulness(answer(), case())

    install()  # a new process over the same cache file, with an empty script: the cache must answer
    again = asserts.faithfulness(answer(), case())

    assert [c["cache_hit"] for c in again["judge"]["claims"]] == [True, True]
    assert again["judge"]["usage"]["input_tokens"] == 2400
    assert again["score"] == 1.0


def test_the_judge_call_takes_the_eval_timeout(install: Install) -> None:
    provider = install(SUPPORTED, SUPPORTED)

    asserts.faithfulness(answer(), case())

    assert {call.timeout_s for call in provider.calls} == {40.0}
    assert {call.temperature for call in provider.calls} == {0.0}


# --- Faithfulness: errored -----------------------------------------------------------------------


def test_a_claim_the_judge_cannot_grade_makes_the_case_errored_not_zero(install: Install) -> None:
    # Claim A: invalid output twice (the one retry). Claim B is still judged.
    install("not json", "still not json", SUPPORTED)

    result = asserts.faithfulness(answer(), case())

    assert (result["errored"], result["not_applicable"], result["pass"]) == (True, False, False)
    assert result["score"] == 0.0  # a placeholder: the component has no score
    error = result["judge"]["error"]
    assert (error["kind"], error["provider_side"], error["is_quota"]) == (
        "ProviderBadOutput",
        False,
        False,
    )
    assert result["reason"].startswith("judge error (ProviderBadOutput): ")
    first, second = result["judge"]["claims"]
    assert (first["verdict"], first["error"]["kind"], first["attempts"]) == (
        None,
        "ProviderBadOutput",
        2,
    )
    assert second["verdict"] == "SUPPORTED"  # what was judged is kept
    assert (result["judge"]["n_supported"], result["judge"]["n_errored"]) == (1, 1)


def test_a_provider_side_error_names_the_case_over_a_bad_output(install: Install) -> None:
    install("not json", "still not json", ProviderTimeout("slow"))

    result = asserts.faithfulness(answer(), case())

    error = result["judge"]["error"]
    assert (error["kind"], error["provider_side"]) == ("ProviderTimeout", True)
    assert result["errored"] is True


def test_after_a_provider_failure_the_other_claims_of_the_answer_are_not_asked(
    install: Install,
) -> None:
    provider = install(ProviderTimeout("slow"))  # one step: a second call would fail the test

    result = asserts.faithfulness(answer(), case())

    assert len(provider.calls) == 1
    assert [c["error"]["kind"] for c in result["judge"]["claims"]] == [
        "ProviderTimeout",
        "ProviderTimeout",
    ]
    assert result["judge"]["n_errored"] == 2


# --- Correctness ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("verdict", "score", "passed"),
    [("CORRECT", 1.0, True), ("PARTIALLY_CORRECT", 0.5, False), ("INCORRECT", 0.0, False)],
)
def test_correctness_maps_the_grade_to_one_half_or_zero(
    install: Install, verdict: str, score: float, passed: bool
) -> None:
    install(CorrectnessVerdict(verdict=verdict, reason="Because."))  # type: ignore[arg-type]

    result = asserts.correctness(answer(), case())

    assert (result["score"], result["pass"]) == (score, passed)
    assert (result["not_applicable"], result["errored"]) == (False, False)
    assert result["reason"] == f"{verdict}: Because."
    assert result["judge"]["verdict"] == verdict


def test_correctness_sends_the_question_the_reference_and_the_answer(install: Install) -> None:
    provider = install(CorrectnessVerdict(verdict="CORRECT", reason="Same."))

    result = asserts.correctness(answer(), case())

    [call] = provider.calls
    assert call.schema is CorrectnessVerdict
    assert "How do I return 201?" in call.user
    assert "Pass status_code=201." in call.user
    assert "Use status_code=201 in the decorator. [1]" in call.user
    assert call.system == load_judge_correctness_prompt().system
    judge = result["judge"]
    assert (judge["metric"], judge["prompt_version"]) == (
        "correctness",
        load_judge_correctness_prompt().version,
    )
    assert (judge["attempts"], judge["cache_hit"]) == (1, False)
    assert judge["usage"] == {"input_tokens": 1200, "output_tokens": 90, "thinking_tokens": 40}
    assert judge["error"] is None


@pytest.mark.parametrize("mode", ["no_rag", "hybrid"])
def test_correctness_is_judged_in_both_configs(install: Install, mode: str) -> None:
    provider = install(CorrectnessVerdict(verdict="INCORRECT", reason="Wrong."))

    result = asserts.correctness(answer(), case(mode=mode))

    assert len(provider.calls) == 1
    assert (result["score"], result["not_applicable"]) == (0.0, False)
    assert result["errored"] is False  # a score of 0 is a score, not an error


def test_correctness_is_judged_for_a_refusal_too(install: Install) -> None:
    provider = install(CorrectnessVerdict(verdict="CORRECT", reason="Both say it is not covered."))
    context = case(claims=[])
    context["test"]["metadata"]["golden"]["answerable"] = False
    context["test"]["metadata"]["golden"]["reference_answer"] = "The docs do not cover this."

    result = asserts.correctness(answer("insufficient_context"), context)

    assert len(provider.calls) == 1
    assert "The docs do not cover this." in provider.calls[0].user
    assert (result["score"], result["errored"]) == (1.0, False)


def test_a_correctness_judge_failure_is_errored_not_zero(install: Install) -> None:
    install(ProviderTimeout("slow"))

    result = asserts.correctness(answer(), case())

    assert (result["errored"], result["not_applicable"], result["pass"]) == (True, False, False)
    assert result["judge"]["verdict"] is None
    assert result["judge"]["error"] == {
        "kind": "ProviderTimeout",
        "is_quota": False,
        "provider_side": True,
        "detail": "slow",
    }


def test_invalid_correctness_output_is_unscored_and_not_provider_side(install: Install) -> None:
    install("nope", "nope again")

    result = asserts.correctness(answer(), case())

    assert result["errored"] is True
    assert result["judge"]["error"]["kind"] == "ProviderBadOutput"
    assert result["judge"]["error"]["provider_side"] is False
    assert result["judge"]["attempts"] == 2


# --- A stop is shared by the processes of one run ------------------------------------------------


def quota() -> ProviderRateLimited:
    return ProviderRateLimited("daily quota", retry_after_s=3600, is_quota=True)


def test_a_daily_quota_stops_the_judging_of_the_run(install: Install) -> None:
    provider = install(quota())  # one step: any later call would fail the test

    first = asserts.correctness(answer(), case())
    second = asserts.correctness(answer(), case())
    third = asserts.faithfulness(answer(), case())

    assert len(provider.calls) == 1
    assert first["judge"]["error"]["is_quota"] is True
    for skipped in (second, third):
        assert skipped["errored"] is True
        error = skipped["judge"]["error"]
        assert (error["kind"], error["is_quota"], error["provider_side"]) == (
            "ProviderRateLimited",
            True,
            True,
        )
        assert error["detail"] == (
            "skipped: not asked, the run stopped on an earlier ProviderRateLimited"
        )
        assert skipped["judge"]["prompt_version"] is None


def test_the_quota_found_by_a_faithfulness_claim_stops_the_rest_of_the_run(
    install: Install,
) -> None:
    provider = install(quota())

    asserts.faithfulness(answer(), case())
    later = asserts.correctness(answer(), case())

    assert len(provider.calls) == 1
    assert later["judge"]["error"]["is_quota"] is True


def test_a_stop_belongs_to_its_run(install: Install) -> None:
    provider = install(quota(), CorrectnessVerdict(verdict="CORRECT", reason="Same."))
    asserts.correctness(answer(), case(run_id="old-run"))

    result = asserts.correctness(answer(), case(run_id="new-run"))

    assert len(provider.calls) == 2
    assert (result["errored"], result["score"]) == (False, 1.0)


def test_a_per_minute_limit_that_the_backoff_gave_up_on_is_not_a_stop(install: Install) -> None:
    # A Retry-After of 600 s is over the eval backoff's total bound (120 s): it gives up at once,
    # without sleeping, and the call fails as BackoffExhaustedError.
    provider = install(
        ProviderRateLimited("per-minute limit", retry_after_s=600, is_quota=False),
        CorrectnessVerdict(verdict="CORRECT", reason="Same."),
    )

    first = asserts.correctness(answer(), case())
    second = asserts.correctness(answer(), case())

    error = first["judge"]["error"]
    assert (error["kind"], error["is_quota"], error["provider_side"]) == (
        "BackoffExhaustedError",
        False,
        True,
    )
    assert second["errored"] is False
    assert len(provider.calls) == 2


def test_a_rejected_key_is_errored_and_stops_the_run(install: Install) -> None:
    provider = install(ProviderRequestRejected("401 bad key", status_code=401))

    first = asserts.correctness(answer(), case())
    second = asserts.faithfulness(answer(), case())

    assert len(provider.calls) == 1
    for result in (first, second):
        error = result["judge"]["error"]
        assert (error["kind"], error["provider_side"], error["is_quota"]) == (
            "ProviderRequestRejected",
            False,
            False,
        )
        assert result["errored"] is True


def test_a_missing_judge_configuration_is_errored_with_its_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = make_settings(**EVAL_SETTINGS, cache_dir=tmp_path, judge_model=GROQ_MODEL)
    monkeypatch.setattr(promptfoo_judge, "get_settings", lambda: settings)
    monkeypatch.setattr(promptfoo_judge, "configure_logging", lambda _level: None)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    result = asserts.correctness(answer(), case())  # no GROQ_API_KEY: the real open_judge

    assert result["errored"] is True
    error = result["judge"]["error"]
    assert (error["kind"], error["provider_side"]) == ("ProviderConfigError", False)
    assert "GROQ_API_KEY" in error["detail"]


def test_outside_eval_mode_the_assertion_refuses_loudly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = make_settings(cache_dir=tmp_path)  # app_env=test
    monkeypatch.setattr(promptfoo_judge, "get_settings", lambda: settings)

    with pytest.raises(ProviderConfigError, match="eval mode"):
        asserts.correctness(answer(), case())


def test_two_judge_sessions_do_not_overlap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(promptfoo_judge, "LOCK_TIMEOUT_S", 0.1)
    state = tmp_path / "judge_run.sqlite"

    turn = promptfoo_judge._turn  # pyright: ignore[reportPrivateUsage]

    with (
        turn(state, RUN),
        pytest.raises(sqlite3.OperationalError, match="locked"),
        turn(state, RUN),
    ):
        pytest.fail("a second session must wait for the first")

    with turn(state, RUN) as second:
        assert second.stopped is None  # released when the first ended


# --- The fake switch -----------------------------------------------------------------------------


def test_a_fake_run_has_a_stub_judge_and_touches_neither_the_network_nor_the_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = make_settings(
        **{**EVAL_SETTINGS, "generator_providers": ["fake"]}, cache_dir=tmp_path
    )
    monkeypatch.setattr(promptfoo_judge, "get_settings", lambda: settings)
    monkeypatch.setattr(promptfoo_judge, "configure_logging", lambda _level: None)

    faithfulness = asserts.faithfulness(answer(), case())
    correctness = asserts.correctness(answer(), case())

    assert (faithfulness["score"], faithfulness["errored"]) == (1.0, False)
    assert (correctness["score"], correctness["errored"]) == (0.5, False)
    assert faithfulness["judge"]["judge_provider"] == "fake-judge"
    assert not (tmp_path / LLM_EVAL_CACHE_FILE).exists()  # canned verdicts are never cached


# --- Malformed input -----------------------------------------------------------------------------


def malformed(result: dict[str, Any]) -> bool:
    error = result["judge"]["error"]
    return (
        result["errored"] is True
        and result["pass"] is False
        and error["kind"] == "MalformedInput"
        and error["provider_side"] is False
        and error["detail"].startswith("malformed input: ")
    )


def test_malformed_input_is_an_errored_component_not_a_score(install: Install) -> None:
    provider = install()
    no_run_id = case(run_id=None)
    unknown_chunk = case(claims=[claim("A.", 99)])
    no_claims_key = case()
    del no_claims_key["providerResponse"]["metadata"]["claims"]
    no_confidence = case(claims=[{"text": "A.", "chunk_ids": [11]}])

    assert malformed(asserts.faithfulness(answer(), no_run_id))
    assert malformed(asserts.faithfulness(answer(), unknown_chunk))
    assert malformed(asserts.faithfulness(answer(), no_claims_key))
    assert malformed(asserts.faithfulness(answer(), no_confidence))
    assert malformed(asserts.faithfulness("not json", case()))
    assert malformed(asserts.correctness(answer(), no_run_id))
    assert malformed(asserts.correctness({"status": "answered"}, case()))
    assert malformed(asserts.correctness(answer(), {"vars": {}, "test": case()["test"]}))
    assert provider.calls == []


# --- Weight --------------------------------------------------------------------------------------


def test_a_not_applicable_case_never_loads_the_judge() -> None:
    """Every assertion call is a new Python process (Tech §15.3): N/A and malformed cases must not
    pay for the judge's imports."""
    probe = (
        "import sys; import grounded.evals.promptfoo_asserts as a; "
        "ctx = {'providerResponse': {'metadata': {'mode': 'no_rag'}}}; "
        "r = a.faithfulness({'status': 'answered'}, ctx); "
        "bad = a.correctness({}, {}); "
        "heavy = [m for m in ('pydantic', 'psycopg', 'google', 'groq', 'grounded.settings', "
        "'grounded.runtime', 'grounded.evals.promptfoo_judge', 'grounded.evals.judge') "
        "if m in sys.modules]; "
        "print(r['not_applicable'], bad['errored'], ','.join(heavy))"
    )
    done = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True, timeout=60
    )

    assert done.stdout.strip() == "True True"
