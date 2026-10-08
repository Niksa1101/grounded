"""The files promptfoo loads (eval/promptfoo/) are thin shims over tested package code. These tests
check that the wiring is whole: everything the config names exists, the shims re-export the package
functions, and the settings the provider shim writes are the eval ones."""

from __future__ import annotations

import importlib.util
import os
import re
from pathlib import Path
from types import ModuleType

import pytest

from grounded.evals import promptfoo_asserts, promptfoo_provider
from grounded.evals.golden import GOLDEN_DIR, load_golden_set

PROMPTFOO_DIR = Path(__file__).resolve().parents[3] / "eval" / "promptfoo"
CONFIG = (PROMPTFOO_DIR / "promptfooconfig.yaml").read_text(encoding="utf-8")


def load_shim(name: str, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Import ``eval/promptfoo/<name>.py`` under a private module name, with the environment the
    provider shim writes restored afterwards."""
    for variable in ("APP_ENV", "GENERATOR_PROVIDERS"):
        monkeypatch.setenv(variable, "placeholder")  # records the original for the teardown
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.delenv("GENERATOR_PROVIDERS")
    spec = importlib.util.spec_from_file_location(
        f"promptfoo_shim_{name}", PROMPTFOO_DIR / f"{name}.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_everything_the_config_names_exists(monkeypatch: pytest.MonkeyPatch) -> None:
    references = re.findall(r"file://(\w+)\.py(?::(\w+))?", CONFIG)
    assert {module for module, _ in references} == {"provider", "tests_loader", "asserts"}

    for module_name, function in references:
        module = load_shim(module_name, monkeypatch)
        assert callable(getattr(module, function or "call_api")), (module_name, function)


def test_each_assertion_has_a_metric_label_equal_to_its_function_name() -> None:
    pairs = re.findall(r"value: file://asserts\.py:(\w+)\s+metric: (\w+)", CONFIG)

    assert [function for function, _ in pairs] == [
        "schema_first_try",
        "citation_validity",
        "refusal_correctness",
        "citation_precision",
        "faithfulness",
        "correctness",
    ]
    assert all(function == metric for function, metric in pairs)


def test_the_config_has_the_two_providers_the_gate_reads() -> None:
    assert re.findall(r"label: (\w+)\s+config:\s+mode: (\w+)", CONFIG) == [
        ("no_rag", "no_rag"),
        ("hybrid", "hybrid"),
    ]


def test_the_config_enforces_concurrency_one_and_no_promptfoo_cache() -> None:
    # The command line has both flags; the config must not depend on them being remembered.
    assert re.search(r"commandLineOptions:\s+maxConcurrency: 1\s+cache: false", CONFIG)


def test_the_asserts_shim_re_exports_the_package_functions(monkeypatch: pytest.MonkeyPatch) -> None:
    shim = load_shim("asserts", monkeypatch)

    for name in (
        "schema_first_try",
        "citation_validity",
        "refusal_correctness",
        "citation_precision",
        "faithfulness",
        "correctness",
    ):
        assert getattr(shim, name) is getattr(promptfoo_asserts, name)
    # The judge assertions read the settings in their own process: same eval environment as the
    # provider (APP_ENV forced, GENERATOR_PROVIDERS defaulted, a shell choice such as `fake` kept).
    assert os.environ["APP_ENV"] == "eval"
    assert os.environ["GENERATOR_PROVIDERS"] == "gemini"


def test_the_asserts_shim_keeps_a_generator_chosen_in_the_shell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for variable in ("APP_ENV", "GENERATOR_PROVIDERS"):
        monkeypatch.setenv(variable, "placeholder")
    monkeypatch.setenv("GENERATOR_PROVIDERS", "fake")
    spec = importlib.util.spec_from_file_location(
        "promptfoo_shim_asserts_fake", PROMPTFOO_DIR / "asserts.py"
    )
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(importlib.util.module_from_spec(spec))

    assert os.environ["GENERATOR_PROVIDERS"] == "fake"
    assert os.environ["APP_ENV"] == "eval"


def test_the_provider_shim_re_exports_call_api_and_sets_eval_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shim = load_shim("provider", monkeypatch)

    assert shim.call_api is promptfoo_provider.call_api
    assert os.environ["APP_ENV"] == "eval"
    assert os.environ["GENERATOR_PROVIDERS"] == "gemini"


def test_the_provider_shim_keeps_a_generator_chosen_in_the_shell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for variable in ("APP_ENV", "GENERATOR_PROVIDERS"):
        monkeypatch.setenv(variable, "placeholder")
    monkeypatch.setenv("GENERATOR_PROVIDERS", "fake")
    spec = importlib.util.spec_from_file_location(
        "promptfoo_shim_provider_fake", PROMPTFOO_DIR / "provider.py"
    )
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(importlib.util.module_from_spec(spec))

    assert os.environ["GENERATOR_PROVIDERS"] == "fake"
    assert os.environ["APP_ENV"] == "eval"


def test_the_loader_runs_the_whole_golden_set_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    shim = load_shim("tests_loader", monkeypatch)
    monkeypatch.delenv("EVAL_QUESTION_IDS", raising=False)

    tests = shim.generate_tests({"golden_set": "golden_set.v1.jsonl"})

    assert len(tests) == len(load_golden_set(GOLDEN_DIR / "golden_set.v1.jsonl"))


def test_the_loader_limits_a_run_to_eval_question_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    shim = load_shim("tests_loader", monkeypatch)
    monkeypatch.setenv("EVAL_QUESTION_IDS", "q045, q003")

    tests = shim.generate_tests({"golden_set": "golden_set.v1.jsonl"})

    assert [t["metadata"]["golden"]["id"] for t in tests] == ["q003", "q045"]


def test_the_loader_refuses_an_unknown_question_id(monkeypatch: pytest.MonkeyPatch) -> None:
    shim = load_shim("tests_loader", monkeypatch)
    monkeypatch.setenv("EVAL_QUESTION_IDS", "q003,q999")

    with pytest.raises(Exception, match="q999"):
        shim.generate_tests({"golden_set": "golden_set.v1.jsonl"})
