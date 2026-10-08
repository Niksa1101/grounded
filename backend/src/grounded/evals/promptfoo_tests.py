"""Test cases for the promptfoo run, built from the golden set (Tech §15.3, 4.05).

``eval/promptfoo/tests_loader.py`` is a shim over ``generate_tests`` here: promptfoo starts the
function itself (``tests: file://tests_loader.py:generate_tests``) and takes the returned list as
its test cases. Nothing is written to disk, so there is no generated file that could go stale.

One test case per golden item:

- ``vars`` holds only what the prompt needs (``question``). promptfoo prints every var as a column
  of its results table, so the labels and the reference answer stay out of it.
- ``metadata.golden`` carries what the assertions and the gate need and the provider must not see:
  the item's id, type, ``answerable``, its labelled sections with grades, the reference answer, and
  the golden-set version and sha256 (so a results file says which set it scored). promptfoo hands
  the test's ``metadata`` to every assertion (``context["test"]["metadata"]``) and copies it into
  the results row, so the assertion processes (a new Python process per call) never reload the file.

The golden file is only read (AGENTS.md §7). Which items run is the caller's choice: the full set by
default, or a list of ids for a cheap smoke run.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from grounded.evals.golden import GOLDEN_DIR, load_golden_set
from grounded.evals.retrieval_runner import golden_set_digest, golden_set_version
from grounded.schemas.eval import GoldenItem

DEFAULT_GOLDEN_SET: Final = "golden_set.v1.jsonl"

_ID_SEPARATORS: Final = re.compile(r"[,\s]+")


class PromptfooTestsError(Exception):
    """The test cases cannot be built: an unknown id, or a golden-set name that is not a file."""


def parse_question_ids(text: str | None) -> list[str] | None:
    """``"q003, q045 q012"`` -> ``["q003", "q045", "q012"]``; blank or ``None`` -> ``None`` (all)"""
    ids = [part for part in _ID_SEPARATORS.split(text or "") if part]
    return ids or None


def golden_case(item: GoldenItem, *, version: str, sha256: str) -> dict[str, Any]:
    """The ``metadata.golden`` payload of one test case (see the module docstring)."""
    return {
        "id": item.id,
        "type": item.type,
        "answerable": item.answerable,
        "relevant": item.relevant,
        "reference_answer": item.reference_answer,
        "golden_set_version": version,
        "golden_set_sha256": sha256,
    }


def build_tests(
    *, golden_set: str = DEFAULT_GOLDEN_SET, ids: Sequence[str] | None = None
) -> list[dict[str, Any]]:
    """The test cases for ``golden_set`` (a file name in ``eval/golden/``), in file order.

    ``ids`` limits the run to those items. An id that is not in the file raises instead of running
    fewer questions than asked for, so a typo in a smoke run cannot pass for a small run.
    """
    if Path(golden_set).name != golden_set:
        raise PromptfooTestsError(
            f"golden_set must be a file name in eval/golden/, got {golden_set!r}"
        )
    path = GOLDEN_DIR / golden_set
    version, sha256 = golden_set_version(path), golden_set_digest(path)
    items = load_golden_set(path)
    if ids is not None:
        unknown = sorted(set(ids) - {item.id for item in items})
        if unknown:
            raise PromptfooTestsError(f"{golden_set} has no item {', '.join(unknown)}")
        wanted = set(ids)
        items = [item for item in items if item.id in wanted]
    return [
        {
            "description": f"{item.id} ({item.type})",
            "vars": {"question": item.question},
            "metadata": {"golden": golden_case(item, version=version, sha256=sha256)},
        }
        for item in items
    ]


def generate_tests(config: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    """promptfoo's test generator. ``config`` is the YAML ``config:`` next to the function path:
    ``golden_set`` (file name, default ``DEFAULT_GOLDEN_SET``) and optionally ``ids``."""
    options = config or {}
    ids = options.get("ids")
    return build_tests(
        golden_set=str(options.get("golden_set", DEFAULT_GOLDEN_SET)),
        ids=None if ids is None else [str(i) for i in ids],
    )
