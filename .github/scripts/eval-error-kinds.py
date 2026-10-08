"""Count the errors of a promptfoo generation run by kind, for the PR comment (Tech.md section 17).

    python3 -I eval-error-kinds.py eval/results/generation.json

The gate says how many cases errored for provider reasons, not what the reason was, and a reader of
an `inconclusive` comment wants to know whether it was the quota (nothing to fix, wait for the reset)
or a 5xx or a timeout of the provider. Prints nothing when there is no error, and a Markdown line per
stage otherwise:

    - generator: `ProviderUnavailable` x24, `ProviderTimeout` x9
    - judge: `ProviderRateLimited` (daily quota) x3

Standard library only, because it runs on the bare runner. It reads promptfoo's file the way
Tech.md section 15.3 documents it (`results[].failureReason`, the tagged `error`, and the judge
components with `errored` and `judge.error`), and it only counts: the gate decides what the errors
mean, and a file it cannot read gives no output rather than an error (the gate reports that itself).
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

TAG = re.compile(r"^\[(?P<kind>\w+)(?: quota=(?P<quota>true|false))?\]")


def generator_kind(row: dict[str, Any]) -> str:
    """`Kind` or `Kind (daily quota)`, from the row's tagged error."""
    match = TAG.match(str(row.get("error") or ""))
    if match is None:
        return "an untagged error"
    return f"{match['kind']} (daily quota)" if match["quota"] == "true" else match["kind"]


def judge_kind(error: dict[str, Any]) -> str:
    kind = str(error.get("kind") or "an unnamed error")
    return f"{kind} (daily quota)" if error.get("is_quota") else kind


def count(document: dict[str, Any]) -> dict[str, Counter[str]]:
    stages: dict[str, Counter[str]] = {"generator": Counter(), "judge": Counter()}
    for row in document["results"]["results"]:
        if row.get("failureReason") == 2:
            stages["generator"][generator_kind(row)] += 1
        for component in (row.get("gradingResult") or {}).get("componentResults") or []:
            error = (component.get("judge") or {}).get("error")
            if component.get("errored") and isinstance(error, dict):
                stages["judge"][judge_kind(error)] += 1
    return stages


def main(path: str) -> None:
    try:
        stages = count(json.loads(Path(path).read_bytes()))
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return
    for stage, kinds in stages.items():
        if kinds:
            listed = ", ".join(f"`{kind}` x{n}" for kind, n in kinds.most_common())
            print(f"- {stage}: {listed}")


if __name__ == "__main__":
    main(sys.argv[1])
