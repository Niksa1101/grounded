"""Structure of the CI workflows that guard a production write and a required check.

``eval.yml`` can insert into the production ``eval_runs`` with the owner connection
(``DATABASE_URL_DIRECT``; ticket 4.10c). That write ships disabled and needs the Author's go-ahead,
so what these tests pin is not YAML style but the guards: the secret is referenced by one step
only, that step runs on a push to ``main`` and only when the repository variable
``EVAL_RECORD_RUNS`` is ``true`` (unset it is empty, so the step is skipped), and no pull request
run can reach it.

The same file's ``eval`` job is a required status check of ``main`` (Tech §17), which only works
if every PR gets an ``eval`` status and a red one cannot be replaced by a green skipped one: the
second half of the file pins the triggers, the job's ``if`` and the job name. The files are read as
text: PyYAML is not a dependency of the project.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

WORKFLOWS = Path(__file__).resolve().parents[3] / ".github" / "workflows"
SECRET = "DATABASE_URL_DIRECT"
RECORD_STEP = "Record the run in eval_runs"


def code(name: str) -> str:
    """A workflow file without its comment lines: a comment that names the secret is not a use."""
    text = (WORKFLOWS / name).read_text(encoding="utf-8")
    return "".join(
        line for line in text.splitlines(keepends=True) if not line.lstrip().startswith("#")
    )


def steps(name: str) -> list[str]:
    """The text of each step of a workflow file: from one ``      - `` line (the indentation of a
    job's steps) to the next. The preamble before the first step is not a step."""
    blocks = re.split(r"(?m)^(?=      - )", code(name))
    return [block for block in blocks if block.startswith("      - ")]


def step_named(name: str, title: str) -> str:
    (block,) = [b for b in steps(name) if re.search(rf"(?m)^      - name: {re.escape(title)}$", b)]
    return block


def step_condition(block: str) -> str:
    """The step's ``if:`` expression, a plain or a folded (``>-``) scalar, on one line."""
    match = re.search(r"(?m)^        if: (?:>-\n((?:          .*\n?)+)|(.+))$", block)
    assert match is not None, "the step has no if:"
    return " ".join((match.group(1) or match.group(2)).split())


def job_condition(name: str) -> str:
    """The job's own ``if:`` (a folded scalar at the indentation of a job's keys), on one line."""
    match = re.search(r"(?m)^    if: >-\n((?:      .*\n)+)", code(name))
    assert match is not None, "the job has no if:"
    return " ".join(match.group(1).split())


@pytest.mark.parametrize("workflow", sorted(p.name for p in WORKFLOWS.glob("*.yml")))
def test_the_production_secret_is_referenced_only_by_the_record_step_of_eval_yml(
    workflow: str,
) -> None:
    holders = [b for b in steps(workflow) if SECRET in b]

    if workflow == "eval.yml":
        assert len(holders) == 1
        assert f"- name: {RECORD_STEP}" in holders[0]
    else:
        assert holders == []  # a PR workflow, the cache seeding, the retrieval eval: none of them


def test_the_secret_is_not_in_the_job_environment_or_anywhere_else_in_eval_yml() -> None:
    block = step_named("eval.yml", RECORD_STEP)

    assert code("eval.yml").count(SECRET) == block.count(SECRET)  # every mention is in that step
    assert f"{SECRET}: ${{{{ secrets.{SECRET} }}}}" in block


def test_the_record_step_runs_only_on_a_push_to_main_and_only_when_the_variable_is_true() -> None:
    condition = step_condition(step_named("eval.yml", RECORD_STEP))

    assert "github.event_name == 'push'" in condition
    assert "github.ref == 'refs/heads/main'" in condition
    assert "vars.EVAL_RECORD_RUNS == 'true'" in condition
    # Only the gate's three verdicts are recorded: not "could not run", not "did not run".
    for status in ("pass", "fail", "inconclusive"):
        assert f"steps.report.outputs.status == '{status}'" in condition
    # Every part is required: the only "or" is between the three verdicts.
    assert condition.count("||") == 2


def test_the_record_step_writes_only_with_the_explicit_flag_and_comes_after_the_report() -> None:
    names = re.findall(r"(?m)^      - name: (.*)$", "".join(steps("eval.yml")))
    block = step_named("eval.yml", RECORD_STEP)

    assert "grounded eval record" in block
    assert "--write" in block  # without it the command is a dry run
    assert names.index("Report") < names.index(RECORD_STEP) < names.index("Verdict")


# --- `eval` as a required status check (Tech §17) -------------------------------------------------


def test_every_new_pr_gets_an_eval_run_and_removing_the_label_starts_nothing() -> None:
    match = re.search(r"(?m)^  pull_request:\n    types: \[(.*)\]$", code("eval.yml"))
    assert match is not None, "eval.yml has no pull_request types"

    types = {t.strip() for t in match.group(1).split(",")}

    # `opened` and `reopened` give a PR its status at once (a required check that is never
    # reported waits for ever); `unlabeled` would let removing the label touch a red status.
    assert types == {"opened", "reopened", "synchronize", "labeled"}


def test_the_job_name_is_eval_because_that_is_the_context_of_the_branch_protection() -> None:
    text = code("eval.yml")
    jobs = text.split("\njobs:\n", 1)[1]

    assert re.findall(r"(?m)^  ([\w-]+):$", jobs) == ["eval"]  # one job, and its id is the context
    assert not re.search(r"(?m)^    name:", jobs)  # a job `name:` would replace the id as context


def test_a_pr_event_runs_the_eval_whenever_the_label_is_on_the_pr_and_otherwise_skips_it() -> None:
    clauses = [clause.strip() for clause in job_condition("eval.yml").split("||")]

    assert clauses == [
        # push to main and workflow_dispatch: as before.
        "github.event_name != 'pull_request'",
        # The label is on the PR now: any event type, so an unrelated label added after a red run
        # re-runs the eval on that commit and cannot leave a green skipped `eval` in its place.
        "contains(github.event.pull_request.labels.*.name, 'run-eval')",
        # The label that was just added, in case the payload's list does not show it yet.
        "(github.event.action == 'labeled' && github.event.label.name == 'run-eval')",
    ]
    # Nothing else decides: no `synchronize` / `labeled` guard on the clause that checks the label.
    assert "github.event.action == 'synchronize'" not in " ".join(clauses)


def test_concurrency_is_on_the_job_so_a_skipped_run_cannot_cancel_an_eval_in_progress() -> None:
    text = code("eval.yml")

    assert not re.search(r"(?m)^concurrency:", text)  # a workflow-level group sees skipped runs
    group = "eval-${{ github.event.pull_request.number || github.ref }}"
    assert f"    concurrency:\n      group: {group}\n" in text
    # Only a PR run is cancelled by a newer one; main always finishes.
    assert "cancel-in-progress: ${{ github.event_name == 'pull_request' }}" in text
