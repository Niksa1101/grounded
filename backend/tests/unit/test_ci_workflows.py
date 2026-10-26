"""Structure of the CI workflows that guard a production write and a required check.

``eval.yml`` can insert into the production ``eval_runs`` with the owner connection
(``DATABASE_URL_DIRECT``; ticket 4.10c). That write ships disabled and needs the Author's go-ahead,
so what these tests pin is not YAML style but the guards: the secret is referenced by one step
only, that step runs on a push to ``main`` and only when the repository variable
``EVAL_RECORD_RUNS`` is ``true`` (unset it is empty, so the step is skipped), and no pull request
run can reach it.

``ingest.yml`` (ticket 5.09) is the other workflow that can write to production, by a manual
dispatch from ``main``: the last section of this file pins who may start it, which steps see the
secret, that user input never reaches a script inline, and that it restores the same caches as
``warm-cache.yml``.

The same file's ``eval`` job is a required status check of ``main`` (Tech §17), which only works
if every PR gets an ``eval`` status and a red one cannot be replaced by a green skipped one: the
second half of the file pins the triggers, the job's ``if`` and the job name. The files are read as
text: PyYAML is not a dependency of the project.

``housekeeping.yml`` (ticket 5.11), the third workflow that writes to production, is pinned in
``test_housekeeping_workflow.py``, which reuses the helpers above; only the list of steps that may
see the secret lives here, with the other two workflows'.

``migrate.yml`` (ticket 8.01), the fourth, applies pending migrations to Neon by a manual dispatch
from ``main`` and does nothing else: the last section pins its trigger, its guard (the one of
``ingest.yml``), the two steps that see the secret and that ``grounded migrate`` is its one command.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
WORKFLOWS = ROOT / ".github" / "workflows"
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


def step_title(block: str) -> str:
    match = re.search(r"(?m)^      - name: (.*)$", block)
    assert match is not None, "the step has no name"
    return match.group(1)


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


INGEST_STEPS_WITH_THE_SECRET = {
    "Guard the production connection",
    "Migrate Neon",
    "Ingest",
    "Index versions",
}


HOUSEKEEPING_STEPS_WITH_THE_SECRET = {"Guard the production connection", "Retention"}


MIGRATE_STEPS_WITH_THE_SECRET = {"Guard the production connection", "Migrate Neon"}


@pytest.mark.parametrize("workflow", sorted(p.name for p in WORKFLOWS.glob("*.yml")))
def test_the_production_secret_is_referenced_only_by_the_steps_that_need_it(
    workflow: str,
) -> None:
    holders = [b for b in steps(workflow) if SECRET in b]

    if workflow == "eval.yml":
        assert len(holders) == 1
        assert f"- name: {RECORD_STEP}" in holders[0]
    elif workflow == "housekeeping.yml":
        # Retention is the second workflow made to write to production: the guard and the command.
        assert {step_title(b) for b in holders} == HOUSEKEEPING_STEPS_WITH_THE_SECRET
        assert all(f"{SECRET}: ${{{{ secrets.{SECRET} }}}}" in b for b in holders)
    elif workflow == "ingest.yml":
        # The one workflow made to write to production: four steps, each with its own env entry.
        assert {step_title(b) for b in holders} == INGEST_STEPS_WITH_THE_SECRET
        assert all(f"{SECRET}: ${{{{ secrets.{SECRET} }}}}" in b for b in holders)
    elif workflow == "migrate.yml":
        # Migrations only: the guard and the command.
        assert {step_title(b) for b in holders} == MIGRATE_STEPS_WITH_THE_SECRET
        assert all(f"{SECRET}: ${{{{ secrets.{SECRET} }}}}" in b for b in holders)
    else:
        assert holders == []  # a PR workflow, the cache seeding, the retrieval eval: none of them


@pytest.mark.parametrize("workflow", sorted(p.name for p in WORKFLOWS.glob("*.yml")))
def test_the_proxy_secret_reaches_only_the_keep_warm_step(workflow: str) -> None:
    # /readyz needs it (PRD D54c); no other step of any workflow may see it.
    holders = [b for b in steps(workflow) if "PROXY_SHARED_SECRET" in b]

    if workflow == "housekeeping.yml":
        assert [step_title(b) for b in holders] == ["Wake the backend and the database"]
        assert "PROXY_SHARED_SECRET: ${{ secrets.PROXY_SHARED_SECRET }}" in holders[0]
    else:
        assert holders == []


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


# --- ingest.yml: the manual write to production Neon (ticket 5.09) --------------------------------


def run_blocks(name: str) -> list[str]:
    """The script text of every ``run:`` step, block scalars and one-liners alike."""
    lines = code(name).splitlines()
    blocks: list[str] = []
    for i, line in enumerate(lines):
        match = re.match(r"^(\s*)run: (.*)$", line)
        if match is None:
            continue
        indent, rest = len(match.group(1)), match.group(2)
        if rest.strip() not in {"|", ">", "|-", ">-"}:
            blocks.append(rest)
            continue
        body: list[str] = []
        for following in lines[i + 1 :]:
            if following.strip() and len(following) - len(following.lstrip()) <= indent:
                break
            body.append(following)
        blocks.append("\n".join(body))
    return blocks


def test_ingest_starts_only_from_a_manual_dispatch_on_main() -> None:
    text = code("ingest.yml")
    triggers = text.split("\njobs:\n", 1)[0]

    assert re.search(r"(?m)^on:\n  workflow_dispatch:", triggers)
    for other in ("pull_request", "push", "schedule", "workflow_run", "workflow_call"):
        assert not re.search(rf"(?m)^  {other}:", triggers)
    assert "if: github.event_name == 'workflow_dispatch' && github.ref == 'refs/heads/main'" in text


def test_ingest_activates_only_on_request_and_never_overlaps_itself() -> None:
    text = code("ingest.yml")

    assert re.search(
        r"(?m)^      activate:\n(?:        .*\n)*?        default: false\n(?:        .*\n)*?"
        r"        type: boolean",
        text,
    )
    assert "group: neon-production-writes\n  cancel-in-progress: false" in text
    ingest = step_named("ingest.yml", "Ingest")
    assert 'if [ "$ACTIVATE" = "true" ]; then' in ingest  # the flag is conditional, never fixed
    assert re.findall(r"--activate", ingest) == ["--activate"]


def test_user_input_reaches_a_script_only_through_env() -> None:
    for script in run_blocks("ingest.yml"):
        assert "${{" not in script, script  # no inputs.*, steps.*, secrets.* inside a script
    mentions = [ln for ln in code("ingest.yml").splitlines() if "inputs.ref" in ln]
    assert mentions == ["          INPUT_REF: ${{ inputs.ref }}"]
    validate = step_named("ingest.yml", "Validate the inputs")
    assert "=~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$" in validate


def test_ingest_never_prints_a_secret_and_pins_every_action() -> None:
    for script in run_blocks("ingest.yml"):
        for line in script.splitlines():
            if re.search(r"\b(echo|printf|tee)\b", line):
                assert not re.search(r"\$\{?(DATABASE_URL_DIRECT|GEMINI_API_KEY|url)\b", line), line
    uses = re.findall(r"(?m)^\s+(?:- )?uses: (\S+)", code("ingest.yml"))
    assert uses
    for action in uses:
        assert action.startswith("./") or re.fullmatch(r".+@[0-9a-f]{40}", action), action


def test_ingest_checks_the_connection_first_and_migrates_before_it_ingests() -> None:
    names = re.findall(r"(?m)^      - name: (.*)$", "".join(steps("ingest.yml")))
    guard = step_named("ingest.yml", "Guard the production connection")

    assert names.index("Guard the production connection") < names.index("Restore the corpus clone")
    assert names.index("Preflight, texts still to embed") < names.index("Migrate Neon")
    assert names.index("Migrate Neon") < names.index("Ingest") < names.index("Index versions")
    for refusal in ('-z "$url"', "*-pooler*", "localhost"):
        assert refusal in guard
    assert "::add-mask::$host" in guard


def test_ingest_uses_the_cache_keys_of_warm_cache_yml() -> None:
    ingest, warm = code("ingest.yml"), code("warm-cache.yml")
    embeddings_key = (
        "embeddings-${{ steps.cfg.outputs.embedding_model }}-"
        "${{ steps.cfg.outputs.embedding_dim }}-${{ github.run_id }}-${{ github.run_attempt }}"
    )

    for text in (ingest, warm):
        assert f"path: .cache/embeddings.sqlite\n          key: {embeddings_key}" in text
        assert (
            "restore-keys: |\n            embeddings-${{ steps.cfg.outputs.embedding_model }}"
            in text
        )
        assert "path: .cache/corpus\n          key: corpus-${{ steps." in text
    # The embeddings are saved by the shared action, which derives the same key.
    assert "uses: ./.github/actions/save-embeddings" in ingest
    save = (ROOT / ".github" / "actions" / "save-embeddings" / "action.yml").read_text(
        encoding="utf-8"
    )
    assert embeddings_key in save


# --- migrate.yml: pending migrations to production Neon, nothing else (ticket 8.01) ---------------

MIGRATE = "migrate.yml"


def test_migrate_starts_only_from_a_manual_dispatch_on_main() -> None:
    text = code(MIGRATE)
    triggers = text.split("\njobs:\n", 1)[0]

    assert re.search(r"(?m)^on:\n  workflow_dispatch:$", triggers)
    assert "inputs:" not in triggers  # nothing to choose: it applies what is pending
    for other in ("pull_request", "push", "schedule", "workflow_run", "workflow_call"):
        assert not re.search(rf"(?m)^  {other}:", triggers)
    assert "if: github.event_name == 'workflow_dispatch' && github.ref == 'refs/heads/main'" in text


def test_migrate_has_read_only_permissions_and_never_overlaps_itself() -> None:
    text = code(MIGRATE)

    assert re.search(r"(?m)^permissions:\n  contents: read$", text)
    assert not re.search(r"(?m)^\s+[\w-]+: write$", text)  # no write permission anywhere
    assert "group: neon-production-writes\n  cancel-in-progress: false" in text
    # The optional approval gate of ingest.yml: a manual production write may wait for a click.
    assert re.search(r"(?m)^    environment: neon-production$", text)


def test_migrate_and_ingest_share_one_concurrency_group_so_writes_never_overlap() -> None:
    pattern = r"(?m)^concurrency:\n  group: (.+)\n  cancel-in-progress: (.+)$"

    groups = {name: re.findall(pattern, code(name)) for name in ("ingest.yml", MIGRATE)}

    assert groups == {
        "ingest.yml": [("neon-production-writes", "false")],
        MIGRATE: [("neon-production-writes", "false")],
    }


def test_migrate_guards_the_connection_like_ingest_before_anything_is_installed() -> None:
    title = "Guard the production connection"
    names = re.findall(r"(?m)^      - name: (.*)$", "".join(steps(MIGRATE)))
    guard = step_named(MIGRATE, title)

    assert guard == step_named("ingest.yml", title)
    for refusal in ('-z "$url"', "postgresql://*", "*-pooler*", "localhost"):
        assert refusal in guard
    assert "::add-mask::$host" in guard
    assert names == [title, "Install dependencies", "Migrate Neon"]


def test_migrate_runs_grounded_migrate_and_no_other_command() -> None:
    commands = re.findall(r"\bgrounded (\S+)", "\n".join(run_blocks(MIGRATE)))
    block = step_named(MIGRATE, "Migrate Neon")

    assert commands == ["migrate"]
    assert "set -o pipefail" in block  # tee must not hide a failed migration
    assert '>> "$GITHUB_STEP_SUMMARY"' in block
    assert code(MIGRATE).count(f"secrets.{SECRET}") == 2  # the guard and the command
    assert set(re.findall(r"secrets\.([A-Z_]+)", code(MIGRATE))) == {SECRET}


def test_migrate_never_prints_a_secret_and_pins_every_action() -> None:
    for script in run_blocks(MIGRATE):
        assert "${{" not in script, script
        for line in script.splitlines():
            if re.search(r"\b(echo|printf|tee)\b", line):
                assert not re.search(r"\$\{?(DATABASE_URL_DIRECT|url)\b", line), line
    uses = re.findall(r"(?m)^\s+(?:- )?uses: (\S+)", code(MIGRATE))
    assert uses
    for action in uses:
        assert re.fullmatch(r".+@[0-9a-f]{40}", action), action
