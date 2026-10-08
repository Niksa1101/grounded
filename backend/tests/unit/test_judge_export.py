"""The files of the judge-agreement sample (4.11a, Tech §15.4): the blind sheet and view, the key,
the judge's calls for the controls and the command that ties them together.

The run is the committed promptfoo recording (scripted models, so the verdicts mean nothing; the
mechanics are what is tested), with its rubric versions rewritten to the current ones so that a
future edit of a rubric does not break this file. The judge for the controls is a scripted
``FakeLLMProvider`` behind the real ``Judge``, or a stand-in for ``ask``: no test calls a model.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from typer.testing import CliRunner

import grounded.cli
from grounded.cli import app
from grounded.evals.agreement import parse_sheet, render_sheet
from grounded.evals.judge import Judge, JudgeConfig
from grounded.evals.judge_export import (
    ControlVerdicts,
    ask_controls,
    export,
    render_view,
    sources_text,
)
from grounded.evals.judge_sample import SampleError, SampleItem, SampleParams, Source
from grounded.generation.prompts import (
    load_judge_correctness_prompt,
    load_judge_faithfulness_prompt,
)
from grounded.generation.providers.fake import FakeLLMProvider
from grounded.infra.provider_errors import ProviderRateLimited, ProviderRequestRejected
from grounded.runtime import ProviderConfigError
from grounded.schemas.judge import FaithfulnessVerdict, JudgeError
from grounded.schemas.judge_agreement import AgreementKey, JudgeRecord
from tests.support import make_settings

FIXTURE = (
    Path(__file__).resolve().parents[1] / "fixtures" / "promptfoo" / "results_sample_judge.json"
)
runner = CliRunner()

# Two faithfulness items (one real, one control) and four correctness items out of the recording's
# four judged claims over three questions and nine judged answers.
PARAMS = SampleParams(seed=411, faithfulness=2, correctness=4, controls=1, max_overlap=0.5)


def run_bytes(text_edit: tuple[str, str] | None = None) -> bytes:
    """The recording, with its two rubric versions set to the repository's current ones."""
    text = FIXTURE.read_text(encoding="utf-8")
    text = text.replace("judge_faithfulness_v1@84103412", load_judge_faithfulness_prompt().version)
    text = text.replace("judge_correctness_v1@1bde5fe4", load_judge_correctness_prompt().version)
    if text_edit:
        text = text.replace(*text_edit)
    return text.encode("utf-8")


def record(verdict: str = "NOT_SUPPORTED", reason: str = "Scripted control reason.") -> JudgeRecord:
    return JudgeRecord(
        verdict=verdict,
        reason=reason,
        prompt_version=load_judge_faithfulness_prompt().version,
        judge_provider="groq",
        judge_model="openai/gpt-oss-120b",
    )


class ScriptedAsk:
    """The ``ask`` of ``export``: records what it was asked and answers every control."""

    def __init__(self, stopped: JudgeError | None = None) -> None:
        self.asked: list[list[str]] = []
        self.stopped = stopped

    def __call__(self, todo: list[SampleItem]) -> ControlVerdicts:
        self.asked.append([i.item_id for i in todo])
        if self.stopped is not None:
            return ControlVerdicts({}, 1, self.stopped)
        return ControlVerdicts({i.item_id: record() for i in todo}, len(todo), None)


class Paths:
    def __init__(self, root: Path) -> None:
        self.sheet = root / "out" / "v1.csv"
        self.view = root / "out" / "v1.md"
        self.key = root / "results" / "key.json"


def do_export(
    paths: Paths,
    ask: ScriptedAsk | None,
    params: SampleParams = PARAMS,
    raw: bytes | None = None,
):
    return export(
        raw or run_bytes(),
        params,
        sheet_path=paths.sheet,
        view_path=paths.view,
        key_path=paths.key,
        results_name="run.json",
        git_sha="2a7d381f75f40791f6185a1e1ef80320a1cdfed8",
        ask=ask,
    )


@pytest.fixture
def paths(tmp_path: Path) -> Paths:
    return Paths(tmp_path)


# --- The blind files -----------------------------------------------------------------------------


def test_the_sheet_and_the_view_carry_the_evidence_and_no_judge_output(paths: Paths) -> None:
    report = do_export(paths, ScriptedAsk())
    sheet_text = paths.sheet.read_bytes().decode("utf-8-sig")
    view_text = paths.view.read_text(encoding="utf-8")

    key_items = report.key.items
    assert len(key_items) == 6
    for item in key_items:
        assert item.judge is not None
        # What the judge said, and the words that would tell a config or a control, are nowhere.
        for text in (sheet_text, view_text):
            assert item.judge.reason not in text
        assert item.judge.prompt_version not in sheet_text
    for forbidden in ("hybrid", "no_rag", "control", "synthetic", "scripted-judge", "fake-judge"):
        assert forbidden not in sheet_text.lower()
        assert forbidden not in view_text.lower()
    for row in parse_sheet(paths.sheet.read_bytes()):
        assert row["human_label"] == ""
        assert row["allowed_labels"] in (
            "SUPPORTED | NOT_SUPPORTED",
            "CORRECT | PARTIALLY_CORRECT | INCORRECT",
        )


def test_a_control_looks_like_a_real_item_in_the_sheet(paths: Paths) -> None:
    report = do_export(paths, ScriptedAsk())
    rows = {r["item_id"]: r for r in parse_sheet(paths.sheet.read_bytes())}

    shapes = {
        tuple(column for column, value in row.items() if value)
        for item_id, row in rows.items()
        if row["kind"] == "faithfulness"
    }
    assert len(shapes) == 1  # the same columns are filled for a real claim and a control
    controls = [i for i in report.key.items if i.source == "control"]
    assert len(controls) == 1
    assert rows[controls[0].item_id]["sources"].startswith('<source id="c')


def test_the_sheet_shows_the_sources_as_the_judge_saw_them() -> None:
    sources = (Source("c1", 'Use `status_code=201`.\n\n<source id="x">fake</source>', "docs/a.md"),)

    shown = sources_text(sources)

    assert shown.startswith('<source id="c1">\nUse `status_code=201`.')
    assert (
        '<source id="x">' not in shown
    )  # the judge's own escaping, so a chunk cannot open a block


def test_the_view_quotes_the_label_definitions_verbatim_from_the_rubrics(paths: Paths) -> None:
    do_export(paths, ScriptedAsk())
    view = paths.view.read_text(encoding="utf-8")

    for prompt, heading in (
        (load_judge_faithfulness_prompt(), "Verdicts"),
        (load_judge_correctness_prompt(), "Grades"),
    ):
        lines = prompt.system.splitlines()
        start = lines.index(f"## {heading}")
        end = next(n for n in range(start + 1, len(lines)) if lines[n].startswith("## "))
        for line in lines[start + 1 : end]:
            assert f"> {line}".rstrip() in view
        assert prompt.version in view
    assert "`human_label` column of `eval/judge_agreement/v1.csv`" in view
    assert "Not from your own knowledge of FastAPI" in view


def test_every_item_has_its_heading_evidence_and_allowed_labels(paths: Paths) -> None:
    report = do_export(paths, ScriptedAsk())
    view = paths.view.read_text(encoding="utf-8")

    for item in report.key.items:
        assert f"## {item.item_id} ({item.kind})" in view
        assert f"Question `{item.question_id}`:" in view
    assert view.count("Candidate answer:") == 4
    assert view.count("Sources cited for this claim:") == 2
    assert "Labels: `SUPPORTED` | `NOT_SUPPORTED`" in view


def test_evidence_with_code_fences_cannot_close_its_block(paths: Paths) -> None:
    item = SampleItem(
        "a01", "correctness", "real", "hybrid", "q001", "Q?", None,
        reference_answer="Ref.", candidate_answer="Use:\n```python\nx = 1\n````\nDone.",
    )  # fmt: skip

    view = render_view([item], do_export(paths, None).key.sample)

    assert "`````text\nUse:\n```python\nx = 1\n````\nDone.\n`````" in view


def test_the_files_are_deterministic(paths: Paths, tmp_path: Path) -> None:
    do_export(paths, ScriptedAsk())
    other = Paths(tmp_path / "again")
    do_export(other, ScriptedAsk())

    assert other.sheet.read_bytes() == paths.sheet.read_bytes()
    assert other.view.read_bytes() == paths.view.read_bytes()
    assert other.key.read_bytes() == paths.key.read_bytes()


def test_the_sheet_and_the_view_do_not_depend_on_what_the_judge_said(
    paths: Paths, tmp_path: Path
) -> None:
    do_export(paths, ScriptedAsk())  # the controls come back NOT_SUPPORTED
    other = Paths(tmp_path / "other")
    do_export(other, None)  # nobody asked: the controls are pending

    assert other.sheet.read_bytes() == paths.sheet.read_bytes()
    assert other.view.read_bytes() == paths.view.read_bytes()


def test_the_item_order_is_a_seeded_shuffle_not_the_order_of_construction(paths: Paths) -> None:
    report = do_export(paths, None)

    order = [(i.kind, i.source) for i in report.key.items]

    assert order != sorted(order)  # not grouped by kind and source
    other = do_export(Paths(paths.sheet.parent / "x"), None, SampleParams(7, 2, 4, 1, 0.5))
    assert [i.item_id for i in other.key.items] == [i.item_id for i in report.key.items]
    assert [(i.question_id, i.kind) for i in other.key.items] != [
        (i.question_id, i.kind) for i in report.key.items
    ]  # a different seed is a different sample under the same ids


# --- The key and the controls' verdicts ----------------------------------------------------------


def test_the_key_says_what_every_item_is_and_how_the_sample_was_made(paths: Paths) -> None:
    raw = run_bytes()
    report = do_export(paths, ScriptedAsk(), raw=raw)
    key = AgreementKey.model_validate_json(paths.key.read_bytes())

    assert key == report.key
    sample = key.sample
    assert sample.results_sha256 == hashlib.sha256(raw).hexdigest()
    assert (sample.seed, sample.controls, sample.results_file) == (411, 1, "run.json")
    assert sample.git_sha == "2a7d381f75f40791f6185a1e1ef80320a1cdfed8"
    assert sample.golden_set_version == "v1"
    assert sample.judge_prompt_versions["faithfulness"] == load_judge_faithfulness_prompt().version
    assert sample.population["faithfulness"] == {"hybrid": {"NOT_SUPPORTED": 1, "SUPPORTED": 3}}
    assert [i.item_id for i in key.items] == [f"a0{n}" for n in range(1, 7)]
    by_source = sorted((i.kind, i.source) for i in key.items)
    assert by_source == [("correctness", "real")] * 4 + [
        ("faithfulness", "control"),
        ("faithfulness", "real"),
    ]
    [control] = [i for i in key.items if i.source == "control"]
    assert control.control is not None
    assert control.question_id != control.control.sources_question_id
    assert control.judge == record()
    [real] = [i for i in key.items if i.kind == "faithfulness" and i.source == "real"]
    assert real.judge is not None
    assert real.judge.reason  # the run's own verdict, from the results file


def test_without_a_judge_the_controls_are_pending_and_the_rest_is_final(paths: Paths) -> None:
    report = do_export(paths, None)

    assert [i.source for i in report.key.pending] == ["control"]
    assert (report.asked, report.stopped) == (0, None)
    assert paths.sheet.exists()
    assert paths.view.exists()


def test_a_second_run_completes_the_key_and_asks_only_for_what_is_missing(paths: Paths) -> None:
    do_export(paths, None)
    first = ScriptedAsk()
    second = ScriptedAsk()

    completed = do_export(paths, first)
    again = do_export(paths, second)

    assert len(first.asked) == 1
    assert len(first.asked[0]) == 1  # the one pending control
    assert completed.key.pending == []
    assert second.asked == []  # nothing is missing: the judge is not asked again
    assert again.key == completed.key


def test_a_quota_stops_the_asking_and_keeps_the_key_incomplete(paths: Paths) -> None:
    quota = JudgeError(
        kind="ProviderRateLimited", is_quota=True, provider_side=True, detail="daily"
    )

    report = do_export(paths, ScriptedAsk(stopped=quota))

    assert report.stopped == quota
    assert len(report.key.pending) == 1
    assert paths.sheet.exists()  # the blind files are final whatever the judge did
    assert AgreementKey.model_validate_json(paths.key.read_bytes()).pending != []


def test_a_key_of_another_sample_is_not_reused(paths: Paths) -> None:
    do_export(paths, ScriptedAsk())
    ask = ScriptedAsk()

    do_export(paths, ask, SampleParams(412, 2, 4, 1, 0.5))  # another seed, so another fingerprint

    assert len(ask.asked) == 1
    assert len(ask.asked[0]) == 1


def test_a_rubric_that_changed_since_the_run_is_refused(paths: Paths) -> None:
    stale = run_bytes((load_judge_faithfulness_prompt().version, "judge_faithfulness_v1@deadbeef"))

    with pytest.raises(SampleError, match=r"faithfulness rubric judge_faithfulness_v1@deadbeef"):
        do_export(paths, None, raw=stale)
    assert not paths.sheet.exists()  # nothing was written


# --- Never overwrite a labeled sheet -------------------------------------------------------------


def test_a_labeled_sheet_of_the_same_items_is_left_alone(paths: Paths) -> None:
    do_export(paths, None)
    rows = parse_sheet(paths.sheet.read_bytes())
    rows[0]["human_label"] = "supported"
    labeled = render_sheet(rows)
    paths.sheet.write_bytes(labeled)

    report = do_export(paths, ScriptedAsk())

    assert report.sheet == "kept"
    assert paths.sheet.read_bytes() == labeled


def test_a_labeled_sheet_of_other_items_is_not_overwritten(paths: Paths) -> None:
    paths.sheet.parent.mkdir(parents=True)
    old = render_sheet(
        [{"item_id": "a01", "kind": "correctness", "question_id": "q999", "human_label": "CORRECT"}]
    )
    paths.sheet.write_bytes(old)

    with pytest.raises(SampleError, match="holds labels of a different sample"):
        do_export(paths, None)
    assert paths.sheet.read_bytes() == old


def test_an_unlabeled_sheet_of_other_items_is_replaced(paths: Paths) -> None:
    paths.sheet.parent.mkdir(parents=True)
    paths.sheet.write_bytes(
        render_sheet([{"item_id": "a01", "kind": "correctness", "question_id": "q999"}])
    )

    assert do_export(paths, None).sheet == "written"
    assert len(parse_sheet(paths.sheet.read_bytes())) == 6


def test_a_sheet_a_spreadsheet_saved_with_semicolons_still_counts_as_the_same_sample(
    paths: Paths,
) -> None:
    report = do_export(paths, None)
    rows = parse_sheet(paths.sheet.read_bytes())
    rows[0]["human_label"] = "NOT_SUPPORTED"
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=list(rows[0]), delimiter=";", lineterminator="\r\n")
    writer.writeheader()
    writer.writerows(rows)
    resaved = out.getvalue().encode("utf-8")  # as a spreadsheet in a Serbian locale may save it
    paths.sheet.write_bytes(resaved)

    again = do_export(paths, None)

    assert again.sheet == "kept"
    assert paths.sheet.read_bytes() == resaved
    assert again.key.sample.fingerprint == report.key.sample.fingerprint


# --- Asking the judge for the controls -----------------------------------------------------------


def control_item(item_id: str, claim_text: str, source_text: str) -> SampleItem:
    return SampleItem(
        item_id, "faithfulness", "control", "hybrid", "q001", "Q?", None,
        claim_index=1, claim=claim_text, sources=(Source("c2", source_text, "docs/b.md#x"),),
    )  # fmt: skip


def opener(provider: FakeLLMProvider):
    judge = Judge.create(provider, JudgeConfig(max_output_tokens=800, timeout_s=10.0))

    @asynccontextmanager
    async def open_judge() -> AsyncIterator[Judge]:
        yield judge

    return open_judge


async def test_each_control_is_judged_on_its_claim_and_the_other_questions_sources() -> None:
    provider = FakeLLMProvider(
        [FaithfulnessVerdict(verdict="NOT_SUPPORTED", reason="c2 is about something else.")] * 2,
        name="groq",
        model="openai/gpt-oss-120b",
    )
    items = [
        control_item("a03", "Claim about alpha.", "Text about beta."),
        control_item("a07", "Claim about gamma.", "Text about delta."),
    ]

    result = await ask_controls(items, opener(provider))

    assert (result.asked, result.stopped) == (2, None)
    assert {k: v.verdict for k, v in result.records.items()} == {
        "a03": "NOT_SUPPORTED",
        "a07": "NOT_SUPPORTED",
    }
    assert result.records["a03"].judge_model == "openai/gpt-oss-120b"
    first = provider.calls[0].user
    assert "Claim about alpha." in first
    assert '<source id="c2">\nText about beta.\n</source>' in first
    assert "delta" not in first


async def test_a_quota_stops_the_asking_and_keeps_the_verdicts_already_obtained() -> None:
    provider = FakeLLMProvider(
        [
            FaithfulnessVerdict(verdict="NOT_SUPPORTED", reason="Unrelated."),
            ProviderRateLimited("daily quota", retry_after_s=3600, is_quota=True),
        ],
        name="groq",
        model="m",
    )
    items = [control_item(f"a0{n}", f"Claim {n}.", f"Text {n}.") for n in (1, 2, 3)]

    result = await ask_controls(items, opener(provider))

    assert result.asked == 2  # the third was never asked
    assert list(result.records) == ["a01"]
    assert result.stopped is not None
    assert (result.stopped.kind, result.stopped.is_quota) == ("ProviderRateLimited", True)
    assert provider.remaining == 0
    assert len(provider.calls) == 2


async def test_a_judge_that_cannot_be_opened_or_is_refused_is_a_stop_not_a_crash() -> None:
    @asynccontextmanager
    async def unconfigured() -> AsyncIterator[Judge]:
        raise ProviderConfigError("JUDGE_MODEL must be set")
        yield  # pragma: no cover

    refused = FakeLLMProvider(
        [ProviderRequestRejected("bad key", status_code=401)], name="groq", model="m"
    )

    nothing = await ask_controls([control_item("a01", "C.", "T.")], unconfigured)
    rejected = await ask_controls([control_item("a01", "C.", "T.")], opener(refused))

    assert nothing.stopped is not None
    assert (nothing.stopped.kind, nothing.stopped.provider_side) == ("ProviderConfigError", False)
    assert "JUDGE_MODEL" in nothing.stopped.detail
    assert rejected.stopped is not None
    assert rejected.stopped.kind == "ProviderRequestRejected"
    assert rejected.records == {}


# --- grounded eval export-verdicts ---------------------------------------------------------------


def command(paths: Paths, results: Path, *extra: str) -> list[str]:
    return [
        "eval", "export-verdicts", "--results", str(results), "--faithfulness", "2",
        "--correctness", "4", "--controls", "1", "--max-overlap", "0.5",
        "--out", str(paths.sheet), "--view-out", str(paths.view), "--key-out", str(paths.key),
        "--git-sha", "2a7d381", *extra,
    ]  # fmt: skip


def use_settings(monkeypatch: pytest.MonkeyPatch, **overrides: object) -> None:
    monkeypatch.setattr(grounded.cli, "get_settings", lambda: make_settings(**overrides))


@pytest.fixture
def results_file(tmp_path: Path) -> Path:
    path = tmp_path / "promptfoo.json"
    path.write_bytes(run_bytes())
    return path


def test_the_command_writes_the_files_and_prints_the_provenance_without_verdicts(
    monkeypatch: pytest.MonkeyPatch, paths: Paths, results_file: Path
) -> None:
    use_settings(monkeypatch)

    result = runner.invoke(app, command(paths, results_file, "--no-judge"))

    assert result.exit_code == 0, result.output
    assert f"sha256 {hashlib.sha256(results_file.read_bytes()).hexdigest()}" in result.output
    assert "golden set v1" in result.output
    assert "faithfulness, hybrid: 4 (NOT_SUPPORTED 1, SUPPORTED 3)" in result.output
    assert (
        "Sample (seed 411): 6 items = 2 faithfulness (1 synthetic controls) + 4 correctness"
        in result.output
    )
    assert "Key: 5 of 6 items have a judge verdict" in result.output
    assert "pending" in result.output
    key = AgreementKey.model_validate_json(paths.key.read_bytes())
    for item in key.items:
        assert item.judge is None or item.judge.reason not in result.output


def test_the_command_judges_the_controls_with_the_stub_judge_of_a_fake_run(
    monkeypatch: pytest.MonkeyPatch, paths: Paths, results_file: Path
) -> None:
    use_settings(monkeypatch, app_env="eval", generator_providers=["fake"])

    result = runner.invoke(app, command(paths, results_file))

    assert result.exit_code == 0, result.output
    assert "Key: 6 of 6 items have a judge verdict" in result.output
    assert "judge calls made now: 1" in result.output
    key = AgreementKey.model_validate_json(paths.key.read_bytes())
    [control] = [i for i in key.items if i.source == "control"]
    assert control.judge is not None
    assert control.judge.judge_provider == "fake-judge"


def test_outside_eval_mode_the_controls_stay_pending_and_the_exit_is_one(
    monkeypatch: pytest.MonkeyPatch, paths: Paths, results_file: Path
) -> None:
    use_settings(monkeypatch)  # APP_ENV=test

    result = runner.invoke(app, command(paths, results_file))

    assert result.exit_code == 1
    assert "set APP_ENV=eval" in result.output
    assert "run this command again later to complete the key" in result.output
    assert paths.sheet.exists()  # final anyway
    assert len(AgreementKey.model_validate_json(paths.key.read_bytes()).pending) == 1


def test_the_command_says_what_it_cannot_read_or_draw(
    monkeypatch: pytest.MonkeyPatch, paths: Paths, tmp_path: Path, results_file: Path
) -> None:
    use_settings(monkeypatch)

    missing = runner.invoke(app, command(paths, tmp_path / "none.json", "--no-judge"))
    too_many = runner.invoke(
        app, [*command(paths, results_file, "--no-judge"), "--controls", "3", "--faithfulness", "2"]
    )

    assert missing.exit_code == 1
    assert "Cannot export" in missing.output
    assert too_many.exit_code == 1
    assert "3 controls do not fit in 2 items" in too_many.output


def test_the_labeled_sheet_goes_through_the_agreement_command(
    monkeypatch: pytest.MonkeyPatch, paths: Paths, results_file: Path
) -> None:
    """Both halves together: export, label every item as the key says, compute."""
    use_settings(monkeypatch, app_env="eval", generator_providers=["fake"])
    assert runner.invoke(app, command(paths, results_file)).exit_code == 0
    key = AgreementKey.model_validate_json(paths.key.read_bytes())
    rows = parse_sheet(paths.sheet.read_bytes())
    for row, item in zip(rows, key.items, strict=True):
        assert item.judge is not None
        row["human_label"] = item.judge.verdict.lower()
    paths.sheet.write_bytes(render_sheet(rows))

    result = runner.invoke(
        app, ["eval", "agreement", "--labels", str(paths.sheet), "--key", str(paths.key)]
    )

    assert result.exit_code == 0, result.output
    assert "| All items (labels of both kinds pooled) | 6 | 100.0% (6/6) |" in result.output
    assert json.loads(paths.key.read_text(encoding="utf-8"))["schema_version"] == 1
