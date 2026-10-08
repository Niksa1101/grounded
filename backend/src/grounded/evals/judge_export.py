"""The files of the judge-agreement sample (Tech.md §15.4, PRD FR-22, ticket 4.11a): the blind
sheet and view a person labels from, and the key that says what the judge said.

``grounded eval export-verdicts`` draws the sample (``judge_sample.py``) and writes three files:

- **The sheet** (``v1.csv``) and **the view** (``v1.md``) hold the evidence (the claim and the
  sources the judge was shown, or the reference and the candidate answer) and no verdict, reason,
  score, config name or control mark. Their content depends only on the seed and the results file,
  so they are final even when the judge could not be asked.
- **The key** says what each ``item_id`` is and what the judge said. It stays out of the repository
  until the labels are in. The real items carry the verdict the judge gave in the run; a control's
  verdict comes from a fresh call of the same judge through the eval cache, because what is
  compared is what the judge says. Re-running the command completes a key whose control calls did
  not finish (a daily quota), asking only for the verdicts it lacks, and never overwrites a sheet
  that already holds labels.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Sequence
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

from pydantic import ValidationError

from grounded.evals.agreement import SheetError, parse_sheet, render_sheet
from grounded.evals.judge import CitedSource, Judge, render_sources
from grounded.evals.judge_sample import (
    Population,
    SampleError,
    SampleItem,
    SampleParams,
    Source,
    build_sample,
    read_population,
)
from grounded.generation.prompts import (
    Prompt,
    load_judge_correctness_prompt,
    load_judge_faithfulness_prompt,
)
from grounded.infra.provider_errors import ProviderRequestRejected
from grounded.runtime import ProviderConfigError
from grounded.schemas.generation_eval import GenerationRunInfo
from grounded.schemas.judge import JudgeError
from grounded.schemas.judge_agreement import ALLOWED_LABELS, AgreementKey, JudgeRecord, SampleInfo

# --- The blind files -----------------------------------------------------------------------------


def sources_text(sources: Sequence[Source]) -> str:
    """The sources as the judge saw them: ``<source id="c1">`` blocks, escaped the same way."""
    return render_sources([CitedSource(s.label, s.text) for s in sources])


def sheet_rows(items: Sequence[SampleItem]) -> list[dict[str, str]]:
    return [
        {
            "item_id": i.item_id,
            "kind": i.kind,
            "question_id": i.question_id,
            "question": i.question,
            "claim": i.claim,
            "sources": sources_text(i.sources),
            "reference_answer": i.reference_answer,
            "candidate_answer": i.candidate_answer,
            "allowed_labels": " | ".join(ALLOWED_LABELS[i.kind]),
            "human_label": "",
        }
        for i in items
    ]


def _section(prompt: Prompt, heading: str) -> str:
    """The text under ``## <heading>`` of a rubric, verbatim, up to the next ``##`` heading."""
    lines = prompt.system.splitlines()
    try:
        start = lines.index(f"## {heading}")
    except ValueError:
        raise SampleError(f"{prompt.name} has no '## {heading}' section") from None
    end = next((n for n in range(start + 1, len(lines)) if lines[n].startswith("## ")), len(lines))
    return "\n".join(lines[start + 1 : end]).strip("\n")


def _quote(text: str) -> str:
    return "\n".join(f"> {line}".rstrip() for line in text.splitlines())


def _block(title: str, text: str) -> list[str]:
    """A titled code block. The fence is longer than any run of backticks in ``text``, so no
    evidence (the docs have code fences) can close it."""
    longest = max((len(run) for run in re.findall(r"`+", text)), default=0)
    fence = "`" * max(3, longest + 1)
    return ["", f"{title}:", "", f"{fence}text\n{text}\n{fence}"]


_HOW_TO_LABEL: Final = """\
## How to label

1. Write one label per item into the `human_label` column of `eval/judge_agreement/v1.csv`, on
   the row with the same `item_id`. This page shows the same items with the evidence laid out to
   be read.
2. Use only the labels listed for the item's kind. Capital letters, spaces and a hyphen instead
   of an underscore do not matter.
3. Judge from the evidence shown on this page, as the judge did: the sources, or the reference
   answer. Not from your own knowledge of FastAPI, and not from anything else.
4. Label every item. `grounded eval agreement` refuses a sheet with a missing or invalid label."""


def render_view(items: Sequence[SampleItem], sample: SampleInfo) -> str:
    """``v1.md``: the same items as the sheet, laid out to be read, with how to label them and
    the label definitions copied verbatim from the two rubrics. No verdict, config or control
    mark."""
    faith, corr = load_judge_faithfulness_prompt(), load_judge_correctness_prompt()
    out = [
        "# Judge agreement sheet v1: the evidence to label",
        "",
        "<!-- Generated by `grounded eval export-verdicts`; do not edit. The labels go in the "
        "`human_label` column of `v1.csv`. -->",
        "",
        _HOW_TO_LABEL,
        "",
        "## Label definitions",
        "",
        "Copied verbatim from the rubrics the judge was given (`backend/prompts/judge_*_v1.md`; "
        "the full rubrics also have numbered rules).",
        "",
        f"### Faithfulness ({faith.version}, section Verdicts)",
        "",
        _quote(_section(faith, "Verdicts")),
        "",
        f"### Correctness ({corr.version}, section Grades)",
        "",
        _quote(_section(corr, "Grades")),
        "",
        f"Sample: seed {sample.seed}, {len(items)} items, drawn from a run whose results file has "
        f"the sha256 `{sample.results_sha256}`. How: `eval/judge_agreement/README.md`.",
    ]
    for item in items:
        out += ["", f"## {item.item_id} ({item.kind})"]
        out += _block(f"Question `{item.question_id}`", item.question)
        if item.kind == "faithfulness":
            out += _block("Claim", item.claim)
            out += _block("Sources cited for this claim", sources_text(item.sources))
        else:
            out += _block("Reference answer", item.reference_answer)
            out += _block("Candidate answer", item.candidate_answer)
        out += ["", "Labels: " + " | ".join(f"`{label}`" for label in ALLOWED_LABELS[item.kind])]
    return "\n".join(out) + "\n"


def write_sheet(path: Path, rows: Sequence[dict[str, str]]) -> Literal["written", "kept"]:
    """Write the sheet unless it already is this sample. A sheet of the same items keeps its
    labels (it is left untouched), and a sheet of different items that holds labels is never
    overwritten."""
    data = render_sheet(rows)
    if path.exists():
        old = path.read_bytes()
        try:
            old_rows = parse_sheet(old)
        except SheetError:
            old_rows = []
        who = [(r["item_id"], r["kind"], r["question_id"]) for r in old_rows]
        if old == data or who == [(r["item_id"], r["kind"], r["question_id"]) for r in rows]:
            return "kept"
        if any(r["human_label"].strip() for r in old_rows):
            raise SampleError(f"{path} holds labels of a different sample: not overwritten")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return "written"


# --- The key, the judge's calls, the whole export ------------------------------------------------


def _check_rubrics(info: GenerationRunInfo) -> None:
    """The definitions on the view are read from the rubric files, so the judge must have used
    these files: a run judged with another version would be shown a different rubric."""
    for metric, prompt in (
        ("faithfulness", load_judge_faithfulness_prompt()),
        ("correctness", load_judge_correctness_prompt()),
    ):
        used = info.judge_prompt_versions.get(metric)
        if used != prompt.version:
            raise SampleError(
                f"the run used the {metric} rubric {used}, this repo has {prompt.version}"
            )


def _sample_info(
    raw: bytes,
    params: SampleParams,
    population: Population,
    items: Sequence[SampleItem],
    results_name: str,
    git_sha: str | None,
) -> SampleInfo:
    digest, info = hashlib.sha256(raw).hexdigest(), population.info
    identity = [digest, params.seed, params.faithfulness, params.correctness, params.controls]
    identity += [params.max_overlap, [[i.item_id, i.natural_key] for i in items]]
    return SampleInfo(
        seed=params.seed,
        faithfulness=params.faithfulness,
        correctness=params.correctness,
        controls=params.controls,
        max_overlap=params.max_overlap,
        fingerprint=hashlib.sha256(json.dumps(identity).encode()).hexdigest(),
        results_file=results_name,
        results_sha256=digest,
        run_date=info.date,
        promptfoo_version=info.promptfoo_version,
        golden_set_version=info.golden_set_version,
        golden_set_sha256=info.golden_set_sha256,
        git_sha=git_sha,
        judge_provider=info.judge_provider,
        judge_model=info.judge_model,
        judge_prompt_versions=info.judge_prompt_versions,
        population=population.counts(),
    )


def _earlier_verdicts(path: Path, sample: SampleInfo) -> dict[str, JudgeRecord]:
    """The control verdicts of an earlier key of this same sample and the current rubric."""
    rubric = load_judge_faithfulness_prompt().version
    try:
        key = AgreementKey.model_validate_json(path.read_bytes())
    except (OSError, ValidationError):
        return {}
    if key.sample.fingerprint != sample.fingerprint:
        return {}
    return {
        i.item_id: i.judge
        for i in key.items
        if i.source == "control" and i.judge is not None and i.judge.prompt_version == rubric
    }


@dataclass(frozen=True, slots=True)
class ControlVerdicts:
    records: dict[str, JudgeRecord]  # item_id -> the verdict obtained now
    asked: int  # judge calls made (each may be an eval-cache hit)
    stopped: JudgeError | None  # why the rest was not asked


async def ask_controls(
    items: Sequence[SampleItem], open_judge: Callable[[], AbstractAsyncContextManager[Judge]]
) -> ControlVerdicts:
    """One judge call per control, in order, stopping at the first failure: a quota is not asked
    again, and a rejected key or a missing judge setting is the same kind of stop. The verdicts
    obtained before it are kept."""
    records: dict[str, JudgeRecord] = {}
    asked = 0
    try:
        async with open_judge() as judge:
            for item in items:
                sources = [CitedSource(s.label, s.text) for s in item.sources]
                judgment = await judge.judge_claim(item.claim_index or 0, item.claim, sources)
                asked += 1
                if judgment.error is not None:
                    return ControlVerdicts(records, asked, judgment.error)
                records[item.item_id] = JudgeRecord.from_claim(judgment)
    except (ProviderRequestRejected, ProviderConfigError) as exc:
        detail = " ".join(str(exc).split())[:300]
        stop = JudgeError(kind=type(exc).__name__, provider_side=False, detail=detail)
        return ControlVerdicts(records, asked, stop)
    return ControlVerdicts(records, asked, None)


@dataclass(frozen=True, slots=True)
class ExportReport:
    key: AgreementKey
    sheet: Literal["written", "kept"]
    asked: int
    stopped: JudgeError | None


def export(
    raw: bytes,
    params: SampleParams,
    *,
    sheet_path: Path,
    view_path: Path,
    key_path: Path,
    results_name: str,
    git_sha: str | None,
    ask: Callable[[list[SampleItem]], ControlVerdicts] | None,
) -> ExportReport:
    """Draw the sample from ``raw`` (the results file's bytes) and write the sheet, the view and
    the key. ``ask`` gets the controls whose verdict the key does not have yet (the real items
    carry the run's verdict) and returns what it obtained; ``None`` asks nobody and leaves them
    pending. The sheet and the view do not depend on any verdict, so they are final either way."""
    population = read_population(raw)
    _check_rubrics(population.info)
    items = build_sample(population, params)
    sample = _sample_info(raw, params, population, items, results_name, git_sha)
    view_path.parent.mkdir(parents=True, exist_ok=True)
    view_path.write_bytes(render_view(items, sample).encode("utf-8"))
    sheet = write_sheet(sheet_path, sheet_rows(items))

    judged = {i.item_id: i.judge for i in items} | _earlier_verdicts(key_path, sample)
    todo = [i for i in items if judged[i.item_id] is None]
    obtained = ask(todo) if todo and ask else ControlVerdicts({}, 0, None)
    judged |= obtained.records
    key = AgreementKey(sample=sample, items=[i.key_item(judged[i.item_id]) for i in items])
    key_path.parent.mkdir(parents=True, exist_ok=True)
    key_path.write_bytes((key.model_dump_json(indent=2) + "\n").encode("utf-8"))
    return ExportReport(key, sheet, obtained.asked, obtained.stopped)
