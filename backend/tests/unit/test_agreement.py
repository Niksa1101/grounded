"""Judge-human agreement (4.11a, Tech §15.4): Cohen's kappa against hand-computed cases, the sheet
file format, the join of labels and key (which refuses instead of dropping), the report, and the
`grounded eval agreement` command.

Every expected number below is computed by hand in the comment next to it, from the definition
``kappa = (p_o - p_e) / (1 - p_e)`` with ``p_o`` the observed agreement and ``p_e`` the chance
agreement from the raters' own label frequencies.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from grounded.cli import app
from grounded.evals.agreement import (
    SHEET_COLUMNS,
    LabelError,
    Rating,
    SheetError,
    cohens_kappa,
    normalize_label,
    parse_sheet,
    rate_items,
    render_report,
    render_sheet,
    summarize,
)
from grounded.schemas.judge_agreement import (
    AgreementKey,
    ControlInfo,
    ItemSource,
    JudgeRecord,
    KeyItem,
    Kind,
    SampleInfo,
)

runner = CliRunner()

SUP, NOT = "SUPPORTED", "NOT_SUPPORTED"
COR, PAR, INC = "CORRECT", "PARTIALLY_CORRECT", "INCORRECT"


def repeat(label_pairs: dict[tuple[str, str], int]) -> tuple[list[str], list[str]]:
    """Two label lists (human, judge) from a confusion table ``{(human, judge): count}``."""
    human: list[str] = []
    judge: list[str] = []
    for (h, j), count in label_pairs.items():
        human += [h] * count
        judge += [j] * count
    return human, judge


# --- Cohen's kappa -------------------------------------------------------------------------------


def test_perfect_agreement_is_one() -> None:
    # p_o = 1; each rater uses SUP twice and NOT twice, so p_e = 2/4 * 2/4 + 2/4 * 2/4 = 0.5.
    # kappa = (1 - 0.5) / (1 - 0.5) = 1.
    result = cohens_kappa([SUP, SUP, NOT, NOT], [SUP, SUP, NOT, NOT])

    assert (result.observed, result.expected, result.value) == (1.0, 0.5, 1.0)
    assert result.reason is None


def test_agreement_no_better_than_chance_is_zero() -> None:
    # Matches on items 1 and 4 only: p_o = 2/4. p_e = 0.5 as above. (0.5 - 0.5) / 0.5 = 0.
    result = cohens_kappa([SUP, SUP, NOT, NOT], [SUP, NOT, SUP, NOT])

    assert (result.observed, result.expected, result.value) == (0.5, 0.5, 0.0)


def test_complete_disagreement_is_minus_one() -> None:
    # p_o = 0, p_e = 0.5: (0 - 0.5) / 0.5 = -1.
    assert cohens_kappa([SUP, SUP, NOT, NOT], [NOT, NOT, SUP, SUP]).value == -1.0


def test_the_textbook_two_by_two_example() -> None:
    """Cohen's kappa, Wikipedia's 50 grant proposals: both yes 20, A yes B no 5, A no B yes 10,
    both no 15. A says yes 25 times, B 30 times. p_o = 35/50 = 0.7, p_e = 0.5*0.6 + 0.5*0.4 = 0.5,
    kappa = 0.2 / 0.5 = 0.4."""
    human, judge = repeat({(SUP, SUP): 20, (SUP, NOT): 5, (NOT, SUP): 10, (NOT, NOT): 15})

    result = cohens_kappa(human, judge)

    assert result.observed == pytest.approx(0.7)
    assert result.expected == pytest.approx(0.5)
    assert result.value == pytest.approx(0.4)


def test_a_three_class_case() -> None:
    """Rows human, columns judge (A, B, COR): [[6, 1, 0], [2, 3, 1], [0, 2, 5]], n = 20.
    p_o = (6 + 3 + 5) / 20 = 0.7. Row totals 7, 6, 7; column totals 8, 6, 6, so
    p_e = (7*8 + 6*6 + 7*6) / 400 = 134 / 400 = 0.335. kappa = 0.365 / 0.665 = 73 / 133."""
    table = {
        (COR, COR): 6, (COR, PAR): 1, (COR, INC): 0,
        (PAR, COR): 2, (PAR, PAR): 3, (PAR, INC): 1,
        (INC, COR): 0, (INC, PAR): 2, (INC, INC): 5,
    }  # fmt: skip
    human, judge = repeat(table)

    result = cohens_kappa(human, judge)

    assert result.observed == pytest.approx(0.7)
    assert result.expected == pytest.approx(0.335)
    assert result.value == pytest.approx(73 / 133)


def test_one_rater_using_a_single_label_gives_zero_not_a_division_by_zero() -> None:
    # Human: SUP x4. Judge: SUP x3, NOT x1. p_o = 3/4, p_e = (4 * 3) / 16 = 3/4, so kappa = 0.
    result = cohens_kappa([SUP] * 4, [SUP, SUP, SUP, NOT])

    assert result.value == 0.0
    assert result.reason is None


def test_kappa_is_undefined_when_chance_agreement_is_certain() -> None:
    # Both raters say SUP for every item: p_e = 1, and 0 / 0 is not a number.
    result = cohens_kappa([SUP] * 4, [SUP] * 4)

    assert result.value is None
    assert result.reason is not None
    assert SUP in result.reason
    assert (result.observed, result.expected) == (1.0, 1.0)


@pytest.mark.parametrize(("human", "judge"), [([], []), ([SUP], [SUP, NOT])])
def test_kappa_needs_the_same_non_zero_number_of_labels(human: list[str], judge: list[str]) -> None:
    with pytest.raises(ValueError, match="same, non-zero number"):
        cohens_kappa(human, judge)


# --- The sheet file format -----------------------------------------------------------------------

AWKWARD = {
    "item_id": "a01",
    "kind": "faithfulness",
    "question_id": "q003",
    "question": 'How do INC say "hello", then "bye"?\nSecond line, with a comma; and a semicolon',
    "claim": "Use `Query(max_length=50)`.\r\nIt returns 422.",
    "sources": '<source id="c1">\nčšž Привет 日本語\n\n    code = "x,y"\n</source>',
    "reference_answer": "",
    "candidate_answer": "",
    "allowed_labels": "SUPPORTED | NOT_SUPPORTED",
    "human_label": "",
}


def test_the_sheet_round_trips_newlines_quotes_commas_and_non_ascii() -> None:
    parsed = parse_sheet(render_sheet([AWKWARD]))

    assert parsed == [{**AWKWARD, "claim": AWKWARD["claim"].replace("\r\n", "\n")}]


def test_the_sheet_has_a_bom_lf_row_ends_and_the_columns_in_order() -> None:
    data = render_sheet([AWKWARD])

    assert data.startswith(b"\xef\xbb\xbf")
    assert data[3:].decode("utf-8").splitlines()[0] == ",".join(SHEET_COLUMNS)
    assert b"\r" not in data  # LF only, also inside the cells: the same bytes on every OS
    assert render_sheet([AWKWARD]) == data


def test_the_reader_accepts_a_sheet_without_bom_and_with_crlf_row_ends() -> None:
    text = ",".join(SHEET_COLUMNS) + "\r\n" + "a01,faithfulness,q003,Q,COR,,,,x,SUPPORTED\r\n\r\n"

    [row] = parse_sheet(text.encode("utf-8"))

    assert (row["item_id"], row["human_label"]) == ("a01", "SUPPORTED")


def test_the_reader_accepts_the_semicolon_a_spreadsheet_saves_in_some_locales() -> None:
    header = ";".join(SHEET_COLUMNS)
    text = f"{header}\na01;faithfulness;q003;Q;COR;;;;x;NOT_SUPPORTED\n"

    [row] = parse_sheet(b"\xef\xbb\xbf" + text.encode("utf-8"))

    assert (row["question"], row["human_label"]) == ("Q", "NOT_SUPPORTED")


def test_blank_rows_are_ignored() -> None:
    text = ",".join(SHEET_COLUMNS) + "\n,,,,,,,,,\n"

    assert parse_sheet(text.encode()) == []


def test_a_sheet_that_is_not_utf8_or_lacks_a_column_is_refused_with_the_reason() -> None:
    with pytest.raises(SheetError, match="not UTF-8"):
        parse_sheet("item_id,human_label\na01,č".encode("cp1250"))
    with pytest.raises(SheetError, match=r"lacks the columns .*kind"):
        parse_sheet(b"item_id,human_label\na01,SUPPORTED\n")


def test_labels_are_read_whatever_the_case_spacing_or_separator() -> None:
    assert normalize_label("  not supported ") == "NOT_SUPPORTED"
    assert normalize_label("Partially-Correct") == "PARTIALLY_CORRECT"
    assert normalize_label("") == ""


# --- Key schema ----------------------------------------------------------------------------------


def record(verdict: str, reason: str = "A reason.") -> JudgeRecord:
    return JudgeRecord(
        verdict=verdict,
        reason=reason,
        prompt_version="judge_faithfulness_v1@84103412",
        judge_provider="groq",
        judge_model="openai/gpt-oss-120b",
    )


def item(
    item_id: str,
    kind: str = "faithfulness",
    verdict: str | None = SUP,
    source: str = "real",
    **extra: object,
) -> KeyItem:
    claim = {"claim_index": 0} if kind == "faithfulness" else {}
    control = (
        {
            "control": ControlInfo(
                sources_config="hybrid",
                sources_question_id="q009",
                sources_claim_index=1,
                sources_labels=("c1",),
                token_overlap=0.1,
                max_overlap=0.25,
            )
        }
        if source == "control"
        else {}
    )
    return KeyItem.model_validate(
        {
            "item_id": item_id,
            "kind": kind,
            "source": source,
            "config": "hybrid",
            "question_id": "q003",
            "judge": record(verdict) if verdict else None,
            **claim,
            **control,
            **extra,
        }
    )


def sample_info(**overrides: object) -> SampleInfo:
    return SampleInfo.model_validate(
        {
            "seed": 411,
            "faithfulness": 2,
            "correctness": 2,
            "controls": 1,
            "max_overlap": 0.25,
            "fingerprint": "f" * 64,
            "results_file": "run.json",
            "results_sha256": "ab" * 32,
            "run_date": datetime(2026, 10, 8, tzinfo=UTC),
            "promptfoo_version": "0.123.1",
            "golden_set_version": "v1",
            "golden_set_sha256": "cd" * 32,
            "judge_provider": "groq",
            "judge_model": "openai/gpt-oss-120b",
            "judge_prompt_versions": {
                "correctness": "judge_correctness_v1@1bde5fe4",
                "faithfulness": "judge_faithfulness_v1@84103412",
            },
            "population": {},
            **overrides,
        }
    )


def test_a_key_item_must_match_its_kind_and_source() -> None:
    with pytest.raises(ValidationError, match="claim index"):
        item("a01", kind="correctness", verdict=COR, claim_index=0)
    with pytest.raises(ValidationError, match="control details"):
        KeyItem(
            item_id="a01",
            kind="faithfulness",
            source="control",
            config="hybrid",
            question_id="q003",
            claim_index=0,
        )
    with pytest.raises(ValidationError, match="only faithfulness items have controls"):
        item("a01", kind="correctness", verdict=COR, source="control")
    with pytest.raises(ValidationError, match="not a correctness label"):
        item("a01", kind="correctness", verdict=SUP)
    with pytest.raises(ValidationError, match="appears twice"):
        AgreementKey(sample=sample_info(), items=[item("a01"), item("a01")])


def test_the_key_round_trips_through_json() -> None:
    key = AgreementKey(sample=sample_info(), items=[item("a01"), item("a02", verdict=None)])

    again = AgreementKey.model_validate_json(key.model_dump_json(indent=2))

    assert again == key
    assert [i.item_id for i in again.pending] == ["a02"]


# --- Joining the labels to the key ---------------------------------------------------------------


def sheet_row(item_id: str, kind: str, label: str) -> dict[str, str]:
    return {c: "" for c in SHEET_COLUMNS} | {"item_id": item_id, "kind": kind, "human_label": label}


def small_key() -> AgreementKey:
    return AgreementKey(
        sample=sample_info(),
        items=[
            item("a01", verdict=SUP),
            item("a02", verdict=NOT, source="control"),
            item("a03", kind="correctness", verdict=COR),
            item("a04", kind="correctness", verdict=INC),
        ],
    )


def test_complete_labels_become_ratings_in_key_order() -> None:
    sheet = [
        sheet_row("a04", "correctness", "partially correct"),
        sheet_row("a03", "correctness", "CORRECT"),
        sheet_row("a02", "faithfulness", "NOT_SUPPORTED"),
        sheet_row("a01", "faithfulness", " supported"),
    ]

    ratings = rate_items(sheet, small_key())

    assert ratings == [
        Rating("a01", "faithfulness", "real", SUP, SUP),
        Rating("a02", "faithfulness", "control", NOT, NOT),
        Rating("a03", "correctness", "real", COR, COR),
        Rating("a04", "correctness", "real", PAR, INC),
    ]


def test_partial_and_invalid_labels_are_refused_with_every_item_at_fault() -> None:
    sheet = [
        sheet_row("a01", "faithfulness", ""),  # missing
        sheet_row("a02", "faithfulness", "CORRECT"),  # a label of the other kind
        sheet_row("a03", "correctness", "mostly"),  # not a label at all
        sheet_row("a99", "faithfulness", SUP),  # not in the key
        # a04 has no row
    ]

    with pytest.raises(LabelError) as caught:
        rate_items(sheet, small_key())

    assert caught.value.problems == (
        "a99: is in the sheet but not in the key",
        "a01: no human_label",
        "a02: 'CORRECT' is not one of SUPPORTED, NOT_SUPPORTED",
        "a03: 'mostly' is not one of CORRECT, PARTIALLY_CORRECT, INCORRECT",
        "a04: has no row in the sheet",
    )


def test_a_duplicate_row_a_changed_kind_and_a_key_without_a_verdict_are_refused() -> None:
    key = AgreementKey(sample=sample_info(), items=[item("a01"), item("a02", verdict=None)])
    sheet = [
        sheet_row("a01", "faithfulness", SUP),
        sheet_row("a01", "faithfulness", SUP),
        sheet_row("a02", "correctness", SUP),
    ]

    with pytest.raises(LabelError) as caught:
        rate_items(sheet, key)

    assert caught.value.problems == (
        "a01: appears twice in the sheet",
        "a02: the sheet says kind 'correctness', the key 'faithfulness'",
        "a02: the key has no judge verdict yet (re-run export-verdicts)",
    )


# --- Summaries and the report --------------------------------------------------------------------


def ratings_of(rows: list[tuple[str, Kind, ItemSource, str, str]]) -> list[Rating]:
    return [Rating(*row) for row in rows]


def test_summary_counts_agreement_and_the_confusion_table() -> None:
    ratings = ratings_of(
        [
            ("a01", "correctness", "real", COR, COR),
            ("a02", "correctness", "real", COR, PAR),
            ("a03", "correctness", "real", PAR, PAR),
            ("a04", "correctness", "real", INC, INC),
        ]
    )

    summary = summarize(ratings, (COR, PAR, INC))

    assert (summary.n, summary.agree, summary.rate) == (4, 3, 0.75)
    assert summary.confusion == {(COR, COR): 1, (COR, PAR): 1, (PAR, PAR): 1, (INC, INC): 1}
    # p_o = 3/4; human COR2 PAR1 INC1, judge COR1 PAR2 INC1; p_e = (2*1 + 1*2 + 1*1) / 16 = 5/16;
    # kappa = (12/16 - 5/16) / (11/16) = 7 / 11.
    assert summary.kappa is not None
    assert summary.kappa.value == pytest.approx(7 / 11)


def test_the_report_has_n_everywhere_and_states_the_target_without_deciding_more() -> None:
    key = small_key()
    ratings = ratings_of(
        [
            ("a01", "faithfulness", "real", SUP, SUP),
            ("a02", "faithfulness", "control", NOT, NOT),
            ("a03", "correctness", "real", COR, COR),
            ("a04", "correctness", "real", PAR, INC),
        ]
    )

    report = render_report(ratings, key)

    # All: 3 of 4 agree = 75.0%. Real only (a01, a03, a04): 2 of 3. Controls (a02): 1 of 1, so
    # both raters used one label and kappa is undefined.
    assert "| All items (labels of both kinds pooled) | 4 | 75.0% (3/4) |" in report
    assert "| Faithfulness | 2 | 100.0% (2/2) | 1.000 |" in report
    assert "| Correctness | 2 | 50.0% (1/2) |" in report
    assert "| Real items only (pooled) | 3 | 66.7% (2/3) |" in report
    assert "| Controls only (synthetic) | 1 | 100.0% (1/1) | undefined |" in report
    assert (
        "Kappa of 'Controls only (synthetic)' is undefined: both raters gave every item" in report
    )
    assert "| Exact agreement, all items | 0.750 | 4 | not met |" in report
    assert "At least 10 items labeled: no (n = 4)." in report
    assert "- a04 (correctness, real): human PARTIALLY_CORRECT, judge INCORRECT" in report
    assert "| human \\ judge | CORRECT | PARTIALLY_CORRECT | INCORRECT | total |" in report
    assert "| PARTIALLY_CORRECT | 0 | 0 | 1 | 1 |" in report


def test_the_report_marks_a_met_target_and_handles_a_scope_without_items() -> None:
    key = AgreementKey(sample=sample_info(), items=[item("a01"), item("a02", verdict=NOT)])
    ratings = ratings_of(
        [("a01", "faithfulness", "real", SUP, SUP), ("a02", "faithfulness", "real", NOT, NOT)]
    )

    report = render_report(ratings, key)

    assert "| Exact agreement, all items | 1.000 | 2 | met |" in report
    assert "| Controls only (synthetic) | 0 | n/a | n/a |" in report
    assert "| Correctness | 0 | n/a | n/a |" in report
    assert "None." in report  # no disagreements


# --- grounded eval agreement ---------------------------------------------------------------------


def write_inputs(tmp_path: Path, labels: dict[str, str]) -> tuple[Path, Path]:
    key = small_key()
    key_path = tmp_path / "key.json"
    key_path.write_text(key.model_dump_json(indent=2), encoding="utf-8")
    kinds = {i.item_id: i.kind for i in key.items}
    sheet_path = tmp_path / "sheet.csv"
    sheet_path.write_bytes(
        render_sheet([sheet_row(i, kinds[i], label) for i, label in labels.items()])
    )
    return sheet_path, key_path


def test_the_command_prints_the_report_for_a_complete_sheet(tmp_path: Path) -> None:
    sheet, key = write_inputs(
        tmp_path, {"a01": SUP, "a02": NOT, "a03": COR, "a04": INC}
    )  # all four agree with the judge

    result = runner.invoke(app, ["eval", "agreement", "--labels", str(sheet), "--key", str(key)])

    assert result.exit_code == 0, result.output
    assert "| All items (labels of both kinds pooled) | 4 | 100.0% (4/4) |" in result.output
    assert "Disagreements (n = 0 of 4)" in result.output


def test_the_command_refuses_incomplete_labels_and_lists_the_items(tmp_path: Path) -> None:
    sheet, key = write_inputs(tmp_path, {"a01": SUP, "a02": "", "a03": "maybe", "a04": INC})

    result = runner.invoke(app, ["eval", "agreement", "--labels", str(sheet), "--key", str(key)])

    assert result.exit_code == 1
    assert "- a02: no human_label" in result.output
    assert "- a03: 'maybe' is not one of CORRECT, PARTIALLY_CORRECT, INCORRECT" in result.output
    assert "| All items" not in result.output  # nothing was computed on a smaller n


def test_the_command_says_what_it_cannot_read(tmp_path: Path) -> None:
    sheet, key = write_inputs(tmp_path, {"a01": SUP})

    missing_key = runner.invoke(
        app, ["eval", "agreement", "--labels", str(sheet), "--key", str(tmp_path / "none.json")]
    )
    bad_sheet = tmp_path / "bad.csv"
    bad_sheet.write_text("item_id\na01\n", encoding="utf-8")
    broken = runner.invoke(
        app, ["eval", "agreement", "--labels", str(bad_sheet), "--key", str(key)]
    )

    assert missing_key.exit_code == 1
    assert "Cannot read the key" in missing_key.output
    assert broken.exit_code == 1
    assert "the header lacks the columns" in broken.output
