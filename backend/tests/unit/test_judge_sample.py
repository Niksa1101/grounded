"""The judge-agreement sample (4.11a, Tech §15.4): reading the claims and answers the judge graded,
the seeded stratified draw, the negative controls and the shuffled item order.

The reading is tested on the committed promptfoo recording (scripted models: it checks the
mechanics, not any result). The draw is tested on populations built here, big enough to tell a
design from luck: the properties asserted (balance across verdicts and configs, different
questions, disjoint sections, low overlap) hold whatever the seed, and determinism is asserted by
running twice.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from grounded.evals.judge_sample import (
    AnswerCandidate,
    ClaimCandidate,
    Population,
    SampleError,
    SampleParams,
    Source,
    _spread,  # pyright: ignore[reportPrivateUsage]
    build_controls,
    build_sample,
    read_population,
    select,
    token_overlap,
)
from grounded.schemas.generation_eval import GenerationRunInfo
from grounded.schemas.judge_agreement import JudgeRecord

FIXTURE = (
    Path(__file__).resolve().parents[1] / "fixtures" / "promptfoo" / "results_sample_judge.json"
)
SUP, NOT = "SUPPORTED", "NOT_SUPPORTED"
COR, PAR, INC = "CORRECT", "PARTIALLY_CORRECT", "INCORRECT"
INFO = GenerationRunInfo(
    date=datetime(2026, 10, 8, tzinfo=UTC),
    promptfoo_version="0.123.1",
    golden_set_version="v1",
    golden_set_sha256="a" * 64,
)


def record(verdict: str) -> JudgeRecord:
    return JudgeRecord(
        verdict=verdict,
        reason="A reason.",
        prompt_version="judge_x_v1@00000000",
        judge_provider="groq",
        judge_model="openai/gpt-oss-120b",
    )


def claim(n: int, verdict: str = SUP, *, config: str = "hybrid", index: int = 0) -> ClaimCandidate:
    """Claim ``n`` of question ``q{n}``: its own section and its own words (``topic<n>``), so two
    claims of different questions have nothing in common."""
    return ClaimCandidate(
        config,
        f"q{n:03d}",
        f"What is topic{n}?",
        record(verdict),
        index,
        f"Topic{n} works through widget{n} and gadget{n}.",
        (Source("c1", f"The page on topic{n} explains widget{n}.", f"docs/p{n}.md#s{n}"),),
    )


def answer(n: int, verdict: str, config: str) -> AnswerCandidate:
    return AnswerCandidate(
        config, f"q{n:03d}", f"What is topic{n}?", record(verdict), "Ref.", "Ans."
    )


def population(
    claims: list[ClaimCandidate] | None = None, answers: list[AnswerCandidate] | None = None
) -> Population:
    return Population(INFO, tuple(claims or []), tuple(answers or []))


def answers_of_a_run() -> list[AnswerCandidate]:
    """30 questions x 2 configs, like the baseline run: hybrid mostly correct, no_rag mixed."""
    hybrid = [answer(n, COR if n % 8 else PAR, "hybrid") for n in range(1, 31)]
    no_rag = [answer(n, (COR, INC, INC, PAR)[n % 4], "no_rag") for n in range(1, 31)]
    return hybrid + no_rag


# --- Reading the run -----------------------------------------------------------------------------


def test_the_judged_claims_and_answers_of_a_recorded_run_are_the_candidates() -> None:
    pop = read_population(FIXTURE.read_bytes())

    claims = {c.key: c for c in pop.claims}
    # q007's second claim cites no valid source (decided without a call), q008 and q047 are errored
    # components and q049 was skipped: none of them is something a person can check.
    assert sorted(claims) == ["hybrid/q003/0", "hybrid/q003/1", "hybrid/q007/0", "hybrid/q015/0"]
    assert claims["hybrid/q003/1"].judge.verdict == NOT
    assert [(s.label, s.section_id) for s in claims["hybrid/q003/1"].sources] == [
        ("c1", "docs/en/docs/tutorial/index.md#install-fastapi"),
        ("c2", "docs/en/docs/index.md#without-fastapi-cloud-cli"),
    ]
    assert claims["hybrid/q003/0"].question.startswith("How do I install FastAPI")
    assert sorted(a.key for a in pop.answers) == sorted(
        f"{config}/{q}"
        for config, q in [
            ("hybrid", "q003"),
            ("hybrid", "q007"),
            ("hybrid", "q008"),
            ("hybrid", "q045"),
        ]
        + [("no_rag", q) for q in ("q003", "q007", "q008", "q045", "q047")]
    )  # q015, q047 (hybrid) and q049 have a correctness error
    assert pop.info.golden_set_version == "v1"


def test_the_population_is_counted_by_kind_config_and_verdict() -> None:
    counts = read_population(FIXTURE.read_bytes()).counts()

    assert counts["faithfulness"] == {"hybrid": {SUP: 3, NOT: 1}}
    assert counts["correctness"] == {
        "no_rag": {INC: 4, PAR: 1},
        "hybrid": {PAR: 1, COR: 3},
    }


def test_a_judge_record_that_names_a_source_the_context_lacks_is_refused() -> None:
    data: dict[str, Any] = json.loads(FIXTURE.read_bytes())
    for row in data["results"]["results"]:
        if row["provider"]["label"] == "hybrid" and row["metadata"]["golden"]["id"] == "q003":
            row["metadata"]["context"] = row["metadata"]["context"][:1]  # drops c2

    with pytest.raises(SampleError, match=r"hybrid q003 claim 1: the judge saw c2"):
        read_population(json.dumps(data).encode())


def test_a_file_that_is_not_a_run_is_refused() -> None:
    with pytest.raises(SampleError, match="not a promptfoo results file"):
        read_population(b'{"results": {}}')


# --- Spreading units -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("total", "capacity", "expected"),
    [
        (10, {"a": 40, "b": 3}, {"a": 7, "b": 3}),  # b is full: a takes the rest
        (5, {"x": 10, "y": 10, "z": 10}, {"x": 2, "y": 2, "z": 1}),  # the remainder goes first
        (4, {"a": 1, "b": 10}, {"a": 1, "b": 3}),
        (0, {"a": 2}, {"a": 0}),
    ],
)
def test_spread_is_even_up_to_the_capacities(
    total: int, capacity: dict[str, int], expected: dict[str, int]
) -> None:
    assert _spread(total, capacity, list(capacity)) == expected


# --- The stratified draw -------------------------------------------------------------------------


def test_a_lopsided_run_is_spread_over_the_verdicts_it_has() -> None:
    pool = [claim(n, SUP) for n in range(1, 13)] + [claim(n, NOT) for n in range(13, 16)]

    chosen = select(pool, 6, 411, "faithfulness")

    assert sorted(c.judge.verdict for c in chosen) == [NOT] * 3 + [SUP] * 3


def test_the_rarer_verdict_gives_what_it_has_and_the_rest_comes_from_the_other() -> None:
    pool = [claim(n, SUP) for n in range(1, 13)] + [claim(13, NOT)]

    chosen = select(pool, 6, 411, "faithfulness")

    assert sorted(c.judge.verdict for c in chosen) == [NOT] + [SUP] * 5


def test_an_odd_count_gives_the_remainder_to_the_bigger_group() -> None:
    pool = [claim(n, SUP) for n in range(1, 13)] + [claim(n, NOT) for n in range(13, 24)]

    chosen = select(pool, 5, 411, "faithfulness")

    assert sorted(c.judge.verdict for c in chosen) == [NOT] * 2 + [SUP] * 3


def test_a_run_with_one_verdict_gives_that_verdict() -> None:
    chosen = select([claim(n) for n in range(1, 9)], 4, 411, "faithfulness")

    assert {c.judge.verdict for c in chosen} == {SUP}
    assert len(chosen) == 4


def test_both_configs_are_drawn_within_a_grade() -> None:
    pool = [answer(n, COR, "hybrid") for n in range(1, 11)]
    pool += [answer(n, COR, "no_rag") for n in range(11, 21)]

    chosen = select(pool, 4, 411, "correctness")

    assert sorted(a.config for a in chosen) == ["hybrid", "hybrid", "no_rag", "no_rag"]


def test_different_questions_are_preferred_to_a_second_claim_of_one() -> None:
    crowded = [claim(1, index=i) for i in range(5)]  # five claims of q001
    pool = crowded + [claim(n) for n in range(2, 5)]

    for seed in range(20):
        chosen = select(pool, 4, seed, "faithfulness")
        assert len({c.question_id for c in chosen}) == 4


def test_asking_for_more_than_the_run_has_is_an_error_and_zero_is_nothing() -> None:
    with pytest.raises(SampleError, match="asked for 3 correctness items, the run has 2"):
        select([answer(1, COR, "hybrid"), answer(2, COR, "hybrid")], 3, 411, "correctness")
    assert select([], 0, 411, "faithfulness") == []


def test_the_draw_is_deterministic_and_the_seed_matters() -> None:
    pool = answers_of_a_run()

    first = select(pool, 10, 411, "correctness")
    again = select(pool, 10, 411, "correctness")
    others = {tuple(a.key for a in select(pool, 10, seed, "correctness")) for seed in range(1, 6)}

    assert first == again
    assert len(others | {tuple(a.key for a in first)}) > 1


# --- Negative controls ---------------------------------------------------------------------------


def test_token_overlap_is_the_share_of_the_claims_content_words_in_the_sources() -> None:
    # "other" is a stop word, so the claim has 4 content words and 2 of them are in the sources.
    assert token_overlap("alpha beta gamma delta", "alpha beta other") == 0.5
    assert (
        token_overlap("Use Query max_length", "query validation max_length") == 1.0
    )  # "use" is a stop word
    assert token_overlap("the and for", "anything at all") == 0.0  # no content words


def test_a_control_pairs_a_claim_with_the_sources_of_a_different_question() -> None:
    pool = [claim(n) for n in range(1, 13)]

    pairs = build_controls(pool, 4, 411, 0.25)

    assert len(pairs) == 4
    questions = [q for donor, partner, _ in pairs for q in (donor.question_id, partner.question_id)]
    assert len(set(questions)) == 8  # every question once: no pair shares one, none is reused
    for donor, partner, overlap in pairs:
        assert not donor.sections & partner.sections
        assert overlap <= 0.25
        assert overlap == token_overlap(donor.claim, " ".join(s.text for s in partner.sources))


def test_the_controls_are_the_same_on_every_run_and_follow_the_seed() -> None:
    pool = [claim(n) for n in range(1, 25)]

    first = build_controls(pool, 4, 411, 0.25)

    assert build_controls(pool, 4, 411, 0.25) == first
    assert any(build_controls(pool, 4, seed, 0.25) != first for seed in range(1, 6))


def test_sources_that_share_the_claims_words_are_not_used() -> None:
    first = claim(
        1
    )  # claim about topic1, widget1, gadget1; its sources: "page ... topic1 ... widget1"
    echo = ClaimCandidate(  # q002's claim and sources are about exactly what q001 says
        "hybrid", "q002", "Q?", record(SUP), 0, "Page explains topic1 widget1.",
        (Source("c1", "Topic1 works through widget1 and gadget1 here.", "docs/other.md#x"),),
    )  # fmt: skip

    with pytest.raises(SampleError, match="could build only 0 of 1 controls"):
        build_controls([first, echo], 1, 411, 0.25)
    # The sections differ, so only the words kept it out: a limit of 1 lets the pair through.
    [(donor, partner, overlap)] = build_controls([first, echo], 1, 411, 1.0)
    assert {donor.question_id, partner.question_id} == {"q001", "q002"}
    assert overlap > 0.25


def test_sources_from_the_same_section_are_not_used() -> None:
    twin = ClaimCandidate(  # unrelated words, but it cites the section q001 cites
        "hybrid", "q003", "Q?", record(SUP), 0, "Different words eta.",
        (Source("c1", "Nothing alike.", "docs/p1.md#s1"),),
    )  # fmt: skip

    with pytest.raises(SampleError, match="could build only 0 of 1 controls"):
        build_controls([claim(1), twin], 1, 411, 1.0)  # not even a limit of 1 allows it


def test_two_claims_of_one_question_are_not_a_control() -> None:
    with pytest.raises(SampleError, match="could build only 0 of 1 controls"):
        build_controls([claim(1, index=0), claim(1, index=1)], 1, 411, 1.0)


def test_no_controls_are_needed_for_zero() -> None:
    assert build_controls([], 0, 411, 0.25) == []


# --- The whole sample ----------------------------------------------------------------------------


PARAMS = SampleParams(seed=411, faithfulness=10, correctness=10, controls=4, max_overlap=0.25)


def a_run() -> Population:
    return population([claim(n) for n in range(1, 25)], answers_of_a_run())


def test_the_sample_has_the_asked_kinds_sources_and_ids() -> None:
    items = build_sample(a_run(), PARAMS)

    assert [i.item_id for i in items] == [f"a{n:02d}" for n in range(1, 21)]
    assert [i.kind for i in items].count("faithfulness") == 10
    assert [i.kind for i in items].count("correctness") == 10
    assert [(i.kind, i.source) for i in items].count(("faithfulness", "control")) == 4
    assert all(i.source == "real" for i in items if i.kind == "correctness")


def test_real_items_carry_the_runs_verdict_and_controls_wait_for_the_judge() -> None:
    items = build_sample(a_run(), PARAMS)

    assert all(i.judge is not None for i in items if i.source == "real")
    assert all(i.judge is None for i in items if i.source == "control")


def test_a_control_is_a_real_claim_with_the_sources_of_another_question() -> None:
    items = build_sample(a_run(), PARAMS)
    claims = {c.key: c for c in a_run().claims}
    real_questions = {
        i.question_id for i in items if i.kind == "faithfulness" and i.source == "real"
    }

    for item in (i for i in items if i.source == "control"):
        assert item.control is not None
        assert item.claim_index is not None
        donor = claims[f"{item.config}/{item.question_id}/{item.claim_index}"]
        partner = claims[
            f"{item.control.sources_config}/{item.control.sources_question_id}/"
            f"{item.control.sources_claim_index}"
        ]
        assert item.claim == donor.claim  # the claim is the donor's own
        assert item.sources == partner.sources  # the evidence is another question's
        assert item.control.sources_labels == tuple(s.label for s in partner.sources)
        assert item.question_id != item.control.sources_question_id
        assert item.question_id not in real_questions  # no real item shows the same question
        assert item.control.sources_question_id not in real_questions


def test_the_correctness_items_cover_every_grade_and_both_configs() -> None:
    items = build_sample(a_run(), PARAMS)
    chosen = [i for i in items if i.kind == "correctness"]

    assert {i.judge.verdict for i in chosen if i.judge} == {COR, PAR, INC}
    assert {i.config for i in chosen} == {"hybrid", "no_rag"}
    assert len({i.question_id for i in chosen}) == 10  # ten different questions


def test_the_sample_is_the_same_for_the_same_seed_and_another_for_another() -> None:
    first = build_sample(a_run(), PARAMS)

    assert build_sample(a_run(), PARAMS) == first
    other = build_sample(a_run(), SampleParams(412, 10, 10, 4, 0.25))
    assert [i.natural_key for i in other] != [i.natural_key for i in first]


@pytest.mark.parametrize("seed", [411, *range(1, 11)])
def test_the_order_does_not_group_kinds_or_controls(seed: int) -> None:
    items = build_sample(a_run(), SampleParams(seed, 10, 10, 4, 0.25))
    kinds = [i.kind for i in items]
    controls = [n for n, i in enumerate(items) if i.source == "control"]

    assert kinds[:10] not in (["faithfulness"] * 10, ["correctness"] * 10)
    assert controls != list(range(controls[0], controls[0] + 4))  # not one block


def test_the_order_is_not_the_order_of_construction() -> None:
    """Real faithfulness items are built first, then the controls, then the answers; the shuffle
    must not leave that order visible in the item ids."""
    items = build_sample(a_run(), PARAMS)
    order = [(i.kind, i.source) for i in items]
    built = (
        [("faithfulness", "real")] * 6
        + [("faithfulness", "control")] * 4
        + [("correctness", "real")] * 10
    )

    assert order != built


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"faithfulness": 3, "controls": 4}, "4 controls do not fit in 3 items"),
        ({"correctness": -1}, "cannot be negative"),
        ({"max_overlap": 1.5}, "share between 0 and 1"),
    ],
)
def test_the_parameters_are_checked(kwargs: dict[str, Any], message: str) -> None:
    params: dict[str, Any] = {
        "seed": 1, "faithfulness": 10, "correctness": 10, "controls": 4, "max_overlap": 0.25,
    }  # fmt: skip

    with pytest.raises(SampleError, match=message):
        SampleParams(**{**params, **kwargs})


def test_a_run_too_small_for_the_sample_is_an_error_not_a_smaller_sample() -> None:
    small = population([claim(n) for n in range(1, 9)], answers_of_a_run())

    # 8 claims give 6 real items; the 2 left (two questions) make one control, not four.
    with pytest.raises(SampleError, match="could build only 1 of 4 controls"):
        build_sample(small, PARAMS)
