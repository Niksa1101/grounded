"""The judge-agreement sample (Tech.md §15.4, PRD FR-22, ticket 4.11a): which verdicts a person
labels, drawn from one promptfoo results file of a real generation run. Pure: no file is written
and no model is called here (``judge_export.py`` does that).

Three rules shape the draw:

- **Seeded and reproducible.** Nothing uses a random-number generator: a candidate's rank is the
  sha256 of ``seed | purpose | its key``, so the same seed and file draw the same items on any
  Python version, and the draw does not move when an unrelated candidate is added. The ``item_id``
  order is a seeded shuffle too, so it says nothing about kind, config or source.
- **Stratified by the judge's verdict.** A real run can be lopsided (the baseline's 53 claims were
  all ``SUPPORTED``) and agreement on a lopsided sample means little, so the draw spreads the items
  over the verdicts the run has, as evenly as the run allows, over the configs within a verdict,
  and over different questions where it can.
- **Candidates are what a person can check.** A claim the judge decided without a call (it had no
  usable source) is mechanical and is not a candidate; neither is anything errored or not
  applicable.

**Negative controls.** Where the run has no ``NOT_SUPPORTED`` claim to show, some faithfulness items
are built from real claims: the claim of one question with the cited sources of a claim from
another question, whose sections differ and whose content words barely overlap with the claim. The
right label is then not in doubt. They are synthetic, and the key says so; the sheet does not.
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Final

from pydantic import AliasPath, BaseModel, ConfigDict, Field, ValidationError

from grounded.evals.generation_results import PromptfooResultsError, parse_results
from grounded.schemas.generation_eval import GenerationRunInfo
from grounded.schemas.judge_agreement import (
    ControlInfo,
    ItemSource,
    JudgeRecord,
    KeyItem,
    Kind,
)

# The seed of the committed sample (the ticket number), fixed before the run was looked at.
DEFAULT_SEED: Final = 411
DEFAULT_MAX_OVERLAP: Final = 0.25

# Words nearly every page of the docs shares; they say nothing about whether a source supports a
# claim. Only used to measure how much a control's claim and sources have in common.
_STOP_TEXT: Final = (
    "the and for you your that this with from are can will not but all any has have was were use "
    "used using when which then than they them their its into also only more other such what where "
    "how does doing each about over under after before these those there here see example fastapi "
    "docs"
)
_STOP_WORDS: Final = frozenset(_STOP_TEXT.split())
_WORD: Final = re.compile(r"[a-z0-9_]{3,}")


class SampleError(ValueError):
    """The sample cannot be drawn or written as asked; the message says why."""


# --- Reading the run -----------------------------------------------------------------------------


class _Wire(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)


class _Golden(_Wire):
    id: str
    reference_answer: str


class _Chunk(_Wire):
    label: str
    section_id: str
    content: str


class _Metadata(_Wire):
    golden: _Golden
    context: list[_Chunk] = []


class _JudgedClaim(_Wire):
    claim_index: int
    claim: str
    cited_labels: list[str]
    verdict: str | None = None
    reason: str | None = None
    decided_locally: bool = False
    cache_hit: bool = False


class _JudgeDetail(_Wire):
    prompt_version: str | None = None
    judge_provider: str | None = None
    judge_model: str | None = None
    verdict: str | None = None  # correctness
    reason: str | None = None
    cache_hit: bool = False
    claims: list[_JudgedClaim] = []  # faithfulness


class _Component(_Wire):
    metric: str = Field(validation_alias=AliasPath("assertion", "metric"))
    not_applicable: bool = False
    errored: bool = False
    judge: _JudgeDetail | None = None


class _Row(_Wire):
    """One question of one config. The nested keys of promptfoo's row are read by path; a failed
    generator call has no ``gradingResult`` and no answer, and just yields no components."""

    config: str = Field(validation_alias=AliasPath("provider", "label"))
    question: str = Field(validation_alias=AliasPath("vars", "question"))
    metadata: _Metadata
    answer: str | None = Field(
        default=None, validation_alias=AliasPath("response", "output", "answer_markdown")
    )
    components: list[_Component] = Field(
        default=[], validation_alias=AliasPath("gradingResult", "componentResults")
    )


class _Document(_Wire):
    rows: list[_Row] = Field(validation_alias=AliasPath("results", "results"))


@dataclass(frozen=True, slots=True)
class Source:
    """A chunk a claim cited, as the judge was shown it."""

    label: str  # "c1": the label the chunk had in the answer it was retrieved for
    text: str
    section_id: str


@dataclass(frozen=True, slots=True)
class ClaimCandidate:
    """One claim of a ``hybrid`` answer that the judge graded against its sources."""

    config: str
    question_id: str
    question: str
    judge: JudgeRecord
    claim_index: int
    claim: str
    sources: tuple[Source, ...]

    @property
    def key(self) -> str:
        return f"{self.config}/{self.question_id}/{self.claim_index}"

    @property
    def sections(self) -> frozenset[str]:
        return frozenset(source.section_id for source in self.sources)


@dataclass(frozen=True, slots=True)
class AnswerCandidate:
    """One answer (either config) that the judge graded against the reference answer."""

    config: str
    question_id: str
    question: str
    judge: JudgeRecord
    reference_answer: str
    answer: str

    @property
    def key(self) -> str:
        return f"{self.config}/{self.question_id}"


@dataclass(frozen=True, slots=True)
class Population:
    """What the run offers: every claim and answer the judge graded, with the verdict it gave."""

    info: GenerationRunInfo
    claims: tuple[ClaimCandidate, ...]
    answers: tuple[AnswerCandidate, ...]

    def counts(self) -> dict[str, dict[str, dict[str, int]]]:
        """kind -> config -> verdict -> how many: the input of the draw, public information."""
        counts: dict[str, dict[str, dict[str, int]]] = {"faithfulness": {}, "correctness": {}}
        for kind, group in (("faithfulness", self.claims), ("correctness", self.answers)):
            for c in group:
                per_verdict = counts[kind].setdefault(c.config, {})
                per_verdict[c.judge.verdict] = per_verdict.get(c.judge.verdict, 0) + 1
        return counts


def read_population(raw: bytes) -> Population:
    """The judged claims and answers of a promptfoo results file.

    A claim is a candidate when a model call gave it a verdict and it had sources: a claim decided
    without a call (no usable source) is mechanical and a person has nothing to check. A component
    that is errored or not applicable has no verdict and is left out, as in the gate. Raises
    ``SampleError`` for a file that is not a readable run, or whose judge record names a source its
    own recorded context does not have.
    """
    try:
        info = parse_results(raw).info
        rows = _Document.model_validate_json(raw).rows
    except (PromptfooResultsError, ValidationError) as exc:
        raise SampleError(f"not a promptfoo results file I can read: {exc}") from exc
    claims: list[ClaimCandidate] = []
    answers: list[AnswerCandidate] = []
    for row in rows:
        config, qid, question = row.config, row.metadata.golden.id, row.question
        for component in row.components:  # none when the generator failed: nothing was judged
            judge = component.judge
            if component.not_applicable or component.errored or judge is None:
                continue
            if component.metric == "correctness" and judge.verdict and judge.reason:
                if row.answer is None:
                    raise SampleError(f"{config} {qid}: a graded answer without an answer")
                record = _record(judge, judge.verdict, judge.reason, judge.cache_hit)
                reference = row.metadata.golden.reference_answer
                answers.append(
                    AnswerCandidate(config, qid, question, record, reference, row.answer)
                )
            elif component.metric == "faithfulness":
                chunks = {chunk.label: chunk for chunk in row.metadata.context}
                for entry in judge.claims:
                    if entry.verdict is None or entry.reason is None or entry.decided_locally:
                        continue
                    unknown = [label for label in entry.cited_labels if label not in chunks]
                    if unknown:
                        raise SampleError(
                            f"{config} {qid} claim {entry.claim_index}: the judge saw "
                            f"{', '.join(unknown)}, which the recorded context does not have"
                        )
                    sources = tuple(
                        Source(label, chunks[label].content, chunks[label].section_id)
                        for label in entry.cited_labels
                    )
                    record = _record(judge, entry.verdict, entry.reason, entry.cache_hit)
                    claims.append(
                        ClaimCandidate(
                            config, qid, question, record, entry.claim_index, entry.claim, sources
                        )
                    )
    return Population(info, tuple(claims), tuple(answers))


def _record(judge: _JudgeDetail, verdict: str, reason: str, cache_hit: bool) -> JudgeRecord:
    if not (judge.prompt_version and judge.judge_provider and judge.judge_model):
        raise SampleError("a scored judge component does not name its rubric, provider and model")
    return JudgeRecord(
        verdict=verdict,
        reason=reason,
        prompt_version=judge.prompt_version,
        judge_provider=judge.judge_provider,
        judge_model=judge.judge_model,
        cache_hit=cache_hit,
    )


# --- The draw ------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SampleParams:
    seed: int
    faithfulness: int  # items of this kind, controls included
    correctness: int
    controls: int  # of the faithfulness items
    max_overlap: float  # the most a control's claim may share with its sources

    def __post_init__(self) -> None:
        if min(self.faithfulness, self.correctness, self.controls) < 0:
            raise SampleError("the item counts cannot be negative")
        if self.controls > self.faithfulness:
            raise SampleError(f"{self.controls} controls do not fit in {self.faithfulness} items")
        if not 0 <= self.max_overlap <= 1:
            raise SampleError("the overlap limit is a share between 0 and 1")


def _rank(seed: int, purpose: str, key: str) -> str:
    return hashlib.sha256(f"{seed}\x1f{purpose}\x1f{key}".encode()).hexdigest()


def _spread(total: int, capacity: dict[str, int], order: Sequence[str]) -> dict[str, int]:
    """``total`` units over the keys of ``capacity`` as evenly as the capacities allow: one at a
    time round the keys in ``order``, so the remainder goes to the keys that come first."""
    given = dict.fromkeys(order, 0)
    left = total
    while left > 0 and any(given[k] < capacity[k] for k in order):
        for key in order:
            if left > 0 and given[key] < capacity[key]:
                given[key] += 1
                left -= 1
    return given


def select[C: (ClaimCandidate, AnswerCandidate)](
    pool: Sequence[C], n: int, seed: int, purpose: str
) -> list[C]:
    """``n`` candidates, spread over the judge's verdicts (the bigger group first when they do not
    divide), within a verdict over the configs, and over different questions while there are any."""
    if n > len(pool):
        raise SampleError(f"asked for {n} {purpose} items, the run has {len(pool)}")
    by_verdict: dict[str, list[C]] = defaultdict(list)
    for candidate in pool:
        by_verdict[candidate.judge.verdict].append(candidate)
    verdicts = sorted(by_verdict, key=lambda v: (-len(by_verdict[v]), v))
    quota = _spread(n, {v: len(by_verdict[v]) for v in verdicts}, verdicts)
    chosen: list[C] = []
    seen: set[str] = set()
    for verdict in verdicts:
        by_config: dict[str, list[C]] = defaultdict(list)
        for candidate in by_verdict[verdict]:
            by_config[candidate.config].append(candidate)
        configs = sorted(by_config, key=lambda c: _rank(seed, f"{purpose}:{verdict}", c))
        share = _spread(quota[verdict], {c: len(by_config[c]) for c in configs}, configs)
        for config in configs:
            ranked = sorted(
                by_config[config], key=lambda c: _rank(seed, f"{purpose}:{verdict}", c.key)
            )
            picked: list[C] = []
            for _ in range(share[config]):
                left = [c for c in ranked if c not in picked]
                pick = next((c for c in left if c.question_id not in seen), left[0])
                picked.append(pick)
                seen.add(pick.question_id)
            chosen += picked
    return chosen


def content_words(text: str) -> frozenset[str]:
    return frozenset(w for w in _WORD.findall(text.lower()) if w not in _STOP_WORDS)


def token_overlap(claim: str, sources_text: str) -> float:
    """The share of the claim's content words that also occur in the sources: a cheap check that
    a claim and some sources are not accidentally about the same thing. 0.0 for no words."""
    words = content_words(claim)
    return len(words & content_words(sources_text)) / len(words) if words else 0.0


def build_controls(
    pool: Sequence[ClaimCandidate], n: int, seed: int, max_overlap: float
) -> list[tuple[ClaimCandidate, ClaimCandidate, float]]:
    """``n`` pairs ``(claim donor, sources donor, overlap)`` from different questions, none reused.

    The two cite different sections (a section cited by both could support the claim) and the claim
    shares at most ``max_overlap`` of its content words with the other claim's sources. Candidates
    are tried in seeded order, so the result is the same on every run.
    """
    made: list[tuple[ClaimCandidate, ClaimCandidate, float]] = []
    used: set[str] = set()
    for donor in sorted(pool, key=lambda c: _rank(seed, "control-claim", c.key)):
        if len(made) == n:
            break
        if donor.question_id in used:
            continue
        partners = sorted(
            (
                b
                for b in pool
                if b.question_id not in used | {donor.question_id}
                and not donor.sections & b.sections
            ),
            key=lambda b: _rank(seed, f"control-sources:{donor.key}", b.key),
        )
        for partner in partners:
            overlap = token_overlap(donor.claim, " ".join(s.text for s in partner.sources))
            if overlap <= max_overlap:
                made.append((donor, partner, overlap))
                used |= {donor.question_id, partner.question_id}
                break
    if len(made) < n:
        raise SampleError(
            f"could build only {len(made)} of {n} controls from the claims left after the real "
            f"items (overlap limit {max_overlap}): ask for fewer or raise the limit"
        )
    return made


@dataclass(frozen=True, slots=True)
class SampleItem:
    """One row of the sheet and its entry in the key."""

    item_id: str
    kind: Kind
    source: ItemSource
    config: str
    question_id: str
    question: str
    judge: JudgeRecord | None  # a control's verdict is asked for later
    claim_index: int | None = None
    claim: str = ""
    sources: tuple[Source, ...] = ()
    reference_answer: str = ""
    candidate_answer: str = ""
    control: ControlInfo | None = None

    @property
    def natural_key(self) -> str:
        """Identifies the item whatever its id and verdict are (the id comes from the shuffle)."""
        partner = self.control.sources_question_id if self.control else ""
        head = f"{self.kind}:{self.source}:{self.config}:{self.question_id}"
        return f"{head}:{self.claim_index}:{partner}"

    def key_item(self, judge: JudgeRecord | None) -> KeyItem:
        return KeyItem(
            item_id=self.item_id,
            kind=self.kind,
            source=self.source,
            config=self.config,
            question_id=self.question_id,
            claim_index=self.claim_index,
            judge=judge,
            control=self.control,
        )


def build_sample(population: Population, params: SampleParams) -> list[SampleItem]:
    """The items, with ids ``a01``... in a seeded shuffle of kind, source and config."""
    seed = params.seed
    real = select(population.claims, params.faithfulness - params.controls, seed, "faithfulness")
    answers = select(population.answers, params.correctness, seed, "correctness")
    # A control must not share a question with a real item: the same claim twice would give it away.
    taken = {c.question_id for c in real}
    pool = [c for c in population.claims if c.question_id not in taken]
    items = [_claim_item(c, "real", c.sources, c.judge, None) for c in real]
    for donor, partner, overlap in build_controls(pool, params.controls, seed, params.max_overlap):
        control = ControlInfo(
            sources_config=partner.config,
            sources_question_id=partner.question_id,
            sources_claim_index=partner.claim_index,
            sources_labels=tuple(s.label for s in partner.sources),
            token_overlap=round(overlap, 4),
            max_overlap=params.max_overlap,
        )
        items.append(_claim_item(donor, "control", partner.sources, None, control))
    items += [
        SampleItem(
            "",
            "correctness",
            "real",
            a.config,
            a.question_id,
            a.question,
            a.judge,
            reference_answer=a.reference_answer,
            candidate_answer=a.answer,
        )
        for a in answers
    ]
    ordered = sorted(items, key=lambda i: _rank(seed, "order", i.natural_key))
    return [replace(item, item_id=f"a{n:02d}") for n, item in enumerate(ordered, 1)]


def _claim_item(
    c: ClaimCandidate,
    source: ItemSource,
    sources: tuple[Source, ...],
    judge: JudgeRecord | None,
    control: ControlInfo | None,
) -> SampleItem:
    return SampleItem(
        "",
        "faithfulness",
        source,
        c.config,
        c.question_id,
        c.question,
        judge,
        claim_index=c.claim_index,
        claim=c.claim,
        sources=sources,
        control=control,
    )
