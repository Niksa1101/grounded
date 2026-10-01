"""Golden-set rows (Tech.md §15.1) and retrieval-eval output (§15.2). Pydantic is the authority on
these files' shape (AGENTS.md §6.2).

Rules that need only the row itself live here. Rules that need the corpus (a label resolves to
a chunk; two labels never match the same chunk, which is how nested labels show up) live in
``evals/golden.py``.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from grounded.retrieval.config import RetrievalConfig

GoldenType = Literal["factual", "how_to", "code", "multi_section", "unanswerable"]

# "docs/en/docs/<page>.md" or "docs/en/docs/<page>.md#<anchor>" (the page alone = the whole page).
_SECTION_PATTERN = r"^docs/en/docs/[^#\s]+\.md(#[^#\s]+)?$"


class RelevantSection(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    section: str = Field(pattern=_SECTION_PATTERN)
    grade: Literal[1, 2]  # 2 = contains the answer, 1 = useful context

    @property
    def page(self) -> str:
        return self.section.partition("#")[0]


class GoldenItem(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(pattern=r"^q\d{3}$")
    question: str = Field(min_length=3, max_length=500)  # the same limits as AskRequest
    type: GoldenType
    answerable: bool
    reference_answer: str = Field(min_length=1)
    relevant_sections: list[RelevantSection]
    # Provenance: the sampled section the question was drafted from (None if written freely,
    # e.g. an unanswerable question). Not used by any metric.
    source_section: str | None = Field(default=None, pattern=_SECTION_PATTERN)
    notes: str = ""

    @model_validator(mode="after")
    def _check_labels(self) -> Self:
        if self.answerable == (self.type == "unanswerable"):
            raise ValueError("answerable must be false exactly when type is 'unanswerable'")
        labels = [s.section for s in self.relevant_sections]
        if len(set(labels)) != len(labels):
            raise ValueError("a section is labeled twice")
        grade_2 = sum(1 for s in self.relevant_sections if s.grade == 2)
        if not self.answerable and labels:
            raise ValueError("an unanswerable item has no relevant sections")
        if self.answerable and grade_2 == 0:
            raise ValueError("an answerable item needs at least one grade-2 section")
        if self.type == "multi_section" and grade_2 < 2:
            raise ValueError("a multi_section item needs at least two grade-2 sections")
        # A whole page and a section on it are nested; H2/H3 nesting needs the corpus to see.
        whole_pages = {s.page for s in self.relevant_sections if "#" not in s.section}
        if any(s.page in whole_pages for s in self.relevant_sections if "#" in s.section):
            raise ValueError("a page and a section on it are both labeled (nested labels)")
        return self

    @property
    def relevant(self) -> dict[str, int]:
        """Label → grade, the shape the metrics take."""
        return {s.section: s.grade for s in self.relevant_sections}


# --- Retrieval eval output (Tech.md §15.2) -----------------------------------------------------
# Results files (eval/results/, gitignored) and the committed baseline
# (eval/baselines/retrieval.json) are read back by later tooling (the Phase 2 gate), so their shape
# is validated like any input.

_FROZEN = ConfigDict(frozen=True, extra="forbid")


class RetrievalQuestionResult(BaseModel):
    """One scored question: enough to diff two runs question by question."""

    model_config = _FROZEN

    id: str
    type: GoldenType
    metrics: dict[str, float]
    ranks: dict[str, int | None]  # label → rank of its first matching chunk; None = not retrieved
    retrieved: list[str]  # section_id of each retrieved chunk, rank 1 first


class RetrievalConfigResult(BaseModel):
    """Every answerable question run through one retrieval mode, and the means over them."""

    model_config = _FROZEN

    config: str  # retrieval mode: "dense" (Phase 2 adds "fts", "hybrid")
    # The settings that produced the rows, and their hash: two results are only comparable when
    # this says they ran the same retrieval.
    retrieval_config: RetrievalConfig
    retrieval_config_hash: str
    k: int = Field(ge=1)  # chunks retrieved per question (K_DENSE for dense)
    n: int = Field(ge=1)  # scored (answerable) questions
    skipped_unanswerable: int = Field(ge=0)
    metrics: dict[str, float]  # mean over the n questions
    questions: list[RetrievalQuestionResult]

    @model_validator(mode="after")
    def _check_config(self) -> Self:
        # A results file is read back by the gate: a hash that doesn't match its config (an edited
        # file) or a config for another mode must not pass as this result's identity.
        if self.retrieval_config_hash != self.retrieval_config.config_hash:
            raise ValueError("retrieval_config_hash does not match retrieval_config")
        if self.retrieval_config.mode != self.config:
            raise ValueError("retrieval_config.mode does not match config")
        return self


class RetrievalRunInfo(BaseModel):
    """What produced a run: golden set, index, embedding model and code version."""

    model_config = _FROZEN

    date: datetime
    git_sha: str | None  # repo HEAD; None outside a git checkout
    git_dirty: bool | None  # uncommitted changes when the run started
    golden_set_version: str  # "v1"
    golden_set_sha256: str  # of the file's bytes (LF-normalized), so an edit in place is visible
    index_version_id: int  # only meaningful in the database the run used
    index_config_hash: str  # identifies the index across databases (local, CI)
    fastapi_ref: str
    fastapi_sha: str
    embedding_model: str
    embedding_dim: int


class RetrievalRun(BaseModel):
    """One ``grounded eval retrieval`` run: the contents of a results file."""

    model_config = _FROZEN

    suite: Literal["retrieval"] = "retrieval"
    info: RetrievalRunInfo
    configs: dict[str, RetrievalConfigResult]


class MetricThreshold(BaseModel):
    """The gate rule for one metric: the run may be at most ``tolerance`` below the baseline
    (``current >= baseline - tolerance``). The tolerance is policy, set by hand in the baseline
    file (AGENTS.md §6.7), never in code."""

    model_config = _FROZEN

    tolerance: float = Field(ge=0)


class RetrievalBaselineEntry(BaseModel):
    """One config's row in ``eval/baselines/retrieval.json``. The numbers are copied from a run,
    never typed; ``thresholds`` is the one hand-written part.

    ``thresholds`` maps a metric name to its gate rule. A config with thresholds is **gated**: the
    gate fails the suite when any of its metrics falls too far. A config without them is only
    reported. Names must be keys of ``metrics``. ``update_baseline`` keeps a row's thresholds when
    it refreshes the numbers.

    ``golden_set_sha256`` and ``git_dirty`` are required (a row written before R.03 can no longer be
    read: regenerate it in a baseline PR). ``retrieval_config_hash`` stays optional: no gate rule
    needs it yet.
    """

    model_config = _FROZEN

    metrics: dict[str, float]
    thresholds: dict[str, MetricThreshold] = Field(default_factory=dict)
    n: int = Field(ge=1)
    k: int = Field(ge=1)
    golden_set_version: str
    golden_set_sha256: str
    index_config_hash: str
    retrieval_config_hash: str | None = None
    fastapi_ref: str
    fastapi_sha: str
    embedding_model: str
    embedding_dim: int
    git_sha: str | None
    git_dirty: bool
    date: datetime

    @model_validator(mode="after")
    def _check_thresholds(self) -> Self:
        unknown = sorted(set(self.thresholds) - set(self.metrics))
        if unknown:
            raise ValueError(f"thresholds for metrics the row doesn't have: {', '.join(unknown)}")
        return self
