"""The key of the judge-agreement sample (Tech.md §15.4, ticket 4.11).

``grounded eval export-verdicts`` writes two things from one sample: the blind labeling sheet
(``eval/judge_agreement/v1.csv``, committed, no verdict in it) and this key, which says what each
``item_id`` is and what the judge said. The key stays out of the repository until the Author has
labeled the sheet (4.11b), so the labels are made without seeing a verdict; ``grounded eval
agreement`` joins the two. Pydantic is the authority on its shape (AGENTS.md §6.2).

An item is a *real* one (a claim or an answer of the baseline run, with the verdict the judge gave
in that run) or a *control* (a real claim paired with the sources of a claim from another question,
so the right verdict is known; its verdict comes from a fresh call of the same judge).
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from types import MappingProxyType
from typing import Literal, Self, get_args

from pydantic import BaseModel, ConfigDict, Field, model_validator

from grounded.schemas.judge import CorrectnessLabel, FaithfulnessJudgment, FaithfulnessLabel

Kind = Literal["faithfulness", "correctness"]
ItemSource = Literal["real", "control"]

# What a person may write for each kind: the labels of the rubrics, taken from the schema that
# validates the judge's reply, so the sheet and the judge cannot disagree on the words.
ALLOWED_LABELS: Mapping[Kind, tuple[str, ...]] = MappingProxyType(
    {"faithfulness": get_args(FaithfulnessLabel), "correctness": get_args(CorrectnessLabel)}
)

_FROZEN = ConfigDict(frozen=True, extra="forbid")


class JudgeRecord(BaseModel):
    """One verdict of the judge and where it came from."""

    model_config = _FROZEN

    verdict: str
    reason: str
    prompt_version: str  # "<rubric>@<8 hex>"
    judge_provider: str
    judge_model: str
    decided_locally: bool = False
    cache_hit: bool = False

    @classmethod
    def from_claim(cls, judgment: FaithfulnessJudgment) -> Self:
        """The record of a claim judged now. Only a judgment with a verdict has one."""
        if judgment.verdict is None or judgment.reason is None:
            raise ValueError("an errored judgment has no verdict to record")
        return cls(
            verdict=judgment.verdict,
            reason=judgment.reason,
            prompt_version=judgment.prompt_version,
            judge_provider=judgment.judge_provider,
            judge_model=judgment.judge_model,
            decided_locally=judgment.decided_locally,
            cache_hit=judgment.cache_hit,
        )


class ControlInfo(BaseModel):
    """How a control was built: the claim is the item's own, the sources are another claim's."""

    model_config = _FROZEN

    sources_config: str
    sources_question_id: str
    sources_claim_index: int = Field(ge=0)
    sources_labels: tuple[str, ...]
    token_overlap: float = Field(ge=0, le=1)  # share of the claim's content words in the sources
    max_overlap: float = Field(ge=0, le=1)  # the limit the pair was chosen under


class KeyItem(BaseModel):
    model_config = _FROZEN

    item_id: str = Field(pattern=r"^a\d{2,}$")
    kind: Kind
    source: ItemSource
    config: str  # the promptfoo config of the claim or answer: "hybrid" or "no_rag"
    question_id: str = Field(pattern=r"^q\d{3}$")
    claim_index: int | None = Field(default=None, ge=0)  # faithfulness only
    judge: JudgeRecord | None = None  # None: not asked yet (a control, quota ran out)
    control: ControlInfo | None = None

    @model_validator(mode="after")
    def _check(self) -> Self:
        if (self.kind == "faithfulness") != (self.claim_index is not None):
            raise ValueError("a claim index exists exactly for a faithfulness item")
        if (self.source == "control") != (self.control is not None):
            raise ValueError("control details exist exactly for a control")
        if self.source == "control" and self.kind != "faithfulness":
            raise ValueError("only faithfulness items have controls")
        if self.judge is not None and self.judge.verdict not in ALLOWED_LABELS[self.kind]:
            raise ValueError(f"{self.judge.verdict!r} is not a {self.kind} label")
        return self


class SampleInfo(BaseModel):
    """What the sample was drawn from and with which parameters: enough to redo it."""

    model_config = _FROZEN

    seed: int
    faithfulness: int = Field(ge=0)
    correctness: int = Field(ge=0)
    controls: int = Field(ge=0)
    max_overlap: float
    fingerprint: str  # sha256 of the run and the sample: a key of another sample is not reused
    results_file: str  # the base name: the file itself is not committed
    results_sha256: str
    run_date: datetime
    promptfoo_version: str
    golden_set_version: str
    golden_set_sha256: str
    git_sha: str | None = None  # the commit that produced the results (promptfoo's file has none)
    judge_provider: str | None = None
    judge_model: str | None = None
    judge_prompt_versions: dict[str, str] = Field(default_factory=dict)
    # The input, not the sample: kind -> config -> verdict -> how many the run has.
    population: dict[str, dict[str, dict[str, int]]]


class AgreementKey(BaseModel):
    model_config = _FROZEN

    schema_version: Literal[1] = 1
    sample: SampleInfo
    items: list[KeyItem]

    @model_validator(mode="after")
    def _check(self) -> Self:
        ids = [item.item_id for item in self.items]
        if len(set(ids)) != len(ids):
            raise ValueError("an item id appears twice")
        return self

    @property
    def pending(self) -> list[KeyItem]:
        """Items without a judge verdict (controls whose call has not been made or failed)."""
        return [item for item in self.items if item.judge is None]
