"""Golden-set rows (Tech.md §15.1). Pydantic is the authority on the file's shape (AGENTS.md §6.2).

Rules that need only the row itself live here. Rules that need the corpus (a label resolves to
a chunk; two labels never match the same chunk, which is how nested labels show up) live in
``evals/golden.py``.
"""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

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
