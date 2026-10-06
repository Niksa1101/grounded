"""What the model must return (Tech.md §9.4).

Kept deliberately simple so every provider's structured-output mode accepts it (Tech §9.1): no
unions, no recursion, enums as string literals. Constraints a provider can't express (the label
pattern, lengths) are enforced here, after the call, because Pydantic is the authority on LLM output
(AGENTS.md §6.2). Rules that need the request (is ``c3`` a label of *this* request?) live in
``generation/citations.py``.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field, StringConstraints

AnswerStatus = Literal["answered", "partial", "insufficient_context"]

# Per-request source labels (Tech §9.3): the model never sees DB IDs or writes URLs. One digit,
# so at most nine sources per request (Tech §9.4).
CitationLabel = Annotated[str, StringConstraints(pattern=r"^c[1-9]$")]


class LLMClaim(BaseModel):
    text: str = Field(min_length=1, max_length=500)
    citation_ids: list[CitationLabel] = Field(max_length=5)
    # The model's own estimate. Never shown as "confidence": the server computes that (§6.6).
    self_confidence: float = Field(ge=0.0, le=1.0)


class LLMAnswer(BaseModel):
    status: AnswerStatus
    answer_markdown: str = Field(max_length=4000)  # contains [cN] markers
    claims: list[LLMClaim] = Field(max_length=8)
    follow_up_questions: list[str] = Field(default_factory=list, max_length=3)
