"""HTTP models of ``POST /v1/ask`` (Tech.md §9.4) and the shared error body (§13).

``AskResponse`` is what the server assembles *after* validating the model's ``LLMAnswer``: labels
are mapped to DB-sourced citations, markers rewritten to display numbers, confidence computed here.
"""

from __future__ import annotations

import unicodedata
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, HttpUrl, StringConstraints, field_validator

from grounded.schemas.llm import AnswerStatus

ErrorCode = Literal[
    "bad_request",
    "unauthorized",
    "rate_limited",
    "budget_exhausted",
    "validation_failed",
    "provider_unavailable",
    "deadline_exceeded",
    "internal_error",
]


# The input limits of Tech.md §12. They are part of the API contract (the OpenAPI schema, the golden
# set's question rule, the UI's counter), not tunables: changing one is a contract change.
MIN_QUESTION_CHARS = 3
MAX_QUESTION_CHARS = 500
MAX_BODY_BYTES = 4096


class AskRequest(BaseModel):
    # Trimmed first: the length rule applies to what the pipeline sees, so "  hi  " is 2 characters.
    question: Annotated[
        str,
        StringConstraints(
            strip_whitespace=True, min_length=MIN_QUESTION_CHARS, max_length=MAX_QUESTION_CHARS
        ),
    ]

    @field_validator("question")
    @classmethod
    def _no_control_characters(cls, value: str) -> str:
        # Unicode category Cc: C0 (NUL, tab, newline, ESC, ...), DEL and C1. Leading and trailing
        # whitespace is already gone, so only a control character *inside* the text is refused.
        # The message names the rule and never echoes the text (api/errors.py, Tech §13).
        if any(unicodedata.category(char) == "Cc" for char in value):
            raise ValueError("must not contain control characters")
        return value


class Citation(BaseModel):
    n: int  # display number [n]
    chunk_id: int
    url: HttpUrl
    title: str
    breadcrumb: str
    snippet: str  # first ~300 chars of the chunk content


class Claim(BaseModel):
    text: str
    citations: list[int]  # display numbers
    confidence: float  # server-computed, never the model's self-confidence (AGENTS.md §6.6)
    confidence_components: dict[str, float]


class Meta(BaseModel):
    request_id: UUID
    provider: str | None
    model: str | None
    fallback_used: bool
    cache_hit: bool
    rerank_used: bool
    prompt_version: str
    index_version: str  # "<git_ref>@<config_hash[:8]>"
    retrieval_config_hash: str
    latency_ms: dict[str, int]  # total, embed, retrieval, rerank, llm
    tokens: dict[str, int]  # input, output
    shadow_cost_usd: float


class AskResponse(BaseModel):
    status: AnswerStatus
    answer_markdown: str  # markers rewritten to [n]
    claims: list[Claim]
    citations: list[Citation]  # ordered by n
    follow_up_questions: list[str]
    min_confidence: float | None
    meta: Meta


class ErrorBody(BaseModel):
    code: ErrorCode
    message: str
    retry_after_s: int | None = None


class ErrorResponse(BaseModel):
    error: ErrorBody
    request_id: UUID
