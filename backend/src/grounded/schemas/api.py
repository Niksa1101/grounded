"""HTTP models of ``POST /v1/ask`` (Tech.md §9.4) and the shared error body (§13).

``AskResponse`` is what the server assembles *after* validating the model's ``LLMAnswer``: labels
are mapped to DB-sourced citations, markers rewritten to display numbers, confidence computed here.
"""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field, HttpUrl

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


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)


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
