"""Shadow cost from ``pricing.toml`` (Tech.md §14, PRD D27).

Shadow cost is what a request would cost at the **paid** list prices, although the demo runs on free
tiers. The formula is

    cost = in/1e6*in_price + out/1e6*out_price + embed/1e6*embed_price + rerank/1000*rerank_price

with thinking tokens inside ``out`` (the provider bills them as output). Money is ``Decimal`` end to
end and rounded to eight places, the precision of ``request_logs.shadow_cost_usd``, so the value in
the response, in the log row and in a hand calculation agree to the digit.

**A price is known or the program refuses to start.** ``Pricing.require_*`` raises ``PricingError``
for a model that is missing from the file or marked ``status = "unverified"``; the pipeline
calls them at construction (3.12), so an unpriced model is a startup error and never a silent
``0`` (Tech §14). The ``fake`` provider is the one exemption: it makes no billable call, so its
true cost is 0, not unknown.
"""

from __future__ import annotations

import tomllib
from datetime import date
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

# backend/pricing.toml; this file is backend/src/grounded/observability/cost.py.
DEFAULT_PRICING_PATH = Path(__file__).resolve().parents[3] / "pricing.toml"

UNBILLED_PROVIDERS = frozenset({"fake"})  # no real call is made, so the cost is exactly 0

_MILLION = Decimal(1_000_000)
_THOUSAND = Decimal(1_000)
_PLACES = Decimal("0.00000001")  # numeric(12, 8)


class PricingError(Exception):
    """``pricing.toml`` is unreadable or invalid, or has no usable price for a model in use."""


class _Row(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    # A price that did not come from today's pricing page carries its own provenance (Tech §14).
    price_source: str | None = None
    price_as_of: date | None = None
    status: Literal["verified", "unverified"] = "verified"


class ModelPrice(_Row):
    input_per_mtok: Decimal | None = Field(default=None, ge=0)
    output_per_mtok: Decimal | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _verified_rows_have_prices(self) -> Self:
        if self.status == "verified" and (
            self.input_per_mtok is None or self.output_per_mtok is None
        ):
            raise ValueError("a verified model price needs input_per_mtok and output_per_mtok")
        return self


class EmbeddingPrice(_Row):
    input_per_mtok: Decimal | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _verified_rows_have_prices(self) -> Self:
        if self.status == "verified" and self.input_per_mtok is None:
            raise ValueError("a verified embedding price needs input_per_mtok")
        return self


class RerankPrice(_Row):
    per_1k_searches: Decimal | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _verified_rows_have_prices(self) -> Self:
        if self.status == "verified" and self.per_1k_searches is None:
            raise ValueError("a verified rerank price needs per_1k_searches")
        return self


class Pricing(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    as_of: date  # the day the prices were checked
    source: str
    models: dict[str, ModelPrice] = Field(default_factory=dict)
    embeddings: dict[str, EmbeddingPrice] = Field(default_factory=dict)
    rerank: dict[str, RerankPrice] = Field(default_factory=dict)

    def require_generator(self, provider: str, model: str) -> None:
        if provider not in UNBILLED_PROVIDERS:
            self._model(model)

    def require_embedding(self, model: str) -> None:
        self._embedding(model)

    def shadow_cost(
        self,
        *,
        provider: str | None = None,
        model: str | None = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        embedding_model: str | None = None,
        embedding_tokens: int = 0,
        rerank_model: str | None = None,
        rerank_calls: int = 0,
    ) -> Decimal:
        """The Tech §14 formula. A part is skipped when its model is ``None`` (nothing of that kind
        ran); a model that did run but has no usable price raises ``PricingError``."""
        total = Decimal(0)
        if model is not None and provider not in UNBILLED_PROVIDERS:
            price = self._model(model)
            assert price.input_per_mtok is not None
            assert price.output_per_mtok is not None
            total += input_tokens / _MILLION * price.input_per_mtok
            total += output_tokens / _MILLION * price.output_per_mtok
        if embedding_model is not None:
            embedding = self._embedding(embedding_model)
            assert embedding.input_per_mtok is not None
            total += embedding_tokens / _MILLION * embedding.input_per_mtok
        if rerank_model is not None:
            rerank = self._rerank(rerank_model)
            assert rerank.per_1k_searches is not None
            total += rerank_calls / _THOUSAND * rerank.per_1k_searches
        return total.quantize(_PLACES, rounding=ROUND_HALF_UP)

    def _model(self, model: str) -> ModelPrice:
        return self._usable("model", model, self.models.get(model))

    def _embedding(self, model: str) -> EmbeddingPrice:
        return self._usable("embedding model", model, self.embeddings.get(model))

    def _rerank(self, model: str) -> RerankPrice:
        return self._usable("rerank model", model, self.rerank.get(model))

    def _usable[R: _Row](self, kind: str, name: str, row: R | None) -> R:
        if row is None:
            raise PricingError(
                f"no price for {kind} {name!r} in pricing.toml (checked {self.as_of}); add a "
                "verified row copied from the provider's pricing page"
            )
        if row.status != "verified":
            raise PricingError(
                f"the price of {kind} {name!r} in pricing.toml is marked unverified; verify it "
                "against the provider's pricing page before using the model"
            )
        return row


def load_pricing(path: Path = DEFAULT_PRICING_PATH) -> Pricing:
    try:
        with path.open("rb") as handle:
            raw = tomllib.load(handle)
        return Pricing.model_validate(raw)
    except (OSError, tomllib.TOMLDecodeError, ValidationError) as exc:
        raise PricingError(f"cannot load {path}: {exc}") from exc
