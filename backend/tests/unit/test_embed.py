from __future__ import annotations

import asyncio
import json
import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import httpx
import pytest
from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types

from grounded.infra.kvcache import KVCache
from grounded.infra.provider_errors import (
    ProviderBadOutput,
    ProviderRateLimited,
    ProviderRequestRejected,
    ProviderTimeout,
    ProviderUnavailable,
)
from grounded.ingest.embed import (
    CachedEmbedder,
    EmbeddingInputTooLongError,
    FakeEmbedder,
    GeminiEmbedder,
    TaskType,
    Vector,
    fake_vector,
    l2_normalize,
)
from tests.support import make_settings

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "gemini" / "embed_3.json"
DIM = 4


def _norm(vector: Sequence[float]) -> float:
    return math.sqrt(math.fsum(v * v for v in vector))


def _raw_values(text: str) -> list[float]:
    """What the fake API returns for ``text``: not unit length, and different per text."""
    return [float(len(text)), 1.0, 2.0, 2.0]


def _response(values: Sequence[Sequence[float]]) -> genai_types.EmbedContentResponse:
    return genai_types.EmbedContentResponse(
        embeddings=[genai_types.ContentEmbedding(values=list(v)) for v in values]
    )


# --- Fakes for the SDK client and for time -------------------------------------------------------


@dataclass
class _Call:
    model: str
    contents: list[str]
    config: genai_types.EmbedContentConfig


class _NeverReturns:
    """A script step that hangs until the embedder's own timeout cancels it."""


class _FakeModels:
    """Stands in for ``client.aio.models``. Each call takes the next script step: an exception is
    raised, a response is returned; with the script used up it answers with ``_raw_values``."""

    def __init__(
        self, script: Sequence[BaseException | genai_types.EmbedContentResponse | _NeverReturns]
    ) -> None:
        self.script = list(script)
        self.calls: list[_Call] = []

    async def embed_content(
        self, *, model: str, contents: list[str], config: genai_types.EmbedContentConfig
    ) -> genai_types.EmbedContentResponse:
        self.calls.append(_Call(model, list(contents), config))
        step = self.script.pop(0) if self.script else None
        if isinstance(step, BaseException):
            raise step
        if isinstance(step, _NeverReturns):
            await asyncio.Event().wait()
        if isinstance(step, genai_types.EmbedContentResponse):
            return step
        return _response([_raw_values(text) for text in contents])


class _FakeClient:
    def __init__(self, models: _FakeModels) -> None:
        self.aio = type("Aio", (), {"models": models})()


class _FakeTime:
    """A monotonic clock that only moves when the embedder sleeps (or a test moves it)."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def _words(text: str) -> int:
    return len(text.split())


def _embedder(
    models: _FakeModels, time: _FakeTime | None = None, **overrides: Any
) -> GeminiEmbedder:
    time = time or _FakeTime()
    options: dict[str, Any] = {
        "model": "gemini-embedding-001",
        "dim": DIM,
        "count_tokens": _words,
        "batch_size": 100,
        "rpm": 100,
        "tpm": 30_000,
        "max_input_tokens": 2048,
        "max_retries": 5,
        "timeout_s": 30.0,
        "sleep": time.sleep,
        "clock": time.clock,
        "rng": random.Random(0),
    } | overrides
    return GeminiEmbedder(cast(genai.Client, _FakeClient(models)), **options)


# --- Gemini error bodies --------------------------------------------------------------------------
# Written by hand from the google.rpc error model (https://cloud.google.com/apis/design/errors):
# a 429 carries a QuotaFailure naming the violated quota and a RetryInfo with the suggested delay.
# Not recorded from a real 429 (that would mean exhausting the quota on purpose). The quota IDs
# follow the "...PerMinute..." / "...PerDay..." naming the embedder keys on.

_PER_MINUTE = "EmbedContentRequestsPerMinutePerProjectPerModel-FreeTier"
_PER_DAY = "EmbedContentRequestsPerDayPerProjectPerModel-FreeTier"
_QUOTA_METRIC = "generativelanguage.googleapis.com/embed_content_free_tier_requests"


def _rate_limited(
    *, retry_delay: str | None = None, quota_id: str | None = None, retry_after: str | None = None
) -> genai_errors.ClientError:
    details: list[dict[str, Any]] = []
    if quota_id is not None:
        details.append(
            {
                "@type": "type.googleapis.com/google.rpc.QuotaFailure",
                "violations": [{"quotaMetric": _QUOTA_METRIC, "quotaId": quota_id}],
            }
        )
    if retry_delay is not None:
        details.append(
            {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": retry_delay}
        )
    body = {
        "error": {
            "code": 429,
            "message": "You exceeded your current quota.",
            "status": "RESOURCE_EXHAUSTED",
            "details": details,
        }
    }
    headers = {"retry-after": retry_after} if retry_after is not None else {}
    return genai_errors.ClientError(429, body, httpx.Response(429, json=body, headers=headers))


def _api_error(code: int, status: str) -> genai_errors.APIError:
    body = {"error": {"code": code, "message": "nope", "status": status}}
    cls = genai_errors.ServerError if code >= 500 else genai_errors.ClientError
    return cls(code, body, httpx.Response(code, json=body))


# --- l2_normalize ---------------------------------------------------------------------------------


def test_l2_normalize_scales_to_unit_length() -> None:
    assert l2_normalize([3.0, 4.0]) == pytest.approx([0.6, 0.8])
    assert _norm(l2_normalize([0.1, -7.0, 2.5])) == pytest.approx(1.0)


@pytest.mark.parametrize("values", [[0.0, 0.0], [math.nan, 1.0], [math.inf, 1.0]])
def test_l2_normalize_rejects_degenerate_vectors(values: list[float]) -> None:
    with pytest.raises(ProviderBadOutput):
        l2_normalize(values)


# --- GeminiEmbedder: request and response ---------------------------------------------------------


async def test_replays_the_recorded_gemini_response() -> None:
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    recorded = fixture["_recorded"]
    response = genai_types.EmbedContentResponse.model_validate(fixture["response"])
    models = _FakeModels([response])
    embedder = _embedder(models, model=recorded["model"], dim=recorded["dim"])

    vectors = await embedder.embed(recorded["texts"], "RETRIEVAL_DOCUMENT")

    (call,) = models.calls
    assert call.model == "gemini-embedding-001"
    assert call.contents == recorded["texts"]
    assert call.config.task_type == "RETRIEVAL_DOCUMENT"
    assert call.config.output_dimensionality == 768
    raw = [e.values or [] for e in response.embeddings or []]
    assert len(vectors) == 3
    for vector, raw_vector in zip(vectors, raw, strict=True):
        # Truncated 768-dim vectors come back well short of unit length; that is why we normalize.
        assert _norm(raw_vector) < 0.9
        assert _norm(vector) == pytest.approx(1.0, abs=1e-12)
        assert [v * _norm(raw_vector) for v in vector] == pytest.approx(raw_vector)


async def test_batches_by_count_and_keeps_order() -> None:
    models = _FakeModels([])
    texts = ["x" * (i + 1) for i in range(250)]

    vectors = await _embedder(models).embed(texts, "RETRIEVAL_DOCUMENT")

    assert [len(c.contents) for c in models.calls] == [100, 100, 50]
    assert [t for c in models.calls for t in c.contents] == texts
    assert vectors == [l2_normalize(_raw_values(t)) for t in texts]


async def test_batches_by_estimated_tokens() -> None:
    models = _FakeModels([])
    texts = ["a b c d"] * 5  # 4 "tokens" each; 3 of them would exceed the 10-token budget

    await _embedder(models, tpm=10, max_input_tokens=10).embed(texts, "RETRIEVAL_DOCUMENT")

    assert [len(c.contents) for c in models.calls] == [2, 2, 1]


async def test_query_task_type_is_passed_through() -> None:
    models = _FakeModels([])
    await _embedder(models).embed(["how?"], "RETRIEVAL_QUERY")
    assert models.calls[0].config.task_type == "RETRIEVAL_QUERY"


async def test_no_texts_no_call() -> None:
    models = _FakeModels([])
    assert await _embedder(models).embed([], "RETRIEVAL_DOCUMENT") == []
    assert models.calls == []


async def test_too_long_input_fails_before_any_call() -> None:
    models = _FakeModels([])
    texts = ["short", "short", "one two three four", "short"]

    with pytest.raises(EmbeddingInputTooLongError) as excinfo:
        await _embedder(models, batch_size=1, max_input_tokens=3).embed(texts, "RETRIEVAL_DOCUMENT")

    assert (excinfo.value.index, excinfo.value.tokens, excinfo.value.limit) == (2, 4, 3)
    assert models.calls == []  # not even the valid texts before it were sent


@pytest.mark.parametrize(
    "response",
    [
        pytest.param(_response([[1.0, 0.0, 0.0, 0.0]]), id="too-few-embeddings"),
        pytest.param(_response([[1.0, 0.0, 0.0]] * 2), id="wrong-dimension"),
        pytest.param(genai_types.EmbedContentResponse(), id="no-embeddings"),
    ],
)
async def test_unusable_response_is_bad_output_and_not_retried(
    response: genai_types.EmbedContentResponse,
) -> None:
    models = _FakeModels([response])
    with pytest.raises(ProviderBadOutput):
        await _embedder(models).embed(["a", "b"], "RETRIEVAL_DOCUMENT")
    assert len(models.calls) == 1


# --- GeminiEmbedder: errors and retries -----------------------------------------------------------


async def test_rate_limit_waits_the_server_delay_then_succeeds() -> None:
    time = _FakeTime()
    models = _FakeModels([_rate_limited(retry_delay="7s", quota_id=_PER_MINUTE)])
    embedder = _embedder(models, time)

    vectors = await embedder.embed(["a"], "RETRIEVAL_DOCUMENT")

    assert vectors == [l2_normalize(_raw_values("a"))]
    assert time.sleeps == [7.0]
    assert embedder.api_calls == 2


async def test_rate_limit_falls_back_to_the_retry_after_header() -> None:
    time = _FakeTime()
    models = _FakeModels([_rate_limited(retry_after="3")])
    await _embedder(models, time).embed(["a"], "RETRIEVAL_DOCUMENT")
    assert time.sleeps == [3.0]


async def test_rate_limit_without_a_delay_uses_backoff() -> None:
    time = _FakeTime()
    models = _FakeModels([_rate_limited(retry_after="Wed, 21 Oct 2026 07:28:00 GMT")])
    await _embedder(models, time).embed(["a"], "RETRIEVAL_DOCUMENT")
    (delay,) = time.sleeps
    assert 1.0 <= delay <= 2.0  # first backoff step: 2 s with equal jitter


async def test_daily_quota_is_raised_without_retrying() -> None:
    time = _FakeTime()
    models = _FakeModels([_rate_limited(retry_delay="30s", quota_id=_PER_DAY)])

    with pytest.raises(ProviderRateLimited) as excinfo:
        await _embedder(models, time).embed(["a"], "RETRIEVAL_DOCUMENT")

    assert excinfo.value.is_quota is True
    assert excinfo.value.retry_after_s == 30.0
    assert len(models.calls) == 1
    assert time.sleeps == []


async def test_server_errors_back_off_exponentially_with_jitter() -> None:
    time = _FakeTime()
    models = _FakeModels([_api_error(503, "UNAVAILABLE")] * 3)

    await _embedder(models, time).embed(["a"], "RETRIEVAL_DOCUMENT")

    assert len(models.calls) == 4
    # Equal jitter: attempt n waits between half and all of 2 s * 2^n.
    for attempt, delay in enumerate(time.sleeps):
        full = 2.0 * 2**attempt
        assert full / 2 <= delay <= full


async def test_gives_up_after_max_retries() -> None:
    time = _FakeTime()
    models = _FakeModels([_api_error(500, "INTERNAL")] * 10)

    with pytest.raises(ProviderUnavailable):
        await _embedder(models, time, max_retries=2).embed(["a"], "RETRIEVAL_DOCUMENT")

    assert len(models.calls) == 3
    assert len(time.sleeps) == 2


@pytest.mark.parametrize(
    ("code", "status"), [(400, "INVALID_ARGUMENT"), (403, "PERMISSION_DENIED")]
)
async def test_client_errors_are_rejected_without_retrying(code: int, status: str) -> None:
    models = _FakeModels([_api_error(code, status)])

    with pytest.raises(ProviderRequestRejected) as excinfo:
        await _embedder(models).embed(["a"], "RETRIEVAL_DOCUMENT")

    assert excinfo.value.status_code == code
    assert len(models.calls) == 1


async def test_transport_timeout_maps_to_provider_timeout() -> None:
    models = _FakeModels([httpx.ReadTimeout("slow")])
    with pytest.raises(ProviderTimeout):
        await _embedder(models, max_retries=0).embed(["a"], "RETRIEVAL_DOCUMENT")


async def test_own_timeout_cancels_a_hanging_call() -> None:
    models = _FakeModels([_NeverReturns()])
    with pytest.raises(ProviderTimeout):
        await _embedder(models, max_retries=0, timeout_s=0.01).embed(["a"], "RETRIEVAL_DOCUMENT")


async def test_connection_errors_are_retried() -> None:
    models = _FakeModels([httpx.ConnectError("reset")])
    vectors = await _embedder(models).embed(["a"], "RETRIEVAL_DOCUMENT")
    assert len(vectors) == 1
    assert len(models.calls) == 2


# --- GeminiEmbedder: pacing -----------------------------------------------------------------------


async def test_waits_for_a_free_request_slot() -> None:
    time = _FakeTime()
    models = _FakeModels([])

    await _embedder(models, time, rpm=2, batch_size=1).embed(["a", "b", "c"], "RETRIEVAL_DOCUMENT")

    assert time.sleeps == [60.0]  # the third request waits for the first to leave the window


async def test_every_text_in_a_batch_counts_toward_rpm() -> None:
    # AI Studio showed one 3-text batch as 3 model requests, so RPM caps texts, not HTTP calls.
    time = _FakeTime()
    models = _FakeModels([])

    await _embedder(models, time, rpm=4, batch_size=4).embed(["t"] * 6, "RETRIEVAL_DOCUMENT")

    assert [len(c.contents) for c in models.calls] == [4, 2]
    assert time.sleeps == [60.0]  # 4 + 2 texts > 4 per minute


async def test_waits_for_token_budget() -> None:
    time = _FakeTime()
    models = _FakeModels([])
    texts = ["one two three four five six"] * 2  # 6 + 6 > 10: one text per batch

    await _embedder(models, time, tpm=10, max_input_tokens=10).embed(texts, "RETRIEVAL_DOCUMENT")

    assert [len(c.contents) for c in models.calls] == [1, 1]
    assert time.sleeps == [60.0]


async def test_window_slides_with_the_oldest_request() -> None:
    time = _FakeTime()
    embedder = _embedder(_FakeModels([]), time, rpm=2, batch_size=1)

    await embedder.embed(["a"], "RETRIEVAL_DOCUMENT")  # t=0
    time.now = 30.0
    await embedder.embed(["b"], "RETRIEVAL_DOCUMENT")  # t=30
    time.now = 40.0
    await embedder.embed(["c"], "RETRIEVAL_DOCUMENT")  # full until the t=0 request expires

    assert time.sleeps == [20.0]


async def test_rejected_requests_count_against_the_window() -> None:
    # The server counts the 429'd request too, so the retry still has to fit in the window.
    time = _FakeTime()
    models = _FakeModels([_rate_limited(retry_delay="1s", quota_id=_PER_MINUTE)])

    await _embedder(models, time, rpm=1, batch_size=1).embed(["a"], "RETRIEVAL_DOCUMENT")

    assert time.sleeps == [1.0, 59.0]


# --- GeminiEmbedder: construction -----------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"rpm": 10, "batch_size": 11}, id="batch-over-rpm"),
        pytest.param({"tpm": 100, "max_input_tokens": 101}, id="input-over-tpm"),
    ],
)
def test_limits_that_could_never_be_met_are_rejected(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="batch_size must be <= rpm"):
        _embedder(_FakeModels([]), **overrides)


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"embedding_model": "gemini-embedding-001"}, id="no-key"),
        pytest.param({"gemini_api_key": "k"}, id="no-model"),
    ],
)
def test_from_settings_requires_key_and_model(
    overrides: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("GEMINI_API_KEY", "EMBEDDING_MODEL"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(ValueError, match="GEMINI_API_KEY and EMBEDDING_MODEL"):
        GeminiEmbedder.from_settings(make_settings(**overrides), _words)


def test_from_settings_builds_without_network() -> None:
    settings = make_settings(gemini_api_key="k", embedding_model="gemini-embedding-001")
    embedder = GeminiEmbedder.from_settings(settings, _words)
    assert (embedder.model, embedder.dim) == ("gemini-embedding-001", 768)


# --- FakeEmbedder ---------------------------------------------------------------------------------


async def test_fake_embedder_is_deterministic_unit_length_and_records_calls() -> None:
    first, second = FakeEmbedder(dim=20), FakeEmbedder(dim=20)

    a = await first.embed(["x", "y"], "RETRIEVAL_DOCUMENT")
    b = await second.embed(["x", "y"], "RETRIEVAL_DOCUMENT")
    query = await first.embed(["x"], "RETRIEVAL_QUERY")

    assert a == b
    assert a[0] != a[1]
    assert query[0] != a[0]  # the task type is part of the seed, like a real asymmetric model
    assert all(len(v) == 20 and _norm(v) == pytest.approx(1.0) for v in [*a, *query])
    assert first.calls == [(["x", "y"], "RETRIEVAL_DOCUMENT"), (["x"], "RETRIEVAL_QUERY")]
    assert fake_vector("x", "RETRIEVAL_DOCUMENT", 20) == a[0]


# --- CachedEmbedder -------------------------------------------------------------------------------


class _FailingEmbedder(FakeEmbedder):
    """Succeeds for ``ok_calls`` calls, then fails like an exhausted daily quota."""

    def __init__(self, ok_calls: int) -> None:
        super().__init__()
        self._ok_calls = ok_calls

    async def embed(self, texts: Sequence[str], task_type: TaskType) -> list[Vector]:
        if len(self.calls) >= self._ok_calls:
            raise ProviderRateLimited("daily quota", retry_after_s=None, is_quota=True)
        return await super().embed(texts, task_type)


@pytest.fixture
def kv(tmp_path: Path) -> KVCache:
    return KVCache(tmp_path / "embeddings.sqlite")


async def test_second_run_makes_no_calls_and_returns_identical_vectors(kv: KVCache) -> None:
    texts = ["a", "b", "c"]
    first_inner, second_inner = FakeEmbedder(), FakeEmbedder()

    first = await CachedEmbedder(first_inner, kv, write_every=2).embed(texts, "RETRIEVAL_DOCUMENT")
    cached = CachedEmbedder(second_inner, kv, write_every=2)
    second = await cached.embed(texts, "RETRIEVAL_DOCUMENT")

    assert [len(t) for t, _ in first_inner.calls] == [2, 1]  # misses sent in write_every slices
    assert second_inner.calls == []
    assert second == first  # exact: a fresh run already returns float32-rounded vectors
    assert (cached.hits, cached.misses) == (3, 0)


async def test_fresh_vectors_are_float32_rounded(kv: KVCache) -> None:
    (vector,) = await CachedEmbedder(FakeEmbedder(), kv, write_every=10).embed(
        ["a"], "RETRIEVAL_DOCUMENT"
    )
    exact = fake_vector("a", "RETRIEVAL_DOCUMENT", 8)
    assert vector != exact
    assert vector == pytest.approx(exact, rel=1e-6)


async def test_duplicates_are_embedded_once(kv: KVCache) -> None:
    inner = FakeEmbedder()
    cached = CachedEmbedder(inner, kv, write_every=10)

    vectors = await cached.embed(["a", "b", "a"], "RETRIEVAL_DOCUMENT")

    assert inner.calls == [(["a", "b"], "RETRIEVAL_DOCUMENT")]
    assert vectors[0] == vectors[2] != vectors[1]
    assert (cached.hits, cached.misses) == (1, 2)


async def test_mixed_hits_and_misses_keep_input_order(kv: KVCache) -> None:
    inner = FakeEmbedder()
    cached = CachedEmbedder(inner, kv, write_every=10)
    await cached.embed(["b"], "RETRIEVAL_DOCUMENT")

    vectors = await cached.embed(["a", "b", "c"], "RETRIEVAL_DOCUMENT")

    assert inner.calls[-1] == (["a", "c"], "RETRIEVAL_DOCUMENT")
    expected = await CachedEmbedder(FakeEmbedder(), kv, write_every=10).embed(
        ["a", "b", "c"], "RETRIEVAL_DOCUMENT"
    )
    assert vectors == expected


async def test_task_type_is_part_of_the_key(kv: KVCache) -> None:
    inner = FakeEmbedder()
    cached = CachedEmbedder(inner, kv, write_every=10)

    await cached.embed(["a"], "RETRIEVAL_DOCUMENT")
    await cached.embed(["a"], "RETRIEVAL_QUERY")

    assert [task for _, task in inner.calls] == ["RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY"]


async def test_model_and_dim_are_part_of_the_key(kv: KVCache) -> None:
    await CachedEmbedder(FakeEmbedder(model="m1"), kv, write_every=10).embed(
        ["a"], "RETRIEVAL_DOCUMENT"
    )
    other_model, other_dim = FakeEmbedder(model="m2"), FakeEmbedder(model="m1", dim=16)

    await CachedEmbedder(other_model, kv, write_every=10).embed(["a"], "RETRIEVAL_DOCUMENT")
    await CachedEmbedder(other_dim, kv, write_every=10).embed(["a"], "RETRIEVAL_DOCUMENT")

    assert len(other_model.calls) == len(other_dim.calls) == 1


def test_cache_key_is_model_dim_task_and_text_hash(kv: KVCache) -> None:
    cached = CachedEmbedder(FakeEmbedder(model="m", dim=8), kv, write_every=1)
    # sha256("abc"), the same digest ingest stores as chunks.content_hash for that text.
    digest = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert cached.cache_key("abc", "RETRIEVAL_DOCUMENT") == f"m|8|RETRIEVAL_DOCUMENT|{digest}"


async def test_finished_slices_survive_a_failure(kv: KVCache) -> None:
    texts = ["a", "b", "c", "d", "e"]

    with pytest.raises(ProviderRateLimited):
        await CachedEmbedder(_FailingEmbedder(ok_calls=1), kv, write_every=2).embed(
            texts, "RETRIEVAL_DOCUMENT"
        )
    assert len(kv) == 2  # the first slice was stored before the second one failed

    resumed = FakeEmbedder()
    await CachedEmbedder(resumed, kv, write_every=2).embed(texts, "RETRIEVAL_DOCUMENT")
    assert [t for t, _ in resumed.calls] == [["c", "d"], ["e"]]


async def test_cached_vector_with_the_wrong_size_is_an_error(kv: KVCache) -> None:
    cached = CachedEmbedder(FakeEmbedder(dim=8), kv, write_every=1)
    kv.put_many({cached.cache_key("a", "RETRIEVAL_DOCUMENT"): b"\x00" * 12})
    with pytest.raises(ValueError, match="3 dimensions, expected 8"):
        await cached.embed(["a"], "RETRIEVAL_DOCUMENT")


def test_write_every_must_be_positive(kv: KVCache) -> None:
    with pytest.raises(ValueError, match="write_every"):
        CachedEmbedder(FakeEmbedder(), kv, write_every=0)
