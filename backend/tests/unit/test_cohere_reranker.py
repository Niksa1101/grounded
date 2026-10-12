"""``CohereReranker`` against the real ``cohere`` SDK over a mock transport (``rerank_rig.py``).

The responses follow the documented shape of ``POST /v2/rerank``; none is recorded (see
``tests/fixtures/cohere/README.md``). What these tests pin: the request that is sent (no ``top_n``),
the ranking, one HTTP request per call whatever happens (the SDK's own retries sleep, so a second
request would be a bug), and the mapping of every failure to a typed provider error.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

from grounded.infra.kvcache import KVCache
from grounded.infra.provider_errors import (
    ProviderBadOutput,
    ProviderRateLimited,
    ProviderRequestRejected,
    ProviderTimeout,
    ProviderUnavailable,
)
from grounded.retrieval.rerank import CachedReranker, rerank_document
from tests.rerank_rig import (
    COHERE_MODEL,
    QUERY,
    CohereRig,
    chunks,
    scores_response,
    success_body,
)

# The fixture's scores by document index, and the order they rank in (best first).
FIXTURE_SCORES = [0.7867867, 0.2771274, 0.0153044, 0.9990712, 0.4812119]
FIXTURE_ORDER = [3, 0, 4, 1, 2]


# --- The request and the ranking ------------------------------------------------------------------


async def test_a_success_ranks_the_chunks_and_sets_the_score() -> None:
    rig = CohereRig("success")
    candidates = chunks(5)

    ranked = await rig.reranker.rerank(QUERY, candidates, 5)

    assert [chunk.chunk_id for chunk in ranked] == FIXTURE_ORDER
    assert [chunk.rerank_score for chunk in ranked] == [FIXTURE_SCORES[i] for i in FIXTURE_ORDER]
    # Only the score is new: the chunk is the same one, and the input list is not touched.
    assert ranked[0].content == candidates[3].content
    assert ranked[0].rrf_score == candidates[3].rrf_score
    assert all(chunk.rerank_score is None for chunk in candidates)
    assert rig.remaining == 0


async def test_the_request_has_the_model_the_query_and_every_document_but_no_top_n() -> None:
    rig = CohereRig("success")
    candidates = chunks(5)

    await rig.reranker.rerank(QUERY, candidates, 2)

    (request,) = rig.http_requests
    assert (request.method, str(request.url)) == ("POST", "https://api.cohere.com/v2/rerank")
    assert request.headers["authorization"] == "Bearer test-key"
    # `top_n` is applied by us (Tech §8): the cache entry has to serve any `top_n`, and the eval
    # needs the whole list. The SDK leaves unset optional fields out of the body.
    assert rig.request_bodies == [
        {
            "model": COHERE_MODEL,
            "query": QUERY,
            "documents": [rerank_document(chunk) for chunk in candidates],
        }
    ]


async def test_top_n_cuts_the_list_after_ranking() -> None:
    rig = CohereRig("success")

    ranked = await rig.reranker.rerank(QUERY, chunks(5), 2)

    assert [chunk.chunk_id for chunk in ranked] == FIXTURE_ORDER[:2]


async def test_a_top_n_above_the_number_of_chunks_returns_them_all() -> None:
    rig = CohereRig("success")

    ranked = await rig.reranker.rerank(QUERY, chunks(5), 40)

    assert [chunk.chunk_id for chunk in ranked] == FIXTURE_ORDER


@pytest.mark.parametrize("top_n", [0, -1])
async def test_a_top_n_below_one_is_refused_before_any_call(top_n: int) -> None:
    rig = CohereRig()

    with pytest.raises(ValueError, match="top_n"):
        await rig.reranker.rerank(QUERY, chunks(5), top_n)

    assert rig.http_requests == []


async def test_no_chunks_means_no_call() -> None:
    rig = CohereRig()

    assert await rig.reranker.rerank(QUERY, [], 5) == []
    assert rig.http_requests == []


async def test_equal_scores_keep_the_rrf_order_whatever_order_cohere_lists_them_in() -> None:
    # `scores_response` lists equal scores in reverse index order.
    rig = CohereRig(scores_response([0.5, 0.5, 0.5, 0.9, 0.5]))

    ranked = await rig.reranker.rerank(QUERY, chunks(5), 5)

    assert [chunk.chunk_id for chunk in ranked] == [3, 0, 1, 2, 4]


async def test_a_response_listed_out_of_order_is_still_matched_by_index() -> None:
    body = success_body()
    body["results"].reverse()  # worst first: the order of the list means nothing, `index` does
    rig = CohereRig(httpx.Response(200, json=body))

    ranked = await rig.reranker.rerank(QUERY, chunks(5), 5)

    assert [chunk.chunk_id for chunk in ranked] == FIXTURE_ORDER


# --- Errors: one attempt, a typed error --------------------------------------------------------


async def test_a_429_with_retry_after_carries_the_wait_and_is_not_retried() -> None:
    rig = CohereRig("429_retry_after")

    with pytest.raises(ProviderRateLimited) as caught:
        await rig.reranker.rerank(QUERY, chunks(5), 5)

    assert caught.value.retry_after_s == 7.0
    assert caught.value.is_quota is False  # Cohere documents no difference for a monthly limit
    assert len(rig.http_requests) == 1  # the SDK's default is two retries that sleep: off


async def test_a_429_without_retry_after_has_no_wait() -> None:
    rig = CohereRig("429_no_retry_after")

    with pytest.raises(ProviderRateLimited) as caught:
        await rig.reranker.rerank(QUERY, chunks(5), 5)

    assert caught.value.retry_after_s is None
    assert len(rig.http_requests) == 1


@pytest.mark.parametrize("value", ["Wed, 21 Oct 2026 07:28:00 GMT", "soon", "-3", "inf", "nan"])
async def test_a_retry_after_that_is_not_a_usable_number_of_seconds_is_dropped(value: str) -> None:
    rig = CohereRig(
        httpx.Response(429, json={"message": "slow down"}, headers={"retry-after": value})
    )

    with pytest.raises(ProviderRateLimited) as caught:
        await rig.reranker.rerank(QUERY, chunks(5), 5)

    assert caught.value.retry_after_s is None


@pytest.mark.parametrize("step", ["500", 502, 503, 504])
async def test_a_5xx_is_unavailable_and_is_not_retried(step: str | int) -> None:
    response = httpx.Response(step, json={"message": "down"}) if isinstance(step, int) else step
    rig = CohereRig(response)

    with pytest.raises(ProviderUnavailable):
        await rig.reranker.rerank(QUERY, chunks(5), 5)

    assert len(rig.http_requests) == 1


async def test_a_5xx_with_an_html_body_is_unavailable() -> None:
    rig = CohereRig(httpx.Response(502, text="<html>Bad Gateway</html>"))

    with pytest.raises(ProviderUnavailable):
        await rig.reranker.rerank(QUERY, chunks(5), 5)


@pytest.mark.parametrize("step", ["401", "404", 400, 403, 422, 498, 499])
async def test_any_other_4xx_is_rejected_with_its_status(step: str | int) -> None:
    response = httpx.Response(step, json={"message": "no"}) if isinstance(step, int) else step
    rig = CohereRig(response)

    with pytest.raises(ProviderRequestRejected) as caught:
        await rig.reranker.rerank(QUERY, chunks(5), 5)

    status = int(step)
    assert caught.value.status_code == status
    assert str(status) in str(caught.value)
    assert len(rig.http_requests) == 1


async def test_a_rejection_says_why_but_never_repeats_the_question_or_the_key() -> None:
    rig = CohereRig("404")

    with pytest.raises(ProviderRequestRejected) as caught:
        await rig.reranker.rerank(QUERY, chunks(5), 5)

    assert "model not found" in str(caught.value)
    assert QUERY not in str(caught.value)
    assert "test-key" not in str(caught.value)


async def test_a_call_that_hangs_times_out_as_provider_timeout() -> None:
    rig = CohereRig("hang", timeout_s=0.05)

    with pytest.raises(ProviderTimeout, match=r"0\.05s"):
        await rig.reranker.rerank(QUERY, chunks(5), 5)

    assert len(rig.http_requests) == 1


async def test_an_httpx_timeout_is_provider_timeout() -> None:
    rig = CohereRig("timeout")

    with pytest.raises(ProviderTimeout):
        await rig.reranker.rerank(QUERY, chunks(5), 5)

    assert len(rig.http_requests) == 1


async def test_a_transport_failure_is_unavailable() -> None:
    rig = CohereRig("connect_error")

    with pytest.raises(ProviderUnavailable):
        await rig.reranker.rerank(QUERY, chunks(5), 5)

    assert len(rig.http_requests) == 1


# --- Errors: an answer that cannot be used -------------------------------------------------------


def _drop_a_result(body: dict[str, Any]) -> None:
    body["results"].pop()


def _add_a_result(body: dict[str, Any]) -> None:
    body["results"].append({"index": 0, "relevance_score": 0.1})


def _repeat_an_index(body: dict[str, Any]) -> None:
    body["results"][1]["index"] = body["results"][0]["index"]


def _index_past_the_end(body: dict[str, Any]) -> None:
    body["results"][0]["index"] = 5


def _negative_index(body: dict[str, Any]) -> None:
    body["results"][0]["index"] = -1


def _score_above_one(body: dict[str, Any]) -> None:
    body["results"][0]["relevance_score"] = 1.5


def _score_below_zero(body: dict[str, Any]) -> None:
    body["results"][0]["relevance_score"] = -0.01


def _score_not_a_number(body: dict[str, Any]) -> None:
    body["results"][0]["relevance_score"] = "high"


def _score_missing(body: dict[str, Any]) -> None:
    del body["results"][0]["relevance_score"]


def _index_missing(body: dict[str, Any]) -> None:
    del body["results"][0]["index"]


def _results_missing(body: dict[str, Any]) -> None:
    del body["results"]


def _results_null(body: dict[str, Any]) -> None:
    body["results"] = None


def _no_results(body: dict[str, Any]) -> None:
    body["results"] = []


MALFORMED: dict[str, Callable[[dict[str, Any]], None]] = {
    "one result too few": _drop_a_result,
    "one result too many": _add_a_result,
    "a repeated index": _repeat_an_index,
    "an index past the end": _index_past_the_end,
    "a negative index": _negative_index,
    "a score above 1": _score_above_one,
    "a score below 0": _score_below_zero,
    "a score that is not a number": _score_not_a_number,
    "a result without a score": _score_missing,
    "a result without an index": _index_missing,
    "no results field": _results_missing,
    "results null": _results_null,
    "an empty results list": _no_results,
}


@pytest.mark.parametrize("mutate", MALFORMED.values(), ids=MALFORMED.keys())
async def test_a_malformed_response_is_bad_output(mutate: Callable[[dict[str, Any]], None]) -> None:
    body = success_body()
    mutate(body)
    rig = CohereRig(httpx.Response(200, json=body))

    with pytest.raises(ProviderBadOutput):
        await rig.reranker.rerank(QUERY, chunks(5), 5)

    assert len(rig.http_requests) == 1


async def test_a_validation_failure_does_not_echo_the_response() -> None:
    body = success_body()
    body["results"][0]["relevance_score"] = "SECRET-LOOKING-VALUE"
    rig = CohereRig(httpx.Response(200, json=body))

    with pytest.raises(ProviderBadOutput) as caught:
        await rig.reranker.rerank(QUERY, chunks(5), 5)

    assert "SECRET-LOOKING-VALUE" not in str(caught.value) + caught.value.validation_error
    assert "relevance_score" in caught.value.validation_error


async def test_a_success_that_is_not_json_is_bad_output() -> None:
    rig = CohereRig(httpx.Response(200, text="<html>ok</html>"))

    with pytest.raises(ProviderBadOutput):
        await rig.reranker.rerank(QUERY, chunks(5), 5)


async def test_a_response_for_another_number_of_documents_is_bad_output() -> None:
    rig = CohereRig("success")  # the fixture answers five documents

    with pytest.raises(ProviderBadOutput, match="5 results for 3 documents"):
        await rig.reranker.rerank(QUERY, chunks(3), 3)


# --- Construction ---------------------------------------------------------------------------------


async def test_the_key_and_the_url_are_the_ones_passed_not_the_environments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The SDK reads CO_API_KEY / CO_API_URL when it is given nothing; we always give it both.
    monkeypatch.setenv("CO_API_KEY", "key-from-the-environment")
    monkeypatch.setenv("CO_API_URL", "https://elsewhere.example")
    rig = CohereRig("success", api_key="key-from-settings")

    await rig.reranker.rerank(QUERY, chunks(5), 5)

    (request,) = rig.http_requests
    assert request.url.host == "api.cohere.com"
    assert request.headers["authorization"] == "Bearer key-from-settings"


async def test_aclose_closes_the_http_client() -> None:
    rig = CohereRig()

    await rig.reranker.aclose()

    assert rig.http_client.is_closed


# --- The cache over the real adapter -------------------------------------------------------------


async def test_a_cache_hit_makes_no_http_request(tmp_path: Path) -> None:
    rig = CohereRig("success")
    with KVCache(tmp_path / "rerank.sqlite") as cache:
        cached = CachedReranker(rig.reranker, cache)

        first = await cached.rerank(QUERY, chunks(5), 5)
        second = await cached.rerank(QUERY, chunks(5), 3)  # another `top_n`: the same entry

    assert len(rig.http_requests) == 1
    assert [chunk.chunk_id for chunk in second] == [chunk.chunk_id for chunk in first][:3]
    assert (cached.hits, cached.misses) == (1, 1)


async def test_an_error_is_not_cached(tmp_path: Path) -> None:
    rig = CohereRig("429_no_retry_after", "success")
    with KVCache(tmp_path / "rerank.sqlite") as cache:
        cached = CachedReranker(rig.reranker, cache)

        with pytest.raises(ProviderRateLimited):
            await cached.rerank(QUERY, chunks(5), 5)
        ranked = await cached.rerank(QUERY, chunks(5), 5)

        assert len(cache) == 1  # only the success was stored

    assert [chunk.chunk_id for chunk in ranked] == FIXTURE_ORDER
    assert len(rig.http_requests) == 2


# --- The SDK is loaded only when the adapter is built --------------------------------------------


def test_importing_the_module_does_not_load_the_sdk() -> None:
    probe = "import sys; import grounded.retrieval.rerank; print('cohere' in sys.modules)"
    done = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True, timeout=60
    )

    assert done.stdout.strip() == "False"
