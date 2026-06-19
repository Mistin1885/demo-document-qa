"""Unit tests for app.vespa.feed (Phase 6.2).

All tests use ``pytest-httpx`` so no real Vespa is required.
"""

from __future__ import annotations

import json
import uuid

import pytest
import pytest_httpx

from app.errors import VespaDimensionMismatch
from app.vespa.feed import VespaChunk, VespaFeedClient, make_vespa_id

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

EMBEDDING_DIM = 4
ENDPOINT = "http://vespa-test:8080"
NAMESPACE = "default"
SCHEMA = "document_chunk"
CLUSTER = "documents"


def make_client(dim: int = EMBEDDING_DIM) -> VespaFeedClient:
    return VespaFeedClient(
        endpoint=ENDPOINT,
        application_name=NAMESPACE,
        schema_name=SCHEMA,
        embedding_dim=dim,
        cluster=CLUSTER,
    )


def make_chunk(
    doc_id: uuid.UUID | None = None,
    chat_id: uuid.UUID | None = None,
    order_index: int = 0,
    embedding_dim: int = EMBEDDING_DIM,
) -> VespaChunk:
    doc_id = doc_id or uuid.uuid4()
    chat_id = chat_id or uuid.uuid4()
    node_id = uuid.uuid4()
    vid = make_vespa_id(doc_id, "raw_block", node_id, order_index)
    return VespaChunk(
        vespa_document_id=vid,
        chat_id=str(chat_id),
        document_id=str(doc_id),
        source_node_id=str(node_id),
        source_type="raw_block",
        content="test content",
        page_start=1,
        page_end=1,
        order_index=order_index,
        token_count=3,
        embedding=[0.1] * embedding_dim,
        created_at=1_700_000_000_000,
    )


# ---------------------------------------------------------------------------
# make_vespa_id — deterministic uuid5
# ---------------------------------------------------------------------------


def test_make_vespa_id_deterministic_and_unique() -> None:
    """Same inputs → identical ID; different order_index or source_type → different ID."""
    doc_id, node_id = uuid.uuid4(), uuid.uuid4()
    id1 = make_vespa_id(doc_id, "raw_block", node_id, 0)
    id2 = make_vespa_id(doc_id, "raw_block", node_id, 0)
    assert id1 == id2
    assert isinstance(id1, str)
    uuid.UUID(id1)  # must be valid UUID
    assert make_vespa_id(doc_id, "raw_block", node_id, 0) != make_vespa_id(doc_id, "raw_block", node_id, 1)
    assert make_vespa_id(doc_id, "raw_block", node_id, 0) != make_vespa_id(doc_id, "chunk", node_id, 0)


# ---------------------------------------------------------------------------
# feed_chunks
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_feed_chunks_put_url_body_and_embedding(
    httpx_mock: pytest_httpx.HTTPXMock,
) -> None:
    """Single chunk: correct POST URL, body fields, embedding tensor format."""
    chunk = make_chunk()
    chunk.embedding = [0.1, 0.2, 0.3, 0.4]
    url = f"{ENDPOINT}/document/v1/{NAMESPACE}/{SCHEMA}/docid/{chunk.vespa_document_id}"
    httpx_mock.add_response(url=url, method="POST", status_code=200, json={"id": chunk.vespa_document_id})

    client = make_client(dim=4)
    report = await client.feed_chunks([chunk])

    assert report.success_count == 1 and report.fail_count == 0
    body = json.loads(httpx_mock.get_requests()[0].content)
    assert body["fields"]["content"] == "test content"
    assert body["fields"]["source_type"] == "raw_block"
    emb = body["fields"]["embedding"]
    assert isinstance(emb, dict) and "values" in emb
    assert emb["values"] == [0.1, 0.2, 0.3, 0.4]


@pytest.mark.asyncio
async def test_feed_chunks_idempotency_and_partial_failure(
    httpx_mock: pytest_httpx.HTTPXMock,
) -> None:
    """Same vespa_document_id → same URL (idempotent upsert). Partial 500 → tracked."""
    doc_id = uuid.uuid4()
    node_id = uuid.uuid4()
    vid = make_vespa_id(doc_id, "raw_block", node_id, 0)
    chunk1 = make_chunk(doc_id=doc_id).model_copy(update={"vespa_document_id": vid})
    chunk2 = make_chunk(doc_id=doc_id).model_copy(update={"vespa_document_id": vid})
    url = f"{ENDPOINT}/document/v1/{NAMESPACE}/{SCHEMA}/docid/{vid}"
    httpx_mock.add_response(url=url, method="POST", status_code=200, json={"id": vid})
    httpx_mock.add_response(url=url, method="POST", status_code=200, json={"id": vid})
    report = await make_client().feed_chunks([chunk1, chunk2])
    reqs = httpx_mock.get_requests()
    assert len(reqs) == 2 and all(str(r.url) == url for r in reqs)
    assert report.success_count == 2

    # Partial failure: two chunks, second returns 500
    chunk_a, chunk_b = make_chunk(order_index=0), make_chunk(order_index=1)
    httpx_mock.add_response(
        url=f"{ENDPOINT}/document/v1/{NAMESPACE}/{SCHEMA}/docid/{chunk_a.vespa_document_id}",
        method="POST", status_code=200, json={"id": chunk_a.vespa_document_id},
    )
    httpx_mock.add_response(
        url=f"{ENDPOINT}/document/v1/{NAMESPACE}/{SCHEMA}/docid/{chunk_b.vespa_document_id}",
        method="POST", status_code=500, text="internal error",
    )
    r2 = await make_client().feed_chunks([chunk_a, chunk_b])
    assert r2.success_count == 1 and r2.fail_count == 1 and len(r2.errors) == 1


@pytest.mark.asyncio
async def test_feed_chunks_empty_list() -> None:
    report = await make_client().feed_chunks([])
    assert report.success_count == 0 and report.fail_count == 0 and report.total == 0


# ---------------------------------------------------------------------------
# VespaDimensionMismatch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dimension_mismatch_raised_before_network(
    httpx_mock: pytest_httpx.HTTPXMock,
) -> None:
    """Wrong embedding dim raises VespaDimensionMismatch before any HTTP call."""
    chunk = make_chunk(embedding_dim=8)
    client = make_client(dim=4)
    with pytest.raises(VespaDimensionMismatch) as exc_info:
        await client.feed_chunks([chunk])
    assert exc_info.value.expected == 4 and exc_info.value.got == 8
    assert len(httpx_mock.get_requests()) == 0
    assert "1024" in str(VespaDimensionMismatch(expected=1024, got=768))


# ---------------------------------------------------------------------------
# delete_by_document — chat_id AND document_id in selection
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delete_by_document_selection_and_cluster(
    httpx_mock: pytest_httpx.HTTPXMock,
) -> None:
    """DELETE selection must contain both chat_id and document_id values; cluster param present."""
    chat_id, doc_id = uuid.uuid4(), uuid.uuid4()
    httpx_mock.add_response(method="DELETE", status_code=200, json={"documentCount": 5})

    client = make_client()
    count = await client.delete_by_document(chat_id, doc_id)

    assert count == 5
    reqs = httpx_mock.get_requests()
    assert len(reqs) == 1 and reqs[0].method == "DELETE"
    params = dict(reqs[0].url.params)
    selection = params["selection"]
    assert "chat_id" in selection and str(chat_id) in selection
    assert "document_id" in selection and str(doc_id) in selection
    assert params.get("cluster") == CLUSTER


@pytest.mark.asyncio
async def test_delete_by_document_404_returns_zero(
    httpx_mock: pytest_httpx.HTTPXMock,
) -> None:
    httpx_mock.add_response(method="DELETE", status_code=404)
    count = await make_client().delete_by_document(uuid.uuid4(), uuid.uuid4())
    assert count == 0


# ---------------------------------------------------------------------------
# health_check
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_health_check(httpx_mock: pytest_httpx.HTTPXMock) -> None:
    """200 → True; 503 → False."""
    url = f"{ENDPOINT}/ApplicationStatus"
    httpx_mock.add_response(url=url, method="GET", status_code=200)
    assert await make_client().health_check() is True
    httpx_mock.add_response(url=url, method="GET", status_code=503)
    assert await make_client().health_check() is False
