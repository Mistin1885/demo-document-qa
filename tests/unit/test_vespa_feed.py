"""Unit tests for app.vespa.feed (Phase 6.2).

All tests use ``pytest-httpx`` (``httpx_mock`` fixture) so no real Vespa is
required.  Tests cover:
- ``feed_chunks``: correct PUT URL, correct body encoding, embedding tensor
  format, idempotent upsert (same vespa_document_id → same URL), partial
  failure tracking.
- ``delete_by_document``: correct DELETE URL and selection string.
- ``health_check``: returns True on 200, False on non-200 / error.
- ``VespaDimensionMismatch``: raised when embedding dim mismatches.
- ``make_vespa_id``: deterministic across multiple calls.
"""

from __future__ import annotations

import json
import uuid

import pytest
import pytest_httpx

from app.errors import VespaDimensionMismatch
from app.vespa.feed import FeedReport, VespaChunk, VespaFeedClient, make_vespa_id

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
# make_vespa_id
# ---------------------------------------------------------------------------


class TestMakeVespaId:
    def test_returns_string(self) -> None:
        doc_id = uuid.uuid4()
        node_id = uuid.uuid4()
        result = make_vespa_id(doc_id, "raw_block", node_id, 0)
        assert isinstance(result, str)

    def test_deterministic_same_input(self) -> None:
        doc_id = uuid.uuid4()
        node_id = uuid.uuid4()
        id1 = make_vespa_id(doc_id, "raw_block", node_id, 0)
        id2 = make_vespa_id(doc_id, "raw_block", node_id, 0)
        assert id1 == id2

    def test_different_order_index_differs(self) -> None:
        doc_id = uuid.uuid4()
        node_id = uuid.uuid4()
        id0 = make_vespa_id(doc_id, "raw_block", node_id, 0)
        id1 = make_vespa_id(doc_id, "raw_block", node_id, 1)
        assert id0 != id1

    def test_different_source_type_differs(self) -> None:
        doc_id = uuid.uuid4()
        node_id = uuid.uuid4()
        id_raw = make_vespa_id(doc_id, "raw_block", node_id, 0)
        id_chunk = make_vespa_id(doc_id, "chunk", node_id, 0)
        assert id_raw != id_chunk

    def test_valid_uuid_format(self) -> None:
        doc_id = uuid.uuid4()
        node_id = uuid.uuid4()
        result = make_vespa_id(doc_id, "chunk", node_id, 5)
        # Should parse as UUID
        parsed = uuid.UUID(result)
        assert str(parsed) == result


# ---------------------------------------------------------------------------
# VespaFeedClient.feed_chunks
# ---------------------------------------------------------------------------


class TestFeedChunks:
    @pytest.mark.asyncio
    async def test_single_chunk_put_url(
        self, httpx_mock: pytest_httpx.HTTPXMock
    ) -> None:
        chunk = make_chunk()
        url = f"{ENDPOINT}/document/v1/{NAMESPACE}/{SCHEMA}/docid/{chunk.vespa_document_id}"
        httpx_mock.add_response(url=url, method="PUT", status_code=200, json={"id": chunk.vespa_document_id})

        client = make_client()
        report = await client.feed_chunks([chunk])

        assert report.success_count == 1
        assert report.fail_count == 0

    @pytest.mark.asyncio
    async def test_single_chunk_put_body_structure(
        self, httpx_mock: pytest_httpx.HTTPXMock
    ) -> None:
        chunk = make_chunk()
        url = f"{ENDPOINT}/document/v1/{NAMESPACE}/{SCHEMA}/docid/{chunk.vespa_document_id}"
        httpx_mock.add_response(url=url, method="PUT", status_code=200, json={"id": chunk.vespa_document_id})

        client = make_client()
        await client.feed_chunks([chunk])

        request = httpx_mock.get_requests()[0]
        body = json.loads(request.content)
        assert "fields" in body
        fields = body["fields"]
        assert fields["content"] == "test content"
        assert fields["source_type"] == "raw_block"

    @pytest.mark.asyncio
    async def test_embedding_encoded_as_values_tensor(
        self, httpx_mock: pytest_httpx.HTTPXMock
    ) -> None:
        chunk = make_chunk()
        chunk.embedding = [0.1, 0.2, 0.3, 0.4]
        url = f"{ENDPOINT}/document/v1/{NAMESPACE}/{SCHEMA}/docid/{chunk.vespa_document_id}"
        httpx_mock.add_response(url=url, method="PUT", status_code=200, json={"id": chunk.vespa_document_id})

        client = make_client(dim=4)
        await client.feed_chunks([chunk])

        request = httpx_mock.get_requests()[0]
        body = json.loads(request.content)
        embedding_field = body["fields"]["embedding"]
        assert isinstance(embedding_field, dict)
        assert "values" in embedding_field
        assert embedding_field["values"] == [0.1, 0.2, 0.3, 0.4]

    @pytest.mark.asyncio
    async def test_multiple_chunks_multiple_puts(
        self, httpx_mock: pytest_httpx.HTTPXMock
    ) -> None:
        chunks = [make_chunk(order_index=i) for i in range(3)]
        for chunk in chunks:
            url = f"{ENDPOINT}/document/v1/{NAMESPACE}/{SCHEMA}/docid/{chunk.vespa_document_id}"
            httpx_mock.add_response(url=url, method="PUT", status_code=200, json={"id": chunk.vespa_document_id})

        client = make_client()
        report = await client.feed_chunks(chunks)

        assert report.success_count == 3
        assert report.fail_count == 0
        assert len(httpx_mock.get_requests()) == 3

    @pytest.mark.asyncio
    async def test_idempotent_same_docid_same_url(
        self, httpx_mock: pytest_httpx.HTTPXMock
    ) -> None:
        """Same vespa_document_id produces the same PUT URL (upsert idempotency)."""
        doc_id = uuid.uuid4()
        node_id = uuid.uuid4()
        # Build two chunks with identical vespa_document_id
        vid = make_vespa_id(doc_id, "raw_block", node_id, 0)
        chunk1 = make_chunk(doc_id=doc_id)
        chunk1 = chunk1.model_copy(update={"vespa_document_id": vid})
        chunk2 = make_chunk(doc_id=doc_id)
        chunk2 = chunk2.model_copy(update={"vespa_document_id": vid})

        url = f"{ENDPOINT}/document/v1/{NAMESPACE}/{SCHEMA}/docid/{vid}"
        # Register the URL twice
        httpx_mock.add_response(url=url, method="PUT", status_code=200, json={"id": vid})
        httpx_mock.add_response(url=url, method="PUT", status_code=200, json={"id": vid})

        client = make_client()
        report = await client.feed_chunks([chunk1, chunk2])

        # Both went to the same URL
        requests = httpx_mock.get_requests()
        assert len(requests) == 2
        assert all(str(r.url) == url for r in requests)
        assert report.success_count == 2

    @pytest.mark.asyncio
    async def test_partial_failure_tracked(
        self, httpx_mock: pytest_httpx.HTTPXMock
    ) -> None:
        chunks = [make_chunk(order_index=i) for i in range(2)]
        url0 = f"{ENDPOINT}/document/v1/{NAMESPACE}/{SCHEMA}/docid/{chunks[0].vespa_document_id}"
        url1 = f"{ENDPOINT}/document/v1/{NAMESPACE}/{SCHEMA}/docid/{chunks[1].vespa_document_id}"
        httpx_mock.add_response(url=url0, method="PUT", status_code=200, json={"id": chunks[0].vespa_document_id})
        httpx_mock.add_response(url=url1, method="PUT", status_code=500, text="internal error")

        client = make_client()
        report = await client.feed_chunks(chunks)

        assert report.success_count == 1
        assert report.fail_count == 1
        assert len(report.errors) == 1

    @pytest.mark.asyncio
    async def test_empty_list_returns_empty_report(self) -> None:
        client = make_client()
        report = await client.feed_chunks([])
        assert report.success_count == 0
        assert report.fail_count == 0
        assert report.total == 0


# ---------------------------------------------------------------------------
# VespaDimensionMismatch
# ---------------------------------------------------------------------------


class TestDimensionMismatch:
    @pytest.mark.asyncio
    async def test_raises_on_wrong_dim(self) -> None:
        chunk = make_chunk(embedding_dim=8)  # 8-dim vector
        client = make_client(dim=4)  # schema expects 4

        with pytest.raises(VespaDimensionMismatch) as exc_info:
            await client.feed_chunks([chunk])

        assert exc_info.value.expected == 4
        assert exc_info.value.got == 8

    @pytest.mark.asyncio
    async def test_raises_before_any_network_call(
        self, httpx_mock: pytest_httpx.HTTPXMock
    ) -> None:
        chunk = make_chunk(embedding_dim=8)
        client = make_client(dim=4)

        with pytest.raises(VespaDimensionMismatch):
            await client.feed_chunks([chunk])

        # No HTTP requests should have been made
        assert len(httpx_mock.get_requests()) == 0

    def test_error_message_contains_dims(self) -> None:
        exc = VespaDimensionMismatch(expected=1024, got=768)
        assert "1024" in str(exc)
        assert "768" in str(exc)


# ---------------------------------------------------------------------------
# VespaFeedClient.delete_by_document
# ---------------------------------------------------------------------------


class TestDeleteByDocument:
    @pytest.mark.asyncio
    async def test_delete_issues_delete_request(
        self, httpx_mock: pytest_httpx.HTTPXMock
    ) -> None:
        chat_id = uuid.uuid4()
        doc_id = uuid.uuid4()
        # Use method-only match to avoid URL query-param conflict with pytest-httpx URL matching
        httpx_mock.add_response(
            method="DELETE",
            status_code=200,
            json={"documentCount": 5},
        )

        client = make_client()
        count = await client.delete_by_document(chat_id, doc_id)

        assert count == 5
        requests = httpx_mock.get_requests()
        assert len(requests) == 1
        assert requests[0].method == "DELETE"

    @pytest.mark.asyncio
    async def test_delete_selection_contains_chat_id(
        self, httpx_mock: pytest_httpx.HTTPXMock
    ) -> None:
        chat_id = uuid.uuid4()
        doc_id = uuid.uuid4()
        httpx_mock.add_response(
            method="DELETE",
            status_code=200,
            json={"documentCount": 0},
        )

        client = make_client()
        await client.delete_by_document(chat_id, doc_id)

        request = httpx_mock.get_requests()[0]
        params = dict(request.url.params)
        selection = params["selection"]
        assert str(chat_id) in selection
        assert "chat_id" in selection

    @pytest.mark.asyncio
    async def test_delete_selection_contains_document_id(
        self, httpx_mock: pytest_httpx.HTTPXMock
    ) -> None:
        chat_id = uuid.uuid4()
        doc_id = uuid.uuid4()
        httpx_mock.add_response(
            method="DELETE",
            status_code=200,
            json={"documentCount": 0},
        )

        client = make_client()
        await client.delete_by_document(chat_id, doc_id)

        request = httpx_mock.get_requests()[0]
        params = dict(request.url.params)
        selection = params["selection"]
        assert str(doc_id) in selection
        assert "document_id" in selection

    @pytest.mark.asyncio
    async def test_delete_selection_has_both_conditions(
        self, httpx_mock: pytest_httpx.HTTPXMock
    ) -> None:
        chat_id = uuid.uuid4()
        doc_id = uuid.uuid4()
        httpx_mock.add_response(
            method="DELETE",
            status_code=200,
            json={"documentCount": 0},
        )

        client = make_client()
        await client.delete_by_document(chat_id, doc_id)

        request = httpx_mock.get_requests()[0]
        params = dict(request.url.params)
        selection = params["selection"]
        # Must contain both tokens
        assert "chat_id" in selection
        assert "document_id" in selection
        assert str(chat_id) in selection
        assert str(doc_id) in selection

    @pytest.mark.asyncio
    async def test_delete_cluster_param_present(
        self, httpx_mock: pytest_httpx.HTTPXMock
    ) -> None:
        chat_id = uuid.uuid4()
        doc_id = uuid.uuid4()
        httpx_mock.add_response(
            method="DELETE",
            status_code=200,
            json={"documentCount": 0},
        )

        client = make_client()
        await client.delete_by_document(chat_id, doc_id)

        request = httpx_mock.get_requests()[0]
        params = dict(request.url.params)
        assert "cluster" in params
        assert params["cluster"] == CLUSTER

    @pytest.mark.asyncio
    async def test_delete_returns_zero_on_404(
        self, httpx_mock: pytest_httpx.HTTPXMock
    ) -> None:
        chat_id = uuid.uuid4()
        doc_id = uuid.uuid4()
        httpx_mock.add_response(
            method="DELETE",
            status_code=404,
        )

        client = make_client()
        count = await client.delete_by_document(chat_id, doc_id)

        assert count == 0


# ---------------------------------------------------------------------------
# VespaFeedClient.health_check
# ---------------------------------------------------------------------------


class TestHealthCheck:
    @pytest.mark.asyncio
    async def test_health_check_true_on_200(
        self, httpx_mock: pytest_httpx.HTTPXMock
    ) -> None:
        httpx_mock.add_response(
            url=f"{ENDPOINT}/ApplicationStatus",
            method="GET",
            status_code=200,
        )

        client = make_client()
        result = await client.health_check()

        assert result is True

    @pytest.mark.asyncio
    async def test_health_check_false_on_503(
        self, httpx_mock: pytest_httpx.HTTPXMock
    ) -> None:
        httpx_mock.add_response(
            url=f"{ENDPOINT}/ApplicationStatus",
            method="GET",
            status_code=503,
        )

        client = make_client()
        result = await client.health_check()

        assert result is False


# ---------------------------------------------------------------------------
# FeedReport dataclass
# ---------------------------------------------------------------------------


class TestFeedReport:
    def test_total_property(self) -> None:
        report = FeedReport(success_count=7, fail_count=3)
        assert report.total == 10

    def test_default_empty(self) -> None:
        report = FeedReport()
        assert report.success_count == 0
        assert report.fail_count == 0
        assert report.errors == []
        assert report.total == 0
