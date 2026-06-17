"""Integration-level isolation test for RetrievalService.

Verifies that when two chats (A and B) execute the same query against a
mock Vespa that returns different documents per chat, the results for chat A
never contain documents belonging to chat B, and vice versa.

CLAUDE.md §2: Chat isolation is the highest-priority invariant.

The mock Vespa server is simulated with ``pytest-httpx``.  Each chat's search
request must include its own ``chat_id`` filter in the YQL, and the response
must only contain hits with the matching ``chat_id``.
"""

from __future__ import annotations

import json
from uuid import UUID

import pytest
from pytest_httpx import HTTPXMock

from app.providers.mock import MockEmbeddingProvider
from app.retrieval.models import RetrievalRequest
from app.retrieval.service import RetrievalService

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VESPA_URL = "http://vespa-isolation-test:8080"
CHAT_A = UUID("aaaaaaaa-0000-0000-0000-aaaaaaaaaaaa")
CHAT_B = UUID("bbbbbbbb-0000-0000-0000-bbbbbbbbbbbb")
DOC_A1 = "doc-a-1"
DOC_A2 = "doc-a-2"
DOC_B1 = "doc-b-1"
DOC_B2 = "doc-b-2"

EMB_DIM = 4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _vespa_response_for_chat(
    chat_id: UUID,
    doc_ids: list[str],
) -> dict[str, object]:
    """Build a Vespa response that contains only documents for the given chat."""
    children = []
    for i, doc_id in enumerate(doc_ids):
        children.append(
            {
                "id": f"id:default:document_chunk::{doc_id}",
                "relevance": float(len(doc_ids) - i),
                "fields": {
                    "vespa_document_id": doc_id,
                    "chat_id": str(chat_id),
                    "document_id": f"document-for-{doc_id}",
                    "source_node_id": f"node-{doc_id}",
                    "source_type": "chunk",
                    "content": f"Content for {doc_id} in chat {chat_id}",
                    "title": f"Title {doc_id}",
                    "heading_path": "Section 1",
                    "page_start": 1,
                    "page_end": 2,
                    "order_index": i,
                },
            }
        )
    return {
        "root": {
            "id": "toplevel",
            "relevance": 1.0,
            "fields": {"totalCount": len(doc_ids)},
            "children": children,
        }
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def svc() -> RetrievalService:
    emb = MockEmbeddingProvider(dimension=EMB_DIM)
    return RetrievalService(
        endpoint=VESPA_URL,
        embedding_provider=emb,
        embedding_dim=EMB_DIM,
    )


# ===========================================================================
# Isolation tests
# ===========================================================================


class TestChatIsolation:
    """Two chats must never see each other's documents."""

    async def test_chat_a_results_contain_only_chat_a_docs(
        self,
        svc: RetrievalService,
        httpx_mock: HTTPXMock,
    ) -> None:
        """Chat A search returns only Chat A documents."""
        # Mock: BM25 and ANN both return Chat A docs
        httpx_mock.add_response(json=_vespa_response_for_chat(CHAT_A, [DOC_A1, DOC_A2]))
        httpx_mock.add_response(json=_vespa_response_for_chat(CHAT_A, [DOC_A1]))

        req_a = RetrievalRequest(
            chat_id=CHAT_A,
            query="test query",
            rerank_mode="none",
            final_top_k=10,
        )
        resp_a = await svc.search(req_a)

        # All returned hits must have chat_id == CHAT_A
        for hit in resp_a.hits:
            assert hit.chat_id == str(CHAT_A), (
                f"Expected chat_id={CHAT_A}, got chat_id={hit.chat_id} "
                f"for document {hit.vespa_document_id}"
            )

        # No Chat B document IDs must appear
        returned_doc_ids = {hit.vespa_document_id for hit in resp_a.hits}
        assert DOC_B1 not in returned_doc_ids
        assert DOC_B2 not in returned_doc_ids

    async def test_chat_b_results_contain_only_chat_b_docs(
        self,
        svc: RetrievalService,
        httpx_mock: HTTPXMock,
    ) -> None:
        """Chat B search returns only Chat B documents."""
        httpx_mock.add_response(json=_vespa_response_for_chat(CHAT_B, [DOC_B1, DOC_B2]))
        httpx_mock.add_response(json=_vespa_response_for_chat(CHAT_B, [DOC_B2]))

        req_b = RetrievalRequest(
            chat_id=CHAT_B,
            query="test query",
            rerank_mode="none",
            final_top_k=10,
        )
        resp_b = await svc.search(req_b)

        for hit in resp_b.hits:
            assert hit.chat_id == str(CHAT_B), (
                f"Expected chat_id={CHAT_B}, got chat_id={hit.chat_id} "
                f"for document {hit.vespa_document_id}"
            )

        returned_doc_ids = {hit.vespa_document_id for hit in resp_b.hits}
        assert DOC_A1 not in returned_doc_ids
        assert DOC_A2 not in returned_doc_ids

    async def test_yql_for_chat_a_contains_chat_a_id_not_chat_b(
        self,
        svc: RetrievalService,
        httpx_mock: HTTPXMock,
    ) -> None:
        """The YQL sent for Chat A must contain Chat A's ID, not Chat B's."""
        httpx_mock.add_response(json=_vespa_response_for_chat(CHAT_A, [DOC_A1]))
        httpx_mock.add_response(json=_vespa_response_for_chat(CHAT_A, []))

        req = RetrievalRequest(chat_id=CHAT_A, query="q", rerank_mode="none")
        await svc.search(req)

        for r in httpx_mock.get_requests():
            body = json.loads(r.content)
            yql = body.get("yql", "")
            assert str(CHAT_A) in yql, f"Chat A ID not in YQL: {yql!r}"
            assert str(CHAT_B) not in yql, f"Chat B ID leaked into Chat A YQL: {yql!r}"

    async def test_yql_for_chat_b_contains_chat_b_id_not_chat_a(
        self,
        svc: RetrievalService,
        httpx_mock: HTTPXMock,
    ) -> None:
        """The YQL sent for Chat B must contain Chat B's ID, not Chat A's."""
        httpx_mock.add_response(json=_vespa_response_for_chat(CHAT_B, [DOC_B1]))
        httpx_mock.add_response(json=_vespa_response_for_chat(CHAT_B, []))

        req = RetrievalRequest(chat_id=CHAT_B, query="q", rerank_mode="none")
        await svc.search(req)

        for r in httpx_mock.get_requests():
            body = json.loads(r.content)
            yql = body.get("yql", "")
            assert str(CHAT_B) in yql
            assert str(CHAT_A) not in yql

    async def test_sequential_searches_do_not_bleed_filters(
        self,
        svc: RetrievalService,
        httpx_mock: HTTPXMock,
    ) -> None:
        """Two back-to-back searches for different chats must not share filters."""
        # Setup 4 mock responses: 2 for Chat A search, 2 for Chat B search
        httpx_mock.add_response(json=_vespa_response_for_chat(CHAT_A, [DOC_A1]))
        httpx_mock.add_response(json=_vespa_response_for_chat(CHAT_A, []))
        httpx_mock.add_response(json=_vespa_response_for_chat(CHAT_B, [DOC_B1]))
        httpx_mock.add_response(json=_vespa_response_for_chat(CHAT_B, []))

        req_a = RetrievalRequest(chat_id=CHAT_A, query="q", rerank_mode="none")
        req_b = RetrievalRequest(chat_id=CHAT_B, query="q", rerank_mode="none")

        resp_a = await svc.search(req_a)
        resp_b = await svc.search(req_b)

        # Chat A response must not contain Chat B documents
        a_ids = {h.vespa_document_id for h in resp_a.hits}
        b_ids = {h.vespa_document_id for h in resp_b.hits}

        # No overlap of Chat B doc IDs in Chat A response
        assert not a_ids.intersection({DOC_B1, DOC_B2}), (
            f"Chat A response contains Chat B documents: {a_ids}"
        )
        # No overlap of Chat A doc IDs in Chat B response
        assert not b_ids.intersection({DOC_A1, DOC_A2}), (
            f"Chat B response contains Chat A documents: {b_ids}"
        )

    async def test_native_rerank_yql_also_enforces_chat_id(
        self,
        svc: RetrievalService,
        httpx_mock: HTTPXMock,
    ) -> None:
        """Even the native-rerank (3rd) Vespa request must have the chat_id filter."""
        hits_a = [
            {
                "vespa_document_id": DOC_A1,
                "chat_id": str(CHAT_A),
                "document_id": "doc-doc-a1",
                "source_node_id": "node-a1",
                "source_type": "chunk",
                "content": "a1 content",
                "title": "",
                "heading_path": "",
                "page_start": 1,
                "page_end": 1,
                "order_index": 0,
            }
        ]

        def _resp(hits: list[dict[str, object]]) -> dict[str, object]:
            children = []
            for i, fields in enumerate(hits):
                f = dict(fields)
                children.append(
                    {
                        "id": f"id:default:document_chunk::{f['vespa_document_id']}",
                        "relevance": float(len(hits) - i),
                        "fields": f,
                    }
                )
            return {"root": {"children": children}}

        httpx_mock.add_response(json=_resp(hits_a))
        httpx_mock.add_response(json=_resp(hits_a))
        httpx_mock.add_response(json=_resp(hits_a))

        req = RetrievalRequest(
            chat_id=CHAT_A,
            query="q",
            rerank_mode="native",
            rerank_top_k=5,
            debug=True,
        )
        await svc.search(req)

        all_requests = httpx_mock.get_requests()
        assert len(all_requests) == 3

        for r in all_requests:
            body = json.loads(r.content)
            yql = body.get("yql", "")
            assert str(CHAT_A) in yql, f"chat_id {CHAT_A} not found in YQL of request: {yql!r}"
            assert str(CHAT_B) not in yql
