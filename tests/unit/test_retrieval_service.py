"""Unit tests for RetrievalService.

Uses ``pytest-httpx`` to mock Vespa REST /search/ calls — no real Vespa or
OpenAI/embedding API is involved.

Coverage:
- rerank_mode="none": exactly 2 requests (BM25 + ANN), fusion scores, 1-based final_rank.
- rerank_mode="native": 3 requests, chat_id in every YQL.
- rerank_mode="cross_encoder": 2 requests + reranker.score() call.
- reranker_provider=None with cross_encoder mode raises RerankerUnavailable.
- chat_id isolation: every YQL contains the expected chat_id; query text never embedded.
- document_ids / source_types filters appear in YQL; invalid source_type raises before network.
- embedding dimension mismatch raises VespaDimensionMismatch.
- SearchHit fields are correctly mapped (all stage scores present).
"""

from __future__ import annotations

import json
from uuid import UUID

import pytest
from pytest_httpx import HTTPXMock

from app.errors import InvalidRetrievalFilter, RerankerUnavailable, VespaDimensionMismatch
from app.providers.mock import MockEmbeddingProvider, MockRerankerProvider
from app.retrieval.models import RetrievalRequest
from app.retrieval.service import RetrievalService

VESPA_URL = "http://vespa-test:8080"
CHAT_A = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
CHAT_B = UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")
DOC_1 = UUID("11111111-1111-1111-1111-111111111111")
DOC_2 = UUID("22222222-2222-2222-2222-222222222222")
EMB_DIM = 4


def _vespa_response(hits: list[dict[str, object]], total: int | None = None) -> dict[str, object]:
    children = []
    for i, fields in enumerate(hits):
        children.append(
            {
                "id": f"id:default:document_chunk::{fields.get('vespa_document_id', f'doc-{i}')}",
                "relevance": fields.pop("relevance", float(len(hits) - i)),
                "fields": fields,
            }
        )
    return {
        "root": {
            "id": "toplevel",
            "relevance": 1.0,
            "fields": {"totalCount": total or len(hits)},
            "children": children,
        }
    }


def _make_hit(
    doc_id: str,
    chat_id: str = str(CHAT_A),
    relevance: float = 1.0,
    source_type: str = "chunk",
    content: str = "test content",
    page_start: int = 1,
    page_end: int = 1,
    order_index: int = 0,
) -> dict[str, object]:
    return {
        "vespa_document_id": doc_id,
        "chat_id": chat_id,
        "document_id": str(DOC_1),
        "source_node_id": f"node-{doc_id}",
        "source_type": source_type,
        "content": content,
        "title": "Test Title",
        "heading_path": "Section 1",
        "page_start": page_start,
        "page_end": page_end,
        "order_index": order_index,
        "relevance": relevance,
    }


@pytest.fixture
def emb_provider() -> MockEmbeddingProvider:
    return MockEmbeddingProvider(dimension=EMB_DIM)


@pytest.fixture
def reranker_provider() -> MockRerankerProvider:
    return MockRerankerProvider()


@pytest.fixture
def service(emb_provider: MockEmbeddingProvider) -> RetrievalService:
    return RetrievalService(
        endpoint=VESPA_URL,
        embedding_provider=emb_provider,
        reranker_provider=None,
        embedding_dim=EMB_DIM,
    )


@pytest.fixture
def service_with_reranker(
    emb_provider: MockEmbeddingProvider,
    reranker_provider: MockRerankerProvider,
) -> RetrievalService:
    return RetrievalService(
        endpoint=VESPA_URL,
        embedding_provider=emb_provider,
        reranker_provider=reranker_provider,
        embedding_dim=EMB_DIM,
    )


# ===========================================================================
# 1. rerank_mode="none" smoke — 2 requests, fusion scores, 1-based rank
# ===========================================================================


class TestRerankNoneSmoke:
    async def test_two_requests_fusion_score_and_rank(
        self,
        service: RetrievalService,
        httpx_mock: HTTPXMock,
    ) -> None:
        httpx_mock.add_response(
            json=_vespa_response([_make_hit("d1", relevance=5.0), _make_hit("d2", relevance=3.0)])
        )
        httpx_mock.add_response(json=_vespa_response([_make_hit("d1", relevance=0.9)]))

        req = RetrievalRequest(chat_id=CHAT_A, query="q", rerank_mode="none", final_top_k=5)
        resp = await service.search(req)

        assert len(httpx_mock.get_requests()) == 2
        assert all(h.fusion_score is not None for h in resp.hits)
        if resp.hits:
            assert resp.hits[0].final_rank == 1
            ranks = [h.final_rank for h in resp.hits]
            assert ranks == list(range(1, len(ranks) + 1))


# ===========================================================================
# 2. rerank_mode="native" smoke — 3 requests, chat_id in every YQL
# ===========================================================================


class TestRerankNativeSmoke:
    async def test_three_requests_all_contain_chat_id(
        self,
        service: RetrievalService,
        httpx_mock: HTTPXMock,
    ) -> None:
        hits = [_make_hit(f"d{i}") for i in range(2)]
        httpx_mock.add_response(json=_vespa_response(hits))
        httpx_mock.add_response(json=_vespa_response(hits))
        httpx_mock.add_response(json=_vespa_response(hits))

        req = RetrievalRequest(chat_id=CHAT_A, query="q", rerank_mode="native", rerank_top_k=5)
        await service.search(req)

        requests = httpx_mock.get_requests()
        assert len(requests) == 3
        for r in requests:
            yql = json.loads(r.content).get("yql", "")
            assert str(CHAT_A) in yql, f"chat_id not found in YQL: {yql!r}"


# ===========================================================================
# 3. rerank_mode="cross_encoder" smoke — 2 requests + reranker, no provider raises
# ===========================================================================


class TestRerankCrossEncoderSmoke:
    async def test_two_requests_cross_encoder_scores(
        self,
        service_with_reranker: RetrievalService,
        httpx_mock: HTTPXMock,
    ) -> None:
        httpx_mock.add_response(
            json=_vespa_response([_make_hit("ce-1", relevance=5.0), _make_hit("ce-2", relevance=3.0)])
        )
        httpx_mock.add_response(json=_vespa_response([]))

        req = RetrievalRequest(
            chat_id=CHAT_A, query="q", rerank_mode="cross_encoder", final_top_k=2
        )
        resp = await service_with_reranker.search(req)

        assert len(httpx_mock.get_requests()) == 2
        for hit in resp.hits:
            assert hit.cross_encoder_score is not None

    async def test_no_reranker_raises(
        self,
        service: RetrievalService,
        httpx_mock: HTTPXMock,
    ) -> None:
        httpx_mock.add_response(json=_vespa_response([_make_hit("d1")]))
        httpx_mock.add_response(json=_vespa_response([]))

        req = RetrievalRequest(chat_id=CHAT_A, query="q", rerank_mode="cross_encoder")
        with pytest.raises(RerankerUnavailable):
            await service.search(req)


# ===========================================================================
# 4. Chat-id isolation + injection safety
# ===========================================================================


class TestChatIdIsolation:
    async def test_correct_chat_id_in_yql_not_other(
        self,
        service: RetrievalService,
        httpx_mock: HTTPXMock,
    ) -> None:
        httpx_mock.add_response(json=_vespa_response([]))
        httpx_mock.add_response(json=_vespa_response([]))

        req = RetrievalRequest(chat_id=CHAT_B, query="q", rerank_mode="none")
        await service.search(req)

        for r in httpx_mock.get_requests():
            yql = json.loads(r.content).get("yql", "")
            assert str(CHAT_B) in yql
            assert str(CHAT_A) not in yql

    async def test_query_text_never_in_yql(
        self,
        service: RetrievalService,
        httpx_mock: HTTPXMock,
    ) -> None:
        evil_query = "'; drop table documents; SELECT * FROM chats --"
        httpx_mock.add_response(json=_vespa_response([]))
        httpx_mock.add_response(json=_vespa_response([]))

        req = RetrievalRequest(chat_id=CHAT_A, query=evil_query, rerank_mode="none")
        await service.search(req)

        for r in httpx_mock.get_requests():
            yql = json.loads(r.content).get("yql", "")
            assert evil_query not in yql
            assert "drop table" not in yql.lower()


# ===========================================================================
# 5. Filters + dimension mismatch + SearchHit mapping
# ===========================================================================


class TestFiltersAndMapping:
    async def test_document_and_source_type_filters_in_yql(
        self,
        service: RetrievalService,
        httpx_mock: HTTPXMock,
    ) -> None:
        httpx_mock.add_response(json=_vespa_response([]))
        httpx_mock.add_response(json=_vespa_response([]))

        req = RetrievalRequest(
            chat_id=CHAT_A,
            query="q",
            document_ids=[DOC_1, DOC_2],
            source_types=["chunk", "section_summary"],
            rerank_mode="none",
        )
        await service.search(req)

        for r in httpx_mock.get_requests():
            yql = json.loads(r.content).get("yql", "")
            assert str(DOC_1) in yql and str(DOC_2) in yql
            assert "chunk" in yql and "section_summary" in yql

    async def test_invalid_source_type_raises_before_network(
        self,
        service: RetrievalService,
        httpx_mock: HTTPXMock,
    ) -> None:
        req = RetrievalRequest(
            chat_id=CHAT_A, query="q", source_types=["invalid_xyz"], rerank_mode="none"
        )
        with pytest.raises(InvalidRetrievalFilter):
            await service.search(req)
        assert len(httpx_mock.get_requests()) == 0

    async def test_dimension_mismatch_raises_before_network(
        self,
        httpx_mock: HTTPXMock,
    ) -> None:
        svc = RetrievalService(
            endpoint=VESPA_URL,
            embedding_provider=MockEmbeddingProvider(dimension=8),
            embedding_dim=4,
        )
        req = RetrievalRequest(chat_id=CHAT_A, query="q", rerank_mode="none")
        with pytest.raises(VespaDimensionMismatch):
            await svc.search(req)
        assert len(httpx_mock.get_requests()) == 0

    async def test_search_hit_all_fields_mapped(
        self,
        service: RetrievalService,
        httpx_mock: HTTPXMock,
    ) -> None:
        hit_data = _make_hit(
            "doc-mapped",
            chat_id=str(CHAT_A),
            relevance=7.5,
            source_type="section_summary",
            content="Section content here",
            page_start=3,
            page_end=5,
            order_index=42,
        )
        httpx_mock.add_response(json=_vespa_response([hit_data]))
        httpx_mock.add_response(json=_vespa_response([]))

        req = RetrievalRequest(chat_id=CHAT_A, query="q", rerank_mode="none", final_top_k=1)
        resp = await service.search(req)

        assert len(resp.hits) == 1
        h = resp.hits[0]
        assert h.vespa_document_id == "doc-mapped"
        assert h.chat_id == str(CHAT_A)
        assert h.source_type == "section_summary"
        assert h.content == "Section content here"
        assert h.page_start == 3
        assert h.page_end == 5
        assert h.order_index == 42
        assert h.final_rank == 1
        assert h.fusion_score is not None
