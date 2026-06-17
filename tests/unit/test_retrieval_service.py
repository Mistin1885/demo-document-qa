"""Unit tests for RetrievalService.

Uses ``pytest-httpx`` to mock Vespa REST /search/ calls — no real Vespa or
OpenAI/embedding API is involved.

Coverage:
- rerank_mode="none": exactly 2 requests (BM25 + ANN), fusion, direct return.
- rerank_mode="native": 3 requests (BM25 + ANN + native rerank by id).
- rerank_mode="cross_encoder": 2 requests + reranker.score() call.
- reranker_provider=None with cross_encoder mode raises RerankerUnavailable.
- chat_id isolation: every YQL contains the expected chat_id.
- document_ids filter appears in YQL.
- source_types filter appears in YQL.
- embedding dimension mismatch raises VespaDimensionMismatch.
- debug=True: RetrievalResponse.debug is populated with counts + queries.
- final_rank is 1-based.
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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VESPA_URL = "http://vespa-test:8080"
CHAT_A = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
CHAT_B = UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")
DOC_1 = UUID("11111111-1111-1111-1111-111111111111")
DOC_2 = UUID("22222222-2222-2222-2222-222222222222")

EMB_DIM = 4  # small for test speed


# ---------------------------------------------------------------------------
# Vespa mock response builder
# ---------------------------------------------------------------------------


def _vespa_response(hits: list[dict[str, object]], total: int | None = None) -> dict[str, object]:
    """Build a minimal Vespa /search/ JSON response."""
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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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
# 1. rerank_mode="none" — 2 requests, no rerank
# ===========================================================================


class TestRerankNone:
    async def test_two_requests_sent(
        self,
        service: RetrievalService,
        httpx_mock: HTTPXMock,
    ) -> None:
        httpx_mock.add_response(json=_vespa_response([_make_hit("doc-bm25-1", relevance=5.0)]))
        httpx_mock.add_response(json=_vespa_response([_make_hit("doc-ann-1", relevance=0.9)]))

        req = RetrievalRequest(
            chat_id=CHAT_A,
            query="test query",
            rerank_mode="none",
            final_top_k=5,
        )
        resp = await service.search(req)

        requests = httpx_mock.get_requests()
        assert len(requests) == 2
        assert len(resp.hits) <= 5

    async def test_hits_have_fusion_score(
        self,
        service: RetrievalService,
        httpx_mock: HTTPXMock,
    ) -> None:
        httpx_mock.add_response(
            json=_vespa_response([_make_hit("doc-shared", relevance=3.0)])
        )  # BM25
        httpx_mock.add_response(
            json=_vespa_response([_make_hit("doc-shared", relevance=0.8)])
        )  # ANN

        req = RetrievalRequest(chat_id=CHAT_A, query="q", rerank_mode="none", final_top_k=3)
        resp = await service.search(req)
        assert all(h.fusion_score is not None for h in resp.hits)

    async def test_final_rank_starts_at_one(
        self,
        service: RetrievalService,
        httpx_mock: HTTPXMock,
    ) -> None:
        httpx_mock.add_response(
            json=_vespa_response([_make_hit("d1", relevance=5.0), _make_hit("d2", relevance=3.0)])
        )
        httpx_mock.add_response(json=_vespa_response([]))

        req = RetrievalRequest(chat_id=CHAT_A, query="q", rerank_mode="none", final_top_k=5)
        resp = await service.search(req)
        if resp.hits:
            ranks = [h.final_rank for h in resp.hits]
            assert ranks[0] == 1
            assert ranks == list(range(1, len(ranks) + 1))


# ===========================================================================
# 2. rerank_mode="native" — 3 requests
# ===========================================================================


class TestRerankNative:
    async def test_three_requests_sent(
        self,
        service: RetrievalService,
        httpx_mock: HTTPXMock,
    ) -> None:
        bm25_hits = [_make_hit(f"doc-{i}", relevance=float(5 - i)) for i in range(3)]
        ann_hits = [_make_hit(f"doc-{i}", relevance=float(3 - i)) for i in range(3)]
        rerank_hits = [_make_hit(f"doc-{i}", relevance=float(10 - i)) for i in range(3)]

        httpx_mock.add_response(json=_vespa_response(bm25_hits))
        httpx_mock.add_response(json=_vespa_response(ann_hits))
        httpx_mock.add_response(json=_vespa_response(rerank_hits))

        req = RetrievalRequest(
            chat_id=CHAT_A,
            query="native rerank test",
            rerank_mode="native",
            rerank_top_k=5,
            final_top_k=3,
        )
        await service.search(req)

        requests = httpx_mock.get_requests()
        assert len(requests) == 3

    async def test_final_hits_sorted_by_native_score(
        self,
        service: RetrievalService,
        httpx_mock: HTTPXMock,
    ) -> None:
        """Hits should be ordered by native_rerank_score descending."""
        bm25_hits = [_make_hit("doc-a", relevance=5.0), _make_hit("doc-b", relevance=4.0)]
        ann_hits = [_make_hit("doc-b", relevance=0.9), _make_hit("doc-a", relevance=0.8)]
        # Rerank: doc-b gets higher native score
        rerank_hits = [
            _make_hit("doc-b", relevance=20.0),
            _make_hit("doc-a", relevance=10.0),
        ]

        httpx_mock.add_response(json=_vespa_response(bm25_hits))
        httpx_mock.add_response(json=_vespa_response(ann_hits))
        httpx_mock.add_response(json=_vespa_response(rerank_hits))

        req = RetrievalRequest(
            chat_id=CHAT_A,
            query="q",
            rerank_mode="native",
            rerank_top_k=5,
            final_top_k=2,
        )
        resp = await service.search(req)

        assert len(resp.hits) == 2
        assert resp.hits[0].vespa_document_id == "doc-b"
        assert resp.hits[1].vespa_document_id == "doc-a"
        assert resp.hits[0].native_rerank_score is not None
        assert resp.hits[1].native_rerank_score is not None
        assert resp.hits[0].native_rerank_score > resp.hits[1].native_rerank_score  # type: ignore[operator]

    async def test_native_rerank_yql_contains_chat_id(
        self,
        service: RetrievalService,
        httpx_mock: HTTPXMock,
    ) -> None:
        """The third (native-rerank) request must also have chat_id in its YQL."""
        hits_for_both = [_make_hit(f"doc-{i}") for i in range(2)]

        httpx_mock.add_response(json=_vespa_response(hits_for_both))
        httpx_mock.add_response(json=_vespa_response(hits_for_both))
        httpx_mock.add_response(json=_vespa_response(hits_for_both))

        req = RetrievalRequest(
            chat_id=CHAT_A,
            query="q",
            rerank_mode="native",
            debug=True,
        )
        await service.search(req)

        # Check via captured requests
        requests = httpx_mock.get_requests()
        assert len(requests) == 3
        for r in requests:
            body = json.loads(r.content)
            yql = body.get("yql", "")
            assert str(CHAT_A) in yql, f"chat_id not found in YQL: {yql!r}"


# ===========================================================================
# 3. rerank_mode="cross_encoder"
# ===========================================================================


class TestRerankCrossEncoder:
    async def test_two_requests_plus_reranker(
        self,
        service_with_reranker: RetrievalService,
        httpx_mock: HTTPXMock,
    ) -> None:
        bm25_hits = [_make_hit("doc-ce-1", relevance=5.0), _make_hit("doc-ce-2", relevance=3.0)]
        ann_hits = [_make_hit("doc-ce-2", relevance=0.9), _make_hit("doc-ce-1", relevance=0.8)]

        httpx_mock.add_response(json=_vespa_response(bm25_hits))
        httpx_mock.add_response(json=_vespa_response(ann_hits))

        req = RetrievalRequest(
            chat_id=CHAT_A,
            query="cross encoder query",
            rerank_mode="cross_encoder",
            rerank_top_k=5,
            final_top_k=2,
        )
        resp = await service_with_reranker.search(req)

        # Only 2 HTTP requests (no third Vespa call)
        requests = httpx_mock.get_requests()
        assert len(requests) == 2

        # Hits have cross_encoder_score
        for hit in resp.hits:
            assert hit.cross_encoder_score is not None

    async def test_cross_encoder_none_raises(
        self,
        service: RetrievalService,  # no reranker
        httpx_mock: HTTPXMock,
    ) -> None:
        """RerankerUnavailable is raised when reranker_provider is None."""
        httpx_mock.add_response(json=_vespa_response([_make_hit("d1")]))
        httpx_mock.add_response(json=_vespa_response([_make_hit("d1")]))

        req = RetrievalRequest(
            chat_id=CHAT_A,
            query="q",
            rerank_mode="cross_encoder",
        )
        with pytest.raises(RerankerUnavailable):
            await service.search(req)

    async def test_cross_encoder_scores_drive_ordering(
        self,
        service_with_reranker: RetrievalService,
        httpx_mock: HTTPXMock,
    ) -> None:
        """Final order is by cross_encoder_score (MockReranker is deterministic)."""
        bm25_hits = [
            _make_hit("doc-alpha", relevance=10.0, content="alpha content"),
            _make_hit("doc-beta", relevance=5.0, content="beta content"),
        ]
        ann_hits: list[dict[str, object]] = []

        httpx_mock.add_response(json=_vespa_response(bm25_hits))
        httpx_mock.add_response(json=_vespa_response(ann_hits))

        req = RetrievalRequest(
            chat_id=CHAT_A,
            query="alpha content",
            rerank_mode="cross_encoder",
            final_top_k=2,
        )
        resp = await service_with_reranker.search(req)

        assert len(resp.hits) == 2
        # Scores must be sorted descending
        scores = [h.cross_encoder_score for h in resp.hits]
        assert scores[0] is not None and scores[1] is not None
        assert scores[0] >= scores[1]  # type: ignore[operator]


# ===========================================================================
# 4. Chat-id isolation assertion
# ===========================================================================


class TestChatIdIsolation:
    """Every YQL sent to Vespa must contain the correct chat_id."""

    async def test_all_yqls_contain_chat_id_none_mode(
        self,
        service: RetrievalService,
        httpx_mock: HTTPXMock,
    ) -> None:
        httpx_mock.add_response(json=_vespa_response([_make_hit("d1")]))
        httpx_mock.add_response(json=_vespa_response([_make_hit("d1")]))

        req = RetrievalRequest(chat_id=CHAT_A, query="q", rerank_mode="none")
        await service.search(req)

        requests = httpx_mock.get_requests()
        for r in requests:
            body = json.loads(r.content)
            yql = body.get("yql", "")
            assert str(CHAT_A) in yql, f"chat_id {CHAT_A} not found in YQL: {yql!r}"

    async def test_chat_b_id_in_yql_not_chat_a(
        self,
        service: RetrievalService,
        httpx_mock: HTTPXMock,
    ) -> None:
        httpx_mock.add_response(json=_vespa_response([]))
        httpx_mock.add_response(json=_vespa_response([]))

        req = RetrievalRequest(chat_id=CHAT_B, query="q", rerank_mode="none")
        await service.search(req)

        requests = httpx_mock.get_requests()
        for r in requests:
            body = json.loads(r.content)
            yql = body.get("yql", "")
            assert str(CHAT_B) in yql
            assert str(CHAT_A) not in yql

    async def test_yql_never_contains_query_text(
        self,
        service: RetrievalService,
        httpx_mock: HTTPXMock,
    ) -> None:
        """Query text must NOT be embedded in the YQL string (XSS / injection safety)."""
        evil_query = "'; drop table documents; SELECT * FROM chats --"

        httpx_mock.add_response(json=_vespa_response([]))
        httpx_mock.add_response(json=_vespa_response([]))

        req = RetrievalRequest(chat_id=CHAT_A, query=evil_query, rerank_mode="none")
        await service.search(req)

        requests = httpx_mock.get_requests()
        for r in requests:
            body = json.loads(r.content)
            yql = body.get("yql", "")
            # The evil text should NOT appear in the YQL string
            assert evil_query not in yql
            assert "drop table" not in yql.lower()


# ===========================================================================
# 5. Filter passthrough tests
# ===========================================================================


class TestFilters:
    async def test_document_ids_filter_in_yql(
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
            rerank_mode="none",
        )
        await service.search(req)

        requests = httpx_mock.get_requests()
        for r in requests:
            body = json.loads(r.content)
            yql = body.get("yql", "")
            assert str(DOC_1) in yql
            assert str(DOC_2) in yql
            assert "document_id in" in yql

    async def test_source_types_filter_in_yql(
        self,
        service: RetrievalService,
        httpx_mock: HTTPXMock,
    ) -> None:
        httpx_mock.add_response(json=_vespa_response([]))
        httpx_mock.add_response(json=_vespa_response([]))

        req = RetrievalRequest(
            chat_id=CHAT_A,
            query="q",
            source_types=["chunk", "section_summary"],
            rerank_mode="none",
        )
        await service.search(req)

        requests = httpx_mock.get_requests()
        for r in requests:
            body = json.loads(r.content)
            yql = body.get("yql", "")
            assert "chunk" in yql
            assert "section_summary" in yql
            assert "source_type in" in yql

    async def test_invalid_source_type_raises_before_network(
        self,
        service: RetrievalService,
        httpx_mock: HTTPXMock,
    ) -> None:
        """InvalidRetrievalFilter is raised before any Vespa request."""
        req = RetrievalRequest(
            chat_id=CHAT_A,
            query="q",
            source_types=["invalid_type_xyz"],
            rerank_mode="none",
        )
        with pytest.raises(InvalidRetrievalFilter):
            await service.search(req)

        # No HTTP requests should have been made
        assert len(httpx_mock.get_requests()) == 0


# ===========================================================================
# 6. Embedding dimension mismatch
# ===========================================================================


class TestDimensionMismatch:
    async def test_dimension_mismatch_raises(
        self,
        httpx_mock: HTTPXMock,
    ) -> None:
        # Service expects dim=4 but provider returns dim=8
        wrong_provider = MockEmbeddingProvider(dimension=8)
        svc = RetrievalService(
            endpoint=VESPA_URL,
            embedding_provider=wrong_provider,
            embedding_dim=4,  # mismatch
        )
        req = RetrievalRequest(chat_id=CHAT_A, query="q", rerank_mode="none")
        with pytest.raises(VespaDimensionMismatch):
            await svc.search(req)

        # No HTTP requests should have been made
        assert len(httpx_mock.get_requests()) == 0

    async def test_matching_dimension_no_error(
        self,
        httpx_mock: HTTPXMock,
    ) -> None:
        provider = MockEmbeddingProvider(dimension=4)
        svc = RetrievalService(
            endpoint=VESPA_URL,
            embedding_provider=provider,
            embedding_dim=4,
        )
        httpx_mock.add_response(json=_vespa_response([]))
        httpx_mock.add_response(json=_vespa_response([]))

        req = RetrievalRequest(chat_id=CHAT_A, query="q", rerank_mode="none")
        resp = await svc.search(req)
        assert isinstance(resp.hits, list)


# ===========================================================================
# 7. Debug mode
# ===========================================================================


class TestDebugMode:
    async def test_debug_none_when_not_requested(
        self,
        service: RetrievalService,
        httpx_mock: HTTPXMock,
    ) -> None:
        httpx_mock.add_response(json=_vespa_response([]))
        httpx_mock.add_response(json=_vespa_response([]))

        req = RetrievalRequest(chat_id=CHAT_A, query="q", rerank_mode="none", debug=False)
        resp = await service.search(req)
        assert resp.debug is None

    async def test_debug_populated_when_requested(
        self,
        service: RetrievalService,
        httpx_mock: HTTPXMock,
    ) -> None:
        bm25_hits = [_make_hit("d1"), _make_hit("d2")]
        ann_hits = [_make_hit("d2"), _make_hit("d3")]

        httpx_mock.add_response(json=_vespa_response(bm25_hits))
        httpx_mock.add_response(json=_vespa_response(ann_hits))

        req = RetrievalRequest(chat_id=CHAT_A, query="q", rerank_mode="none", debug=True)
        resp = await service.search(req)

        assert resp.debug is not None
        assert resp.debug.bm25_hits_count == 2
        assert resp.debug.ann_hits_count == 2
        assert resp.debug.fused_hits_count >= 2
        assert len(resp.debug.queries) >= 2
        # Verify timings are present
        assert resp.debug.timings_ms is not None

    async def test_debug_queries_contain_yql(
        self,
        service: RetrievalService,
        httpx_mock: HTTPXMock,
    ) -> None:
        httpx_mock.add_response(json=_vespa_response([]))
        httpx_mock.add_response(json=_vespa_response([]))

        req = RetrievalRequest(chat_id=CHAT_A, query="test query", rerank_mode="none", debug=True)
        resp = await service.search(req)

        assert resp.debug is not None
        for yql_str in resp.debug.queries:
            assert "select" in yql_str.lower()

    async def test_debug_native_rerank_has_three_queries(
        self,
        service: RetrievalService,
        httpx_mock: HTTPXMock,
    ) -> None:
        hits = [_make_hit(f"d{i}") for i in range(2)]

        httpx_mock.add_response(json=_vespa_response(hits))
        httpx_mock.add_response(json=_vespa_response(hits))
        httpx_mock.add_response(json=_vespa_response(hits))

        req = RetrievalRequest(
            chat_id=CHAT_A, query="q", rerank_mode="native", debug=True, rerank_top_k=5
        )
        resp = await service.search(req)

        assert resp.debug is not None
        assert len(resp.debug.queries) == 3

    async def test_debug_after_rerank_count(
        self,
        service: RetrievalService,
        httpx_mock: HTTPXMock,
    ) -> None:
        hits = [_make_hit(f"d{i}") for i in range(5)]

        httpx_mock.add_response(json=_vespa_response(hits))
        httpx_mock.add_response(json=_vespa_response(hits))

        req = RetrievalRequest(
            chat_id=CHAT_A, query="q", rerank_mode="none", final_top_k=3, debug=True
        )
        resp = await service.search(req)

        assert resp.debug is not None
        assert resp.debug.after_rerank_count == len(resp.hits)


# ===========================================================================
# 8. BM25 and ANN rank profiles
# ===========================================================================


class TestRankProfiles:
    async def test_bm25_uses_bm25_only_profile(
        self,
        service: RetrievalService,
        httpx_mock: HTTPXMock,
    ) -> None:
        httpx_mock.add_response(json=_vespa_response([]))
        httpx_mock.add_response(json=_vespa_response([]))

        req = RetrievalRequest(chat_id=CHAT_A, query="q", rerank_mode="none")
        await service.search(req)

        requests = httpx_mock.get_requests()
        bodies = [json.loads(r.content) for r in requests]
        profiles = [b.get("ranking.profile") for b in bodies]
        assert "bm25_only" in profiles

    async def test_ann_uses_semantic_only_profile(
        self,
        service: RetrievalService,
        httpx_mock: HTTPXMock,
    ) -> None:
        httpx_mock.add_response(json=_vespa_response([]))
        httpx_mock.add_response(json=_vespa_response([]))

        req = RetrievalRequest(chat_id=CHAT_A, query="q", rerank_mode="none")
        await service.search(req)

        requests = httpx_mock.get_requests()
        bodies = [json.loads(r.content) for r in requests]
        profiles = [b.get("ranking.profile") for b in bodies]
        assert "semantic_only" in profiles

    async def test_native_rerank_uses_correct_profile(
        self,
        service: RetrievalService,
        httpx_mock: HTTPXMock,
    ) -> None:
        hits = [_make_hit("d1")]
        httpx_mock.add_response(json=_vespa_response(hits))
        httpx_mock.add_response(json=_vespa_response(hits))
        httpx_mock.add_response(json=_vespa_response(hits))

        req = RetrievalRequest(chat_id=CHAT_A, query="q", rerank_mode="native", rerank_top_k=3)
        await service.search(req)

        requests = httpx_mock.get_requests()
        assert len(requests) == 3
        bodies = [json.loads(r.content) for r in requests]
        profiles = [b.get("ranking.profile") for b in bodies]
        assert "hybrid_with_native_rerank" in profiles


# ===========================================================================
# 9. SearchHit fields are correctly mapped
# ===========================================================================


class TestSearchHitMapping:
    async def test_hit_fields_mapped_correctly(
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
