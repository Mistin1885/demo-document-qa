"""Unit tests for Phase 7.2 agent tools — happy paths + overflow + registry.

CLAUDE.md §12.1: ≤10 items per file.
Items:
  1–7.  Happy-path test for each of the 7 tools (mock deps, no real DB/Vespa).
  8.    token_estimate is > 0 after retrieval with content.
  9.    overflow status when evidence content exceeds max_tokens.
  10.   TOOL_REGISTRY has exactly 7 entries.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.agent.state import AgentState, EvidenceItem, make_evidence_id
from app.agent.tools import TOOL_REGISTRY, ToolDeps
from app.agent.tools._models import (
    AggregateSourcesParams,
    ExpandEvidenceParams,
    FetchStructuralNodesParams,
    InspectChatParams,
    InspectDocumentParams,
    QueryStructuredFactsParams,
    SearchHybridParams,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

CHAT_ID = uuid.uuid4()
SESSION_ID = uuid.uuid4()
DOC_ID = uuid.uuid4()
NODE_ID = uuid.uuid4()


def _state(**kwargs: object) -> AgentState:
    return AgentState(
        chat_id=CHAT_ID,
        session_id=SESSION_ID,
        question="Test question?",
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Fake session factory (in-memory SQLAlchemy — async mock)
# ---------------------------------------------------------------------------


class _FakeSession:
    """Fake AsyncSession that returns empty results by default."""

    async def execute(self, stmt: object) -> MagicMock:
        result = MagicMock()
        result.scalars.return_value.all.return_value = []
        result.scalar_one_or_none.return_value = None
        result.scalar_one.return_value = 0
        return result

    async def __aenter__(self) -> _FakeSession:
        return self

    async def __aexit__(self, *args: object) -> None:
        pass


@asynccontextmanager
async def _fake_session_factory() -> AsyncGenerator[_FakeSession, None]:
    yield _FakeSession()


# ---------------------------------------------------------------------------
# Fake RetrievalService
# ---------------------------------------------------------------------------


def _fake_retrieval_service(hits: list[dict] | None = None) -> MagicMock:
    from app.retrieval.models import RetrievalResponse, SearchHit

    svc = MagicMock()
    hit_list: list[SearchHit] = []
    if hits:
        for h in hits:
            hit_list.append(
                SearchHit(
                    vespa_document_id=h.get("vid", str(uuid.uuid4())),
                    chat_id=str(CHAT_ID),
                    document_id=str(DOC_ID),
                    source_node_id=h.get("node_id", str(uuid.uuid4())),
                    source_type=h.get("source_type", "chunk"),
                    content=h.get("content", "some content"),
                    page_start=h.get("page_start", 1),
                    page_end=h.get("page_end", 2),
                    order_index=0,
                )
            )
    svc.search = AsyncMock(return_value=RetrievalResponse(hits=hit_list))
    return svc


def _fake_deps(retrieval_svc: object = None) -> ToolDeps:
    provider = MagicMock()
    return ToolDeps(
        retrieval_service=retrieval_svc or _fake_retrieval_service(),
        chat_provider=provider,
        session_factory=_fake_session_factory,
    )


# ---------------------------------------------------------------------------
# Fake manifest service
# ---------------------------------------------------------------------------


def _mock_manifest_service(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch manifest service to return a lightweight fake ChatManifest."""
    import importlib

    # importlib.import_module always returns the submodule (not the re-exported function)
    inspect_chat_mod = importlib.import_module("app.agent.tools.inspect_chat")

    from app.enrichment.models import ChatManifest

    fake_manifest = ChatManifest(
        chat_id=CHAT_ID,
        generated_at=__import__("datetime")
        .datetime.now(__import__("datetime").timezone.utc)
        .replace(tzinfo=None),
        document_count=1,
        total_token_estimate=500,
        documents=[],
        ingestion_summary={},
    )

    async def _fake_get_manifest(session: object, *, current_chat_id: uuid.UUID) -> ChatManifest:
        return fake_manifest

    monkeypatch.setattr(inspect_chat_mod, "get_chat_manifest", _fake_get_manifest)


# ---------------------------------------------------------------------------
# Test 1: inspect_chat happy path
# ---------------------------------------------------------------------------


async def test_inspect_chat_happy(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.agent.tools.inspect_chat import inspect_chat

    _mock_manifest_service(monkeypatch)
    state = _state()
    params = InspectChatParams(include_topics=True)
    result = await inspect_chat(state, params, deps=_fake_deps())
    assert result.record.status == "ok"
    assert result.chat_manifest is not None
    assert result.chat_manifest.document_count == 1
    assert result.evidence == []


# ---------------------------------------------------------------------------
# Test 2: inspect_document happy path
# ---------------------------------------------------------------------------


async def test_inspect_document_happy(monkeypatch: pytest.MonkeyPatch) -> None:
    import datetime
    import importlib

    inspect_doc_mod = importlib.import_module("app.agent.tools.inspect_document")

    from app.agent.tools.inspect_document import inspect_document
    from app.models.domain import DocumentRead

    fake_doc = DocumentRead(
        id=DOC_ID,
        chat_id=CHAT_ID,
        source_type="upload",
        original_filename="paper.pdf",
        storage_path="/data/paper.pdf",
        mime_type="application/pdf",
        page_count=10,
        status="indexed",
        checksum_sha256="abc" * 21 + "ab",
        created_at=datetime.datetime.now(datetime.UTC).replace(tzinfo=None),
        updated_at=datetime.datetime.now(datetime.UTC).replace(tzinfo=None),
    )

    async def _fake_get_doc(
        session: object, chat_id: uuid.UUID, document_id: uuid.UUID
    ) -> DocumentRead:
        return fake_doc

    monkeypatch.setattr(inspect_doc_mod, "get_document", _fake_get_doc)

    state = _state()
    params = InspectDocumentParams(document_id=DOC_ID, include_section_tree=False)
    result = await inspect_document(state, params, deps=_fake_deps())
    assert result.record.status == "ok"
    assert result.document_manifest is not None
    assert result.document_manifest.document_id == DOC_ID


# ---------------------------------------------------------------------------
# Test 3: fetch_structural_nodes happy path
# ---------------------------------------------------------------------------


async def test_fetch_structural_nodes_happy() -> None:
    from app.agent.tools.fetch_structural_nodes import fetch_structural_nodes

    state = _state()
    params = FetchStructuralNodesParams(limit=10)
    result = await fetch_structural_nodes(state, params, deps=_fake_deps())
    # No nodes in fake session → empty status
    assert result.record.status in ("ok", "empty")
    assert result.evidence == []


# ---------------------------------------------------------------------------
# Test 4: search_hybrid happy path
# ---------------------------------------------------------------------------


async def test_search_hybrid_happy() -> None:
    from app.agent.tools.search_hybrid import search_hybrid

    svc = _fake_retrieval_service(
        hits=[{"content": "test content", "node_id": str(uuid.uuid4()), "vid": str(uuid.uuid4())}]
    )
    state = _state()
    params = SearchHybridParams(query="transformer attention mechanism")
    result = await search_hybrid(state, params, deps=_fake_deps(svc))
    assert result.record.status == "ok"
    assert len(result.evidence) == 1
    assert result.evidence[0].origin_tool == "search_hybrid"
    # Verify chat_id was injected from state (not params)
    call_args = svc.search.call_args[0][0]
    assert call_args.chat_id == CHAT_ID


# ---------------------------------------------------------------------------
# Test 5: query_structured_facts happy path
# ---------------------------------------------------------------------------


async def test_query_structured_facts_happy(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib

    qsf_mod = importlib.import_module("app.agent.tools.query_structured_facts")

    from app.agent.tools.query_structured_facts import query_structured_facts

    async def _fake_query_facts(
        session: object, *, current_chat_id: uuid.UUID, filt: object
    ) -> list:
        return []

    monkeypatch.setattr(qsf_mod, "query_facts", _fake_query_facts)

    state = _state()
    params = QueryStructuredFactsParams(limit=10)
    result = await query_structured_facts(state, params, deps=_fake_deps())
    assert result.record.status in ("ok", "empty")
    assert result.facts == []


# ---------------------------------------------------------------------------
# Test 6: aggregate_sources happy path
# ---------------------------------------------------------------------------


async def test_aggregate_sources_happy() -> None:
    from app.agent.tools.aggregate_sources import aggregate_sources

    ev = EvidenceItem(
        evidence_id=make_evidence_id("search_hybrid", str(NODE_ID), str(DOC_ID)),
        source_type="chunk",
        document_id=DOC_ID,
        source_node_id=str(NODE_ID),
        page_start=1,
        page_end=3,
        content="Attention mechanism description.",
        origin_tool="search_hybrid",
        section_title="Methods",
    )
    state = _state(evidence_items=[ev])
    params = AggregateSourcesParams(strategy="per_section")
    result = await aggregate_sources(state, params, deps=_fake_deps())
    assert result.record.status == "ok"
    assert len(result.evidence) == 1
    assert result.evidence[0].source_type == "aggregated_evidence"


# ---------------------------------------------------------------------------
# Test 7: expand_evidence happy path
# ---------------------------------------------------------------------------


async def test_expand_evidence_happy() -> None:
    from app.agent.tools.expand_evidence import expand_evidence

    ev_id = make_evidence_id("search_hybrid", str(NODE_ID), str(DOC_ID))
    ev = EvidenceItem(
        evidence_id=ev_id,
        source_type="section",
        document_id=DOC_ID,
        source_node_id=str(NODE_ID),
        page_start=2,
        page_end=4,
        content="Section content.",
        origin_tool="search_hybrid",
    )
    state = _state(evidence_items=[ev])
    params = ExpandEvidenceParams(evidence_id=ev_id, neighborhood="section")
    result = await expand_evidence(state, params, deps=_fake_deps())
    # No neighbors in fake session → empty
    assert result.record.status in ("ok", "empty")


# ---------------------------------------------------------------------------
# Test 8: token_estimate is > 0 when evidence has content
# ---------------------------------------------------------------------------


async def test_token_estimate_positive() -> None:
    from app.agent.tools.search_hybrid import search_hybrid

    long_content = "word " * 100  # 100 words ≈ >25 tokens
    svc = _fake_retrieval_service(
        hits=[{"content": long_content, "node_id": str(uuid.uuid4()), "vid": str(uuid.uuid4())}]
    )
    state = _state()
    params = SearchHybridParams(query="test query")
    result = await search_hybrid(state, params, deps=_fake_deps(svc))
    assert result.record.token_estimate > 0


# ---------------------------------------------------------------------------
# Test 9: overflow status when evidence exceeds max_tokens
# ---------------------------------------------------------------------------


async def test_overflow_status_on_large_evidence() -> None:
    from app.agent.tools.search_hybrid import search_hybrid

    huge_content = "word " * 5000  # 5000 words >> 10 tokens
    svc = _fake_retrieval_service(
        hits=[{"content": huge_content, "node_id": str(uuid.uuid4()), "vid": str(uuid.uuid4())}]
    )
    state = _state()
    params = SearchHybridParams(query="test", max_tokens=10)  # tiny budget
    result = await search_hybrid(state, params, deps=_fake_deps(svc))
    assert result.record.status == "overflow"


# ---------------------------------------------------------------------------
# Test 10: TOOL_REGISTRY contains all retrieval and support tools
# ---------------------------------------------------------------------------


def test_tool_registry_size() -> None:
    assert len(TOOL_REGISTRY) == 8
    expected = {
        "inspect_chat",
        "inspect_document",
        "fetch_structural_nodes",
        "grep_document_chunks",
        "search_hybrid",
        "query_structured_facts",
        "aggregate_sources",
        "expand_evidence",
    }
    assert set(TOOL_REGISTRY.keys()) == expected
