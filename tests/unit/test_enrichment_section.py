"""Unit tests for Phase 5.1 — Section-level enrichment.

Reduced to ≤10 tests covering: skipping non-section nodes, count correctness,
determinism, structured fields, chat_id invariant, persist idempotency,
retry-on-garbage, and to_summary_rows kinds.
"""

from __future__ import annotations

from uuid import NAMESPACE_OID, UUID, uuid5

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.enrichment.models import (
    ClaimItem,
    DefinitionItem,
    LimitationItem,
    MethodItem,
    PerformanceFactItem,
    SectionEnrichment,
)
from app.enrichment.section import enrich_sections
from app.parsing.models import (
    BBox,
    BlockType,
    DocumentNodeOut,
    HierarchyResult,
    NodeType,
    ParsedBlock,
)
from app.providers.base import ChatMessage, ChatProvider  # noqa: F401
from app.providers.mock import MockChatProvider
from app.services.enrichment_service import persist_section_summaries

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CHAT_ID = UUID("cccccccc-cccc-cccc-cccc-cccccccccccc")
_DOC_ID = UUID("dddddddd-dddd-dddd-dddd-dddddddddddd")

SQLITE_URL = "sqlite+aiosqlite:///:memory:"

_mock_provider = MockChatProvider(model="test-model")


# ---------------------------------------------------------------------------
# SQLite helper
# ---------------------------------------------------------------------------


def _sqlite_summaries_ddl() -> str:
    return """
    CREATE TABLE IF NOT EXISTS summaries (
        id TEXT PRIMARY KEY,
        chat_id TEXT NOT NULL,
        document_id TEXT NOT NULL,
        source_node_id TEXT,
        kind TEXT NOT NULL,
        content TEXT NOT NULL,
        keywords TEXT NOT NULL DEFAULT '[]',
        entities TEXT NOT NULL DEFAULT '[]',
        token_count INTEGER NOT NULL DEFAULT 0,
        created_at TEXT NOT NULL DEFAULT (datetime('now'))
    )
    """


async def _make_sqlite_session() -> tuple[AsyncSession, object]:  # type: ignore[type-arg]
    from sqlalchemy import text as _sql_text

    engine = create_async_engine(SQLITE_URL, echo=False, future=True)
    async with engine.begin() as conn:
        await conn.execute(_sql_text(_sqlite_summaries_ddl()))
    session_factory = async_sessionmaker(
        bind=engine, expire_on_commit=False, class_=AsyncSession
    )
    session = session_factory()
    return session, engine


# ---------------------------------------------------------------------------
# Synthetic hierarchy builder
# ---------------------------------------------------------------------------


def _make_bbox() -> BBox:
    return BBox(x0=0.0, y0=0.0, x1=100.0, y1=20.0, page_width=612.0, page_height=792.0)


def _make_block(
    doc_id: UUID,
    chat_id: UUID,
    block_type: BlockType,
    text: str,
    *,
    page: int = 1,
    order: int = 0,
) -> ParsedBlock:
    block_id = uuid5(NAMESPACE_OID, f"{doc_id}:block:{page}:{order}:{block_type}")
    return ParsedBlock(
        block_id=block_id,
        chat_id=chat_id,
        document_id=doc_id,
        page_number=page,
        block_type=block_type,
        text=text,
        bbox=_make_bbox(),
        reading_order=order,
    )


def _make_node(
    doc_id: UUID,
    chat_id: UUID,
    node_type: NodeType,
    title: str | None,
    block_ids: list[UUID],
    *,
    order_index: int = 0,
    parent_id: UUID | None = None,
    level: int = 1,
) -> DocumentNodeOut:
    node_id = uuid5(NAMESPACE_OID, f"{doc_id}:node:{order_index}")
    return DocumentNodeOut(
        id=node_id,
        chat_id=chat_id,
        document_id=doc_id,
        parent_id=parent_id,
        node_type=node_type,
        title=title,
        content="",
        page_start=1,
        page_end=1,
        order_index=order_index,
        level=level,
        bbox=None,
        source_block_ids=block_ids,
        metadata_={"source": "heuristic"},
    )


def _build_synthetic() -> tuple[list[ParsedBlock], HierarchyResult]:
    """Build a minimal synthetic hierarchy for deterministic tests."""
    blocks = [
        _make_block(_DOC_ID, _CHAT_ID, BlockType.paragraph,
                    "We propose a novel retrieval method.", order=0),
        _make_block(_DOC_ID, _CHAT_ID, BlockType.paragraph,
                    "However, the approach has limitations in low-resource settings.", order=1),
        _make_block(_DOC_ID, _CHAT_ID, BlockType.paragraph,
                    "Our results show 92% accuracy on the TREC benchmark.", order=2),
        _make_block(_DOC_ID, _CHAT_ID, BlockType.paragraph,
                    "We demonstrate that attention is a useful mechanism.", order=3),
        _make_block(_DOC_ID, _CHAT_ID, BlockType.paragraph,
                    "Authors: Alice, Bob, Carol.", order=4),
    ]
    block_ids = [b.block_id for b in blocks]
    doc_root_id = uuid5(NAMESPACE_OID, f"{_DOC_ID}:node:0")

    nodes: list[DocumentNodeOut] = [
        DocumentNodeOut(
            id=doc_root_id,
            chat_id=_CHAT_ID,
            document_id=_DOC_ID,
            parent_id=None,
            node_type=NodeType.document,
            title="Test Paper",
            content="",
            page_start=1,
            page_end=2,
            order_index=0,
            level=0,
            bbox=None,
            source_block_ids=[],
            metadata_={"source": "heuristic"},
        ),
        _make_node(
            _DOC_ID, _CHAT_ID, NodeType.abstract, "Abstract",
            [block_ids[3]], order_index=1, parent_id=doc_root_id,
        ),
        _make_node(
            _DOC_ID, _CHAT_ID, NodeType.authors, None,
            [block_ids[4]], order_index=2, parent_id=doc_root_id,
        ),
        _make_node(
            _DOC_ID, _CHAT_ID, NodeType.section, "Introduction",
            [block_ids[0], block_ids[1]], order_index=3, parent_id=doc_root_id,
        ),
        _make_node(
            _DOC_ID, _CHAT_ID, NodeType.subsection, "Results",
            [block_ids[2]], order_index=4,
            parent_id=uuid5(NAMESPACE_OID, f"{_DOC_ID}:node:3"),
            level=2,
        ),
        _make_node(
            _DOC_ID, _CHAT_ID, NodeType.appendix, "Appendix A",
            [], order_index=5, parent_id=doc_root_id,
        ),
        _make_node(
            _DOC_ID, _CHAT_ID, NodeType.reference, None,
            [], order_index=6, parent_id=doc_root_id,
        ),
    ]

    hierarchy = HierarchyResult(
        document_id=_DOC_ID,
        chat_id=_CHAT_ID,
        nodes=nodes,
        references_start_index=6,
        appendix_start_index=5,
        heuristics_applied=["synthetic"],
    )
    return blocks, hierarchy


# ---------------------------------------------------------------------------
# Test 1 — skipping non-enrichable node types
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_enrich_sections_skips_non_section_nodes() -> None:
    """abstract / authors / document / reference nodes must NOT produce enrichments."""
    blocks, hierarchy = _build_synthetic()
    enrichments = await enrich_sections(hierarchy, blocks, chat_provider=_mock_provider)
    enriched_ids = {e.node_id for e in enrichments}
    skip_types = {NodeType.abstract, NodeType.authors, NodeType.document, NodeType.reference}
    for node in hierarchy.nodes:
        if node.node_type in skip_types:
            assert node.id not in enriched_ids, (
                f"Node {node.id} of type {node.node_type} should be skipped"
            )


# ---------------------------------------------------------------------------
# Test 2 — every section / subsection / appendix produces one enrichment,
#           chat_id invariant, and structured fields are lists
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_enrich_sections_count_chat_id_and_lists() -> None:
    """Count matches enrichable nodes; chat_id == hierarchy.chat_id; lists not None."""
    blocks, hierarchy = _build_synthetic()
    enrichments = await enrich_sections(hierarchy, blocks, chat_provider=_mock_provider)

    enrichable = [
        n for n in hierarchy.nodes
        if n.node_type in {NodeType.section, NodeType.subsection, NodeType.appendix}
    ]
    assert len(enrichments) == len(enrichable)

    for e in enrichments:
        assert e.chat_id == hierarchy.chat_id
        assert isinstance(e.claims, list)
        assert isinstance(e.definitions, list)
        assert isinstance(e.methods, list)
        assert isinstance(e.limitations, list)
        assert isinstance(e.performance_facts, list)


# ---------------------------------------------------------------------------
# Test 3 — determinism
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_enrich_sections_deterministic() -> None:
    """Two identical calls produce identical keywords, entities, and summaries."""
    blocks, hierarchy = _build_synthetic()
    run1 = await enrich_sections(hierarchy, blocks, chat_provider=_mock_provider)
    run2 = await enrich_sections(hierarchy, blocks, chat_provider=_mock_provider)
    for e1, e2 in zip(run1, run2, strict=True):
        assert e1.keywords == e2.keywords
        assert e1.entities == e2.entities
        assert e1.detailed_summary == e2.detailed_summary
        assert e1.compact_summary == e2.compact_summary


# ---------------------------------------------------------------------------
# Test 4 — persist_section_summaries idempotency
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_persist_section_summaries_idempotent() -> None:
    """Two writes produce identical row counts (idempotent)."""
    from app.models.orm import Summary

    blocks, hierarchy = _build_synthetic()
    enrichments = await enrich_sections(hierarchy, blocks, chat_provider=_mock_provider)

    session, engine = await _make_sqlite_session()
    try:
        await persist_section_summaries(session, enrichments)
        count_result = await session.execute(select(func.count()).select_from(Summary))
        count_after_first: int = count_result.scalar_one()

        await persist_section_summaries(session, enrichments)
        count_result2 = await session.execute(select(func.count()).select_from(Summary))
        count_after_second: int = count_result2.scalar_one()

        assert count_after_first == count_after_second
        assert count_after_first == len(enrichments) * 2
    finally:
        await session.close()
        await engine.dispose()  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Test 5 — retry on parse failure (garbage-first provider)
# ---------------------------------------------------------------------------


class _GarbageOnFirstChatProvider(ChatProvider):
    def __init__(self) -> None:
        self._call_count = 0
        self._fixture_provider: ChatProvider | None = None

    @property
    def name(self) -> str:
        return "garbage-first"

    @property
    def model(self) -> str:
        return "mock-chat"

    @property
    def context_window(self) -> int:
        return 8192

    async def complete(
        self,
        messages: list[ChatMessage],  # type: ignore[name-defined]
        *,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        stop: list[str] | None = None,
    ) -> object:
        from app.providers.base import ChatCompletion, Usage
        from app.providers.mock import FixtureChatProvider

        self._call_count += 1
        if self._call_count == 1:
            return ChatCompletion(
                content="NOT VALID JSON {{{",
                usage=Usage(prompt_tokens=5, completion_tokens=5),
                model="mock-chat",
            )
        if self._fixture_provider is None:
            self._fixture_provider = FixtureChatProvider(model="mock-chat")
        return await self._fixture_provider.complete(
            messages, temperature=temperature, max_tokens=max_tokens
        )

    def stream(self, messages: object, **kwargs: object) -> object:  # type: ignore[override]
        raise NotImplementedError

    async def test_connection(self) -> object:
        from app.providers.base import ProviderTestResult
        return ProviderTestResult(ok=True, model="mock-chat", latency_ms=0)


@pytest.mark.asyncio
async def test_enrich_section_retry_on_parse_failure() -> None:
    """First call returns garbage; second call returns valid JSON → success."""
    from app.enrichment.section import enrich_section

    blocks, hierarchy = _build_synthetic()
    section_node = next(
        n for n in hierarchy.nodes
        if n.node_type in {NodeType.section, NodeType.subsection}
    )
    block_map = {b.block_id: b for b in blocks}

    provider = _GarbageOnFirstChatProvider()
    result = await enrich_section(
        section_node,
        child_paragraphs=[],
        parsed_blocks_index=block_map,
        chat_provider=provider,
        chat_id=_CHAT_ID,
        document_id=_DOC_ID,
    )
    assert result is not None
    assert provider._call_count == 2


# ---------------------------------------------------------------------------
# Test 6 — always-garbage provider raises EnrichmentParseError
# ---------------------------------------------------------------------------


class _AlwaysGarbageChatProvider(ChatProvider):
    @property
    def name(self) -> str:
        return "always-garbage"

    @property
    def model(self) -> str:
        return "mock-chat"

    @property
    def context_window(self) -> int:
        return 8192

    async def complete(
        self,
        messages: list[ChatMessage],  # type: ignore[name-defined]
        *,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        stop: list[str] | None = None,
    ) -> object:
        from app.providers.base import ChatCompletion, Usage
        return ChatCompletion(
            content="{ this is not valid json @@@",
            usage=Usage(prompt_tokens=5, completion_tokens=5),
            model="mock-chat",
        )

    def stream(self, messages: object, **kwargs: object) -> object:  # type: ignore[override]
        raise NotImplementedError

    async def test_connection(self) -> object:
        from app.providers.base import ProviderTestResult
        return ProviderTestResult(ok=True, model="mock-chat", latency_ms=0)


@pytest.mark.asyncio
async def test_enrich_section_raises_after_all_retries() -> None:
    """Always-garbage provider → EnrichmentParseError after retries exhausted."""
    from app.enrichment.section import EnrichmentParseError, enrich_section

    blocks, hierarchy = _build_synthetic()
    section_node = next(
        n for n in hierarchy.nodes
        if n.node_type in {NodeType.section, NodeType.subsection}
    )
    block_map = {b.block_id: b for b in blocks}

    with pytest.raises(EnrichmentParseError):
        await enrich_section(
            section_node,
            child_paragraphs=[],
            parsed_blocks_index=block_map,
            chat_provider=_AlwaysGarbageChatProvider(),
            chat_id=_CHAT_ID,
            document_id=_DOC_ID,
        )


# ---------------------------------------------------------------------------
# Test 7 — to_summary_rows produces correct unique kinds
# ---------------------------------------------------------------------------


def test_to_summary_rows_produces_unique_kinds() -> None:
    """to_summary_rows returns >= 2 rows with unique kinds including mandatory ones."""
    from app.enrichment._orm_bridge import to_summary_rows

    blocks, hierarchy = _build_synthetic()
    section_node = next(
        n for n in hierarchy.nodes if n.node_type == NodeType.section
    )
    enrichment = SectionEnrichment(
        node_id=section_node.id,
        chat_id=_CHAT_ID,
        document_id=_DOC_ID,
        node_type="section",
        title="Test Section",
        page_start=1,
        page_end=2,
        source_block_ids=list(section_node.source_block_ids),
        detailed_summary="This is a detailed summary of the test section.",
        compact_summary="Test section compact.",
        keywords=["retrieval", "attention"],
        technical_keywords=["BERT"],
        entities=["BERT", "TREC"],
        definitions=[DefinitionItem(term="RAG", definition="Retrieval Augmented Generation")],
        claims=[ClaimItem(text="We show improvement over baseline.")],
        methods=[MethodItem(name="HybridRAG", description="Combines BM25 and dense.")],
        limitations=[LimitationItem(text="However, high GPU memory is needed.")],
        performance_facts=[PerformanceFactItem(metric="F1", value="0.812", context=None)],
        related_figure_ids=[],
        related_table_ids=[],
        token_count=50,
        token_count_estimate=50,
        model_used="mock-chat",
    )

    rows = to_summary_rows(enrichment)
    kinds = [r.kind for r in rows]
    assert len(kinds) == len(set(kinds)), f"Duplicate kinds: {kinds}"
    assert "section_detailed" in kinds
    assert "section_compact" in kinds
    for row in rows:
        assert row.chat_id == _CHAT_ID
        assert row.document_id == _DOC_ID
        assert row.source_node_id == section_node.id


# ---------------------------------------------------------------------------
# Test 8 — FixtureChatProvider happy-path: traceable + model_used
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_enrich_document_sections_with_fixture_provider() -> None:
    """FixtureChatProvider: each enrichment has traceable=True and correct model_used."""
    from app.enrichment.section import enrich_document_sections
    from app.providers.mock import FixtureChatProvider

    fixture_provider = FixtureChatProvider(model="mock-chat")
    blocks, hierarchy = _build_synthetic()
    enrichments = await enrich_document_sections(
        hierarchy, blocks, chat_provider=fixture_provider
    )
    assert len(enrichments) > 0
    for e in enrichments:
        assert e.traceable is True
        assert e.model_used == "mock-chat"
        assert e.token_count_estimate >= 0
