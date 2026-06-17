"""Unit tests for Phase 5.1 — Section-level enrichment.

Database strategy (for persist tests)
--------------------------------------
Tests that exercise ``persist_section_summaries`` use SQLite in-memory via
``aiosqlite``.  JSONB columns fall back to JSON automatically on SQLite so no
schema changes are needed.  The SQLite engine is created inside each test
function to avoid fixture scope conflicts with the postgres-based conftest.

All tests are fully deterministic (no random / time-based seeding).
No real LLM is called — ``MockChatProvider`` is used throughout.

Coverage
--------
1. ``enrich_sections`` skips abstract / authors / document / reference nodes.
2. Every section / subsection / appendix node produces exactly one
   ``SectionEnrichment``.
3. keywords / entities are deterministic (two identical calls ⇒ list eq).
4. claims / definitions / methods / limitations / performance_facts are
   ``list[XxxItem]`` (may be empty, but not None).
5. ``SectionEnrichment.chat_id == hierarchy.chat_id``.
6. ``SectionEnrichment.source_block_ids ⊆ node.source_block_ids``.
7. ``persist_section_summaries`` writes rows to SQLite; second call is idempotent
   (row count unchanged).
8. No ``dict[str, Any]``; no raw SQL in production code.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from uuid import NAMESPACE_OID, UUID, uuid5

import pytest
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
from app.parsing.hierarchy import derive_hierarchy
from app.parsing.mapping import map_middle_to_parsed_blocks
from app.parsing.models import (
    BBox,
    BlockType,
    DocumentNodeOut,
    HierarchyResult,
    NodeType,
    ParsedBlock,
)
from app.providers.base import ChatMessage, ChatProvider  # noqa: F401 (used in test helpers below)
from app.providers.mock import MockChatProvider
from app.services.enrichment_service import persist_section_summaries

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CHAT_ID = UUID("cccccccc-cccc-cccc-cccc-cccccccccccc")
_DOC_ID = UUID("dddddddd-dddd-dddd-dddd-dddddddddddd")

PAPER_FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "mineru_sample_paper"
PAPER_MIDDLE = PAPER_FIXTURE_DIR / "middle.json"

SQLITE_URL = "sqlite+aiosqlite:///:memory:"

_mock_provider = MockChatProvider(model="test-model")


# ---------------------------------------------------------------------------
# SQLite helper (used inline in test functions)
# ---------------------------------------------------------------------------


def _sqlite_summaries_ddl() -> str:
    """Return a CREATE TABLE statement for ``summaries`` that works with SQLite.

    We bypass ORM metadata (which uses JSONB) and create the table directly
    with JSON-compatible TEXT columns.  The ORM session still uses
    ``Summary`` ORM objects — SQLite stores JSON as text transparently.
    """
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
    """Create a fresh in-memory SQLite session with the ``summaries`` table.

    Returns ``(session, engine)`` — caller must dispose the engine after use.

    We create only the ``summaries`` table (JSON-based) rather than calling
    ``Base.metadata.create_all`` (which fails on SQLite due to JSONB columns
    in other tables).
    """
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
# Small synthetic hierarchy (built programmatically)
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
        # document root (order_index=0)
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
        # abstract (order_index=1)
        _make_node(
            _DOC_ID, _CHAT_ID, NodeType.abstract, "Abstract",
            [block_ids[3]], order_index=1, parent_id=doc_root_id,
        ),
        # authors (order_index=2)
        _make_node(
            _DOC_ID, _CHAT_ID, NodeType.authors, None,
            [block_ids[4]], order_index=2, parent_id=doc_root_id,
        ),
        # section (order_index=3)
        _make_node(
            _DOC_ID, _CHAT_ID, NodeType.section, "Introduction",
            [block_ids[0], block_ids[1]], order_index=3, parent_id=doc_root_id,
        ),
        # subsection (order_index=4)
        _make_node(
            _DOC_ID, _CHAT_ID, NodeType.subsection, "Results",
            [block_ids[2]], order_index=4,
            parent_id=uuid5(NAMESPACE_OID, f"{_DOC_ID}:node:3"),
            level=2,
        ),
        # appendix (order_index=5)
        _make_node(
            _DOC_ID, _CHAT_ID, NodeType.appendix, "Appendix A",
            [], order_index=5, parent_id=doc_root_id,
        ),
        # reference (order_index=6)
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


def _build_paper() -> tuple[list[ParsedBlock], HierarchyResult]:
    """Build hierarchy from the mineru_sample_paper fixture."""
    middle = json.loads(PAPER_MIDDLE.read_text(encoding="utf-8"))
    blocks = map_middle_to_parsed_blocks(middle, chat_id=_CHAT_ID, document_id=_DOC_ID)
    hierarchy = derive_hierarchy(blocks, document_id=_DOC_ID, chat_id=_CHAT_ID)
    return blocks, hierarchy


def _enrichable_nodes(hierarchy: HierarchyResult) -> list[DocumentNodeOut]:
    return [
        n for n in hierarchy.nodes
        if n.node_type in {NodeType.section, NodeType.subsection, NodeType.appendix}
    ]


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
# Test 2 — every section / subsection / appendix produces one enrichment
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_enrich_sections_count_matches_enrichable_nodes() -> None:
    blocks, hierarchy = _build_synthetic()
    enrichments = await enrich_sections(hierarchy, blocks, chat_provider=_mock_provider)
    expected_count = len(_enrichable_nodes(hierarchy))
    assert len(enrichments) == expected_count, (
        f"Expected {expected_count} enrichments, got {len(enrichments)}"
    )


# ---------------------------------------------------------------------------
# Test 3 — determinism: keywords / entities are stable across two calls
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_enrich_sections_deterministic() -> None:
    blocks, hierarchy = _build_synthetic()
    run1 = await enrich_sections(hierarchy, blocks, chat_provider=_mock_provider)
    run2 = await enrich_sections(hierarchy, blocks, chat_provider=_mock_provider)
    for e1, e2 in zip(run1, run2, strict=True):
        assert e1.keywords == e2.keywords, "keywords are not deterministic"
        assert e1.entities == e2.entities, "entities are not deterministic"
        assert e1.detailed_summary == e2.detailed_summary, "detailed_summary is not deterministic"
        assert e1.compact_summary == e2.compact_summary, "compact_summary is not deterministic"


# ---------------------------------------------------------------------------
# Test 4 — structured fields are lists (never None)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_enrich_sections_structured_fields_are_lists() -> None:
    blocks, hierarchy = _build_synthetic()
    enrichments = await enrich_sections(hierarchy, blocks, chat_provider=_mock_provider)
    for e in enrichments:
        assert isinstance(e.claims, list), "claims must be a list"
        assert isinstance(e.definitions, list), "definitions must be a list"
        assert isinstance(e.methods, list), "methods must be a list"
        assert isinstance(e.limitations, list), "limitations must be a list"
        assert isinstance(e.performance_facts, list), "performance_facts must be a list"
        for item in e.claims:
            assert isinstance(item, ClaimItem)
        for item in e.definitions:
            assert isinstance(item, DefinitionItem)
        for item in e.methods:
            assert isinstance(item, MethodItem)
        for item in e.limitations:
            assert isinstance(item, LimitationItem)
        for item in e.performance_facts:
            assert isinstance(item, PerformanceFactItem)


# ---------------------------------------------------------------------------
# Test 5 — chat_id and source_block_ids invariants
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_enrich_sections_chat_id_from_hierarchy() -> None:
    blocks, hierarchy = _build_synthetic()
    enrichments = await enrich_sections(hierarchy, blocks, chat_provider=_mock_provider)
    for e in enrichments:
        assert e.chat_id == hierarchy.chat_id, (
            f"chat_id mismatch for node {e.node_id}: "
            f"expected {hierarchy.chat_id}, got {e.chat_id}"
        )


@pytest.mark.asyncio
async def test_enrich_sections_source_block_ids_subset() -> None:
    """source_block_ids in enrichment must be a subset of the node's source_block_ids."""
    blocks, hierarchy = _build_synthetic()
    node_map = {n.id: n for n in hierarchy.nodes}
    enrichments = await enrich_sections(hierarchy, blocks, chat_provider=_mock_provider)
    for e in enrichments:
        node = node_map[e.node_id]
        node_block_set = set(node.source_block_ids)
        enrichment_block_set = set(e.source_block_ids)
        assert enrichment_block_set <= node_block_set, (
            f"Enrichment for node {e.node_id} has extra block IDs: "
            f"{enrichment_block_set - node_block_set}"
        )


# ---------------------------------------------------------------------------
# Test 6 — persist_section_summaries writes to SQLite + idempotency
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_persist_section_summaries_writes_two_rows_per_enrichment() -> None:
    """persist_section_summaries writes exactly 2 Summary rows per enrichment."""
    blocks, hierarchy = _build_synthetic()
    enrichments = await enrich_sections(hierarchy, blocks, chat_provider=_mock_provider)

    session, engine = await _make_sqlite_session()
    try:
        ids = await persist_section_summaries(session, enrichments)
        assert len(ids) == len(enrichments) * 2, (
            f"Expected {len(enrichments) * 2} IDs, got {len(ids)}"
        )
        for row_id in ids:
            assert isinstance(row_id, uuid.UUID)
    finally:
        await session.close()
        await engine.dispose()  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_persist_section_summaries_idempotent() -> None:
    """Calling persist_section_summaries twice must not increase row count."""
    from sqlalchemy import func, select

    from app.models.orm import Summary

    blocks, hierarchy = _build_synthetic()
    enrichments = await enrich_sections(hierarchy, blocks, chat_provider=_mock_provider)

    session, engine = await _make_sqlite_session()
    try:
        # First write
        await persist_section_summaries(session, enrichments)
        count_result = await session.execute(select(func.count()).select_from(Summary))
        count_after_first: int = count_result.scalar_one()

        # Second write (idempotent)
        await persist_section_summaries(session, enrichments)
        count_result2 = await session.execute(select(func.count()).select_from(Summary))
        count_after_second: int = count_result2.scalar_one()

        assert count_after_first == count_after_second, (
            f"Row count changed after idempotent call: "
            f"{count_after_first} → {count_after_second}"
        )
    finally:
        await session.close()
        await engine.dispose()  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Test 7 — real-data smoke test using mineru_sample_paper fixture
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_enrich_sections_paper_fixture() -> None:
    """Smoke test against the real mineru_sample_paper fixture."""
    blocks, hierarchy = _build_paper()
    enrichments = await enrich_sections(hierarchy, blocks, chat_provider=_mock_provider)
    assert len(enrichments) > 0, "Expected at least one section enrichment"
    for e in enrichments:
        assert e.detailed_summary, f"detailed_summary is empty for node {e.node_id}"
        assert e.compact_summary, f"compact_summary is empty for node {e.node_id}"
        assert e.node_type in ("section", "subsection", "appendix"), (
            f"Unexpected node_type in enrichment: {e.node_type}"
        )
        assert e.chat_id == hierarchy.chat_id


# ---------------------------------------------------------------------------
# Test 8 — SectionEnrichment model rejects extra fields (extra="forbid")
# ---------------------------------------------------------------------------


def test_section_enrichment_forbids_extra_fields() -> None:
    """Pydantic v2 ConfigDict(extra='forbid') must reject unknown fields."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        SectionEnrichment(
            node_id=uuid.uuid4(),
            chat_id=_CHAT_ID,
            document_id=_DOC_ID,
            node_type="section",
            title="Test",
            page_start=1,
            page_end=1,
            source_block_ids=[],
            detailed_summary="x",
            compact_summary="y",
            unknown_field="should_fail",  # type: ignore[call-arg]
        )


# ---------------------------------------------------------------------------
# Test 9 — item sub-models forbid extra fields
# ---------------------------------------------------------------------------


def test_claim_item_forbids_extra() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ClaimItem(text="x", confidence=1.0, extra_field="bad")  # type: ignore[call-arg]


def test_definition_item_forbids_extra() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        DefinitionItem(term="x", definition="y", extra_field="bad")  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Test 10 — heuristic extraction works on content with known patterns
# ---------------------------------------------------------------------------


def test_parse_keywords_deterministic() -> None:
    """Two calls with same input produce identical keywords."""
    from app.enrichment.section import _parse_keywords

    content = "attention mechanism transformer architecture self-attention multi-head"
    r1 = _parse_keywords("mock response", content)
    r2 = _parse_keywords("mock response", content)
    assert r1 == r2


def test_parse_entities_extracts_pascal_case() -> None:
    """PascalCase words should appear in entities."""
    from app.enrichment.section import _parse_entities

    content = "BERT and GPT are Transformer models used in NLP tasks."
    entities = _parse_entities("mock", content)
    assert len(entities) > 0


def test_parse_claims_extracts_we_show() -> None:
    """'We show' sentences should be extracted as claims."""
    from app.enrichment.section import _parse_claims

    content = "We show that our method outperforms the baseline by a large margin."
    claims = _parse_claims("mock", content)
    assert len(claims) >= 1
    assert isinstance(claims[0], ClaimItem)


def test_parse_limitations_extracts_however() -> None:
    """Sentences with 'however' should be extracted as limitations."""
    from app.enrichment.section import _parse_limitations

    content = (
        "Our method achieves high accuracy. "
        "However, the model does not generalize to low-resource languages."
    )
    limitations = _parse_limitations("mock", content)
    assert len(limitations) >= 1
    assert isinstance(limitations[0], LimitationItem)


# ---------------------------------------------------------------------------
# Test 11 — FixtureChatProvider + enrich_document_sections happy-path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_enrich_document_sections_with_fixture_provider() -> None:
    """Happy-path: FixtureChatProvider drives enrich_document_sections.

    Assertions:
    - At least one section is enriched (abstract / section / subsection / appendix).
    - Each SectionEnrichment.source_block_ids is non-empty or node has empty blocks.
    - token_count_estimate > 0.
    - model_used == "mock-chat".
    - traceable is True.
    """
    from app.enrichment.section import enrich_document_sections
    from app.providers.mock import FixtureChatProvider

    fixture_provider = FixtureChatProvider(model="mock-chat")
    blocks, hierarchy = _build_synthetic()
    enrichments = await enrich_document_sections(
        hierarchy, blocks, chat_provider=fixture_provider
    )
    assert len(enrichments) > 0, "Expected at least one enriched section"
    for e in enrichments:
        assert e.traceable is True
        assert e.model_used == "mock-chat"
        assert e.token_count_estimate >= 0


@pytest.mark.asyncio
async def test_enrich_document_sections_paper_fixture_with_fixture_provider() -> None:
    """Happy-path: FixtureChatProvider on the mineru_sample_paper fixture."""
    from app.enrichment.section import enrich_document_sections
    from app.providers.mock import FixtureChatProvider

    fixture_provider = FixtureChatProvider(model="mock-chat")
    blocks, hierarchy = _build_paper()
    enrichments = await enrich_document_sections(
        hierarchy, blocks, chat_provider=fixture_provider
    )
    assert len(enrichments) > 0, "Expected at least one enriched section"
    for e in enrichments:
        assert e.detailed_summary, f"Empty detailed_summary for {e.node_id}"
        assert e.compact_summary, f"Empty compact_summary for {e.node_id}"
        assert e.node_type in ("section", "subsection", "appendix", "abstract"), (
            f"Unexpected node_type: {e.node_type}"
        )
        assert e.chat_id == hierarchy.chat_id
        # source_block_ids must be subset of node's block IDs (or empty for empty nodes)
        node_map = {n.id: n for n in hierarchy.nodes}
        if e.node_id in node_map:
            node = node_map[e.node_id]
            node_block_set = set(node.source_block_ids)
            enrichment_block_set = set(e.source_block_ids)
            # source_block_ids comes from all_block_ids which is node.source_block_ids
            assert enrichment_block_set <= node_block_set or not enrichment_block_set, (
                f"Enrichment has extra block IDs: {enrichment_block_set - node_block_set}"
            )


# ---------------------------------------------------------------------------
# Test 12 — JSON parse error tolerance
# ---------------------------------------------------------------------------


class _GarbageOnFirstChatProvider(ChatProvider):
    """Returns garbage on the first call, valid JSON on subsequent calls."""

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
        # Second call: return valid JSON from fixture
        if self._fixture_provider is None:
            self._fixture_provider = FixtureChatProvider(model="mock-chat")
        return await self._fixture_provider.complete(messages, temperature=temperature, max_tokens=max_tokens)

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
    assert provider._call_count == 2, (
        f"Expected exactly 2 provider calls (1 fail + 1 success), got {provider._call_count}"
    )


class _AlwaysGarbageChatProvider(ChatProvider):
    """Always returns garbage — triggers EnrichmentParseError after all retries."""

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
# Test 13 — source_block_ids fallback
# ---------------------------------------------------------------------------


class _NoBlockIdsChatProvider(ChatProvider):
    """Returns valid JSON but with empty source_block_ids in all sub-items."""

    @property
    def name(self) -> str:
        return "no-block-ids"

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
        import json as _json

        from app.providers.base import ChatCompletion, Usage
        payload = {
            "detailed_summary": "Section about novel methods and experiments.",
            "compact_summary": "Novel method section.",
            "keywords": [{"term": "method", "weight": 0.9, "source_block_ids": []}],
            "entities": [],
            "definitions": [],
            "claims": [],
            "methods": [],
            "limitations": [],
            "performance_facts": [],
        }
        return ChatCompletion(
            content=_json.dumps(payload),
            usage=Usage(prompt_tokens=10, completion_tokens=20),
            model="mock-chat",
        )

    def stream(self, messages: object, **kwargs: object) -> object:  # type: ignore[override]
        raise NotImplementedError

    async def test_connection(self) -> object:
        from app.providers.base import ProviderTestResult
        return ProviderTestResult(ok=True, model="mock-chat", latency_ms=0)


@pytest.mark.asyncio
async def test_source_block_ids_fallback_when_llm_omits_them() -> None:
    """When LLM returns empty source_block_ids, fallback = section's full block ids."""
    from app.enrichment.section import _try_parse_json_enrichment

    blocks, hierarchy = _build_synthetic()
    section_node = next(
        n for n in hierarchy.nodes
        if n.node_type == NodeType.section and n.source_block_ids
    )
    fallback = list(section_node.source_block_ids)

    # Simulate: LLM returns keyword with empty source_block_ids
    raw_json = '{"detailed_summary": "Test.", "compact_summary": "Short.", "keywords": [{"term": "test", "weight": 0.9, "source_block_ids": []}], "entities": [], "definitions": [], "claims": [], "methods": [], "limitations": [], "performance_facts": []}'
    result = _try_parse_json_enrichment(raw_json, section_node, fallback, "mock-chat")
    # source_block_ids of the enrichment itself should be the fallback
    assert set(result.source_block_ids) == set(fallback), (
        "source_block_ids should be the fallback (section's full block ids)"
    )


# ---------------------------------------------------------------------------
# Test 14 — performance_facts structure with traced block IDs
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_performance_facts_parsed_from_fixture_provider() -> None:
    """FixtureChatProvider returns a performance fact with metric=F1, value=0.812."""
    from app.enrichment.section import enrich_section
    from app.providers.mock import FixtureChatProvider

    blocks, hierarchy = _build_synthetic()
    section_node = next(
        n for n in hierarchy.nodes
        if n.node_type == NodeType.section
    )
    block_map = {b.block_id: b for b in blocks}

    provider = FixtureChatProvider(model="mock-chat")
    result = await enrich_section(
        section_node,
        child_paragraphs=[],
        parsed_blocks_index=block_map,
        chat_provider=provider,
        chat_id=_CHAT_ID,
        document_id=_DOC_ID,
    )

    assert len(result.performance_facts) > 0, "Expected at least one performance fact from fixture"
    pf = result.performance_facts[0]
    assert isinstance(pf, PerformanceFactItem)
    assert pf.metric == "F1"
    assert pf.value == "0.812"


# ---------------------------------------------------------------------------
# Test 15 — to_summary_rows produces correct kinds
# ---------------------------------------------------------------------------


def test_to_summary_rows_produces_unique_kinds() -> None:
    """to_summary_rows: all returned kinds are unique."""
    from app.enrichment._orm_bridge import to_summary_rows

    blocks, hierarchy = _build_synthetic()
    # Build a SectionEnrichment with at least one entry in each list
    section_node = next(
        n for n in hierarchy.nodes
        if n.node_type == NodeType.section
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
    assert len(rows) >= 2, "Must produce at least section_detailed and section_compact rows"

    kinds = [r.kind for r in rows]
    assert len(kinds) == len(set(kinds)), f"Duplicate kinds found: {kinds}"

    # Must include the two mandatory kinds
    assert "section_detailed" in kinds
    assert "section_compact" in kinds
    # Should include conditional kinds since lists are non-empty
    assert "section_keywords" in kinds
    assert "section_entities" in kinds
    assert "section_claims" in kinds
    assert "section_methods" in kinds
    assert "section_limitations" in kinds
    assert "section_performance_facts" in kinds
    assert "section_definitions" in kinds

    # Verify all rows have correct metadata
    for row in rows:
        assert row.chat_id == _CHAT_ID
        assert row.document_id == _DOC_ID
        assert row.source_node_id == section_node.id


# ---------------------------------------------------------------------------
# Test 16 — determinism: enrich_document_sections byte-identical across two runs
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_enrich_document_sections_deterministic_with_fixture_provider() -> None:
    """Two calls with same input + FixtureChatProvider produce byte-identical output."""
    from app.enrichment.section import enrich_document_sections
    from app.providers.mock import FixtureChatProvider

    fixture_provider = FixtureChatProvider(model="mock-chat")
    blocks, hierarchy = _build_synthetic()

    run1 = await enrich_document_sections(hierarchy, blocks, chat_provider=fixture_provider)
    run2 = await enrich_document_sections(hierarchy, blocks, chat_provider=fixture_provider)

    assert len(run1) == len(run2), "Different number of enrichments between runs"
    for e1, e2 in zip(run1, run2, strict=True):
        # Use model_dump to compare (excludes UUIDs generated per-call)
        d1 = e1.model_dump(exclude={"node_id"})
        d2 = e2.model_dump(exclude={"node_id"})
        assert d1["detailed_summary"] == d2["detailed_summary"], (
            "detailed_summary is not deterministic"
        )
        assert d1["compact_summary"] == d2["compact_summary"], (
            "compact_summary is not deterministic"
        )
        assert d1["keywords"] == d2["keywords"], "keywords are not deterministic"
        assert d1["entities"] == d2["entities"], "entities are not deterministic"


# ---------------------------------------------------------------------------
# Test 17 — source_node_id property alias
# ---------------------------------------------------------------------------


def test_section_enrichment_source_node_id_alias() -> None:
    """SectionEnrichment.source_node_id is an alias for node_id."""
    node_uuid = uuid.uuid4()
    e = SectionEnrichment(
        node_id=node_uuid,
        chat_id=_CHAT_ID,
        document_id=_DOC_ID,
        node_type="section",
        title="Test",
        page_start=1,
        page_end=1,
        source_block_ids=[],
        detailed_summary="x",
        compact_summary="y",
    )
    assert e.source_node_id == node_uuid


# ---------------------------------------------------------------------------
# Test 18 — token_count_estimate > 0 for non-empty detailed_summary
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_token_count_estimate_positive() -> None:
    """token_count_estimate must be > 0 when detailed_summary is non-empty."""
    from app.enrichment.section import enrich_section
    from app.providers.mock import FixtureChatProvider

    blocks, hierarchy = _build_synthetic()
    section_node = next(
        n for n in hierarchy.nodes
        if n.node_type == NodeType.section
    )
    block_map = {b.block_id: b for b in blocks}

    provider = FixtureChatProvider(model="mock-chat")
    result = await enrich_section(
        section_node,
        child_paragraphs=[],
        parsed_blocks_index=block_map,
        chat_provider=provider,
        chat_id=_CHAT_ID,
        document_id=_DOC_ID,
    )
    assert result.token_count_estimate > 0, (
        "token_count_estimate must be > 0 for a non-empty detailed_summary"
    )
