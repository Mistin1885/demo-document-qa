"""Unit tests for Phase 5.2 — Document-level enrichment.

Reduced to ≤10 tests covering: determinism, no-abstract path, union-dedup
of methods/limitations/contributions (parametrized), document_overview length,
section-title-driven results/conclusions, persist idempotency, and chat_id.
"""

from __future__ import annotations

from uuid import NAMESPACE_OID, UUID, uuid5

import pytest
from sqlalchemy import func, select
from sqlalchemy import text as _sql_text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.enrichment.document import enrich_document
from app.enrichment.models import (
    ClaimItem,
    LimitationItem,
    MethodItem,
    PerformanceFactItem,
    SectionEnrichment,
)
from app.models.orm import Summary
from app.parsing.models import (
    BBox,
    BlockType,
    DocumentNodeOut,
    HierarchyResult,
    NodeType,
    ParsedBlock,
)
from app.providers.base import ChatCompletion, ChatMessage, ChatProvider, ProviderTestResult
from app.services.enrichment_service import persist_document_summaries

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CHAT_ID = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
_DOC_ID = UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")

SQLITE_URL = "sqlite+aiosqlite:///:memory:"


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
    engine = create_async_engine(SQLITE_URL, echo=False, future=True)
    async with engine.begin() as conn:
        await conn.execute(_sql_text(_sqlite_summaries_ddl()))
    session_factory = async_sessionmaker(bind=engine, expire_on_commit=False, class_=AsyncSession)
    session = session_factory()
    return session, engine


# ---------------------------------------------------------------------------
# Fixtures helpers
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


def _make_section_enrichment(
    node_id: UUID,
    title: str | None,
    *,
    methods: list[str] | None = None,
    limitations: list[str] | None = None,
    claims: list[str] | None = None,
    performance_facts: list[tuple[str, str]] | None = None,
    technical_keywords: list[str] | None = None,
    entities: list[str] | None = None,
    order_index: int = 1,
) -> SectionEnrichment:
    return SectionEnrichment(
        node_id=node_id,
        chat_id=_CHAT_ID,
        document_id=_DOC_ID,
        node_type="section",
        title=title,
        page_start=1,
        page_end=2,
        source_block_ids=[],
        detailed_summary=f"Detailed summary of {title or 'untitled'}.",
        compact_summary=f"Compact summary of {title or 'untitled'}.",
        keywords=[],
        technical_keywords=technical_keywords or [],
        entities=entities or [],
        definitions=[],
        claims=[ClaimItem(text=c) for c in (claims or [])],
        methods=[MethodItem(name=m, description=f"desc of {m}") for m in (methods or [])],
        limitations=[LimitationItem(text=lim) for lim in (limitations or [])],
        performance_facts=[
            PerformanceFactItem(metric=m, value=v) for m, v in (performance_facts or [])
        ],
        related_figure_ids=[],
        related_table_ids=[],
        token_count=10,
    )


# ---------------------------------------------------------------------------
# Standard mini-hierarchy with abstract + two sections
# ---------------------------------------------------------------------------


def _build_mini() -> tuple[list[ParsedBlock], HierarchyResult, list[SectionEnrichment]]:
    abs_block = _make_block(
        _DOC_ID, _CHAT_ID, BlockType.paragraph, "We study attention. It is efficient.",
        page=1, order=0,
    )

    doc_root_id = uuid5(NAMESPACE_OID, f"{_DOC_ID}:node:0")
    abstract_node_id = uuid5(NAMESPACE_OID, f"{_DOC_ID}:node:1")
    intro_node_id = uuid5(NAMESPACE_OID, f"{_DOC_ID}:node:2")
    exp_node_id = uuid5(NAMESPACE_OID, f"{_DOC_ID}:node:3")
    conc_node_id = uuid5(NAMESPACE_OID, f"{_DOC_ID}:node:4")

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
            page_end=5,
            order_index=0,
            level=0,
            bbox=None,
            source_block_ids=[],
            metadata_={"source": "heuristic"},
        ),
        DocumentNodeOut(
            id=abstract_node_id,
            chat_id=_CHAT_ID,
            document_id=_DOC_ID,
            parent_id=doc_root_id,
            node_type=NodeType.abstract,
            title="Abstract",
            content="",
            page_start=1,
            page_end=1,
            order_index=1,
            level=1,
            bbox=None,
            source_block_ids=[abs_block.block_id],
            metadata_={"source": "heuristic"},
        ),
        _make_node(_DOC_ID, _CHAT_ID, NodeType.section, "Introduction", [],
                   order_index=2, parent_id=doc_root_id),
        _make_node(_DOC_ID, _CHAT_ID, NodeType.section, "Experiments", [],
                   order_index=3, parent_id=doc_root_id),
        _make_node(_DOC_ID, _CHAT_ID, NodeType.section, "Conclusions", [],
                   order_index=4, parent_id=doc_root_id),
    ]

    hierarchy = HierarchyResult(
        document_id=_DOC_ID,
        chat_id=_CHAT_ID,
        nodes=nodes,
        references_start_index=None,
        appendix_start_index=None,
        heuristics_applied=["synthetic"],
    )

    section_enrichments = [
        _make_section_enrichment(
            intro_node_id, "Introduction",
            methods=["FlashAttention"],
            limitations=["Does not scale to very long sequences."],
            claims=["We show that FlashAttention improves throughput."],
            performance_facts=[("Throughput", "2x")],
            technical_keywords=["CUDA", "FlashAttention"],
            entities=["BERT", "GPT2"],
            order_index=2,
        ),
        _make_section_enrichment(
            exp_node_id, "Experiments",
            methods=["FlashAttention", "BERT"],
            limitations=["Requires A100 GPU."],
            claims=["We demonstrate SOTA results on GLUE."],
            performance_facts=[("Accuracy", "92%"), ("F1", "0.87")],
            technical_keywords=["A100", "CUDA"],
            entities=["GLUE", "SQuAD"],
            order_index=3,
        ),
        _make_section_enrichment(
            conc_node_id, "Conclusions",
            methods=[],
            limitations=[],
            claims=["We find that efficiency gains are consistent."],
            performance_facts=[],
            technical_keywords=[],
            entities=[],
            order_index=4,
        ),
    ]

    return [abs_block], hierarchy, section_enrichments


def _build_no_abstract() -> tuple[list[ParsedBlock], HierarchyResult, list[SectionEnrichment]]:
    doc_root_id = uuid5(NAMESPACE_OID, f"{_DOC_ID}:node:0")
    sec_node_id = uuid5(NAMESPACE_OID, f"{_DOC_ID}:node:1")

    nodes: list[DocumentNodeOut] = [
        DocumentNodeOut(
            id=doc_root_id,
            chat_id=_CHAT_ID,
            document_id=_DOC_ID,
            parent_id=None,
            node_type=NodeType.document,
            title="No-Abstract Paper",
            content="",
            page_start=1,
            page_end=3,
            order_index=0,
            level=0,
            bbox=None,
            source_block_ids=[],
            metadata_={"source": "heuristic"},
        ),
        _make_node(_DOC_ID, _CHAT_ID, NodeType.section, "Introduction", [],
                   order_index=1, parent_id=doc_root_id),
    ]

    hierarchy = HierarchyResult(
        document_id=_DOC_ID,
        chat_id=_CHAT_ID,
        nodes=nodes,
        references_start_index=None,
        appendix_start_index=None,
        heuristics_applied=["synthetic"],
    )

    enrichments = [
        _make_section_enrichment(sec_node_id, "Introduction", methods=["LinearAttention"],
                                 order_index=1)
    ]
    return [], hierarchy, enrichments


# ---------------------------------------------------------------------------
# Counting provider wrapper
# ---------------------------------------------------------------------------


class _CountingChatProvider(ChatProvider):
    def __init__(self) -> None:
        from app.providers.mock import MockChatProvider
        self._inner = MockChatProvider(model="counting-model")
        self.call_count: int = 0
        self.abstract_calls: int = 0

    @property
    def name(self) -> str:
        return "counting"

    @property
    def model(self) -> str:
        return "counting-model"

    @property
    def context_window(self) -> int:
        return 8192

    async def complete(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        stop: list[str] | None = None,
    ) -> ChatCompletion:
        self.call_count += 1
        for msg in messages:
            if "Abstract" in msg.content and msg.role == "user":
                self.abstract_calls += 1
                break
        return await self._inner.complete(
            messages, temperature=temperature, max_tokens=max_tokens, stop=stop
        )

    async def stream(  # type: ignore[override]
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        stop: list[str] | None = None,
    ) -> object:
        raise NotImplementedError

    async def test_connection(self) -> ProviderTestResult:
        return ProviderTestResult(ok=True, model="counting-model", latency_ms=0)


from app.providers.mock import MockChatProvider  # noqa: E402

_mock_provider = MockChatProvider(model="test-model")


# ---------------------------------------------------------------------------
# Test 1 — determinism
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_enrich_document_deterministic() -> None:
    """Two calls with identical inputs must produce identical model_dump_json."""
    blocks, hierarchy, section_enrichments = _build_mini()
    run1 = await enrich_document(hierarchy, blocks, section_enrichments, chat_provider=_mock_provider)
    run2 = await enrich_document(hierarchy, blocks, section_enrichments, chat_provider=_mock_provider)
    assert run1.model_dump_json() == run2.model_dump_json()


# ---------------------------------------------------------------------------
# Test 2 — no abstract node: abstract_summary is None, no abstract provider call
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_enrich_document_no_abstract() -> None:
    """When hierarchy has no abstract node, abstract_summary must be None."""
    blocks, hierarchy, enrichments = _build_no_abstract()
    counting = _CountingChatProvider()
    result = await enrich_document(hierarchy, blocks, enrichments, chat_provider=counting)
    assert result.abstract_summary is None
    assert counting.abstract_calls == 0


# ---------------------------------------------------------------------------
# Test 3 — union-dedup: methods, limitations, contributions (parametrized)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("field,expected_dup", [
    ("main_methods", "FlashAttention"),
    ("main_limitations", None),
    ("main_contributions", None),
])
async def test_enrich_document_fields_deduped(field: str, expected_dup: str | None) -> None:
    """Union-dedup: no duplicates in methods/limitations/contributions."""
    blocks, hierarchy, enrichments = _build_mini()
    result = await enrich_document(hierarchy, blocks, enrichments, chat_provider=_mock_provider)
    values: list[str] = getattr(result, field)
    assert isinstance(values, list)
    assert len(values) == len(set(values)), f"{field} contains duplicates"
    if expected_dup is not None:
        assert values.count(expected_dup) == 1, (
            f"Expected '{expected_dup}' exactly once in {field}"
        )


# ---------------------------------------------------------------------------
# Test 4 — document_overview non-empty and ≤ 1500 chars
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_enrich_document_overview_length() -> None:
    """document_overview must be non-empty and ≤ 1500 chars."""
    blocks, hierarchy, enrichments = _build_mini()
    result = await enrich_document(hierarchy, blocks, enrichments, chat_provider=_mock_provider)
    assert result.document_overview
    assert len(result.document_overview) <= 1500


# ---------------------------------------------------------------------------
# Test 5 — section-title-driven fields: experiments → non-empty; no-abstract → empty
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("build_fn,field,should_be_nonempty", [
    ("mini", "main_experimental_results", True),
    ("no_abstract", "main_conclusions", False),
])
async def test_enrich_document_section_title_fields(
    build_fn: str, field: str, should_be_nonempty: bool
) -> None:
    """main_experimental_results / main_conclusions depend on matching section titles."""
    if build_fn == "mini":
        blocks, hierarchy, enrichments = _build_mini()
    else:
        blocks, hierarchy, enrichments = _build_no_abstract()
    result = await enrich_document(hierarchy, blocks, enrichments, chat_provider=_mock_provider)
    values: list[str] = getattr(result, field)
    if should_be_nonempty:
        assert len(values) > 0, f"Expected non-empty {field}"
    else:
        assert values == [], f"Expected empty {field}"


# ---------------------------------------------------------------------------
# Test 6 — persist_document_summaries idempotency
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_persist_document_summaries_idempotent() -> None:
    """Two persist calls must not increase row count."""
    blocks, hierarchy, enrichments = _build_mini()
    result = await enrich_document(hierarchy, blocks, enrichments, chat_provider=_mock_provider)

    abstract_node_id: UUID | None = None
    for node in hierarchy.nodes:
        if node.node_type == NodeType.abstract:
            abstract_node_id = node.id
            break

    session, engine = await _make_sqlite_session()
    try:
        await persist_document_summaries(session, result, abstract_node_id=abstract_node_id)
        count_result = await session.execute(select(func.count()).select_from(Summary))
        count_first: int = count_result.scalar_one()

        await persist_document_summaries(session, result, abstract_node_id=abstract_node_id)
        count_result2 = await session.execute(select(func.count()).select_from(Summary))
        count_second: int = count_result2.scalar_one()

        assert count_first == count_second
        assert count_first == 2
    finally:
        await session.close()
        await engine.dispose()  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Test 7 — chat_id and document_id from hierarchy
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_enrich_document_ids_from_hierarchy() -> None:
    """chat_id and document_id must equal hierarchy values."""
    blocks, hierarchy, enrichments = _build_mini()
    result = await enrich_document(hierarchy, blocks, enrichments, chat_provider=_mock_provider)
    assert result.chat_id == hierarchy.chat_id
    assert result.document_id == hierarchy.document_id
