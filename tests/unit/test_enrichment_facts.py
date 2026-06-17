"""Unit tests for app.enrichment.facts.extract_structured_facts.

These tests are fully deterministic and require no running database.
The extractor is a pure function: same inputs → same outputs (including IDs
when computed by persist_facts).

Coverage
--------
1. Same (hierarchy, blocks) inputs → same list[StructuredFactCreate] (determinism).
2. At least one metric, one hyperparameter, and one dataset fact in the fixture.
3. Every fact's context_excerpt is ≤ 200 chars and non-empty.
4. Persist idempotency test uses SQLite + aiosqlite (no FK constraints).

Fixture design
--------------
The mini fixture contains:
- A document node (root).
- A section titled "Experimental Setup and Datasets" (triggers dataset extractor).
- A subsection titled "Implementation Details" (triggers hyperparameter extractor).
- A section titled "Results and Evaluation" (triggers metric extractor).
- Paragraph blocks with text that matches metric / hyperparameter / dataset patterns.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from uuid import NAMESPACE_OID, UUID, uuid5

import pytest
import pytest_asyncio
from sqlalchemy import JSON, Column, Integer, String, Text, event, func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.enrichment.facts import extract_structured_facts
from app.models.domain import StructuredFactCreate
from app.parsing.models import (
    BBox,
    BlockType,
    DocumentNodeOut,
    HierarchyResult,
    NodeType,
    ParsedBlock,
)

# ---------------------------------------------------------------------------
# SQLite-compatible ORM base for persist tests
# ---------------------------------------------------------------------------


class _TestBase(DeclarativeBase):
    pass


class _StructuredFactSQLite(_TestBase):
    """Minimal SQLite-compatible mirror of StructuredFact (no FK, JSON not JSONB)."""

    __tablename__ = "structured_facts"

    id = Column(String(36), primary_key=True)
    chat_id = Column(String(36), nullable=False, index=True)
    document_id = Column(String(36), nullable=False, index=True)
    source_node_id = Column(String(36), nullable=True)
    kind = Column(String(50), nullable=False)
    key = Column(String(255), nullable=False)
    value = Column(JSON, nullable=False)
    unit = Column(String(100), nullable=True)
    context_excerpt = Column(Text, nullable=True)
    page = Column(Integer, nullable=True)
    created_at = Column(
        String(50), nullable=False, default=lambda: datetime.now(UTC).isoformat()
    )


# ---------------------------------------------------------------------------
# Stable fixture UUIDs
# ---------------------------------------------------------------------------

_CHAT_ID: UUID = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
_DOC_ID: UUID = UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")

# Section node IDs (deterministic)
_SECTION_DATASET_ID: UUID = uuid5(NAMESPACE_OID, f"{_DOC_ID}:node:0")
_SECTION_IMPL_ID: UUID = uuid5(NAMESPACE_OID, f"{_DOC_ID}:node:1")
_SECTION_RESULTS_ID: UUID = uuid5(NAMESPACE_OID, f"{_DOC_ID}:node:2")

# Block IDs
_BLK_DATASET_ID: UUID = uuid5(NAMESPACE_OID, f"{_DOC_ID}:blk:dataset")
_BLK_IMPL_ID: UUID = uuid5(NAMESPACE_OID, f"{_DOC_ID}:blk:impl")
_BLK_METRIC_ID: UUID = uuid5(NAMESPACE_OID, f"{_DOC_ID}:blk:metric")


def _bbox() -> BBox:
    return BBox(x0=0.0, y0=0.0, x1=100.0, y1=20.0, page_width=595.0, page_height=842.0)


def _make_blocks() -> list[ParsedBlock]:
    """Return a list of ParsedBlock instances that trigger all three fact kinds."""
    return [
        # Dataset block — inside "Experimental Setup and Datasets" section
        ParsedBlock(
            block_id=_BLK_DATASET_ID,
            chat_id=_CHAT_ID,
            document_id=_DOC_ID,
            page_number=3,
            block_type=BlockType.paragraph,
            text=(
                "We evaluate on the MS-MARCO passage ranking dataset as well as "
                "Natural Questions (NQ) and TriviaQA benchmarks. "
                "The BEIR benchmark is also included for zero-shot evaluation."
            ),
            bbox=_bbox(),
            reading_order=0,
        ),
        # Hyperparameter block — inside "Implementation Details" section
        ParsedBlock(
            block_id=_BLK_IMPL_ID,
            chat_id=_CHAT_ID,
            document_id=_DOC_ID,
            page_number=4,
            block_type=BlockType.paragraph,
            text=(
                "We train the model with a learning rate of 1e-4 and batch size of 64. "
                "The temperature is set to 0.0 for greedy decoding. "
                "We use top-k 8 for candidate retrieval."
            ),
            bbox=_bbox(),
            reading_order=1,
        ),
        # Metric block — inside "Results and Evaluation" section
        ParsedBlock(
            block_id=_BLK_METRIC_ID,
            chat_id=_CHAT_ID,
            document_id=_DOC_ID,
            page_number=5,
            block_type=BlockType.paragraph,
            text=(
                "Our model achieves F1 score of 87.3% and accuracy of 0.923 on the test set. "
                "The nDCG@10 is 92.1 and MRR = 0.756."
            ),
            bbox=_bbox(),
            reading_order=2,
        ),
    ]


def _make_hierarchy(blocks: list[ParsedBlock]) -> HierarchyResult:
    """Build a minimal hierarchy with 3 sections covering the blocks above."""
    blk_dataset = blocks[0]
    blk_impl = blocks[1]
    blk_metric = blocks[2]

    # Root document node (no source_block_ids — structural container)
    root = DocumentNodeOut(
        id=uuid5(NAMESPACE_OID, f"{_DOC_ID}:node:root"),
        chat_id=_CHAT_ID,
        document_id=_DOC_ID,
        parent_id=None,
        node_type=NodeType.document,
        title="Test Paper",
        content="",
        page_start=1,
        page_end=6,
        order_index=0,
        level=0,
        bbox=None,
        source_block_ids=[],
        metadata_={"source": "heuristic"},
    )

    # Section: "Experimental Setup and Datasets" — owns dataset block
    sec_dataset = DocumentNodeOut(
        id=_SECTION_DATASET_ID,
        chat_id=_CHAT_ID,
        document_id=_DOC_ID,
        parent_id=root.id,
        node_type=NodeType.section,
        title="Experimental Setup and Datasets",
        content="",
        page_start=3,
        page_end=3,
        order_index=1,
        level=1,
        bbox=None,
        source_block_ids=[blk_dataset.block_id],
        metadata_={"source": "heuristic"},
    )

    # Subsection: "Implementation Details" — owns hyperparameter block
    sec_impl = DocumentNodeOut(
        id=_SECTION_IMPL_ID,
        chat_id=_CHAT_ID,
        document_id=_DOC_ID,
        parent_id=root.id,
        node_type=NodeType.subsection,
        title="Implementation Details",
        content="",
        page_start=4,
        page_end=4,
        order_index=2,
        level=2,
        bbox=None,
        source_block_ids=[blk_impl.block_id],
        metadata_={"source": "heuristic"},
    )

    # Section: "Results and Evaluation" — owns metric block
    sec_results = DocumentNodeOut(
        id=_SECTION_RESULTS_ID,
        chat_id=_CHAT_ID,
        document_id=_DOC_ID,
        parent_id=root.id,
        node_type=NodeType.section,
        title="Results and Evaluation",
        content="",
        page_start=5,
        page_end=5,
        order_index=3,
        level=1,
        bbox=None,
        source_block_ids=[blk_metric.block_id],
        metadata_={"source": "heuristic"},
    )

    return HierarchyResult(
        document_id=_DOC_ID,
        chat_id=_CHAT_ID,
        nodes=[root, sec_dataset, sec_impl, sec_results],
        references_start_index=None,
        appendix_start_index=None,
        heuristics_applied=["test"],
    )


# ---------------------------------------------------------------------------
# Module-scoped fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def blocks() -> list[ParsedBlock]:
    return _make_blocks()


@pytest.fixture(scope="module")
def hierarchy(blocks: list[ParsedBlock]) -> HierarchyResult:
    return _make_hierarchy(blocks)


@pytest.fixture(scope="module")
def facts(hierarchy: HierarchyResult, blocks: list[ParsedBlock]) -> list[StructuredFactCreate]:
    return extract_structured_facts(hierarchy, blocks)


# ---------------------------------------------------------------------------
# SQLite async session for idempotency tests
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture(scope="module")
async def sqlite_session() -> AsyncGenerator[AsyncSession, None]:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)

    @event.listens_for(engine.sync_engine, "connect")
    def _set_sqlite_pragma(dbapi_conn, connection_record):  # type: ignore[no-untyped-def]
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA foreign_keys=OFF")
        cursor.close()

    async with engine.begin() as conn:
        await conn.run_sync(_TestBase.metadata.create_all)

    factory = async_sessionmaker(bind=engine, expire_on_commit=False, class_=AsyncSession)
    async with factory() as session:
        try:
            yield session
        finally:
            await session.rollback()

    await engine.dispose()


# ---------------------------------------------------------------------------
# Helper: persist facts into SQLite (mirrors facts_service.persist_facts logic
# but uses the SQLite-compatible ORM class)
# ---------------------------------------------------------------------------


async def _persist_sqlite(
    session: AsyncSession,
    facts_list: list[StructuredFactCreate],
    chat_id: UUID,
) -> list[str]:
    ids: list[str] = []
    for seq, fact in enumerate(facts_list):
        page_str = str(fact.page) if fact.page is not None else "None"
        fact_id = str(
            uuid5(NAMESPACE_OID, f"{fact.document_id}:fact:{fact.kind}:{fact.key}:{page_str}:{seq}")
        )
        ids.append(fact_id)
        existing = (
            await session.execute(
                select(_StructuredFactSQLite).where(_StructuredFactSQLite.id == fact_id)
            )
        ).scalar_one_or_none()

        if existing is not None:
            existing.value = fact.value.model_dump()
            existing.unit = fact.unit
            existing.context_excerpt = fact.context_excerpt
            existing.page = fact.page
        else:
            row = _StructuredFactSQLite(
                id=fact_id,
                chat_id=str(chat_id),
                document_id=str(fact.document_id),
                source_node_id=str(fact.source_node_id) if fact.source_node_id else None,
                kind=fact.kind,
                key=fact.key,
                value=fact.value.model_dump(),
                unit=fact.unit,
                context_excerpt=fact.context_excerpt,
                page=fact.page,
            )
            session.add(row)
    await session.flush()
    return ids


# ---------------------------------------------------------------------------
# 1. Determinism tests
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_inputs_produce_same_length(
        self, hierarchy: HierarchyResult, blocks: list[ParsedBlock]
    ) -> None:
        """Same (hierarchy, blocks) → same number of facts."""
        run1 = extract_structured_facts(hierarchy, blocks)
        run2 = extract_structured_facts(hierarchy, blocks)
        assert len(run1) == len(run2), (
            f"Run 1 produced {len(run1)} facts; run 2 produced {len(run2)}"
        )

    def test_same_inputs_produce_same_content(
        self, hierarchy: HierarchyResult, blocks: list[ParsedBlock]
    ) -> None:
        """Same inputs → identical fact keys and values."""
        run1 = extract_structured_facts(hierarchy, blocks)
        run2 = extract_structured_facts(hierarchy, blocks)

        def _sig(
            f: StructuredFactCreate,
        ) -> tuple[str, str, str | None, float | None, int | None]:
            return (f.kind, f.key, f.value.raw, f.value.numeric, f.page)

        sigs1 = [_sig(f) for f in run1]
        sigs2 = [_sig(f) for f in run2]
        assert sigs1 == sigs2, "Facts differ between two runs with identical inputs"


# ---------------------------------------------------------------------------
# 2. Kind coverage tests
# ---------------------------------------------------------------------------


class TestKindCoverage:
    def test_at_least_one_metric(self, facts: list[StructuredFactCreate]) -> None:
        metrics = [f for f in facts if f.kind == "metric"]
        assert metrics, (
            f"Expected at least 1 metric fact; got none. All facts: {[f.kind for f in facts]}"
        )

    def test_at_least_one_hyperparameter(self, facts: list[StructuredFactCreate]) -> None:
        hypers = [f for f in facts if f.kind == "hyperparameter"]
        assert hypers, (
            f"Expected at least 1 hyperparameter fact; got none. "
            f"All facts: {[(f.kind, f.key) for f in facts]}"
        )

    def test_at_least_one_dataset(self, facts: list[StructuredFactCreate]) -> None:
        datasets = [f for f in facts if f.kind == "dataset"]
        assert datasets, (
            f"Expected at least 1 dataset fact; got none. "
            f"All facts: {[(f.kind, f.key) for f in facts]}"
        )


# ---------------------------------------------------------------------------
# 3. context_excerpt constraints
# ---------------------------------------------------------------------------


class TestContextExcerpt:
    def test_all_excerpts_le_200_chars(self, facts: list[StructuredFactCreate]) -> None:
        violations = [
            (f.kind, f.key, len(f.context_excerpt or ""))
            for f in facts
            if f.context_excerpt is not None and len(f.context_excerpt) > 200
        ]
        assert not violations, f"Excerpts > 200 chars: {violations}"

    def test_all_excerpts_non_empty(self, facts: list[StructuredFactCreate]) -> None:
        empty = [(f.kind, f.key) for f in facts if not (f.context_excerpt or "").strip()]
        assert not empty, f"Facts with empty context_excerpt: {empty}"


# ---------------------------------------------------------------------------
# 4. Idempotency (persist twice → COUNT unchanged)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_persist_twice_no_duplicates(
    sqlite_session: AsyncSession,
    hierarchy: HierarchyResult,
    blocks: list[ParsedBlock],
) -> None:
    """Persisting the same facts twice must not increase the row count."""
    facts_list = extract_structured_facts(hierarchy, blocks)
    assert facts_list, "Need at least one fact for this test"

    # First persist
    ids1 = await _persist_sqlite(sqlite_session, facts_list, _CHAT_ID)
    await sqlite_session.flush()

    count_after_first: int = (
        await sqlite_session.execute(
            select(func.count()).select_from(_StructuredFactSQLite)
        )
    ).scalar_one()

    # Second persist (same data)
    ids2 = await _persist_sqlite(sqlite_session, facts_list, _CHAT_ID)
    await sqlite_session.flush()

    count_after_second: int = (
        await sqlite_session.execute(
            select(func.count()).select_from(_StructuredFactSQLite)
        )
    ).scalar_one()

    assert count_after_first == count_after_second, (
        f"Row count changed after second persist: {count_after_first} → {count_after_second}"
    )
    # IDs must match
    assert ids1 == ids2, "IDs must be deterministic across runs"
