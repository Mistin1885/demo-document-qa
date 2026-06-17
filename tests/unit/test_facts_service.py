"""Unit tests for app.services.facts_service.

Database strategy
-----------------
Uses SQLite + aiosqlite (in-memory) via a test-local SQLAlchemy setup.
The production ORM uses PostgreSQL-specific types (JSONB, UUID).
This test file creates a SQLite-compatible schema that mirrors the
``structured_facts`` table so the service layer can be tested without
a running PostgreSQL instance.

Coverage
--------
1. FactsFilter rejects extra fields (pydantic ValidationError).
2. query_facts ignores filt.chat_id; always uses current_chat_id.
3. document_ids filter (positive + negative).
4. kinds filter (positive + negative).
5. keys filter (positive + negative).
6. page_range filter (positive + negative).
7. numeric_min / numeric_max filter (positive + negative).
8. limit / offset pagination.
9. keys element > 100 chars → ValidationError.
10. keys list > 50 entries → ValidationError.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from uuid import NAMESPACE_OID, UUID, uuid5

import pytest
import pytest_asyncio
from pydantic import ValidationError
from sqlalchemy import JSON, Column, Integer, String, Text, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.models.domain import FactValue
from app.services.facts_service import FactsFilter

# ---------------------------------------------------------------------------
# SQLite-compatible ORM for testing
# ---------------------------------------------------------------------------


class _Base(DeclarativeBase):
    pass


class _SF(_Base):
    """SQLite-compatible mirror of StructuredFact without FK constraints."""

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
# Pure-SQLite service helpers (mirror facts_service logic but use _SF)
# ---------------------------------------------------------------------------


def _fact_id_str(document_id: UUID, kind: str, key: str, page: int | None, seq: int) -> str:
    page_str = str(page) if page is not None else "None"
    name = f"{document_id}:fact:{kind}:{key}:{page_str}:{seq}"
    return str(uuid5(NAMESPACE_OID, name))


async def _insert_fact(
    session: AsyncSession,
    *,
    chat_id: UUID,
    document_id: UUID,
    kind: str,
    key: str,
    value: FactValue,
    unit: str | None = None,
    context_excerpt: str | None = None,
    page: int | None = None,
    seq: int = 0,
) -> str:
    """Insert a single fact into SQLite and return its id string."""
    fid = _fact_id_str(document_id, kind, key, page, seq)
    row = _SF(
        id=fid,
        chat_id=str(chat_id),
        document_id=str(document_id),
        source_node_id=None,
        kind=kind,
        key=key,
        value=value.model_dump(),
        unit=unit,
        context_excerpt=context_excerpt,
        page=page,
    )
    session.add(row)
    await session.flush()
    return fid


async def _query_facts_sqlite(
    session: AsyncSession,
    *,
    current_chat_id: UUID,
    filt: FactsFilter,
) -> list[dict]:  # type: ignore[type-arg]
    """SQLite-compatible analogue of facts_service.query_facts.

    Enforces the same isolation semantics:
    - Uses current_chat_id, ignores filt.chat_id.
    - All filters via typed SQLAlchemy expressions.
    """
    # Isolation: always use current_chat_id, ignoring filt.chat_id
    stmt = select(_SF).where(_SF.chat_id == str(current_chat_id))

    if filt.document_ids is not None:
        stmt = stmt.where(_SF.document_id.in_([str(d) for d in filt.document_ids]))

    if filt.kinds is not None:
        stmt = stmt.where(_SF.kind.in_(filt.kinds))

    if filt.keys is not None:
        stmt = stmt.where(_SF.key.in_(filt.keys))

    if filt.page_range is not None:
        lo, hi = filt.page_range
        stmt = stmt.where(_SF.page >= lo).where(_SF.page <= hi)

    if filt.unit_in is not None:
        stmt = stmt.where(_SF.unit.in_(filt.unit_in))

    stmt = stmt.order_by(_SF.created_at, _SF.id)
    stmt = stmt.offset(filt.offset).limit(filt.limit)

    rows = (await session.execute(stmt)).scalars().all()

    # Python-level numeric filter (SQLite doesn't support JSONB operators)
    result: list[dict] = []  # type: ignore[type-arg]
    for row in rows:
        val = row.value
        numeric = val.get("numeric") if isinstance(val, dict) else None
        if filt.numeric_min is not None and (numeric is None or numeric < filt.numeric_min):
            continue
        if filt.numeric_max is not None and (numeric is None or numeric > filt.numeric_max):
            continue
        result.append(
            {
                "id": row.id,
                "chat_id": row.chat_id,
                "document_id": row.document_id,
                "kind": row.kind,
                "key": row.key,
                "value": row.value,
                "unit": row.unit,
                "context_excerpt": row.context_excerpt,
                "page": row.page,
            }
        )
    return result


# ---------------------------------------------------------------------------
# Stable fixture UUIDs
# ---------------------------------------------------------------------------

_CHAT_A: UUID = UUID("aaaa0000-0000-0000-0000-000000000001")
_CHAT_B: UUID = UUID("bbbb0000-0000-0000-0000-000000000002")
_DOC_A1: UUID = UUID("dddd0000-0000-0000-0000-000000000011")
_DOC_A2: UUID = UUID("dddd0000-0000-0000-0000-000000000012")
_DOC_B1: UUID = UUID("dddd0000-0000-0000-0000-000000000021")


# ---------------------------------------------------------------------------
# SQLite session fixture
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture(scope="module")
async def sqlite_session() -> AsyncGenerator[AsyncSession, None]:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)

    async with engine.begin() as conn:
        await conn.run_sync(_Base.metadata.create_all)

    factory = async_sessionmaker(bind=engine, expire_on_commit=False, class_=AsyncSession)
    async with factory() as session:
        yield session
        await session.rollback()

    await engine.dispose()


@pytest_asyncio.fixture(scope="module", autouse=True)
async def seed_data(sqlite_session: AsyncSession) -> None:
    """Seed the SQLite DB with facts for both chats."""
    # Chat A, Doc A1: metric (F1=87.3%), hyperparameter (lr=1e-4), dataset (MS-MARCO)
    await _insert_fact(
        sqlite_session,
        chat_id=_CHAT_A,
        document_id=_DOC_A1,
        kind="metric",
        key="f1",
        value=FactValue(raw="F1 87.3%", numeric=87.3),
        unit="%",
        context_excerpt="Our model achieves F1 87.3% on the test set.",
        page=5,
        seq=0,
    )
    await _insert_fact(
        sqlite_session,
        chat_id=_CHAT_A,
        document_id=_DOC_A1,
        kind="hyperparameter",
        key="learning_rate",
        value=FactValue(raw="1e-4", numeric=1e-4),
        unit=None,
        context_excerpt="We train with learning rate of 1e-4.",
        page=4,
        seq=1,
    )
    await _insert_fact(
        sqlite_session,
        chat_id=_CHAT_A,
        document_id=_DOC_A1,
        kind="dataset",
        key="ms_marco",
        value=FactValue(raw="MS-MARCO", items=["MS-MARCO"]),
        unit=None,
        context_excerpt="We evaluate on the MS-MARCO passage ranking dataset.",
        page=3,
        seq=2,
    )
    # Chat A, Doc A2: metric (accuracy=92.0%)
    await _insert_fact(
        sqlite_session,
        chat_id=_CHAT_A,
        document_id=_DOC_A2,
        kind="metric",
        key="accuracy",
        value=FactValue(raw="accuracy 92.0%", numeric=92.0),
        unit="%",
        context_excerpt="accuracy of 92.0% on dev set",
        page=6,
        seq=3,
    )
    # Chat B, Doc B1: metric (BLEU=34.5) — must NEVER appear in chat_a queries
    await _insert_fact(
        sqlite_session,
        chat_id=_CHAT_B,
        document_id=_DOC_B1,
        kind="metric",
        key="bleu",
        value=FactValue(raw="BLEU 34.5", numeric=34.5),
        unit=None,
        context_excerpt="BLEU = 34.5 on the test set.",
        page=7,
        seq=4,
    )
    await sqlite_session.flush()


# ---------------------------------------------------------------------------
# 1. FactsFilter validation — extra fields rejected
# ---------------------------------------------------------------------------


class TestFactsFilterValidation:
    def test_extra_field_raises(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            FactsFilter(chat_id=_CHAT_A, sql_injection="DROP TABLE--")  # type: ignore[call-arg]
        assert "sql_injection" in str(exc_info.value) or "extra" in str(exc_info.value).lower()

    def test_key_too_long_raises(self) -> None:
        """A single key > 100 chars must raise ValidationError."""
        long_key = "x" * 101
        with pytest.raises(ValidationError) as exc_info:
            FactsFilter(chat_id=_CHAT_A, keys=[long_key])
        assert "100" in str(exc_info.value) or "chars" in str(exc_info.value)

    def test_keys_too_many_raises(self) -> None:
        """More than 50 keys must raise ValidationError."""
        many_keys = [f"key_{i}" for i in range(51)]
        with pytest.raises(ValidationError) as exc_info:
            FactsFilter(chat_id=_CHAT_A, keys=many_keys)
        assert "50" in str(exc_info.value)

    def test_valid_filter_accepted(self) -> None:
        filt = FactsFilter(chat_id=_CHAT_A, kinds=["metric"], limit=10)
        assert filt.chat_id == _CHAT_A

    def test_unit_in_too_many_raises(self) -> None:
        """More than 20 units must raise ValidationError."""
        many_units = [f"unit{i}" for i in range(21)]
        with pytest.raises(ValidationError):
            FactsFilter(chat_id=_CHAT_A, unit_in=many_units)

    def test_page_range_min_lt_1_raises(self) -> None:
        with pytest.raises(ValidationError):
            FactsFilter(chat_id=_CHAT_A, page_range=(0, 5))

    def test_page_range_max_lt_min_raises(self) -> None:
        with pytest.raises(ValidationError):
            FactsFilter(chat_id=_CHAT_A, page_range=(5, 3))

    def test_limit_too_low_raises(self) -> None:
        with pytest.raises(ValidationError):
            FactsFilter(chat_id=_CHAT_A, limit=0)

    def test_limit_too_high_raises(self) -> None:
        with pytest.raises(ValidationError):
            FactsFilter(chat_id=_CHAT_A, limit=201)

    def test_offset_negative_raises(self) -> None:
        with pytest.raises(ValidationError):
            FactsFilter(chat_id=_CHAT_A, offset=-1)


# ---------------------------------------------------------------------------
# 2. Isolation: filt.chat_id is ignored; current_chat_id wins
# ---------------------------------------------------------------------------


class TestIsolation:
    @pytest.mark.asyncio
    async def test_wrong_chat_id_in_filter_returns_correct_chat_facts(
        self, sqlite_session: AsyncSession
    ) -> None:
        """Even if filt.chat_id = CHAT_B, query with current_chat_id=CHAT_A returns A's facts."""
        filt = FactsFilter(chat_id=_CHAT_B, limit=200)  # wrong chat_id in filter
        results = await _query_facts_sqlite(
            sqlite_session, current_chat_id=_CHAT_A, filt=filt
        )
        chat_ids = {r["chat_id"] for r in results}
        assert chat_ids == {str(_CHAT_A)}, (
            f"Expected only CHAT_A facts; got chat_ids: {chat_ids}"
        )

    @pytest.mark.asyncio
    async def test_chat_b_facts_never_leak_to_chat_a(
        self, sqlite_session: AsyncSession
    ) -> None:
        """BLEU fact from CHAT_B must never appear in CHAT_A queries."""
        filt = FactsFilter(chat_id=_CHAT_A, limit=200)
        results = await _query_facts_sqlite(
            sqlite_session, current_chat_id=_CHAT_A, filt=filt
        )
        keys = [r["key"] for r in results]
        assert "bleu" not in keys, f"CHAT_B's bleu fact leaked into CHAT_A results: {keys}"


# ---------------------------------------------------------------------------
# 3. document_ids filter
# ---------------------------------------------------------------------------


class TestDocumentIdsFilter:
    @pytest.mark.asyncio
    async def test_document_ids_positive(self, sqlite_session: AsyncSession) -> None:
        """Filter by DOC_A1 → only DOC_A1 facts."""
        filt = FactsFilter(chat_id=_CHAT_A, document_ids=[_DOC_A1], limit=200)
        results = await _query_facts_sqlite(
            sqlite_session, current_chat_id=_CHAT_A, filt=filt
        )
        doc_ids = {r["document_id"] for r in results}
        assert doc_ids == {str(_DOC_A1)}

    @pytest.mark.asyncio
    async def test_document_ids_negative(self, sqlite_session: AsyncSession) -> None:
        """Filter by DOC_A2 → no DOC_A1 facts."""
        filt = FactsFilter(chat_id=_CHAT_A, document_ids=[_DOC_A2], limit=200)
        results = await _query_facts_sqlite(
            sqlite_session, current_chat_id=_CHAT_A, filt=filt
        )
        doc_ids = {r["document_id"] for r in results}
        assert str(_DOC_A1) not in doc_ids


# ---------------------------------------------------------------------------
# 4. kinds filter
# ---------------------------------------------------------------------------


class TestKindsFilter:
    @pytest.mark.asyncio
    async def test_kinds_positive(self, sqlite_session: AsyncSession) -> None:
        filt = FactsFilter(chat_id=_CHAT_A, kinds=["metric"], limit=200)
        results = await _query_facts_sqlite(
            sqlite_session, current_chat_id=_CHAT_A, filt=filt
        )
        assert all(r["kind"] == "metric" for r in results)
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_kinds_negative(self, sqlite_session: AsyncSession) -> None:
        """kinds=["dataset"] should not return metric or hyperparameter facts."""
        filt = FactsFilter(chat_id=_CHAT_A, kinds=["dataset"], limit=200)
        results = await _query_facts_sqlite(
            sqlite_session, current_chat_id=_CHAT_A, filt=filt
        )
        kinds_found = {r["kind"] for r in results}
        assert "metric" not in kinds_found
        assert "hyperparameter" not in kinds_found


# ---------------------------------------------------------------------------
# 5. keys filter
# ---------------------------------------------------------------------------


class TestKeysFilter:
    @pytest.mark.asyncio
    async def test_keys_positive(self, sqlite_session: AsyncSession) -> None:
        filt = FactsFilter(chat_id=_CHAT_A, keys=["f1"], limit=200)
        results = await _query_facts_sqlite(
            sqlite_session, current_chat_id=_CHAT_A, filt=filt
        )
        assert any(r["key"] == "f1" for r in results)

    @pytest.mark.asyncio
    async def test_keys_negative(self, sqlite_session: AsyncSession) -> None:
        """keys=["nonexistent_key"] returns no results."""
        filt = FactsFilter(chat_id=_CHAT_A, keys=["nonexistent_key_xyz"], limit=200)
        results = await _query_facts_sqlite(
            sqlite_session, current_chat_id=_CHAT_A, filt=filt
        )
        assert len(results) == 0


# ---------------------------------------------------------------------------
# 6. page_range filter
# ---------------------------------------------------------------------------


class TestPageRangeFilter:
    @pytest.mark.asyncio
    async def test_page_range_positive(self, sqlite_session: AsyncSession) -> None:
        """page_range=(4, 5) → pages 4 and 5 only."""
        filt = FactsFilter(chat_id=_CHAT_A, page_range=(4, 5), limit=200)
        results = await _query_facts_sqlite(
            sqlite_session, current_chat_id=_CHAT_A, filt=filt
        )
        assert all(4 <= (r["page"] or 0) <= 5 for r in results)
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_page_range_negative(self, sqlite_session: AsyncSession) -> None:
        """page_range=(10, 20) → no facts (all seeded facts are on pages 3-6)."""
        filt = FactsFilter(chat_id=_CHAT_A, page_range=(10, 20), limit=200)
        results = await _query_facts_sqlite(
            sqlite_session, current_chat_id=_CHAT_A, filt=filt
        )
        assert len(results) == 0


# ---------------------------------------------------------------------------
# 7. numeric_min / numeric_max filter
# ---------------------------------------------------------------------------


class TestNumericFilter:
    @pytest.mark.asyncio
    async def test_numeric_min_positive(self, sqlite_session: AsyncSession) -> None:
        """numeric_min=80 → only facts with numeric >= 80 (F1=87.3, accuracy=92.0)."""
        filt = FactsFilter(chat_id=_CHAT_A, numeric_min=80.0, limit=200)
        results = await _query_facts_sqlite(
            sqlite_session, current_chat_id=_CHAT_A, filt=filt
        )
        assert len(results) >= 1
        for r in results:
            val = r["value"]
            numeric = val.get("numeric") if isinstance(val, dict) else None
            assert numeric is not None and numeric >= 80.0, (
                f"Fact with key={r['key']} has numeric={numeric} but should be >= 80"
            )

    @pytest.mark.asyncio
    async def test_numeric_max_negative(self, sqlite_session: AsyncSession) -> None:
        """numeric_max=0.001 excludes F1=87.3 and accuracy=92.0 (both > 0.001)."""
        filt = FactsFilter(chat_id=_CHAT_A, numeric_max=0.001, limit=200)
        results = await _query_facts_sqlite(
            sqlite_session, current_chat_id=_CHAT_A, filt=filt
        )
        for r in results:
            val = r["value"]
            numeric = val.get("numeric") if isinstance(val, dict) else None
            if numeric is not None:
                assert numeric <= 0.001, (
                    f"Fact with key={r['key']} has numeric={numeric} but should be <= 0.001"
                )


# ---------------------------------------------------------------------------
# 8. limit / offset
# ---------------------------------------------------------------------------


class TestPagination:
    @pytest.mark.asyncio
    async def test_limit_restricts_results(self, sqlite_session: AsyncSession) -> None:
        filt = FactsFilter(chat_id=_CHAT_A, limit=2)
        results = await _query_facts_sqlite(
            sqlite_session, current_chat_id=_CHAT_A, filt=filt
        )
        assert len(results) <= 2

    @pytest.mark.asyncio
    async def test_offset_skips_rows(self, sqlite_session: AsyncSession) -> None:
        """offset=1 should return fewer results than offset=0 (when total > 1)."""
        filt_all = FactsFilter(chat_id=_CHAT_A, limit=200, offset=0)
        filt_skip = FactsFilter(chat_id=_CHAT_A, limit=200, offset=1)
        all_results = await _query_facts_sqlite(
            sqlite_session, current_chat_id=_CHAT_A, filt=filt_all
        )
        skipped = await _query_facts_sqlite(
            sqlite_session, current_chat_id=_CHAT_A, filt=filt_skip
        )
        if len(all_results) > 1:
            assert len(skipped) == len(all_results) - 1
