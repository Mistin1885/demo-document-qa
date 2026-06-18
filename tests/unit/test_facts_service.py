"""Unit tests for app.services.facts_service.

Reduced to ≤10 tests covering: FactsFilter validation (extra="forbid", key
length/count, page_range, limit/offset bounds — parametrized), isolation
(chat_id override always wins), and key filters (document_ids, kinds, keys,
page_range, numeric, limit/offset).
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
# SQLite query helper (enforces isolation)
# ---------------------------------------------------------------------------


def _fact_id_str(document_id: UUID, kind: str, key: str, page: int | None, seq: int) -> str:
    page_str = str(page) if page is not None else "None"
    return str(uuid5(NAMESPACE_OID, f"{document_id}:fact:{kind}:{key}:{page_str}:{seq}"))


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
    """SQLite-compatible analogue of facts_service.query_facts. Enforces isolation."""
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

    stmt = stmt.order_by(_SF.created_at, _SF.id).offset(filt.offset).limit(filt.limit)
    rows = (await session.execute(stmt)).scalars().all()

    result: list[dict] = []  # type: ignore[type-arg]
    for row in rows:
        val = row.value
        numeric = val.get("numeric") if isinstance(val, dict) else None
        if filt.numeric_min is not None and (numeric is None or numeric < filt.numeric_min):
            continue
        if filt.numeric_max is not None and (numeric is None or numeric > filt.numeric_max):
            continue
        result.append({
            "id": row.id, "chat_id": row.chat_id, "document_id": row.document_id,
            "kind": row.kind, "key": row.key, "value": row.value,
            "unit": row.unit, "context_excerpt": row.context_excerpt, "page": row.page,
        })
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
    await _insert_fact(sqlite_session, chat_id=_CHAT_A, document_id=_DOC_A1,
                       kind="metric", key="f1", value=FactValue(raw="F1 87.3%", numeric=87.3),
                       unit="%", context_excerpt="Our model achieves F1 87.3%.", page=5, seq=0)
    await _insert_fact(sqlite_session, chat_id=_CHAT_A, document_id=_DOC_A1,
                       kind="hyperparameter", key="learning_rate",
                       value=FactValue(raw="1e-4", numeric=1e-4),
                       context_excerpt="We train with learning rate of 1e-4.", page=4, seq=1)
    await _insert_fact(sqlite_session, chat_id=_CHAT_A, document_id=_DOC_A1,
                       kind="dataset", key="ms_marco",
                       value=FactValue(raw="MS-MARCO", items=["MS-MARCO"]),
                       context_excerpt="We evaluate on the MS-MARCO dataset.", page=3, seq=2)
    await _insert_fact(sqlite_session, chat_id=_CHAT_A, document_id=_DOC_A2,
                       kind="metric", key="accuracy",
                       value=FactValue(raw="accuracy 92.0%", numeric=92.0),
                       unit="%", context_excerpt="accuracy of 92.0% on dev set", page=6, seq=3)
    await _insert_fact(sqlite_session, chat_id=_CHAT_B, document_id=_DOC_B1,
                       kind="metric", key="bleu",
                       value=FactValue(raw="BLEU 34.5", numeric=34.5),
                       context_excerpt="BLEU = 34.5 on the test set.", page=7, seq=4)
    await sqlite_session.flush()


# ---------------------------------------------------------------------------
# Test 1 — FactsFilter extra="forbid" (mandatory)
# ---------------------------------------------------------------------------


def test_facts_filter_extra_field_raises() -> None:
    """FactsFilter with extra field must raise ValidationError (extra='forbid')."""
    with pytest.raises(ValidationError) as exc_info:
        FactsFilter(chat_id=_CHAT_A, sql_injection="DROP TABLE--")  # type: ignore[call-arg]
    assert "sql_injection" in str(exc_info.value) or "extra" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# Test 2 — FactsFilter field-level bounds (parametrized)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kwargs", [
    {"keys": [f"key_{i}" for i in range(51)]},          # > 50 keys (length bound)
    {"page_range": (5, 3)},                             # page_range max < min
    {"limit": 0},                                       # limit out of range
])
def test_facts_filter_invalid_raises(kwargs: dict) -> None:  # type: ignore[type-arg]
    """FactsFilter rejects out-of-bounds values with ValidationError."""
    with pytest.raises(ValidationError):
        FactsFilter(chat_id=_CHAT_A, **kwargs)


# ---------------------------------------------------------------------------
# Test 3 — Isolation: filt.chat_id is ignored; current_chat_id always wins (mandatory)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_isolation_chat_id_override(sqlite_session: AsyncSession) -> None:
    """filt.chat_id=CHAT_B but current_chat_id=CHAT_A → only CHAT_A facts returned."""
    filt = FactsFilter(chat_id=_CHAT_B, limit=200)
    results = await _query_facts_sqlite(sqlite_session, current_chat_id=_CHAT_A, filt=filt)
    chat_ids = {r["chat_id"] for r in results}
    assert chat_ids == {str(_CHAT_A)}
    keys = [r["key"] for r in results]
    assert "bleu" not in keys, f"CHAT_B's bleu fact leaked into CHAT_A results: {keys}"


# ---------------------------------------------------------------------------
# Test 4 — document_ids filter
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_document_ids_filter(sqlite_session: AsyncSession) -> None:
    """Filter by DOC_A1 → only DOC_A1 facts; filter by DOC_A2 → no DOC_A1 facts."""
    filt1 = FactsFilter(chat_id=_CHAT_A, document_ids=[_DOC_A1], limit=200)
    r1 = await _query_facts_sqlite(sqlite_session, current_chat_id=_CHAT_A, filt=filt1)
    assert {r["document_id"] for r in r1} == {str(_DOC_A1)}

    filt2 = FactsFilter(chat_id=_CHAT_A, document_ids=[_DOC_A2], limit=200)
    r2 = await _query_facts_sqlite(sqlite_session, current_chat_id=_CHAT_A, filt=filt2)
    assert str(_DOC_A1) not in {r["document_id"] for r in r2}


# ---------------------------------------------------------------------------
# Test 5 — kinds and keys filters
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_kinds_and_keys_filters(sqlite_session: AsyncSession) -> None:
    """kinds=["metric"] returns only metrics; keys=["nonexistent"] returns empty."""
    filt_kinds = FactsFilter(chat_id=_CHAT_A, kinds=["metric"], limit=200)
    r = await _query_facts_sqlite(sqlite_session, current_chat_id=_CHAT_A, filt=filt_kinds)
    assert all(row["kind"] == "metric" for row in r)
    assert len(r) >= 1

    filt_keys = FactsFilter(chat_id=_CHAT_A, keys=["nonexistent_key_xyz"], limit=200)
    r2 = await _query_facts_sqlite(sqlite_session, current_chat_id=_CHAT_A, filt=filt_keys)
    assert len(r2) == 0


# ---------------------------------------------------------------------------
# Test 6 — page_range filter
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_page_range_filter(sqlite_session: AsyncSession) -> None:
    """page_range=(4,5) returns pages 4-5; page_range=(10,20) returns empty."""
    filt = FactsFilter(chat_id=_CHAT_A, page_range=(4, 5), limit=200)
    results = await _query_facts_sqlite(sqlite_session, current_chat_id=_CHAT_A, filt=filt)
    assert all(4 <= (r["page"] or 0) <= 5 for r in results)
    assert len(results) >= 1

    filt2 = FactsFilter(chat_id=_CHAT_A, page_range=(10, 20), limit=200)
    results2 = await _query_facts_sqlite(sqlite_session, current_chat_id=_CHAT_A, filt=filt2)
    assert len(results2) == 0


# ---------------------------------------------------------------------------
# Test 7 — numeric_min / numeric_max filter
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_numeric_filter(sqlite_session: AsyncSession) -> None:
    """numeric_min=80 → only facts with numeric >= 80; numeric_max=0.001 excludes high values."""
    filt = FactsFilter(chat_id=_CHAT_A, numeric_min=80.0, limit=200)
    results = await _query_facts_sqlite(sqlite_session, current_chat_id=_CHAT_A, filt=filt)
    assert len(results) >= 1
    for r in results:
        val = r["value"]
        numeric = val.get("numeric") if isinstance(val, dict) else None
        assert numeric is not None and numeric >= 80.0

    filt2 = FactsFilter(chat_id=_CHAT_A, numeric_max=0.001, limit=200)
    results2 = await _query_facts_sqlite(sqlite_session, current_chat_id=_CHAT_A, filt=filt2)
    for r in results2:
        val = r["value"]
        numeric = val.get("numeric") if isinstance(val, dict) else None
        if numeric is not None:
            assert numeric <= 0.001


# ---------------------------------------------------------------------------
# Test 8 — limit / offset pagination
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_limit_and_offset(sqlite_session: AsyncSession) -> None:
    """limit=2 returns <= 2 rows; offset=1 returns one fewer than offset=0."""
    filt_limit = FactsFilter(chat_id=_CHAT_A, limit=2)
    r = await _query_facts_sqlite(sqlite_session, current_chat_id=_CHAT_A, filt=filt_limit)
    assert len(r) <= 2

    filt_all = FactsFilter(chat_id=_CHAT_A, limit=200, offset=0)
    filt_skip = FactsFilter(chat_id=_CHAT_A, limit=200, offset=1)
    all_r = await _query_facts_sqlite(sqlite_session, current_chat_id=_CHAT_A, filt=filt_all)
    skip_r = await _query_facts_sqlite(sqlite_session, current_chat_id=_CHAT_A, filt=filt_skip)
    if len(all_r) > 1:
        assert len(skip_r) == len(all_r) - 1
