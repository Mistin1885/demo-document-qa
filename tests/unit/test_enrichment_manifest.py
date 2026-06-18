"""Unit tests for Phase 5.4 — Chat-level manifest.

Reduced to ≤10 tests covering: empty chat, dual-document order,
abstract_summary presence/absence, main_topics cap+dedup, ingestion
summary counts, isolation (Chat-B not visible to Chat-A), no raw SQL,
and section_count/available_source_types.
"""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timedelta

import pytest
from sqlalchemy import text as _sql_text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.enrichment.manifest import build_chat_manifest
from app.enrichment.models import ChatManifest

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SQLITE_URL = "sqlite+aiosqlite:///:memory:"

_CHAT_A = uuid.UUID("aaaaaaaa-0000-0000-0000-000000000000")
_CHAT_B = uuid.UUID("bbbbbbbb-0000-0000-0000-000000000000")

_DOC_A1 = uuid.UUID("00000000-a001-0000-0000-000000000000")
_DOC_A2 = uuid.UUID("00000000-a002-0000-0000-000000000000")
_DOC_B1 = uuid.UUID("00000000-b001-0000-0000-000000000000")

_BASE_TIME = datetime(2024, 1, 1, 0, 0, 0)


# ---------------------------------------------------------------------------
# UUID format helper
# ---------------------------------------------------------------------------

def _h(u: uuid.UUID) -> str:
    return u.hex


# ---------------------------------------------------------------------------
# SQLite DDL
# ---------------------------------------------------------------------------


def _all_ddl() -> list[str]:
    return [
        """
        CREATE TABLE IF NOT EXISTS chats (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY,
            chat_id TEXT NOT NULL,
            source_type TEXT NOT NULL DEFAULT 'upload',
            original_filename TEXT NOT NULL,
            storage_path TEXT NOT NULL DEFAULT '',
            mime_type TEXT NOT NULL DEFAULT 'application/pdf',
            page_count INTEGER,
            status TEXT NOT NULL DEFAULT 'uploaded',
            checksum_sha256 TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS chat_documents (
            chat_id TEXT NOT NULL,
            document_id TEXT NOT NULL,
            PRIMARY KEY (chat_id, document_id)
        )
        """,
        """
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
        """,
        """
        CREATE TABLE IF NOT EXISTS document_nodes (
            id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL,
            chat_id TEXT NOT NULL,
            parent_id TEXT,
            node_type TEXT NOT NULL,
            title TEXT,
            content TEXT NOT NULL DEFAULT '',
            page_start INTEGER NOT NULL DEFAULT 1,
            page_end INTEGER NOT NULL DEFAULT 1,
            order_index INTEGER NOT NULL DEFAULT 0,
            level INTEGER NOT NULL DEFAULT 0,
            bbox TEXT,
            metadata TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS structured_facts (
            id TEXT PRIMARY KEY,
            chat_id TEXT NOT NULL,
            document_id TEXT NOT NULL,
            source_node_id TEXT,
            kind TEXT NOT NULL,
            key TEXT NOT NULL,
            value TEXT NOT NULL DEFAULT '{}',
            unit TEXT,
            context_excerpt TEXT,
            page INTEGER,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS ingestion_jobs (
            id TEXT PRIMARY KEY,
            chat_id TEXT NOT NULL,
            document_id TEXT NOT NULL,
            state TEXT NOT NULL DEFAULT 'pending',
            attempt INTEGER NOT NULL DEFAULT 0,
            last_error TEXT,
            started_at TEXT,
            finished_at TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
        """,
    ]


# ---------------------------------------------------------------------------
# Session factory
# ---------------------------------------------------------------------------


async def _make_session() -> tuple[AsyncSession, object]:  # type: ignore[type-arg]
    engine = create_async_engine(SQLITE_URL, echo=False, future=True)
    async with engine.begin() as conn:
        for ddl in _all_ddl():
            await conn.execute(_sql_text(ddl))
    factory = async_sessionmaker(bind=engine, expire_on_commit=False, class_=AsyncSession)
    return factory(), engine


# ---------------------------------------------------------------------------
# Row-insert helpers
# ---------------------------------------------------------------------------


async def _insert_chat(session: AsyncSession, chat_id: uuid.UUID, name: str = "Test") -> None:
    await session.execute(
        _sql_text("INSERT OR IGNORE INTO chats (id, name) VALUES (:id, :name)"),
        {"id": _h(chat_id), "name": name},
    )


async def _insert_document(
    session: AsyncSession,
    *,
    doc_id: uuid.UUID,
    chat_id: uuid.UUID,
    filename: str = "paper.pdf",
    page_count: int | None = 10,
    created_at: datetime | None = None,
) -> None:
    ts = (created_at or _BASE_TIME).isoformat()
    await session.execute(
        _sql_text(
            "INSERT OR IGNORE INTO documents "
            "(id, chat_id, original_filename, page_count, created_at, updated_at) "
            "VALUES (:id, :chat_id, :fn, :pc, :ca, :ca)"
        ),
        {"id": _h(doc_id), "chat_id": _h(chat_id), "fn": filename, "pc": page_count, "ca": ts},
    )


async def _insert_chat_document(
    session: AsyncSession, *, chat_id: uuid.UUID, doc_id: uuid.UUID
) -> None:
    await session.execute(
        _sql_text(
            "INSERT OR IGNORE INTO chat_documents (chat_id, document_id) "
            "VALUES (:chat_id, :doc_id)"
        ),
        {"chat_id": _h(chat_id), "doc_id": _h(doc_id)},
    )


async def _insert_summary(
    session: AsyncSession,
    *,
    summary_id: uuid.UUID | None = None,
    chat_id: uuid.UUID,
    doc_id: uuid.UUID,
    kind: str,
    content: str,
    keywords: list[str] | None = None,
    entities: list[str] | None = None,
    token_count: int = 0,
    source_node_id: uuid.UUID | None = None,
) -> None:
    sid = _h(summary_id or uuid.uuid4())
    await session.execute(
        _sql_text(
            "INSERT OR IGNORE INTO summaries "
            "(id, chat_id, document_id, source_node_id, kind, content, keywords, entities, token_count) "
            "VALUES (:id, :cid, :did, :snid, :kind, :content, :kw, :ent, :tc)"
        ),
        {
            "id": sid,
            "cid": _h(chat_id),
            "did": _h(doc_id),
            "snid": _h(source_node_id) if source_node_id else None,
            "kind": kind,
            "content": content,
            "kw": json.dumps(keywords or []),
            "ent": json.dumps(entities or []),
            "tc": token_count,
        },
    )


async def _insert_document_node(
    session: AsyncSession,
    *,
    node_id: uuid.UUID | None = None,
    chat_id: uuid.UUID,
    doc_id: uuid.UUID,
    node_type: str,
    title: str | None = None,
    order_index: int = 0,
) -> None:
    nid = _h(node_id or uuid.uuid4())
    await session.execute(
        _sql_text(
            "INSERT OR IGNORE INTO document_nodes "
            "(id, document_id, chat_id, node_type, title, order_index) "
            "VALUES (:id, :did, :cid, :nt, :title, :oi)"
        ),
        {"id": nid, "did": _h(doc_id), "cid": _h(chat_id),
         "nt": node_type, "title": title, "oi": order_index},
    )


async def _insert_sf(
    session: AsyncSession,
    *,
    chat_id: uuid.UUID,
    doc_id: uuid.UUID,
    kind: str,
    key: str = "acc",
) -> None:
    await session.execute(
        _sql_text(
            "INSERT OR IGNORE INTO structured_facts "
            "(id, chat_id, document_id, kind, key, value) "
            "VALUES (:id, :cid, :did, :kind, :key, :val)"
        ),
        {
            "id": _h(uuid.uuid4()),
            "cid": _h(chat_id),
            "did": _h(doc_id),
            "kind": kind,
            "key": key,
            "val": json.dumps({"raw": "42"}),
        },
    )


async def _insert_ingestion_job(
    session: AsyncSession,
    *,
    chat_id: uuid.UUID,
    doc_id: uuid.UUID,
    state: str,
    created_at: datetime | None = None,
) -> None:
    ts = (created_at or _BASE_TIME).isoformat()
    await session.execute(
        _sql_text(
            "INSERT OR IGNORE INTO ingestion_jobs "
            "(id, chat_id, document_id, state, created_at) "
            "VALUES (:id, :cid, :did, :state, :ca)"
        ),
        {"id": _h(uuid.uuid4()), "cid": _h(chat_id), "did": _h(doc_id), "state": state, "ca": ts},
    )


# ---------------------------------------------------------------------------
# Test 1 — empty Chat
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_chat_manifest() -> None:
    """Empty chat yields ChatManifest(documents=[], document_count=0, ingestion_summary={})."""
    session, engine = await _make_session()
    try:
        await _insert_chat(session, _CHAT_A)
        await session.flush()
        manifest = await build_chat_manifest(session, current_chat_id=_CHAT_A)
        assert isinstance(manifest, ChatManifest)
        assert manifest.chat_id == _CHAT_A
        assert manifest.document_count == 0
        assert manifest.documents == []
        assert manifest.total_token_estimate == 0
        assert manifest.ingestion_summary == {}
    finally:
        await session.close()
        await engine.dispose()  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Test 2 — dual-document chat: count = 2, deterministic order
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_two_document_manifest_order() -> None:
    """Two-document chat yields 2 entries ordered by created_at asc."""
    session, engine = await _make_session()
    try:
        await _insert_chat(session, _CHAT_A)
        await _insert_document(session, doc_id=_DOC_A1, chat_id=_CHAT_A, filename="first.pdf",
                               created_at=_BASE_TIME)
        await _insert_document(session, doc_id=_DOC_A2, chat_id=_CHAT_A, filename="second.pdf",
                               created_at=_BASE_TIME + timedelta(seconds=1))
        await _insert_chat_document(session, chat_id=_CHAT_A, doc_id=_DOC_A1)
        await _insert_chat_document(session, chat_id=_CHAT_A, doc_id=_DOC_A2)
        await session.flush()
        manifest = await build_chat_manifest(session, current_chat_id=_CHAT_A)
        assert manifest.document_count == 2
        assert manifest.documents[0].document_id == _DOC_A1
        assert manifest.documents[1].document_id == _DOC_A2
    finally:
        await session.close()
        await engine.dispose()  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Test 3 — abstract_summary: present from chapter_summary, None when absent
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("include_chapter_summary,expected", [
    (True, "Abstract of the paper."),
    (False, None),
])
async def test_abstract_summary(include_chapter_summary: bool, expected: str | None) -> None:
    """abstract_summary equals chapter_summary content or None when absent."""
    session, engine = await _make_session()
    try:
        await _insert_chat(session, _CHAT_A)
        await _insert_document(session, doc_id=_DOC_A1, chat_id=_CHAT_A)
        await _insert_chat_document(session, chat_id=_CHAT_A, doc_id=_DOC_A1)
        await _insert_summary(session, chat_id=_CHAT_A, doc_id=_DOC_A1,
                              kind="document_overview", content="Overview.", token_count=50)
        if include_chapter_summary:
            await _insert_summary(session, chat_id=_CHAT_A, doc_id=_DOC_A1,
                                  kind="chapter_summary", content="Abstract of the paper.",
                                  token_count=10)
        await session.flush()
        manifest = await build_chat_manifest(session, current_chat_id=_CHAT_A)
        entry = manifest.documents[0]
        assert entry.abstract_summary == expected
    finally:
        await session.close()
        await engine.dispose()  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Test 4 — main_topics <= 8, deduplicated
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_main_topics_capped_and_deduped() -> None:
    """main_topics has at most 8 entries and no duplicates."""
    session, engine = await _make_session()
    try:
        await _insert_chat(session, _CHAT_A)
        await _insert_document(session, doc_id=_DOC_A1, chat_id=_CHAT_A)
        await _insert_chat_document(session, chat_id=_CHAT_A, doc_id=_DOC_A1)
        many_kw = ["kw1", "kw2", "kw3", "kw4", "kw5", "kw6"]
        many_ent = ["kw1", "ent2", "ent3", "ent4", "ent5", "ent6"]
        await _insert_summary(session, chat_id=_CHAT_A, doc_id=_DOC_A1,
                              kind="document_overview", content="Overview.",
                              keywords=many_kw, entities=many_ent, token_count=100)
        await session.flush()
        manifest = await build_chat_manifest(session, current_chat_id=_CHAT_A)
        entry = manifest.documents[0]
        assert len(entry.main_topics) <= 8
        assert len(entry.main_topics) == len(set(entry.main_topics))
    finally:
        await session.close()
        await engine.dispose()  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Test 5 — ingestion_summary counts: latest-per-document state
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ingestion_summary_counts() -> None:
    """ingestion_summary reflects latest-per-document state counts."""
    session, engine = await _make_session()
    try:
        await _insert_chat(session, _CHAT_A)
        # doc_a1: older=failed, newer=succeeded → latest = succeeded
        await _insert_document(session, doc_id=_DOC_A1, chat_id=_CHAT_A)
        await _insert_chat_document(session, chat_id=_CHAT_A, doc_id=_DOC_A1)
        await _insert_ingestion_job(session, chat_id=_CHAT_A, doc_id=_DOC_A1, state="failed",
                                    created_at=_BASE_TIME)
        await _insert_ingestion_job(session, chat_id=_CHAT_A, doc_id=_DOC_A1, state="succeeded",
                                    created_at=_BASE_TIME + timedelta(seconds=10))
        # doc_a2: one job = pending
        await _insert_document(session, doc_id=_DOC_A2, chat_id=_CHAT_A,
                               created_at=_BASE_TIME + timedelta(seconds=1))
        await _insert_chat_document(session, chat_id=_CHAT_A, doc_id=_DOC_A2)
        await _insert_ingestion_job(session, chat_id=_CHAT_A, doc_id=_DOC_A2, state="pending",
                                    created_at=_BASE_TIME)
        await session.flush()
        manifest = await build_chat_manifest(session, current_chat_id=_CHAT_A)
        assert manifest.ingestion_summary.get("succeeded") == 1
        assert manifest.ingestion_summary.get("pending") == 1
        assert manifest.ingestion_summary.get("failed", 0) == 0
    finally:
        await session.close()
        await engine.dispose()  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Test 6 — isolation: Chat-B data does not appear in Chat-A manifest (mandatory)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_isolation_chat_b_not_visible_to_chat_a() -> None:
    """build_chat_manifest(current_chat_id=_CHAT_A) must not expose Chat-B data."""
    session, engine = await _make_session()
    try:
        await _insert_chat(session, _CHAT_A, "Chat A")
        await _insert_document(session, doc_id=_DOC_A1, chat_id=_CHAT_A, filename="a1.pdf")
        await _insert_chat_document(session, chat_id=_CHAT_A, doc_id=_DOC_A1)
        await _insert_summary(session, chat_id=_CHAT_A, doc_id=_DOC_A1,
                              kind="document_overview", content="Chat A document.",
                              keywords=["topic_a"], token_count=100)
        await _insert_ingestion_job(session, chat_id=_CHAT_A, doc_id=_DOC_A1, state="succeeded")

        await _insert_chat(session, _CHAT_B, "Chat B")
        await _insert_document(session, doc_id=_DOC_B1, chat_id=_CHAT_B, filename="b1.pdf")
        await _insert_chat_document(session, chat_id=_CHAT_B, doc_id=_DOC_B1)
        await _insert_summary(session, chat_id=_CHAT_B, doc_id=_DOC_B1,
                              kind="document_overview", content="Chat B secret document.",
                              keywords=["secret_topic"], token_count=999)
        await _insert_summary(session, chat_id=_CHAT_B, doc_id=_DOC_B1,
                              kind="chapter_summary", content="Secret abstract.", token_count=50)
        await _insert_sf(session, chat_id=_CHAT_B, doc_id=_DOC_B1, kind="metric")
        await _insert_ingestion_job(session, chat_id=_CHAT_B, doc_id=_DOC_B1, state="running")
        await session.flush()

        manifest = await build_chat_manifest(session, current_chat_id=_CHAT_A)

        assert manifest.document_count == 1
        doc_ids = {e.document_id for e in manifest.documents}
        assert _DOC_B1 not in doc_ids
        entry = manifest.documents[0]
        assert "secret_topic" not in entry.main_topics
        assert entry.abstract_summary is None
        assert manifest.ingestion_summary == {"succeeded": 1}
    finally:
        await session.close()
        await engine.dispose()  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Test 7 — no raw SQL in production code
# ---------------------------------------------------------------------------


def test_no_raw_sql_in_production_code() -> None:
    """Production manifest.py must not contain text() calls or f-string SQL."""
    import pathlib

    manifest_path = (
        pathlib.Path(__file__).parent.parent.parent
        / "src" / "app" / "enrichment" / "manifest.py"
    )
    source = manifest_path.read_text()
    stripped = re.sub(r'""".*?"""', "", source, flags=re.DOTALL)
    stripped = re.sub(r"'''.*?'''", "", stripped, flags=re.DOTALL)
    stripped = re.sub(r"#.*", "", stripped)
    assert "text(" not in stripped
    fstring_sql = re.search(
        r'f["\'].*\b(SELECT|INSERT|UPDATE|DELETE|WHERE)\b', stripped, re.IGNORECASE
    )
    assert fstring_sql is None


# ---------------------------------------------------------------------------
# Test 8 — section_count and available_source_types
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_section_count_and_source_types() -> None:
    """section_count = 2 sections + 1 subsection; available_source_types includes fact:metric."""
    session, engine = await _make_session()
    try:
        await _insert_chat(session, _CHAT_A)
        await _insert_document(session, doc_id=_DOC_A1, chat_id=_CHAT_A)
        await _insert_chat_document(session, chat_id=_CHAT_A, doc_id=_DOC_A1)
        await _insert_document_node(session, chat_id=_CHAT_A, doc_id=_DOC_A1,
                                    node_type="document", title="My Paper", order_index=0)
        await _insert_document_node(session, chat_id=_CHAT_A, doc_id=_DOC_A1,
                                    node_type="section", title="Intro", order_index=1)
        await _insert_document_node(session, chat_id=_CHAT_A, doc_id=_DOC_A1,
                                    node_type="subsection", title="Background", order_index=2)
        await _insert_document_node(session, chat_id=_CHAT_A, doc_id=_DOC_A1,
                                    node_type="section", title="Conclusion", order_index=3)
        await _insert_summary(session, chat_id=_CHAT_A, doc_id=_DOC_A1,
                              kind="document_overview", content="Overview.", token_count=50)
        await _insert_sf(session, chat_id=_CHAT_A, doc_id=_DOC_A1, kind="metric")
        await session.flush()
        manifest = await build_chat_manifest(session, current_chat_id=_CHAT_A)
        entry = manifest.documents[0]
        assert entry.section_count == 3
        assert "document_overview" in entry.available_source_types
        assert "fact:metric" in entry.available_source_types
        assert entry.title == "My Paper"
    finally:
        await session.close()
        await engine.dispose()  # type: ignore[union-attr]
