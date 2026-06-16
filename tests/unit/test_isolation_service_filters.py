"""Unit-level isolation tests — service layer filter enforcement.

CLAUDE.md §2 isolation layers covered
--------------------------------------
- **Repository / Service layer**: the single trusted injection point.
  All public service functions that accept ``chat_id`` must enforce it via
  WHERE clauses before returning any data.  Callers (routes, agents) MUST NOT
  bypass the service.

These tests call service functions directly (no HTTP) and assert that
cross-chat lookups raise the appropriate ``*NotFound`` error.

Scenarios covered
-----------------
- (j) ``document_service.get_document(session, chat_id=B, document_id=X)``
      raises ``DocumentNotFound`` even though Chat A truly owns X.
- (k) ``session_service.get_session_by_id(session, chat_id=B, session_id=S)``
      raises ``SessionNotFound`` even though Chat A truly owns S.
- Service list functions: list scoped to Chat B never returns Chat A's items.
- (n) Message ORM session isolation: a SELECT filtered by session_a.id never
      returns messages that belong to session_b (regression gate).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.errors import DocumentNotFound, SessionNotFound
from app.models.orm import Message
from app.services import document_service, session_service
from tests._helpers import SpyIndexer, make_chat, make_document, make_session

# ---------------------------------------------------------------------------
# (j) document_service.get_document — cross-chat raises DocumentNotFound
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_document_cross_chat_raises_document_not_found(
    db_session: AsyncSession,
    tmp_path: Path,
) -> None:
    """(j) document_service.get_document(chat_id=B, doc_id=X_from_A)
    raises DocumentNotFound — layer: Service filter.

    Even though document X truly exists in the DB under Chat A, the service
    layer enforces the chat_id constraint and refuses to return it when called
    with Chat B's id.
    """
    chat_a = await make_chat(db_session, "A-service-doc-j")
    chat_b = await make_chat(db_session, "B-service-doc-j")
    doc_x = await make_document(db_session, chat_a, tmp_path=tmp_path, filename="x.pdf")

    with pytest.raises(DocumentNotFound) as exc_info:
        await document_service.get_document(db_session, chat_b.id, doc_x.id)

    # Error message must reference the document id and the caller chat id
    assert str(doc_x.id) in str(exc_info.value)
    assert str(chat_b.id) in str(exc_info.value)


# ---------------------------------------------------------------------------
# (k) session_service.get_session_by_id — cross-chat raises SessionNotFound
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_session_cross_chat_raises_session_not_found(
    db_session: AsyncSession,
) -> None:
    """(k) session_service.get_session_by_id(chat_id=B, session_id=S_from_A)
    raises SessionNotFound — layer: Service filter.

    Session S truly exists under Chat A.  Calling with Chat B's scope must
    raise SessionNotFound without revealing that the session exists at all.
    """
    chat_a = await make_chat(db_session, "A-service-sess-k")
    chat_b = await make_chat(db_session, "B-service-sess-k")
    session_a = await make_session(db_session, chat_a, name="secret-session")

    with pytest.raises(SessionNotFound) as exc_info:
        await session_service.get_session_by_id(
            db_session, chat_id=chat_b.id, session_id=session_a.id
        )

    assert str(session_a.id) in str(exc_info.value)
    assert str(chat_b.id) in str(exc_info.value)


# ---------------------------------------------------------------------------
# Service list — cross-chat items never leak into list results
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_documents_service_never_leaks_cross_chat(
    db_session: AsyncSession,
    tmp_path: Path,
) -> None:
    """document_service.list_documents(chat_id=B) excludes Chat A's documents.

    Layer: Service / DB query (WHERE chat_documents.chat_id = :chat_id).
    """
    chat_a = await make_chat(db_session, "A-svc-list-doc")
    chat_b = await make_chat(db_session, "B-svc-list-doc")

    await make_document(
        db_session, chat_a, tmp_path=tmp_path, filename="a_only.pdf"
    )

    docs_b = await document_service.list_documents(db_session, chat_b.id)

    ids_returned = {d.id for d in docs_b}
    assert len(docs_b) == 0, (
        "list_documents(chat_b) must return an empty list when Chat B owns nothing; "
        f"got {ids_returned}"
    )


@pytest.mark.asyncio
async def test_list_sessions_service_never_leaks_cross_chat(
    db_session: AsyncSession,
) -> None:
    """session_service.list_sessions(chat_id=B) excludes Chat A's sessions.

    Layer: Service / DB query (WHERE sessions.chat_id = :chat_id).
    """
    chat_a = await make_chat(db_session, "A-svc-list-sess")
    chat_b = await make_chat(db_session, "B-svc-list-sess")

    await make_session(db_session, chat_a, name="a-sess-1")
    await make_session(db_session, chat_a, name="a-sess-2")

    sessions_b = await session_service.list_sessions(db_session, chat_id=chat_b.id)

    assert sessions_b == [], (
        "list_sessions(chat_b) must not return Chat A's sessions; "
        f"got {[s.id for s in sessions_b]}"
    )


# ---------------------------------------------------------------------------
# (n) Message ORM — session-level isolation via ORM filter
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_message_orm_session_filter_isolates_history(
    db_session: AsyncSession,
) -> None:
    """(n) SELECT messages WHERE session_id = sess_a.id never returns sess_b's messages.

    This is an ORM / DB-layer regression gate.  Phase 7 will add a message
    service; this test validates the underlying ORM contract so that higher
    layers can rely on it.

    Layer: Relational DB query (WHERE session_id = :session_id).
    """
    chat = await make_chat(db_session, "chat-msg-isolation")
    sess_a = await make_session(db_session, chat, name="sess-a")
    sess_b = await make_session(db_session, chat, name="sess-b")

    # Seed one message per session directly via ORM
    msg_a = Message(
        session_id=sess_a.id,
        role="user",
        content="message from session A",
    )
    msg_b = Message(
        session_id=sess_b.id,
        role="user",
        content="message from session B",
    )
    db_session.add(msg_a)
    db_session.add(msg_b)
    await db_session.flush()

    # Query scoped to session A only
    result = await db_session.scalars(
        select(Message).where(Message.session_id == sess_a.id)
    )
    rows = list(result.all())

    assert len(rows) == 1, (
        f"Expected exactly 1 message for session A, got {len(rows)}"
    )
    assert rows[0].session_id == sess_a.id
    assert rows[0].content == "message from session A"

    # Explicitly assert session B's message is absent
    session_ids_returned = {r.session_id for r in rows}
    assert sess_b.id not in session_ids_returned, (
        "Session B's message must never appear in a query filtered to session A"
    )


@pytest.mark.asyncio
async def test_message_orm_query_scoped_to_session_b_excludes_session_a(
    db_session: AsyncSession,
) -> None:
    """Symmetric test: session B query must not return session A's messages.

    Layer: Relational DB query (WHERE session_id = :session_id).
    """
    chat = await make_chat(db_session, "chat-msg-iso-b")
    sess_a = await make_session(db_session, chat, name="sess-a-sym")
    sess_b = await make_session(db_session, chat, name="sess-b-sym")

    msg_a = Message(
        session_id=sess_a.id,
        role="assistant",
        content="reply in session A",
    )
    msg_b = Message(
        session_id=sess_b.id,
        role="assistant",
        content="reply in session B",
    )
    db_session.add(msg_a)
    db_session.add(msg_b)
    await db_session.flush()

    result = await db_session.scalars(
        select(Message).where(Message.session_id == sess_b.id)
    )
    rows = list(result.all())

    assert len(rows) == 1
    assert rows[0].session_id == sess_b.id
    session_ids_returned = {r.session_id for r in rows}
    assert sess_a.id not in session_ids_returned


# ---------------------------------------------------------------------------
# Document service: delete cross-chat raises DocumentNotFound (service layer)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delete_document_cross_chat_raises_document_not_found(
    db_session: AsyncSession,
    tmp_path: Path,
) -> None:
    """document_service.delete_document(chat_id=B, doc_id=X_from_A) raises DocumentNotFound.

    Layer: Service filter — cannot delete a document outside the caller's scope.
    The document must remain intact under Chat A after the failed attempt.
    """
    chat_a = await make_chat(db_session, "A-svc-del-cross")
    chat_b = await make_chat(db_session, "B-svc-del-cross")
    doc_x = await make_document(db_session, chat_a, tmp_path=tmp_path, filename="keep.pdf")
    spy = SpyIndexer()
    storage = __import__("app.storage.local", fromlist=["LocalBlobStorage"]).LocalBlobStorage(
        root=tmp_path
    )

    with pytest.raises(DocumentNotFound):
        await document_service.delete_document(
            db_session, chat_b.id, doc_x.id, storage=storage, indexer=spy
        )

    # indexer must NOT have been called
    assert spy.call_count == 0, "indexer.delete_by_document must not be called on a failed delete"

    # document still retrievable via Chat A
    still_there = await document_service.get_document(db_session, chat_a.id, doc_x.id)
    assert still_there.id == doc_x.id
