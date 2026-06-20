"""Session document-scope locking tests (≤10 tests)."""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.domain import SessionCreate
from app.models.orm import Chat, Document
from app.services import session_service


def _doc(chat_id: uuid.UUID, filename: str) -> Document:
    return Document(
        chat_id=chat_id,
        source_type="upload",
        original_filename=filename,
        storage_path=f"/tmp/{filename}",
        mime_type="application/pdf",
        checksum_sha256=uuid.uuid4().hex + uuid.uuid4().hex,
        status="indexed",
    )


@pytest.mark.asyncio
async def test_first_qa_locks_requested_document_scope(db_session: AsyncSession) -> None:
    chat = Chat(name="scope chat")
    db_session.add(chat)
    await db_session.flush()
    doc_a = _doc(chat.id, "a.pdf")
    doc_b = _doc(chat.id, "b.pdf")
    db_session.add_all([doc_a, doc_b])
    await db_session.flush()
    session = await session_service.create_session(
        db_session,
        chat_id=chat.id,
        data=SessionCreate(chat_id=chat.id, name="s"),
    )

    selected = await session_service.lock_document_scope_for_qa(
        db_session,
        chat_id=chat.id,
        session_id=session.id,
        requested_document_ids=[doc_b.id],
    )

    assert selected == [doc_b.id]
    locked = await session_service.get_session_by_id(
        db_session,
        chat_id=chat.id,
        session_id=session.id,
    )
    assert locked.document_scope_locked is True
    assert locked.selected_document_ids == [doc_b.id]


@pytest.mark.asyncio
async def test_locked_scope_ignores_later_request(db_session: AsyncSession) -> None:
    chat = Chat(name="scope chat 2")
    db_session.add(chat)
    await db_session.flush()
    doc_a = _doc(chat.id, "a.pdf")
    doc_b = _doc(chat.id, "b.pdf")
    db_session.add_all([doc_a, doc_b])
    await db_session.flush()
    session = await session_service.create_session(
        db_session,
        chat_id=chat.id,
        data=SessionCreate(chat_id=chat.id, name="s"),
    )

    await session_service.lock_document_scope_for_qa(
        db_session,
        chat_id=chat.id,
        session_id=session.id,
        requested_document_ids=[doc_a.id],
    )
    selected = await session_service.lock_document_scope_for_qa(
        db_session,
        chat_id=chat.id,
        session_id=session.id,
        requested_document_ids=[doc_b.id],
    )

    assert selected == [doc_a.id]


@pytest.mark.asyncio
async def test_scope_rejects_documents_outside_chat(db_session: AsyncSession) -> None:
    chat_a = Chat(name="scope chat A")
    chat_b = Chat(name="scope chat B")
    db_session.add_all([chat_a, chat_b])
    await db_session.flush()
    doc_b = _doc(chat_b.id, "b.pdf")
    db_session.add(doc_b)
    await db_session.flush()
    session = await session_service.create_session(
        db_session,
        chat_id=chat_a.id,
        data=SessionCreate(chat_id=chat_a.id, name="s"),
    )

    with pytest.raises(ValueError):
        await session_service.lock_document_scope_for_qa(
            db_session,
            chat_id=chat_a.id,
            session_id=session.id,
            requested_document_ids=[doc_b.id],
        )
