"""Session service — domain logic for Session CRUD.

Isolation contract (CLAUDE.md §2)
----------------------------------
- Every query that touches Session rows **must** filter on ``chat_id``.
- The service is the single point where this filter is injected; callers
  (routers, agents, tools) MUST NOT bypass the service.
- Session history is never shared across Sessions; the service only exposes
  metadata (name, profile), not message content.

Rules (CLAUDE.md §12)
----------------------
- All public functions have full type hints.
- No SQL strings — only ORM expressions.
- No FastAPI dependency; the ``session`` argument is an ``AsyncSession``
  injected by the caller.
- No logging of session content.
"""

from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.errors import ChatNotFound, SessionNotFound
from app.models.domain import MessageRead, SessionCreate, SessionRead, SessionUpdate
from app.models.orm import Chat, Document, Message
from app.models.orm import Session as SessionORM


async def _require_chat(db: AsyncSession, chat_id: uuid.UUID) -> Chat:
    """Fetch a Chat by PK; raise :exc:`ChatNotFound` if absent."""
    result = await db.execute(select(Chat).where(Chat.id == chat_id))
    chat = result.scalar_one_or_none()
    if chat is None:
        raise ChatNotFound(chat_id)
    return chat


async def _require_session(
    db: AsyncSession,
    *,
    chat_id: uuid.UUID,
    session_id: uuid.UUID,
) -> SessionORM:
    """Fetch a Session scoped to ``chat_id``; raise :exc:`SessionNotFound` if absent.

    The WHERE clause enforces isolation: a session belonging to a different chat
    is indistinguishable from a non-existent session.
    """
    result = await db.execute(
        select(SessionORM).where(
            SessionORM.id == session_id,
            SessionORM.chat_id == chat_id,
        )
    )
    orm_session = result.scalar_one_or_none()
    if orm_session is None:
        raise SessionNotFound(session_id, chat_id)
    return orm_session


async def _validate_document_scope(
    db: AsyncSession,
    *,
    chat_id: uuid.UUID,
    document_ids: list[uuid.UUID] | None,
) -> list[str] | None:
    if document_ids is None:
        return None
    unique_ids = list(dict.fromkeys(document_ids))
    if not unique_ids:
        return []
    rows = (
        await db.scalars(
            select(Document.id).where(
                Document.chat_id == chat_id,
                Document.id.in_(unique_ids),
            )
        )
    ).all()
    found = set(rows)
    missing = [str(doc_id) for doc_id in unique_ids if doc_id not in found]
    if missing:
        raise ValueError(f"document scope contains documents outside this chat: {missing}")
    return [str(doc_id) for doc_id in unique_ids]


async def create_session(
    db: AsyncSession,
    *,
    chat_id: uuid.UUID,
    data: SessionCreate,
) -> SessionRead:
    """Create a new Session under ``chat_id``.

    Raises
    ------
    ChatNotFound
        If no Chat with the given ``chat_id`` exists.
    """
    await _require_chat(db, chat_id)

    orm_session = SessionORM(
        chat_id=chat_id,
        name=data.name,
        chat_profile_id=data.chat_profile_id,
        selected_document_ids=await _validate_document_scope(
            db,
            chat_id=chat_id,
            document_ids=data.selected_document_ids,
        ),
    )
    db.add(orm_session)
    await db.flush()  # populate server-generated defaults without committing
    await db.refresh(orm_session)
    return SessionRead.model_validate(orm_session)


async def get_session_by_id(
    db: AsyncSession,
    *,
    chat_id: uuid.UUID,
    session_id: uuid.UUID,
) -> SessionRead:
    """Return a Session by ID, scoped to ``chat_id``.

    Raises
    ------
    SessionNotFound
        If no Session with ``session_id`` exists under ``chat_id``.
        This is also raised when the session exists but belongs to a
        *different* chat, preserving isolation.
    """
    orm_session = await _require_session(db, chat_id=chat_id, session_id=session_id)
    return SessionRead.model_validate(orm_session)


async def list_sessions(
    db: AsyncSession,
    *,
    chat_id: uuid.UUID,
    limit: int = 50,
    offset: int = 0,
) -> list[SessionRead]:
    """List Sessions belonging to ``chat_id``, ordered by ``updated_at`` desc.

    Raises
    ------
    ChatNotFound
        If no Chat with the given ``chat_id`` exists.
    """
    await _require_chat(db, chat_id)

    result = await db.execute(
        select(SessionORM)
        .where(SessionORM.chat_id == chat_id)
        .order_by(SessionORM.updated_at.desc())
        .limit(limit)
        .offset(offset)
    )
    rows = result.scalars().all()
    return [SessionRead.model_validate(row) for row in rows]


async def update_session(
    db: AsyncSession,
    *,
    chat_id: uuid.UUID,
    session_id: uuid.UUID,
    data: SessionUpdate,
) -> SessionRead:
    """Partially update a Session; verify chat-scope before writing.

    Raises
    ------
    SessionNotFound
        If the session does not exist within ``chat_id``.
    """
    orm_session = await _require_session(db, chat_id=chat_id, session_id=session_id)

    update_data = data.model_dump(exclude_unset=True)
    if "selected_document_ids" in update_data:
        if orm_session.document_scope_locked:
            raise ValueError("document scope is locked after the first QA")
        update_data["selected_document_ids"] = await _validate_document_scope(
            db,
            chat_id=chat_id,
            document_ids=data.selected_document_ids,
        )
    for field, value in update_data.items():
        setattr(orm_session, field, value)

    await db.flush()
    await db.refresh(orm_session)
    return SessionRead.model_validate(orm_session)


async def lock_document_scope_for_qa(
    db: AsyncSession,
    *,
    chat_id: uuid.UUID,
    session_id: uuid.UUID,
    requested_document_ids: list[uuid.UUID] | None,
) -> list[uuid.UUID]:
    """Lock and return the effective document scope for a session QA.

    The first QA fixes the scope. If the caller does not provide a scope, all
    current chat documents are selected and locked. Later QAs ignore new
    request scopes and reuse the locked value.
    """
    orm_session = await _require_session(db, chat_id=chat_id, session_id=session_id)

    if orm_session.document_scope_locked:
        return [uuid.UUID(str(doc_id)) for doc_id in (orm_session.selected_document_ids or [])]

    effective: list[str]
    if requested_document_ids is None:
        rows = (
            await db.scalars(
                select(Document.id)
                .where(Document.chat_id == chat_id)
                .order_by(Document.created_at)
            )
        ).all()
        effective = [str(doc_id) for doc_id in rows]
    else:
        effective = await _validate_document_scope(
            db,
            chat_id=chat_id,
            document_ids=requested_document_ids,
        ) or []

    orm_session.selected_document_ids = effective
    orm_session.document_scope_locked = True
    await db.flush()
    return [uuid.UUID(str(doc_id)) for doc_id in (effective or [])]


async def delete_session(
    db: AsyncSession,
    *,
    chat_id: uuid.UUID,
    session_id: uuid.UUID,
) -> None:
    """Delete a Session; verify chat-scope before deleting.

    Raises
    ------
    SessionNotFound
        If the session does not exist within ``chat_id``.
    """
    orm_session = await _require_session(db, chat_id=chat_id, session_id=session_id)
    await db.delete(orm_session)
    await db.flush()


async def list_messages(
    db: AsyncSession,
    *,
    session_id: uuid.UUID,
    chat_id: uuid.UUID,
    limit: int = 200,
    offset: int = 0,
) -> list[MessageRead]:
    """Return messages for ``session_id`` scoped to ``chat_id``.

    The query joins through ``Session`` to verify ``Session.chat_id == chat_id``,
    ensuring session history from a different chat cannot be read.

    Raises
    ------
    SessionNotFound
        If the session does not exist within ``chat_id``.
    """
    # Verify scope first (raises SessionNotFound if absent or wrong chat)
    await _require_session(db, chat_id=chat_id, session_id=session_id)

    stmt = (
        select(Message)
        .join(SessionORM, SessionORM.id == Message.session_id)
        .where(
            Message.session_id == session_id,
            SessionORM.chat_id == chat_id,
        )
        .order_by(Message.created_at)
        .limit(limit)
        .offset(offset)
    )
    rows = (await db.scalars(stmt)).all()
    return [MessageRead.model_validate(row) for row in rows]


__all__ = [
    "create_session",
    "get_session_by_id",
    "list_sessions",
    "update_session",
    "delete_session",
    "list_messages",
    "lock_document_scope_for_qa",
]
