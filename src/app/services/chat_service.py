"""Chat service — pure async domain logic for the Chat resource.

Design contract (CLAUDE.md §12)
--------------------------------
- No FastAPI imports; no HTTP-specific types.
- All public functions are async and accept an ``AsyncSession``.
- All inputs/outputs are domain models (``ChatCreate / ChatUpdate / ChatRead``)
  or primitive types.
- ORM queries use SQLAlchemy Core/ORM expressions — no raw SQL strings.
- ``ChatNotFound`` is raised (not swallowed) so the API layer can map it to 404.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.errors import ChatNotFound
from app.models.domain import ChatCreate, ChatRead, ChatUpdate
from app.models.orm import Chat


async def create_chat(session: AsyncSession, data: ChatCreate) -> ChatRead:
    """Persist a new Chat and return its read model.

    Args:
        session: Active async DB session (caller commits / rolls back).
        data: Validated creation payload.

    Returns:
        ``ChatRead`` populated from the newly-created ORM row.
    """
    now = datetime.now(UTC).replace(tzinfo=None)
    chat = Chat(
        name=data.name,
        description=data.description,
        default_chat_profile_id=data.default_chat_profile_id,
        default_embedding_profile_id=data.default_embedding_profile_id,
        default_reranker_profile_id=data.default_reranker_profile_id,
        created_at=now,
        updated_at=now,
    )
    session.add(chat)
    await session.flush()  # get server-generated id + timestamps without committing
    await session.refresh(chat)
    return ChatRead.model_validate(chat)


async def get_chat(session: AsyncSession, chat_id: uuid.UUID) -> ChatRead:
    """Return the Chat read model for the given ``chat_id``.

    Args:
        session: Active async DB session.
        chat_id: Primary key of the Chat to retrieve.

    Raises:
        ChatNotFound: When no Chat row exists for ``chat_id``.
    """
    result = await session.scalar(select(Chat).where(Chat.id == chat_id))
    if result is None:
        raise ChatNotFound(chat_id)
    return ChatRead.model_validate(result)


async def list_chats(
    session: AsyncSession,
    *,
    limit: int = 50,
    offset: int = 0,
) -> list[ChatRead]:
    """Return a paginated list of Chats ordered by ``updated_at`` descending.

    Args:
        session: Active async DB session.
        limit: Maximum rows to return (default 50).
        offset: Number of rows to skip (default 0).

    Returns:
        List of ``ChatRead`` instances, possibly empty.
    """
    stmt = select(Chat).order_by(desc(Chat.updated_at)).limit(limit).offset(offset)
    rows = await session.scalars(stmt)
    return [ChatRead.model_validate(row) for row in rows.all()]


async def update_chat(
    session: AsyncSession,
    chat_id: uuid.UUID,
    data: ChatUpdate,
) -> ChatRead:
    """Apply a partial update to an existing Chat.

    Only fields explicitly set (non-None) in ``data`` are written.  This
    allows partial-patch semantics — unset fields are left unchanged.

    Args:
        session: Active async DB session.
        chat_id: Primary key of the Chat to update.
        data: Validated patch payload (all fields are Optional).

    Raises:
        ChatNotFound: When no Chat row exists for ``chat_id``.
    """
    chat = await session.scalar(select(Chat).where(Chat.id == chat_id))
    if chat is None:
        raise ChatNotFound(chat_id)

    update_fields = data.model_dump(exclude_unset=True)
    for field, value in update_fields.items():
        setattr(chat, field, value)
    chat.updated_at = datetime.now(UTC).replace(tzinfo=None)

    await session.flush()
    await session.refresh(chat)
    return ChatRead.model_validate(chat)


async def delete_chat(session: AsyncSession, chat_id: uuid.UUID) -> None:
    """Delete a Chat and all cascade-dependent records.

    ORM cascade rules (``all, delete-orphan``) handle Sessions, Documents,
    Summaries, StructuredFacts, IngestionJobs, and ChatDocument associations.

    Args:
        session: Active async DB session.
        chat_id: Primary key of the Chat to delete.

    Raises:
        ChatNotFound: When no Chat row exists for ``chat_id``.
    """
    chat = await session.scalar(select(Chat).where(Chat.id == chat_id))
    if chat is None:
        raise ChatNotFound(chat_id)

    await session.delete(chat)
    await session.flush()


__all__ = [
    "create_chat",
    "get_chat",
    "list_chats",
    "update_chat",
    "delete_chat",
]
