"""Unit tests for app.services.session_service.

These tests run against a real PostgreSQL instance (via Docker Compose) but
use SAVEPOINT-based rollback (see ``tests/conftest.py``) so each test starts
from a clean state without schema teardown.

Isolation coverage (CLAUDE.md §2 mandatory gates)
--------------------------------------------------
- ``list_sessions`` for Chat A must NOT return sessions from Chat B.
- ``get_session_by_id`` with Chat A's ``chat_id`` and Chat B's ``session_id``
  must raise ``SessionNotFound`` — never leak cross-chat data.
- ``update_session`` / ``delete_session`` with mismatched chat scope must raise
  ``SessionNotFound``.
"""

from __future__ import annotations

import uuid

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession

from app.errors import ChatNotFound, SessionNotFound
from app.models.domain import SessionCreate, SessionUpdate
from app.models.orm import Chat
from app.services import session_service

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _make_chat(db: AsyncSession, name: str = "Test Chat") -> Chat:
    """Create and persist a bare Chat row."""
    chat = Chat(name=name)
    db.add(chat)
    await db.flush()
    await db.refresh(chat)
    return chat


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def chat_a(db_session: AsyncSession) -> Chat:
    return await _make_chat(db_session, "Chat A")


@pytest_asyncio.fixture
async def chat_b(db_session: AsyncSession) -> Chat:
    return await _make_chat(db_session, "Chat B")


# ---------------------------------------------------------------------------
# create_session
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_session_success(db_session: AsyncSession, chat_a: Chat) -> None:
    """Creating a session under a valid chat returns the new SessionRead."""
    data = SessionCreate(chat_id=chat_a.id, name="Session 1")
    result = await session_service.create_session(db_session, chat_id=chat_a.id, data=data)

    assert result.chat_id == chat_a.id
    assert result.name == "Session 1"
    assert result.id is not None


@pytest.mark.asyncio
async def test_create_session_chat_not_found(db_session: AsyncSession) -> None:
    """Creating a session under a non-existent chat raises ChatNotFound."""
    missing_id = uuid.uuid4()
    data = SessionCreate(chat_id=missing_id, name="Ghost Session")

    with pytest.raises(ChatNotFound):
        await session_service.create_session(db_session, chat_id=missing_id, data=data)


# ---------------------------------------------------------------------------
# list_sessions — isolation core test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_sessions_isolation(
    db_session: AsyncSession, chat_a: Chat, chat_b: Chat
) -> None:
    """list_sessions(chat_id=A) must return ONLY Chat A's sessions.

    This is the primary isolation guard for CLAUDE.md §2.
    """
    # Create 2 sessions under Chat A, 1 under Chat B
    await session_service.create_session(
        db_session, chat_id=chat_a.id, data=SessionCreate(chat_id=chat_a.id, name="A-1")
    )
    await session_service.create_session(
        db_session, chat_id=chat_a.id, data=SessionCreate(chat_id=chat_a.id, name="A-2")
    )
    await session_service.create_session(
        db_session, chat_id=chat_b.id, data=SessionCreate(chat_id=chat_b.id, name="B-1")
    )

    sessions_a = await session_service.list_sessions(db_session, chat_id=chat_a.id)

    assert len(sessions_a) == 2
    chat_ids = {s.chat_id for s in sessions_a}
    assert chat_ids == {chat_a.id}, (
        "list_sessions must not leak sessions from Chat B into Chat A"
    )
    names = {s.name for s in sessions_a}
    assert names == {"A-1", "A-2"}


@pytest.mark.asyncio
async def test_list_sessions_chat_not_found(db_session: AsyncSession) -> None:
    """list_sessions for a non-existent chat raises ChatNotFound."""
    with pytest.raises(ChatNotFound):
        await session_service.list_sessions(db_session, chat_id=uuid.uuid4())


@pytest.mark.asyncio
async def test_list_sessions_empty(db_session: AsyncSession, chat_a: Chat) -> None:
    """list_sessions returns an empty list when no sessions exist yet."""
    result = await session_service.list_sessions(db_session, chat_id=chat_a.id)
    assert result == []


# ---------------------------------------------------------------------------
# get_session_by_id — isolation core test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_session_cross_chat_raises(
    db_session: AsyncSession, chat_a: Chat, chat_b: Chat
) -> None:
    """get_session_by_id(chat_id=A, session_id=<B session>) must raise SessionNotFound.

    Chat A must never be able to retrieve a session belonging to Chat B.
    This is the core cross-chat isolation test mandated by CLAUDE.md §2.
    """
    session_b = await session_service.create_session(
        db_session, chat_id=chat_b.id, data=SessionCreate(chat_id=chat_b.id, name="B-only")
    )

    with pytest.raises(SessionNotFound):
        await session_service.get_session_by_id(
            db_session, chat_id=chat_a.id, session_id=session_b.id
        )


@pytest.mark.asyncio
async def test_get_session_success(db_session: AsyncSession, chat_a: Chat) -> None:
    """get_session_by_id returns the correct SessionRead for valid scope."""
    created = await session_service.create_session(
        db_session, chat_id=chat_a.id, data=SessionCreate(chat_id=chat_a.id, name="My Session")
    )
    fetched = await session_service.get_session_by_id(
        db_session, chat_id=chat_a.id, session_id=created.id
    )
    assert fetched.id == created.id
    assert fetched.chat_id == chat_a.id
    assert fetched.name == "My Session"


@pytest.mark.asyncio
async def test_get_session_not_found(db_session: AsyncSession, chat_a: Chat) -> None:
    """get_session_by_id raises SessionNotFound for a random UUID."""
    with pytest.raises(SessionNotFound):
        await session_service.get_session_by_id(
            db_session, chat_id=chat_a.id, session_id=uuid.uuid4()
        )


# ---------------------------------------------------------------------------
# update_session
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_update_session_success(db_session: AsyncSession, chat_a: Chat) -> None:
    """update_session applies partial changes within the correct chat scope."""
    created = await session_service.create_session(
        db_session, chat_id=chat_a.id, data=SessionCreate(chat_id=chat_a.id, name="Before")
    )
    updated = await session_service.update_session(
        db_session,
        chat_id=chat_a.id,
        session_id=created.id,
        data=SessionUpdate(name="After"),
    )
    assert updated.id == created.id
    assert updated.name == "After"


@pytest.mark.asyncio
async def test_update_session_cross_chat_raises(
    db_session: AsyncSession, chat_a: Chat, chat_b: Chat
) -> None:
    """update_session with Chat A scope on a Chat B session raises SessionNotFound."""
    session_b = await session_service.create_session(
        db_session,
        chat_id=chat_b.id,
        data=SessionCreate(chat_id=chat_b.id, name="B-session"),
    )
    with pytest.raises(SessionNotFound):
        await session_service.update_session(
            db_session,
            chat_id=chat_a.id,
            session_id=session_b.id,
            data=SessionUpdate(name="Hacked"),
        )


# ---------------------------------------------------------------------------
# delete_session
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delete_session_success(db_session: AsyncSession, chat_a: Chat) -> None:
    """delete_session removes the session; subsequent fetch raises SessionNotFound."""
    created = await session_service.create_session(
        db_session, chat_id=chat_a.id, data=SessionCreate(chat_id=chat_a.id, name="To Delete")
    )
    await session_service.delete_session(
        db_session, chat_id=chat_a.id, session_id=created.id
    )
    with pytest.raises(SessionNotFound):
        await session_service.get_session_by_id(
            db_session, chat_id=chat_a.id, session_id=created.id
        )


@pytest.mark.asyncio
async def test_delete_session_cross_chat_raises(
    db_session: AsyncSession, chat_a: Chat, chat_b: Chat
) -> None:
    """delete_session with Chat A scope on a Chat B session raises SessionNotFound."""
    session_b = await session_service.create_session(
        db_session,
        chat_id=chat_b.id,
        data=SessionCreate(chat_id=chat_b.id, name="B-only"),
    )
    with pytest.raises(SessionNotFound):
        await session_service.delete_session(
            db_session, chat_id=chat_a.id, session_id=session_b.id
        )
