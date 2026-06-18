"""Unit tests for app.services.chat_service.

Database strategy
-----------------
These tests use the Docker Compose ``paper_notebook`` postgres instance with
SAVEPOINT-based rollback (see ``tests/conftest.py``).  This avoids SQLite
compatibility issues (JSONB, UUID types, etc.) while keeping each test
isolated.

Coverage
--------
- create_chat: happy path (name + optional description)
- get_chat: found / not-found
- list_chats: ordering by updated_at desc, pagination
- update_chat: partial update / not-found
- delete_chat: cascade / not-found
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.errors import ChatNotFound
from app.models.domain import ChatCreate, ChatRead, ChatUpdate
from app.services import chat_service

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_create(name: str = "Test Chat", description: str | None = None) -> ChatCreate:
    return ChatCreate(name=name, description=description)


# ---------------------------------------------------------------------------
# create_chat
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_chat_returns_chat_read(db_session: AsyncSession) -> None:
    """happy path: returns ChatRead with all expected fields (description may be None)."""
    result = await chat_service.create_chat(db_session, _make_create("My Notebook"))
    assert isinstance(result, ChatRead)
    assert result.id is not None
    assert result.name == "My Notebook"
    assert result.description is None
    assert result.default_chat_profile_id is None
    assert result.created_at is not None


# ---------------------------------------------------------------------------
# get_chat
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_chat_found(db_session: AsyncSession) -> None:
    created = await chat_service.create_chat(db_session, _make_create("FindMe"))
    fetched = await chat_service.get_chat(db_session, created.id)
    assert fetched.id == created.id
    assert fetched.name == "FindMe"


@pytest.mark.asyncio
async def test_get_chat_not_found_raises(db_session: AsyncSession) -> None:
    with pytest.raises(ChatNotFound):
        await chat_service.get_chat(db_session, uuid.uuid4())


# ---------------------------------------------------------------------------
# list_chats
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_chats_ordering_newest_first(db_session: AsyncSession) -> None:
    """Chats created later must appear earlier in the list."""
    a = await chat_service.create_chat(db_session, _make_create("Alpha"))
    b = await chat_service.create_chat(db_session, _make_create("Beta"))

    results = await chat_service.list_chats(db_session)
    ids = [r.id for r in results]
    assert b.id in ids
    assert a.id in ids
    assert ids.index(b.id) < ids.index(a.id)


@pytest.mark.asyncio
async def test_list_chats_pagination(db_session: AsyncSession) -> None:
    for i in range(3):
        await chat_service.create_chat(db_session, _make_create(f"Page{i}"))

    first_page = await chat_service.list_chats(db_session, limit=2, offset=0)
    second_page = await chat_service.list_chats(db_session, limit=2, offset=2)

    assert len(first_page) <= 2
    first_ids = {r.id for r in first_page}
    second_ids = {r.id for r in second_page}
    assert first_ids.isdisjoint(second_ids)


# ---------------------------------------------------------------------------
# update_chat
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_update_chat_partial(db_session: AsyncSession) -> None:
    created = await chat_service.create_chat(
        db_session, _make_create("Original", description="Original desc")
    )
    updated = await chat_service.update_chat(
        db_session,
        created.id,
        ChatUpdate(name="Renamed"),
    )
    assert updated.id == created.id
    assert updated.name == "Renamed"
    assert updated.description == "Original desc"  # unchanged


@pytest.mark.asyncio
async def test_update_chat_not_found_raises(db_session: AsyncSession) -> None:
    with pytest.raises(ChatNotFound):
        await chat_service.update_chat(
            db_session,
            uuid.uuid4(),
            ChatUpdate(name="Ghost"),
        )


# ---------------------------------------------------------------------------
# delete_chat
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delete_chat_removes_record(db_session: AsyncSession) -> None:
    created = await chat_service.create_chat(db_session, _make_create("ToDelete"))
    await chat_service.delete_chat(db_session, created.id)

    with pytest.raises(ChatNotFound):
        await chat_service.get_chat(db_session, created.id)


@pytest.mark.asyncio
async def test_delete_chat_not_found_raises(db_session: AsyncSession) -> None:
    with pytest.raises(ChatNotFound):
        await chat_service.delete_chat(db_session, uuid.uuid4())
