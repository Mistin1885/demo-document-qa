"""Integration tests for the Session API endpoints.

Tests the full HTTP → service → DB round-trip using the FastAPI test client
and the ``db_session`` / ``api_client`` fixtures from ``tests/conftest.py``.

Isolation coverage (CLAUDE.md §2 mandatory gates)
--------------------------------------------------
- GET /chats/{A}/sessions/{B_session_id} must return 404 — must not leak that
  the session exists in Chat B.
- POST must accept ``chat_id`` only from the URL; body ``chat_id`` is ignored
  in routing (isolation contract).
"""

from __future__ import annotations

import uuid

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.domain import ChatCreate
from app.services import chat_service

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _create_chat(db: AsyncSession, name: str = "Test Chat") -> uuid.UUID:
    """Create a chat via the service layer and return its UUID."""
    chat = await chat_service.create_chat(db, ChatCreate(name=name))
    return chat.id


# ---------------------------------------------------------------------------
# POST /chats/{chat_id}/sessions
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_session_201(
    api_client: AsyncClient, db_session: AsyncSession
) -> None:
    """POST creates a session and returns 201 with the SessionRead body."""
    chat_id = await _create_chat(db_session, "Chat for create-test")

    resp = await api_client.post(
        f"/chats/{chat_id}/sessions",
        json={"chat_id": str(chat_id), "name": "My First Session"},
    )

    assert resp.status_code == 201
    body = resp.json()
    assert body["chat_id"] == str(chat_id)
    assert body["name"] == "My First Session"
    assert "id" in body
    assert "created_at" in body


@pytest.mark.asyncio
async def test_create_session_unknown_chat_404(api_client: AsyncClient) -> None:
    """POST to a non-existent chat_id returns 404."""
    fake_id = uuid.uuid4()
    resp = await api_client.post(
        f"/chats/{fake_id}/sessions",
        json={"chat_id": str(fake_id), "name": "Ghost"},
    )
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# GET /chats/{chat_id}/sessions
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_sessions_empty(
    api_client: AsyncClient, db_session: AsyncSession
) -> None:
    """GET returns an empty list when no sessions exist."""
    chat_id = await _create_chat(db_session, "Empty Chat")
    resp = await api_client.get(f"/chats/{chat_id}/sessions")
    assert resp.status_code == 200
    assert resp.json() == []


@pytest.mark.asyncio
async def test_list_sessions_returns_own_sessions_only(
    api_client: AsyncClient, db_session: AsyncSession
) -> None:
    """GET /chats/A/sessions returns only Chat A sessions (isolation check)."""
    chat_a_id = await _create_chat(db_session, "Chat-A-List")
    chat_b_id = await _create_chat(db_session, "Chat-B-List")

    # Create 2 sessions in A, 1 in B
    for name in ("A-1", "A-2"):
        await api_client.post(
            f"/chats/{chat_a_id}/sessions",
            json={"chat_id": str(chat_a_id), "name": name},
        )
    await api_client.post(
        f"/chats/{chat_b_id}/sessions",
        json={"chat_id": str(chat_b_id), "name": "B-1"},
    )

    resp = await api_client.get(f"/chats/{chat_a_id}/sessions")
    assert resp.status_code == 200
    sessions = resp.json()
    assert len(sessions) == 2
    chat_ids_returned = {s["chat_id"] for s in sessions}
    assert chat_ids_returned == {str(chat_a_id)}, (
        "GET /chats/A/sessions must not leak sessions from Chat B"
    )


@pytest.mark.asyncio
async def test_list_sessions_unknown_chat_404(api_client: AsyncClient) -> None:
    """GET for a non-existent chat returns 404."""
    resp = await api_client.get(f"/chats/{uuid.uuid4()}/sessions")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# GET /chats/{chat_id}/sessions/{session_id}
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_session_success(
    api_client: AsyncClient, db_session: AsyncSession
) -> None:
    """GET by ID returns the correct session."""
    chat_id = await _create_chat(db_session, "Chat for get-test")
    create_resp = await api_client.post(
        f"/chats/{chat_id}/sessions",
        json={"chat_id": str(chat_id), "name": "Fetch Me"},
    )
    session_id = create_resp.json()["id"]

    resp = await api_client.get(f"/chats/{chat_id}/sessions/{session_id}")
    assert resp.status_code == 200
    assert resp.json()["id"] == session_id
    assert resp.json()["name"] == "Fetch Me"


@pytest.mark.asyncio
async def test_get_session_cross_chat_returns_404(
    api_client: AsyncClient, db_session: AsyncSession
) -> None:
    """GET /chats/A/sessions/{B_session_id} returns 404.

    This is the core isolation test: Chat A must not learn about a session
    that belongs to Chat B — the response must be 404, not a different chat's
    session data or a 403.
    """
    chat_a_id = await _create_chat(db_session, "Chat-A-Iso")
    chat_b_id = await _create_chat(db_session, "Chat-B-Iso")

    # Create a session under Chat B
    create_resp = await api_client.post(
        f"/chats/{chat_b_id}/sessions",
        json={"chat_id": str(chat_b_id), "name": "B-secret"},
    )
    b_session_id = create_resp.json()["id"]

    # Chat A tries to access Chat B's session — must get 404
    resp = await api_client.get(f"/chats/{chat_a_id}/sessions/{b_session_id}")
    assert resp.status_code == 404, (
        f"Expected 404, got {resp.status_code}. "
        "Chat A must not be able to retrieve Chat B's session."
    )


@pytest.mark.asyncio
async def test_get_session_not_found_404(
    api_client: AsyncClient, db_session: AsyncSession
) -> None:
    """GET for a non-existent session_id returns 404."""
    chat_id = await _create_chat(db_session, "Chat for miss-test")
    resp = await api_client.get(f"/chats/{chat_id}/sessions/{uuid.uuid4()}")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# PATCH /chats/{chat_id}/sessions/{session_id}
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_patch_session_success(
    api_client: AsyncClient, db_session: AsyncSession
) -> None:
    """PATCH updates the session name."""
    chat_id = await _create_chat(db_session, "Chat for patch")
    create_resp = await api_client.post(
        f"/chats/{chat_id}/sessions",
        json={"chat_id": str(chat_id), "name": "Old Name"},
    )
    session_id = create_resp.json()["id"]

    patch_resp = await api_client.patch(
        f"/chats/{chat_id}/sessions/{session_id}",
        json={"name": "New Name"},
    )
    assert patch_resp.status_code == 200
    assert patch_resp.json()["name"] == "New Name"
    assert patch_resp.json()["id"] == session_id


@pytest.mark.asyncio
async def test_patch_session_cross_chat_404(
    api_client: AsyncClient, db_session: AsyncSession
) -> None:
    """PATCH with cross-chat session_id returns 404."""
    chat_a_id = await _create_chat(db_session, "PatchA")
    chat_b_id = await _create_chat(db_session, "PatchB")

    create_resp = await api_client.post(
        f"/chats/{chat_b_id}/sessions",
        json={"chat_id": str(chat_b_id), "name": "B-patch-target"},
    )
    b_session_id = create_resp.json()["id"]

    resp = await api_client.patch(
        f"/chats/{chat_a_id}/sessions/{b_session_id}",
        json={"name": "Injected"},
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_patch_session_not_found_404(
    api_client: AsyncClient, db_session: AsyncSession
) -> None:
    """PATCH for a non-existent session_id returns 404."""
    chat_id = await _create_chat(db_session, "Patch miss chat")
    resp = await api_client.patch(
        f"/chats/{chat_id}/sessions/{uuid.uuid4()}",
        json={"name": "Ghost"},
    )
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# DELETE /chats/{chat_id}/sessions/{session_id}
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delete_session_success(
    api_client: AsyncClient, db_session: AsyncSession
) -> None:
    """DELETE returns 204; subsequent GET returns 404."""
    chat_id = await _create_chat(db_session, "Chat for delete")
    create_resp = await api_client.post(
        f"/chats/{chat_id}/sessions",
        json={"chat_id": str(chat_id), "name": "To Be Deleted"},
    )
    session_id = create_resp.json()["id"]

    del_resp = await api_client.delete(f"/chats/{chat_id}/sessions/{session_id}")
    assert del_resp.status_code == 204

    get_resp = await api_client.get(f"/chats/{chat_id}/sessions/{session_id}")
    assert get_resp.status_code == 404


@pytest.mark.asyncio
async def test_delete_session_cross_chat_404(
    api_client: AsyncClient, db_session: AsyncSession
) -> None:
    """DELETE with cross-chat session_id returns 404."""
    chat_a_id = await _create_chat(db_session, "DelA")
    chat_b_id = await _create_chat(db_session, "DelB")

    create_resp = await api_client.post(
        f"/chats/{chat_b_id}/sessions",
        json={"chat_id": str(chat_b_id), "name": "B-del-target"},
    )
    b_session_id = create_resp.json()["id"]

    resp = await api_client.delete(f"/chats/{chat_a_id}/sessions/{b_session_id}")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_delete_session_not_found_404(
    api_client: AsyncClient, db_session: AsyncSession
) -> None:
    """DELETE for a non-existent session_id returns 404."""
    chat_id = await _create_chat(db_session, "Del miss chat")
    resp = await api_client.delete(f"/chats/{chat_id}/sessions/{uuid.uuid4()}")
    assert resp.status_code == 404
