"""Integration isolation tests — Chat A must not see Chat B's sessions.

CLAUDE.md §2 isolation layers covered
--------------------------------------
- **API authorization scope**: route layer enforces that the ``session_id``
  in the URL path belongs to the ``chat_id`` in the URL path.
- **Relational DB query**: ``session_service`` enforces
  ``WHERE sessions.chat_id = :current_chat_id``.

Scenarios
---------
(d) GET  /chats/{B}/sessions/{S}    → 404  (S belongs to A)
    PATCH /chats/{B}/sessions/{S}   → 404
    DELETE /chats/{B}/sessions/{S}  → 404
(e) GET  /chats/{B}/sessions        → list does not contain S
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from tests._helpers import make_chat

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


async def _create_session(
    api_client: AsyncClient,
    chat_id,  # type: ignore[no-untyped-def]
    name: str = "Test Session",
) -> str:
    """Create a session via the API and return its id string."""
    resp = await api_client.post(
        f"/chats/{chat_id}/sessions",
        json={"chat_id": str(chat_id), "name": name},
    )
    assert resp.status_code == 201, f"Session creation failed: {resp.text}"
    return resp.json()["id"]


# ---------------------------------------------------------------------------
# (d) GET/PATCH/DELETE /chats/{B}/sessions/{S} → 404
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_session_cross_chat_returns_404(
    api_client: AsyncClient,
    db_session: AsyncSession,
) -> None:
    """(d-GET) Chat A owns session S; GET /chats/{B}/sessions/{S} → 404.

    Layer: API authorization scope + Service layer DB query.
    """
    chat_a = await make_chat(db_session, "iso-a-sess-get")
    chat_b = await make_chat(db_session, "iso-b-sess-get")
    session_s_id = await _create_session(api_client, chat_a.id, "secret-A")

    resp = await api_client.get(f"/chats/{chat_b.id}/sessions/{session_s_id}")
    assert resp.status_code == 404, (
        f"Expected 404, got {resp.status_code}; "
        "Chat B must not retrieve Chat A's session"
    )


@pytest.mark.asyncio
async def test_patch_session_cross_chat_returns_404(
    api_client: AsyncClient,
    db_session: AsyncSession,
) -> None:
    """(d-PATCH) Chat A owns session S; PATCH /chats/{B}/sessions/{S} → 404.

    Layer: API authorization scope + Service layer DB query.
    """
    chat_a = await make_chat(db_session, "iso-a-sess-patch")
    chat_b = await make_chat(db_session, "iso-b-sess-patch")
    session_s_id = await _create_session(api_client, chat_a.id, "patch-target-A")

    resp = await api_client.patch(
        f"/chats/{chat_b.id}/sessions/{session_s_id}",
        json={"name": "injected-name"},
    )
    assert resp.status_code == 404, (
        f"Expected 404, got {resp.status_code}; "
        "Chat B must not be able to PATCH Chat A's session"
    )


@pytest.mark.asyncio
async def test_delete_session_cross_chat_returns_404(
    api_client: AsyncClient,
    db_session: AsyncSession,
) -> None:
    """(d-DELETE) Chat A owns session S; DELETE /chats/{B}/sessions/{S} → 404.

    Layer: API authorization scope + Service layer DB query.
    """
    chat_a = await make_chat(db_session, "iso-a-sess-del")
    chat_b = await make_chat(db_session, "iso-b-sess-del")
    session_s_id = await _create_session(api_client, chat_a.id, "delete-target-A")

    resp = await api_client.delete(f"/chats/{chat_b.id}/sessions/{session_s_id}")
    assert resp.status_code == 404, (
        f"Expected 404, got {resp.status_code}; "
        "Chat B must not be able to DELETE Chat A's session"
    )


@pytest.mark.asyncio
async def test_session_still_exists_after_cross_chat_delete_attempt(
    api_client: AsyncClient,
    db_session: AsyncSession,
) -> None:
    """After a failed cross-chat DELETE, session S still exists under Chat A.

    Verifies that the failed cross-chat delete does not mutate the DB row.
    """
    chat_a = await make_chat(db_session, "iso-a-sess-survives")
    chat_b = await make_chat(db_session, "iso-b-sess-survives")
    session_s_id = await _create_session(api_client, chat_a.id, "survive-session")

    # Cross-chat delete (must fail)
    del_resp = await api_client.delete(f"/chats/{chat_b.id}/sessions/{session_s_id}")
    assert del_resp.status_code == 404

    # Session still accessible from Chat A
    get_resp = await api_client.get(f"/chats/{chat_a.id}/sessions/{session_s_id}")
    assert get_resp.status_code == 200, (
        f"Session must still exist under Chat A after failed cross-chat delete; "
        f"got {get_resp.status_code}"
    )
    assert get_resp.json()["id"] == session_s_id


# ---------------------------------------------------------------------------
# (e) GET /chats/{B}/sessions → list does not contain S
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_sessions_cross_chat_excludes_other_chat_sessions(
    api_client: AsyncClient,
    db_session: AsyncSession,
) -> None:
    """(e) Chat A owns session S; GET /chats/{B}/sessions list does not contain S.

    Layer: API authorization scope + Service layer DB query.
    """
    chat_a = await make_chat(db_session, "iso-a-sess-list")
    chat_b = await make_chat(db_session, "iso-b-sess-list")

    session_s_id = await _create_session(api_client, chat_a.id, "only-in-A")

    resp = await api_client.get(f"/chats/{chat_b.id}/sessions")
    assert resp.status_code == 200, resp.text
    listed_ids = {s["id"] for s in resp.json()}
    assert session_s_id not in listed_ids, (
        f"Session {session_s_id} from Chat A must not appear in Chat B's list; "
        f"got ids: {listed_ids}"
    )
