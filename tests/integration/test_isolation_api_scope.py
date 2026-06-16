"""Integration isolation tests — API path scope validation.

CLAUDE.md §2 isolation layers covered
--------------------------------------
- **API authorization scope**: FastAPI path parameters enforce UUID parsing
  (422 for malformed values); the route layer verifies resource ownership
  via the ``chat_id`` path parameter (404 for non-existent chats).

Scenarios
---------
(l) Malformed UUID in path → 422 (FastAPI automatic validation).
(l) Non-existent chat_id in path → 404.
(l) Valid chat_id but non-existent resource_id → 404.
"""

from __future__ import annotations

import uuid

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from tests._helpers import make_chat

# ---------------------------------------------------------------------------
# (l) Malformed UUID → 422
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_malformed_chat_id_in_sessions_path_returns_422(
    api_client: AsyncClient,
) -> None:
    """(l) GET /chats/not-a-uuid/sessions → 422.

    FastAPI automatically validates UUID path parameters and returns 422
    Unprocessable Entity when the value cannot be parsed as a UUID.
    """
    resp = await api_client.get("/chats/not-a-uuid/sessions")
    assert resp.status_code == 422, (
        f"Malformed chat_id must return 422, got {resp.status_code}"
    )


@pytest.mark.asyncio
async def test_malformed_chat_id_in_documents_path_returns_422(
    api_client: AsyncClient,
) -> None:
    """(l) GET /chats/not-a-uuid/documents → 422."""
    resp = await api_client.get("/chats/not-a-uuid/documents")
    assert resp.status_code == 422, (
        f"Malformed chat_id must return 422, got {resp.status_code}"
    )


@pytest.mark.asyncio
async def test_malformed_session_id_in_path_returns_422(
    api_client: AsyncClient,
    db_session: AsyncSession,
) -> None:
    """(l) GET /chats/{valid_id}/sessions/bad-uuid → 422."""
    chat = await make_chat(db_session, "valid-chat-malformed-sess")
    resp = await api_client.get(f"/chats/{chat.id}/sessions/bad-uuid")
    assert resp.status_code == 422, (
        f"Malformed session_id must return 422, got {resp.status_code}"
    )


@pytest.mark.asyncio
async def test_malformed_document_id_in_path_returns_422(
    api_client: AsyncClient,
    db_session: AsyncSession,
) -> None:
    """(l) GET /chats/{valid_id}/documents/bad-uuid → 422."""
    chat = await make_chat(db_session, "valid-chat-malformed-doc")
    resp = await api_client.get(f"/chats/{chat.id}/documents/bad-uuid")
    assert resp.status_code == 422, (
        f"Malformed document_id must return 422, got {resp.status_code}"
    )


# ---------------------------------------------------------------------------
# (l) Non-existent chat_id → 404
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_nonexistent_chat_id_in_sessions_list_returns_404(
    api_client: AsyncClient,
) -> None:
    """(l) GET /chats/{random_uuid}/sessions → 404 when chat does not exist."""
    resp = await api_client.get(f"/chats/{uuid.uuid4()}/sessions")
    assert resp.status_code == 404, (
        f"Non-existent chat_id must return 404, got {resp.status_code}"
    )


@pytest.mark.asyncio
async def test_nonexistent_chat_id_in_documents_list_returns_404(
    api_client: AsyncClient,
) -> None:
    """(l) GET /chats/{random_uuid}/documents → 404 when chat does not exist."""
    resp = await api_client.get(f"/chats/{uuid.uuid4()}/documents")
    assert resp.status_code == 404, (
        f"Non-existent chat_id must return 404, got {resp.status_code}"
    )


@pytest.mark.asyncio
async def test_nonexistent_chat_id_in_session_get_returns_404(
    api_client: AsyncClient,
) -> None:
    """(l) GET /chats/{random}/sessions/{random} → 404 when neither exists."""
    resp = await api_client.get(f"/chats/{uuid.uuid4()}/sessions/{uuid.uuid4()}")
    assert resp.status_code == 404, (
        f"Non-existent chat_id + session_id must return 404, got {resp.status_code}"
    )


@pytest.mark.asyncio
async def test_nonexistent_chat_id_in_document_get_returns_404(
    api_client: AsyncClient,
) -> None:
    """(l) GET /chats/{random}/documents/{random} → 404 when neither exists."""
    resp = await api_client.get(f"/chats/{uuid.uuid4()}/documents/{uuid.uuid4()}")
    assert resp.status_code == 404, (
        f"Non-existent chat_id + document_id must return 404, got {resp.status_code}"
    )


# ---------------------------------------------------------------------------
# (l) Valid chat, non-existent resource → 404
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_valid_chat_nonexistent_session_returns_404(
    api_client: AsyncClient,
    db_session: AsyncSession,
) -> None:
    """(l) GET /chats/{existing}/sessions/{random} → 404."""
    chat = await make_chat(db_session, "real-chat-missing-sess")
    resp = await api_client.get(f"/chats/{chat.id}/sessions/{uuid.uuid4()}")
    assert resp.status_code == 404, (
        f"Non-existent session_id under real chat must return 404, got {resp.status_code}"
    )


@pytest.mark.asyncio
async def test_valid_chat_nonexistent_document_returns_404(
    api_client: AsyncClient,
    db_session: AsyncSession,
) -> None:
    """(l) GET /chats/{existing}/documents/{random} → 404."""
    chat = await make_chat(db_session, "real-chat-missing-doc")
    resp = await api_client.get(f"/chats/{chat.id}/documents/{uuid.uuid4()}")
    assert resp.status_code == 404, (
        f"Non-existent document_id under real chat must return 404, got {resp.status_code}"
    )


# ---------------------------------------------------------------------------
# Cross-chat: wrong chat_id in path + correct resource_id → 404 (not 403)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_correct_resource_wrong_chat_id_returns_404_not_403(
    api_client: AsyncClient,
    db_session: AsyncSession,
) -> None:
    """Using Chat B's id with Chat A's session id returns 404, never 403.

    403 would reveal that the resource *exists*; CLAUDE.md §2 requires 404
    to preserve isolation (no existence leakage).
    """
    chat_a = await make_chat(db_session, "correct-res-wrong-chat-A")
    chat_b = await make_chat(db_session, "correct-res-wrong-chat-B")

    # Create session in Chat A
    create_resp = await api_client.post(
        f"/chats/{chat_a.id}/sessions",
        json={"chat_id": str(chat_a.id), "name": "A-resource"},
    )
    session_id = create_resp.json()["id"]

    # Access with Chat B's id — must be 404 (not 403, not 200)
    resp = await api_client.get(f"/chats/{chat_b.id}/sessions/{session_id}")
    assert resp.status_code == 404, (
        f"Expected 404 (not 403 or 200), got {resp.status_code}; "
        "using wrong chat_id must never reveal resource existence"
    )
    assert resp.status_code != 403, "403 is forbidden — it leaks resource existence"
