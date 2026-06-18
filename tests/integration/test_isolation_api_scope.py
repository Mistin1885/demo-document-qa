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
(l) Correct resource + wrong chat_id → 404 (not 403, preserves isolation).
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
@pytest.mark.parametrize(
    "path",
    [
        "/chats/not-a-uuid/sessions",
        "/chats/not-a-uuid/documents",
    ],
)
async def test_malformed_chat_id_returns_422(
    api_client: AsyncClient, path: str
) -> None:
    """(l) Malformed chat_id in path must return 422."""
    resp = await api_client.get(path)
    assert resp.status_code == 422, (
        f"Malformed chat_id in {path!r} must return 422, got {resp.status_code}"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "sub_path",
    [
        "sessions/bad-uuid",
        "documents/bad-uuid",
    ],
)
async def test_malformed_resource_id_returns_422(
    api_client: AsyncClient,
    db_session: AsyncSession,
    sub_path: str,
) -> None:
    """(l) Malformed resource_id in path must return 422."""
    chat = await make_chat(db_session, f"valid-chat-{sub_path.split('/')[0]}")
    resp = await api_client.get(f"/chats/{chat.id}/{sub_path}")
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# (l) Non-existent chat_id → 404
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "sub_path_factory",
    [
        lambda: "sessions",
        lambda: "documents",
        lambda: f"sessions/{uuid.uuid4()}",
        lambda: f"documents/{uuid.uuid4()}",
    ],
)
async def test_nonexistent_chat_id_returns_404(
    api_client: AsyncClient, sub_path_factory: object
) -> None:
    """(l) Non-existent chat_id in any path must return 404."""
    sub_path = sub_path_factory()  # type: ignore[operator]
    resp = await api_client.get(f"/chats/{uuid.uuid4()}/{sub_path}")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# (l) Correct resource + wrong chat_id → 404 (not 403)
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

    create_resp = await api_client.post(
        f"/chats/{chat_a.id}/sessions",
        json={"chat_id": str(chat_a.id), "name": "A-resource"},
    )
    session_id = create_resp.json()["id"]

    resp = await api_client.get(f"/chats/{chat_b.id}/sessions/{session_id}")
    assert resp.status_code == 404, (
        f"Expected 404 (not 403 or 200), got {resp.status_code}; "
        "using wrong chat_id must never reveal resource existence"
    )
    assert resp.status_code != 403, "403 is forbidden — it leaks resource existence"
