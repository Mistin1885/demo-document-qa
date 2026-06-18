"""Phase 9.2 — Session isolation E2E (GUIDE §22 / §19.7).

Same chat, two sessions.  Each session POSTs its own question; the per-session
GET /messages endpoint must return only its own messages and never bleed user
turns from the sibling session.

Test density cap: ≤ 10 items per file (CLAUDE.md §12.1).
"""

from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable
from typing import Any
from unittest.mock import patch

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from tests.e2e.conftest import make_hit, make_provider_builder

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _create_chat(client: AsyncClient, name: str) -> uuid.UUID:
    resp = await client.post("/chats/", json={"name": name})
    assert resp.status_code == 201, resp.text
    return uuid.UUID(resp.json()["id"])


async def _create_session(client: AsyncClient, chat_id: uuid.UUID, name: str) -> uuid.UUID:
    resp = await client.post(
        f"/chats/{chat_id}/sessions",
        json={"chat_id": str(chat_id), "name": name},
    )
    assert resp.status_code == 201, resp.text
    return uuid.UUID(resp.json()["id"])


async def _upload_doc(
    client: AsyncClient, chat_id: uuid.UUID, filename: str, contents: bytes
) -> uuid.UUID:
    resp = await client.post(
        f"/chats/{chat_id}/documents",
        files={"file": (filename, contents, "application/pdf")},
    )
    assert resp.status_code == 201, resp.text
    return uuid.UUID(resp.json()["id"])


def _patch_providers(builder: Callable[[Any, uuid.UUID], Awaitable[Any]]):  # type: ignore[no-untyped-def]
    return patch("app.api.messages._build_providers", side_effect=builder)


# ---------------------------------------------------------------------------
# 1. Two sessions in the same chat — each GET /messages returns only its own
#    question/answer pair.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_each_session_sees_only_its_own_messages(
    api_client: AsyncClient, db_session: AsyncSession
) -> None:
    chat_id = await _create_chat(api_client, "session-iso-chat")
    doc_id = await _upload_doc(api_client, chat_id, "p.pdf", b"%PDF-1.4 paper")
    s1 = await _create_session(api_client, chat_id, "s1")
    s2 = await _create_session(api_client, chat_id, "s2")
    await db_session.commit()

    hits = [make_hit(chat_id=chat_id, document_id=doc_id, content="Shared doc content.")]
    builder = make_provider_builder(hits)

    with _patch_providers(builder):
        r1 = await api_client.post(
            f"/chats/{chat_id}/sessions/{s1}/messages",
            json={"question": "Question One in S1.", "stream": False},
        )
        r2 = await api_client.post(
            f"/chats/{chat_id}/sessions/{s2}/messages",
            json={"question": "Question Two in S2.", "stream": False},
        )
    assert r1.status_code == 200 and r2.status_code == 200

    list1 = (await api_client.get(f"/chats/{chat_id}/sessions/{s1}/messages")).json()
    list2 = (await api_client.get(f"/chats/{chat_id}/sessions/{s2}/messages")).json()
    text1 = " ".join(m["content"] for m in list1)
    text2 = " ".join(m["content"] for m in list2)
    assert "Question One" in text1 and "Question Two" not in text1
    assert "Question Two" in text2 and "Question One" not in text2


# ---------------------------------------------------------------------------
# 2. Cross-chat session URL on the GET endpoint → 404 (no shape leakage).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_messages_cross_chat_session_returns_404(
    api_client: AsyncClient, db_session: AsyncSession
) -> None:
    chat_a = await _create_chat(api_client, "iso-A")
    chat_b = await _create_chat(api_client, "iso-B")
    sess_a = await _create_session(api_client, chat_a, "sa")
    await db_session.commit()

    resp = await api_client.get(f"/chats/{chat_b}/sessions/{sess_a}/messages")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# 3. Empty session — GET /messages returns an empty list (never the sibling's).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_session_returns_empty_list_even_if_sibling_has_messages(
    api_client: AsyncClient, db_session: AsyncSession
) -> None:
    chat_id = await _create_chat(api_client, "empty-iso")
    doc_id = await _upload_doc(api_client, chat_id, "p.pdf", b"%PDF-1.4 ep")
    s1 = await _create_session(api_client, chat_id, "with-msgs")
    s2 = await _create_session(api_client, chat_id, "empty")
    await db_session.commit()

    hits = [make_hit(chat_id=chat_id, document_id=doc_id, content="content e")]
    with _patch_providers(make_provider_builder(hits)):
        await api_client.post(
            f"/chats/{chat_id}/sessions/{s1}/messages",
            json={"question": "Only in S1.", "stream": False},
        )

    list_s2 = (await api_client.get(f"/chats/{chat_id}/sessions/{s2}/messages")).json()
    assert list_s2 == []


# ---------------------------------------------------------------------------
# 4. Both sessions can still cite documents in the shared chat scope.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_both_sessions_share_document_scope(
    api_client: AsyncClient, db_session: AsyncSession
) -> None:
    chat_id = await _create_chat(api_client, "shared-scope")
    doc_id = await _upload_doc(api_client, chat_id, "p.pdf", b"%PDF-1.4 shared")
    s1 = await _create_session(api_client, chat_id, "sa")
    s2 = await _create_session(api_client, chat_id, "sb")
    await db_session.commit()

    hits = [make_hit(chat_id=chat_id, document_id=doc_id, content="Shared evidence span.")]
    with _patch_providers(make_provider_builder(hits)):
        r1 = await api_client.post(
            f"/chats/{chat_id}/sessions/{s1}/messages",
            json={"question": "Describe paper from S1.", "stream": False},
        )
        r2 = await api_client.post(
            f"/chats/{chat_id}/sessions/{s2}/messages",
            json={"question": "Describe paper from S2.", "stream": False},
        )

    for resp in (r1, r2):
        assert resp.status_code == 200
        body = resp.json()
        assert any(c["document_id"] == str(doc_id) for c in body["citations"])
        assert all(c["chat_id"] == str(chat_id) for c in body["citations"])


# ---------------------------------------------------------------------------
# 5. Persistence — POST /messages writes user + assistant messages that GET sees.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_post_then_get_returns_both_roles(
    api_client: AsyncClient, db_session: AsyncSession
) -> None:
    chat_id = await _create_chat(api_client, "post-then-get")
    doc_id = await _upload_doc(api_client, chat_id, "p.pdf", b"%PDF-1.4 ptg")
    sess = await _create_session(api_client, chat_id, "s")
    await db_session.commit()

    hits = [make_hit(chat_id=chat_id, document_id=doc_id, content="Body of paper.")]
    with _patch_providers(make_provider_builder(hits)):
        post = await api_client.post(
            f"/chats/{chat_id}/sessions/{sess}/messages",
            json={"question": "Round-trip question.", "stream": False},
        )
        assert post.status_code == 200

    rows = (await api_client.get(f"/chats/{chat_id}/sessions/{sess}/messages")).json()
    roles = [m["role"] for m in rows]
    assert "user" in roles and "assistant" in roles
