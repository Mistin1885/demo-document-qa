"""Phase 9.2 — Chat isolation E2E (GUIDE §22 / §19.6).

Two chats are created via the real FastAPI router stack (with real
Postgres + SAVEPOINT rollback per test).  Each chat uploads its own
document, opens a session, and POSTs a question.  The messages endpoint
runs through the real QAService + LangGraph with patched ``_build_providers``
returning a ``ChatScopedRetrievalMock`` (only emits hits whose chat_id
matches the request) and a ``CitingMockChatProvider`` (emits inline
``[c<idx>]`` markers from the evidence block).

Asserted invariants:
- ``ChatA`` answers cite only ``doc_a``; ``ChatB`` answers cite only ``doc_b``.
- A session from chat A cannot be used under a chat B URL (cross-chat 404).
- GET /chats/{B}/documents never lists doc A and vice versa.
- Forged retrieval hits with the wrong chat_id are filtered out by the
  scoped retrieval mock (mirroring the production Vespa filter).

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
# Helpers — small, file-local; conftest holds the cross-file machinery.
# ---------------------------------------------------------------------------


async def _create_chat(client: AsyncClient, name: str) -> uuid.UUID:
    resp = await client.post("/chats/", json={"name": name})
    assert resp.status_code == 201, resp.text
    return uuid.UUID(resp.json()["id"])


async def _create_session(client: AsyncClient, chat_id: uuid.UUID, name: str = "s1") -> uuid.UUID:
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
    """Patch the messages endpoint provider factory in-place."""
    return patch("app.api.messages._build_providers", side_effect=builder)


# ---------------------------------------------------------------------------
# 1. Two-chat happy path — each chat answers using only its own document.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_two_chats_cite_only_their_own_document(
    api_client: AsyncClient, db_session: AsyncSession
) -> None:
    chat_a = await _create_chat(api_client, "Chat A")
    chat_b = await _create_chat(api_client, "Chat B")
    doc_a = await _upload_doc(api_client, chat_a, "paper_a.pdf", b"%PDF-1.4 A")
    doc_b = await _upload_doc(api_client, chat_b, "paper_b.pdf", b"%PDF-1.4 B")
    sess_a = await _create_session(api_client, chat_a)
    sess_b = await _create_session(api_client, chat_b)
    await db_session.commit()

    hits = [
        make_hit(chat_id=chat_a, document_id=doc_a, content="Paper A unique content."),
        make_hit(chat_id=chat_b, document_id=doc_b, content="Paper B unique content."),
    ]
    builder = make_provider_builder(hits)

    with _patch_providers(builder):
        resp_a = await api_client.post(
            f"/chats/{chat_a}/sessions/{sess_a}/messages",
            json={"question": "Describe the paper.", "stream": False},
        )
        resp_b = await api_client.post(
            f"/chats/{chat_b}/sessions/{sess_b}/messages",
            json={"question": "Describe the paper.", "stream": False},
        )

    assert resp_a.status_code == 200 and resp_b.status_code == 200
    cits_a = resp_a.json()["citations"]
    cits_b = resp_b.json()["citations"]
    assert {c["document_id"] for c in cits_a} == {str(doc_a)}
    assert {c["document_id"] for c in cits_b} == {str(doc_b)}
    assert all(c["chat_id"] == str(chat_a) for c in cits_a)
    assert all(c["chat_id"] == str(chat_b) for c in cits_b)


# ---------------------------------------------------------------------------
# 2. Asking Chat A about Chat B's unique content → refusal, no citations.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_chat_a_cannot_retrieve_chat_b_content(
    api_client: AsyncClient, db_session: AsyncSession
) -> None:
    chat_a = await _create_chat(api_client, "Chat A iso")
    chat_b = await _create_chat(api_client, "Chat B iso")
    _doc_a = await _upload_doc(api_client, chat_a, "a.pdf", b"%PDF-1.4 A")
    doc_b = await _upload_doc(api_client, chat_b, "b.pdf", b"%PDF-1.4 B")
    sess_a = await _create_session(api_client, chat_a)
    await db_session.commit()

    # Seed *only* a chat B hit — the scoped mock will drop it for any chat A query.
    hits = [
        make_hit(chat_id=chat_b, document_id=doc_b, content="QPro evaluation protocol from paper B."),
    ]
    builder = make_provider_builder(hits)

    with _patch_providers(builder):
        resp = await api_client.post(
            f"/chats/{chat_a}/sessions/{sess_a}/messages",
            json={"question": "Describe the unique QPro evaluation protocol from paper B.", "stream": False},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["citations"] == []
    assert "not enough information" in body["answer"].lower()


# ---------------------------------------------------------------------------
# 3. Cross-chat session URL → 404.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cross_chat_session_url_returns_404(
    api_client: AsyncClient, db_session: AsyncSession
) -> None:
    chat_a = await _create_chat(api_client, "Chat A url")
    chat_b = await _create_chat(api_client, "Chat B url")
    sess_a = await _create_session(api_client, chat_a)
    await db_session.commit()

    resp = await api_client.post(
        f"/chats/{chat_b}/sessions/{sess_a}/messages",
        json={"question": "test", "stream": False},
    )
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# 4. Documents list isolation: chat A does not see chat B's docs.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_documents_listing_is_chat_scoped(
    api_client: AsyncClient, db_session: AsyncSession
) -> None:
    chat_a = await _create_chat(api_client, "Chat A docs")
    chat_b = await _create_chat(api_client, "Chat B docs")
    doc_a = await _upload_doc(api_client, chat_a, "a.pdf", b"%PDF-1.4 A list")
    doc_b = await _upload_doc(api_client, chat_b, "b.pdf", b"%PDF-1.4 B list")
    await db_session.commit()

    resp_a = await api_client.get(f"/chats/{chat_a}/documents")
    resp_b = await api_client.get(f"/chats/{chat_b}/documents")
    assert resp_a.status_code == 200 and resp_b.status_code == 200
    a_ids = {d["id"] for d in resp_a.json()}
    b_ids = {d["id"] for d in resp_b.json()}
    assert str(doc_a) in a_ids and str(doc_b) not in a_ids
    assert str(doc_b) in b_ids and str(doc_a) not in b_ids


# ---------------------------------------------------------------------------
# 5. Forged retrieval seed: a hit with chat_id=A is filtered out when the
#    request targets chat B — proving the scoped retrieval mock honours the
#    production isolation contract.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_forged_cross_chat_hit_is_dropped(
    api_client: AsyncClient, db_session: AsyncSession
) -> None:
    chat_a = await _create_chat(api_client, "Chat A forge")
    chat_b = await _create_chat(api_client, "Chat B forge")
    _doc_a = await _upload_doc(api_client, chat_a, "a.pdf", b"%PDF-1.4 forge A")
    _doc_b = await _upload_doc(api_client, chat_b, "b.pdf", b"%PDF-1.4 forge B")
    sess_b = await _create_session(api_client, chat_b)
    await db_session.commit()

    # Seed a hit whose chat_id is A but request targets B.
    forged = make_hit(chat_id=chat_a, document_id=_doc_a, content="Should never reach chat B.")
    builder = make_provider_builder([forged])

    with _patch_providers(builder):
        resp = await api_client.post(
            f"/chats/{chat_b}/sessions/{sess_b}/messages",
            json={"question": "Describe the paper.", "stream": False},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["citations"] == []
    assert "not enough information" in body["answer"].lower()


# ---------------------------------------------------------------------------
# 6. SSE streaming path: citations in the `done` event are chat-scoped.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sse_done_event_carries_only_in_scope_citations(
    api_client: AsyncClient, db_session: AsyncSession
) -> None:
    chat_a = await _create_chat(api_client, "Chat A sse")
    chat_b = await _create_chat(api_client, "Chat B sse")
    doc_a = await _upload_doc(api_client, chat_a, "a.pdf", b"%PDF-1.4 sse A")
    doc_b = await _upload_doc(api_client, chat_b, "b.pdf", b"%PDF-1.4 sse B")
    sess_a = await _create_session(api_client, chat_a)
    await db_session.commit()

    hits = [
        make_hit(chat_id=chat_a, document_id=doc_a, content="A SSE evidence."),
        make_hit(chat_id=chat_b, document_id=doc_b, content="B SSE evidence — must not appear."),
    ]
    builder = make_provider_builder(hits)

    with _patch_providers(builder):
        resp = await api_client.post(
            f"/chats/{chat_a}/sessions/{sess_a}/messages",
            json={"question": "Describe the paper.", "stream": True},
        )

    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers.get("content-type", "")
    body = resp.text
    # The `done` event carries the full QAResponse JSON — assert no foreign doc id slipped in.
    assert str(doc_b) not in body
    assert "event: done" in body or "event: token" in body
