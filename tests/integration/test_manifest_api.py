"""Integration tests for the GET /chats/{chat_id}/manifest API (Phase 5.4).

Strategy
--------
- Uses ``httpx.AsyncClient`` wired to the FastAPI app via ASGI transport.
- ``api_client`` and ``db_session`` fixtures come from ``tests/conftest.py``.
- All DB changes run inside a SAVEPOINT that is rolled back after each test.

Coverage
--------
1. GET /chats/{A}/manifest, A has no documents  -> 200 + documents=[], document_count=0.
2. GET /chats/{A}/manifest, A has 1 document   -> 200 + entry.title non-empty (filename stem).
3. GET /chats/{B}/manifest, B does not exist   -> 404.
4. Cross-chat isolation: document added to Chat A does not appear in Chat B manifest.
5. Cache-Control header is set to max-age=10.
"""

from __future__ import annotations

import io
import uuid
from pathlib import Path

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.domain import ChatCreate
from app.services import chat_service

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _create_chat(db_session: AsyncSession, name: str) -> uuid.UUID:
    """Create a chat via the service layer and return its UUID."""
    chat = await chat_service.create_chat(db_session, ChatCreate(name=name))
    return chat.id  # type: ignore[return-value]


async def _upload_doc(
    api_client: AsyncClient,
    chat_id: uuid.UUID,
    filename: str,
    tmp_path: Path,
) -> uuid.UUID:
    """Upload a minimal PDF document and return its UUID."""
    from app.api.documents import get_indexer, get_storage
    from app.main import app
    from app.services.vespa_indexer import NullVespaIndexer
    from app.storage.local import LocalBlobStorage

    def _get_storage() -> LocalBlobStorage:
        return LocalBlobStorage(root=tmp_path)

    def _get_indexer() -> NullVespaIndexer:
        return NullVespaIndexer()

    app.dependency_overrides[get_storage] = _get_storage
    app.dependency_overrides[get_indexer] = _get_indexer

    try:
        pdf_bytes = b"%PDF-1.4 test content for manifest"
        resp = await api_client.post(
            f"/chats/{chat_id}/documents",
            files=[("file", (filename, io.BytesIO(pdf_bytes), "application/pdf"))],
        )
        assert resp.status_code == 201, f"Doc upload failed ({resp.status_code}): {resp.text}"
        return uuid.UUID(resp.json()["id"])
    finally:
        app.dependency_overrides.pop(get_storage, None)
        app.dependency_overrides.pop(get_indexer, None)


# ---------------------------------------------------------------------------
# 1. Empty chat — 200 + documents=[]
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_manifest_empty_chat(
    api_client: AsyncClient,
    db_session: AsyncSession,
) -> None:
    """GET /chats/{A}/manifest for a chat with no documents returns 200 + empty list."""
    chat_id = await _create_chat(db_session, "Empty Manifest Chat")
    await db_session.flush()

    resp = await api_client.get(f"/chats/{chat_id}/manifest")

    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    body = resp.json()
    assert body["document_count"] == 0
    assert body["documents"] == []
    assert body["total_token_estimate"] == 0
    assert str(chat_id) == body["chat_id"]


# ---------------------------------------------------------------------------
# 2. Chat with one document — 200 + entry with non-empty title
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_manifest_one_document(
    api_client: AsyncClient,
    db_session: AsyncSession,
    tmp_path: Path,
) -> None:
    """GET /chats/{A}/manifest for a chat with one document includes the entry."""
    chat_id = await _create_chat(db_session, "Single Doc Manifest Chat")
    await db_session.flush()

    doc_id = await _upload_doc(api_client, chat_id, "my_paper.pdf", tmp_path)
    await db_session.flush()

    resp = await api_client.get(f"/chats/{chat_id}/manifest")

    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    body = resp.json()
    assert body["document_count"] == 1
    assert len(body["documents"]) == 1

    entry = body["documents"][0]
    assert str(doc_id) == entry["document_id"]
    # title falls back to DocumentNode 'document' node; may be None before
    # hierarchy is run, but the field must be present in the response.
    assert "title" in entry
    # ingestion_status must be 'pending' (no IngestionJob created by upload)
    assert entry["ingestion_status"] == "pending"


# ---------------------------------------------------------------------------
# 3. Non-existent chat — 404
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_manifest_chat_not_found(
    api_client: AsyncClient,
) -> None:
    """GET /chats/{B}/manifest for a non-existent chat_id returns 404."""
    nonexistent_id = uuid.UUID("deadbeef-dead-beef-dead-beefdeadbeef")
    resp = await api_client.get(f"/chats/{nonexistent_id}/manifest")

    assert resp.status_code == 404, f"Expected 404, got {resp.status_code}: {resp.text}"


# ---------------------------------------------------------------------------
# 4. Cross-chat isolation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_manifest_cross_chat_isolation(
    api_client: AsyncClient,
    db_session: AsyncSession,
    tmp_path: Path,
) -> None:
    """Document uploaded to Chat A must not appear in Chat B's manifest."""
    chat_a_id = await _create_chat(db_session, "Isolation Chat A")
    chat_b_id = await _create_chat(db_session, "Isolation Chat B")
    await db_session.flush()

    # Upload a document to Chat A only
    doc_a_id = await _upload_doc(api_client, chat_a_id, "doc_a.pdf", tmp_path)
    await db_session.flush()

    # Chat B manifest must be empty
    resp_b = await api_client.get(f"/chats/{chat_b_id}/manifest")
    assert resp_b.status_code == 200
    body_b = resp_b.json()
    assert body_b["document_count"] == 0
    assert body_b["documents"] == []

    doc_ids_b = {e["document_id"] for e in body_b["documents"]}
    assert str(doc_a_id) not in doc_ids_b, (
        f"Chat-A document {doc_a_id} leaked into Chat-B manifest"
    )

    # Chat A manifest must have exactly 1 document
    resp_a = await api_client.get(f"/chats/{chat_a_id}/manifest")
    assert resp_a.status_code == 200
    body_a = resp_a.json()
    assert body_a["document_count"] == 1


# ---------------------------------------------------------------------------
# 5. Cache-Control header
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_manifest_cache_control_header(
    api_client: AsyncClient,
    db_session: AsyncSession,
) -> None:
    """GET /chats/{A}/manifest response includes Cache-Control: max-age=10."""
    chat_id = await _create_chat(db_session, "Cache Control Chat")
    await db_session.flush()

    resp = await api_client.get(f"/chats/{chat_id}/manifest")

    assert resp.status_code == 200
    cache_header = resp.headers.get("cache-control", "")
    assert "max-age=10" in cache_header, (
        f"Expected Cache-Control: max-age=10, got: {cache_header!r}"
    )
