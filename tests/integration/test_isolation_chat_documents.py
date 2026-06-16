"""Integration isolation tests — Chat A must not see Chat B's documents.

CLAUDE.md §2 isolation layers covered
--------------------------------------
- **API authorization scope**: route layer rejects requests where the
  ``chat_id`` path parameter does not own the requested resource.
- **Relational DB query**: ``document_service`` enforces
  ``WHERE chat_documents.chat_id = :current_chat_id``.

Scenarios
---------
(a) GET  /chats/{B}/documents/{X}  → 404  (X belongs to A)
(b) GET  /chats/{B}/documents      → list does not contain X
(c) DELETE /chats/{B}/documents/{X} → 404; X still exists in Chat A's scope
"""

from __future__ import annotations

import io
import uuid
from pathlib import Path

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from tests._helpers import make_chat

# ---------------------------------------------------------------------------
# Dependency override helpers (reuse pattern from test_documents_api.py)
# ---------------------------------------------------------------------------


def _override_storage(tmp_path: Path):  # type: ignore[no-untyped-def]
    from app.storage.local import LocalBlobStorage

    def _get() -> LocalBlobStorage:
        return LocalBlobStorage(root=tmp_path)

    return _get


def _override_indexer():  # type: ignore[no-untyped-def]
    from app.services.vespa_indexer import NullVespaIndexer

    def _get() -> NullVespaIndexer:
        return NullVespaIndexer()

    return _get


def _pdf_file(content: bytes = b"%PDF-1.4 isolation-test", filename: str = "iso.pdf"):  # type: ignore[no-untyped-def]
    return ("file", (filename, io.BytesIO(content), "application/pdf"))


async def _upload_to_chat(
    api_client: AsyncClient,
    chat_id: uuid.UUID,
    content: bytes = b"%PDF-1.4 isolation-test",
    filename: str = "iso.pdf",
) -> str:
    """Upload a PDF to the chat and return the document id string."""
    resp = await api_client.post(
        f"/chats/{chat_id}/documents",
        files=[_pdf_file(content, filename)],
    )
    assert resp.status_code == 201, f"Upload failed: {resp.text}"
    return resp.json()["id"]


# ---------------------------------------------------------------------------
# (a) GET /chats/{B}/documents/{X} → 404
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_document_cross_chat_returns_404(
    api_client: AsyncClient,
    db_session: AsyncSession,
    tmp_path: Path,
) -> None:
    """(a) Chat A uploads doc X; GET /chats/{B}/documents/{X} → 404.

    Layer: API authorization scope + Service layer DB query.
    Chat B must receive 404 — not the document, not a 403 (existence leak).
    """
    from app.api.documents import get_indexer, get_storage
    from app.main import app

    app.dependency_overrides[get_storage] = _override_storage(tmp_path)
    app.dependency_overrides[get_indexer] = _override_indexer()
    try:
        chat_a = await make_chat(db_session, "iso-a-get-doc")
        chat_b = await make_chat(db_session, "iso-b-get-doc")

        doc_x_id = await _upload_to_chat(api_client, chat_a.id)

        resp = await api_client.get(f"/chats/{chat_b.id}/documents/{doc_x_id}")
        assert resp.status_code == 404, (
            f"Expected 404 but got {resp.status_code}; "
            "Chat B must not retrieve Chat A's document"
        )
    finally:
        app.dependency_overrides.pop(get_storage, None)
        app.dependency_overrides.pop(get_indexer, None)


# ---------------------------------------------------------------------------
# (b) GET /chats/{B}/documents → list does not contain X
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_documents_cross_chat_excludes_other_chat_docs(
    api_client: AsyncClient,
    db_session: AsyncSession,
    tmp_path: Path,
) -> None:
    """(b) Chat A uploads doc X; GET /chats/{B}/documents list does not contain X.

    Layer: API authorization scope + Service layer DB query.
    """
    from app.api.documents import get_indexer, get_storage
    from app.main import app

    app.dependency_overrides[get_storage] = _override_storage(tmp_path)
    app.dependency_overrides[get_indexer] = _override_indexer()
    try:
        chat_a = await make_chat(db_session, "iso-a-list")
        chat_b = await make_chat(db_session, "iso-b-list")

        doc_x_id = await _upload_to_chat(
            api_client, chat_a.id, filename="x_in_a.pdf"
        )

        resp = await api_client.get(f"/chats/{chat_b.id}/documents")
        assert resp.status_code == 200, resp.text
        listed_ids = {d["id"] for d in resp.json()}
        assert doc_x_id not in listed_ids, (
            f"Document {doc_x_id} from Chat A must not appear in Chat B's list; "
            f"got ids: {listed_ids}"
        )
    finally:
        app.dependency_overrides.pop(get_storage, None)
        app.dependency_overrides.pop(get_indexer, None)


# ---------------------------------------------------------------------------
# (c) DELETE /chats/{B}/documents/{X} → 404; X still in Chat A
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delete_document_cross_chat_returns_404_and_doc_survives(
    api_client: AsyncClient,
    db_session: AsyncSession,
    tmp_path: Path,
) -> None:
    """(c) Chat A uploads doc X; DELETE /chats/{B}/documents/{X} → 404.
    Doc X must still be retrievable via Chat A.

    Layer: API authorization scope + Service layer DB query.
    """
    from app.api.documents import get_indexer, get_storage
    from app.main import app

    app.dependency_overrides[get_storage] = _override_storage(tmp_path)
    app.dependency_overrides[get_indexer] = _override_indexer()
    try:
        chat_a = await make_chat(db_session, "iso-a-del")
        chat_b = await make_chat(db_session, "iso-b-del")

        doc_x_id = await _upload_to_chat(api_client, chat_a.id, filename="del_target.pdf")

        # Cross-chat delete must fail
        del_resp = await api_client.delete(f"/chats/{chat_b.id}/documents/{doc_x_id}")
        assert del_resp.status_code == 404, (
            f"Expected 404 but got {del_resp.status_code}; "
            "Chat B must not be able to delete Chat A's document"
        )

        # Document still accessible from Chat A
        get_resp = await api_client.get(f"/chats/{chat_a.id}/documents/{doc_x_id}")
        assert get_resp.status_code == 200, (
            f"Doc X must still exist under Chat A after a failed cross-chat delete; "
            f"got {get_resp.status_code}"
        )
        assert get_resp.json()["id"] == doc_x_id
    finally:
        app.dependency_overrides.pop(get_storage, None)
        app.dependency_overrides.pop(get_indexer, None)
