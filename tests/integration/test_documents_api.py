"""Integration tests for the Document API endpoints.

Tests the full HTTP → service → DB round-trip using the FastAPI test client
and the ``db_session`` / ``api_client`` fixtures from ``tests/conftest.py``.

Isolation coverage (CLAUDE.md §2 mandatory gates)
--------------------------------------------------
- GET /chats/{A}/documents/{B_doc_id} must return 404 — must not leak that
  the document exists in Chat B.
- DELETE /chats/{A}/documents/{B_doc_id} must return 404.
- Only after a document has no more associations is it fully deleted.
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
from app.services.vespa_indexer import NullVespaIndexer
from app.storage.local import LocalBlobStorage

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FAKE_PDF_BYTES = b"%PDF-1.4 integration test pdf content"
_FAKE_PDF_BYTES_2 = b"%PDF-1.4 second integration test pdf content"


async def _create_chat(db: AsyncSession, name: str = "Test Chat") -> uuid.UUID:
    chat = await chat_service.create_chat(db, ChatCreate(name=name))
    return chat.id


def _pdf_file(content: bytes = _FAKE_PDF_BYTES, filename: str = "test.pdf"):  # type: ignore[no-untyped-def]
    """Return a tuple suitable for httpx multipart upload."""
    return ("file", (filename, io.BytesIO(content), "application/pdf"))


# ---------------------------------------------------------------------------
# Override storage dependency with tmp_path-based storage
# ---------------------------------------------------------------------------


def _override_storage(tmp_path: Path):  # type: ignore[no-untyped-def]
    """Return a DI override that injects a LocalBlobStorage backed by tmp_path."""

    def _get_storage() -> LocalBlobStorage:
        return LocalBlobStorage(root=tmp_path)

    return _get_storage


def _override_indexer():  # type: ignore[no-untyped-def]
    def _get_indexer() -> NullVespaIndexer:
        return NullVespaIndexer()

    return _get_indexer


# ---------------------------------------------------------------------------
# POST /chats/{chat_id}/documents — upload
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_upload_document_201(
    api_client: AsyncClient, db_session: AsyncSession, tmp_path: Path
) -> None:
    """POST upload returns 201 with a DocumentRead body."""
    from app.api.documents import get_indexer, get_storage
    from app.main import app

    app.dependency_overrides[get_storage] = _override_storage(tmp_path)
    app.dependency_overrides[get_indexer] = _override_indexer()

    try:
        chat_id = await _create_chat(db_session, "Upload Chat")
        resp = await api_client.post(
            f"/chats/{chat_id}/documents",
            files=[_pdf_file()],
        )

        assert resp.status_code == 201, resp.text
        body = resp.json()
        assert body["chat_id"] == str(chat_id)
        assert body["original_filename"] == "test.pdf"
        assert body["status"] == "uploaded"
        assert body["mime_type"] == "application/pdf"
        assert "id" in body
        assert "checksum_sha256" in body
    finally:
        app.dependency_overrides.pop(get_storage, None)
        app.dependency_overrides.pop(get_indexer, None)


@pytest.mark.asyncio
async def test_upload_document_invalid_mime_400(
    api_client: AsyncClient, db_session: AsyncSession, tmp_path: Path
) -> None:
    """POST with a non-PDF file returns 400."""
    from app.api.documents import get_indexer, get_storage
    from app.main import app

    app.dependency_overrides[get_storage] = _override_storage(tmp_path)
    app.dependency_overrides[get_indexer] = _override_indexer()

    try:
        chat_id = await _create_chat(db_session, "MIME Check Chat")
        resp = await api_client.post(
            f"/chats/{chat_id}/documents",
            files=[("file", ("page.html", io.BytesIO(b"<html/>"), "text/html"))],
        )
        assert resp.status_code == 400, resp.text
    finally:
        app.dependency_overrides.pop(get_storage, None)
        app.dependency_overrides.pop(get_indexer, None)


@pytest.mark.asyncio
async def test_upload_document_unknown_chat_404(
    api_client: AsyncClient, tmp_path: Path
) -> None:
    """POST to a non-existent chat returns 404."""
    from app.api.documents import get_indexer, get_storage
    from app.main import app

    app.dependency_overrides[get_storage] = _override_storage(tmp_path)
    app.dependency_overrides[get_indexer] = _override_indexer()

    try:
        resp = await api_client.post(
            f"/chats/{uuid.uuid4()}/documents",
            files=[_pdf_file()],
        )
        assert resp.status_code == 404, resp.text
    finally:
        app.dependency_overrides.pop(get_storage, None)
        app.dependency_overrides.pop(get_indexer, None)


@pytest.mark.asyncio
async def test_upload_document_duplicate_409(
    api_client: AsyncClient, db_session: AsyncSession, tmp_path: Path
) -> None:
    """POST with duplicate checksum returns 409 with the existing document_id."""
    from app.api.documents import get_indexer, get_storage
    from app.main import app

    app.dependency_overrides[get_storage] = _override_storage(tmp_path)
    app.dependency_overrides[get_indexer] = _override_indexer()

    try:
        chat_id = await _create_chat(db_session, "Dup Chat")

        first_resp = await api_client.post(
            f"/chats/{chat_id}/documents",
            files=[_pdf_file()],
        )
        assert first_resp.status_code == 201
        first_id = first_resp.json()["id"]

        dup_resp = await api_client.post(
            f"/chats/{chat_id}/documents",
            files=[_pdf_file(filename="paper_dup.pdf")],  # same bytes, different name
        )
        assert dup_resp.status_code == 409, dup_resp.text
        # FastAPI wraps dict detail under {"detail": {...}}
        dup_body = dup_resp.json()
        inner = dup_body.get("detail", dup_body)
        if isinstance(inner, dict):
            assert inner.get("document_id") == first_id
        else:
            assert first_id in str(inner)
    finally:
        app.dependency_overrides.pop(get_storage, None)
        app.dependency_overrides.pop(get_indexer, None)


# ---------------------------------------------------------------------------
# GET /chats/{chat_id}/documents — list
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_documents(
    api_client: AsyncClient, db_session: AsyncSession, tmp_path: Path
) -> None:
    """GET list returns uploaded documents."""
    from app.api.documents import get_indexer, get_storage
    from app.main import app

    app.dependency_overrides[get_storage] = _override_storage(tmp_path)
    app.dependency_overrides[get_indexer] = _override_indexer()

    try:
        chat_id = await _create_chat(db_session, "List Chat")

        await api_client.post(
            f"/chats/{chat_id}/documents",
            files=[_pdf_file(content=_FAKE_PDF_BYTES, filename="first.pdf")],
        )
        await api_client.post(
            f"/chats/{chat_id}/documents",
            files=[_pdf_file(content=_FAKE_PDF_BYTES_2, filename="second.pdf")],
        )

        resp = await api_client.get(f"/chats/{chat_id}/documents")
        assert resp.status_code == 200, resp.text
        docs = resp.json()
        assert len(docs) == 2
        filenames = {d["original_filename"] for d in docs}
        assert filenames == {"first.pdf", "second.pdf"}
    finally:
        app.dependency_overrides.pop(get_storage, None)
        app.dependency_overrides.pop(get_indexer, None)


@pytest.mark.asyncio
async def test_list_documents_empty(
    api_client: AsyncClient, db_session: AsyncSession
) -> None:
    """GET list for a chat with no documents returns an empty list."""
    chat_id = await _create_chat(db_session, "Empty Doc Chat")
    resp = await api_client.get(f"/chats/{chat_id}/documents")
    assert resp.status_code == 200
    assert resp.json() == []


# ---------------------------------------------------------------------------
# GET /chats/{chat_id}/documents/{doc_id} — get single
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_document_success(
    api_client: AsyncClient, db_session: AsyncSession, tmp_path: Path
) -> None:
    """GET by ID returns the correct document."""
    from app.api.documents import get_indexer, get_storage
    from app.main import app

    app.dependency_overrides[get_storage] = _override_storage(tmp_path)
    app.dependency_overrides[get_indexer] = _override_indexer()

    try:
        chat_id = await _create_chat(db_session, "Get Doc Chat")
        upload_resp = await api_client.post(
            f"/chats/{chat_id}/documents",
            files=[_pdf_file()],
        )
        doc_id = upload_resp.json()["id"]

        get_resp = await api_client.get(f"/chats/{chat_id}/documents/{doc_id}")
        assert get_resp.status_code == 200, get_resp.text
        assert get_resp.json()["id"] == doc_id
    finally:
        app.dependency_overrides.pop(get_storage, None)
        app.dependency_overrides.pop(get_indexer, None)


# ---------------------------------------------------------------------------
# Cross-chat isolation tests (CLAUDE.md §2 mandatory gate)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_document_cross_chat_404(
    api_client: AsyncClient, db_session: AsyncSession, tmp_path: Path
) -> None:
    """GET /chats/A/documents/{B_doc_id} returns 404.

    Chat A must not learn about a document that belongs to Chat B.
    """
    from app.api.documents import get_indexer, get_storage
    from app.main import app

    app.dependency_overrides[get_storage] = _override_storage(tmp_path)
    app.dependency_overrides[get_indexer] = _override_indexer()

    try:
        chat_a_id = await _create_chat(db_session, "Iso-A")
        chat_b_id = await _create_chat(db_session, "Iso-B")

        # Upload a document to Chat B.
        upload_resp = await api_client.post(
            f"/chats/{chat_b_id}/documents",
            files=[_pdf_file()],
        )
        b_doc_id = upload_resp.json()["id"]

        # Try to access from Chat A — must be 404.
        resp = await api_client.get(f"/chats/{chat_a_id}/documents/{b_doc_id}")
        assert resp.status_code == 404, (
            f"Expected 404, got {resp.status_code}. "
            "Chat A must not be able to retrieve Chat B's document."
        )
    finally:
        app.dependency_overrides.pop(get_storage, None)
        app.dependency_overrides.pop(get_indexer, None)


@pytest.mark.asyncio
async def test_delete_document_cross_chat_404(
    api_client: AsyncClient, db_session: AsyncSession, tmp_path: Path
) -> None:
    """DELETE /chats/A/documents/{B_doc_id} returns 404."""
    from app.api.documents import get_indexer, get_storage
    from app.main import app

    app.dependency_overrides[get_storage] = _override_storage(tmp_path)
    app.dependency_overrides[get_indexer] = _override_indexer()

    try:
        chat_a_id = await _create_chat(db_session, "Del-Iso-A")
        chat_b_id = await _create_chat(db_session, "Del-Iso-B")

        upload_resp = await api_client.post(
            f"/chats/{chat_b_id}/documents",
            files=[_pdf_file()],
        )
        b_doc_id = upload_resp.json()["id"]

        resp = await api_client.delete(f"/chats/{chat_a_id}/documents/{b_doc_id}")
        assert resp.status_code == 404, (
            f"Expected 404, got {resp.status_code}. "
            "Chat A must not be able to delete Chat B's document."
        )
    finally:
        app.dependency_overrides.pop(get_storage, None)
        app.dependency_overrides.pop(get_indexer, None)


# ---------------------------------------------------------------------------
# DELETE /chats/{chat_id}/documents/{doc_id}
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delete_document_204_then_404(
    api_client: AsyncClient, db_session: AsyncSession, tmp_path: Path
) -> None:
    """DELETE returns 204; subsequent GET returns 404."""
    from app.api.documents import get_indexer, get_storage
    from app.main import app

    app.dependency_overrides[get_storage] = _override_storage(tmp_path)
    app.dependency_overrides[get_indexer] = _override_indexer()

    try:
        chat_id = await _create_chat(db_session, "Del Chat")
        upload_resp = await api_client.post(
            f"/chats/{chat_id}/documents",
            files=[_pdf_file()],
        )
        doc_id = upload_resp.json()["id"]

        del_resp = await api_client.delete(f"/chats/{chat_id}/documents/{doc_id}")
        assert del_resp.status_code == 204, del_resp.text

        get_resp = await api_client.get(f"/chats/{chat_id}/documents/{doc_id}")
        assert get_resp.status_code == 404
    finally:
        app.dependency_overrides.pop(get_storage, None)
        app.dependency_overrides.pop(get_indexer, None)


# ---------------------------------------------------------------------------
# POST /chats/{chat_id}/documents/{doc_id}/associate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_associate_document_201(
    api_client: AsyncClient, db_session: AsyncSession, tmp_path: Path
) -> None:
    """POST .../associate creates cross-chat sharing and returns 201."""
    from app.api.documents import get_indexer, get_storage
    from app.main import app

    app.dependency_overrides[get_storage] = _override_storage(tmp_path)
    app.dependency_overrides[get_indexer] = _override_indexer()

    try:
        chat_a_id = await _create_chat(db_session, "Assoc-A")
        chat_b_id = await _create_chat(db_session, "Assoc-B")

        # Upload to Chat A.
        upload_resp = await api_client.post(
            f"/chats/{chat_a_id}/documents",
            files=[_pdf_file()],
        )
        doc_id = upload_resp.json()["id"]

        # Associate to Chat B.
        assoc_resp = await api_client.post(
            f"/chats/{chat_b_id}/documents/{doc_id}/associate",
            json={"source_chat_id": str(chat_a_id)},
        )
        assert assoc_resp.status_code == 201, assoc_resp.text
        body = assoc_resp.json()
        assert body["chat_id"] == str(chat_b_id)
        assert body["document_id"] == doc_id

        # Chat B can now access the document.
        get_resp = await api_client.get(f"/chats/{chat_b_id}/documents/{doc_id}")
        assert get_resp.status_code == 200
    finally:
        app.dependency_overrides.pop(get_storage, None)
        app.dependency_overrides.pop(get_indexer, None)


@pytest.mark.asyncio
async def test_associate_document_duplicate_409(
    api_client: AsyncClient, db_session: AsyncSession, tmp_path: Path
) -> None:
    """POST .../associate twice returns 409."""
    from app.api.documents import get_indexer, get_storage
    from app.main import app

    app.dependency_overrides[get_storage] = _override_storage(tmp_path)
    app.dependency_overrides[get_indexer] = _override_indexer()

    try:
        chat_a_id = await _create_chat(db_session, "DupAssoc-A")
        chat_b_id = await _create_chat(db_session, "DupAssoc-B")

        upload_resp = await api_client.post(
            f"/chats/{chat_a_id}/documents",
            files=[_pdf_file()],
        )
        doc_id = upload_resp.json()["id"]

        await api_client.post(
            f"/chats/{chat_b_id}/documents/{doc_id}/associate",
            json={"source_chat_id": str(chat_a_id)},
        )

        dup_resp = await api_client.post(
            f"/chats/{chat_b_id}/documents/{doc_id}/associate",
            json={"source_chat_id": str(chat_a_id)},
        )
        assert dup_resp.status_code == 409, dup_resp.text
    finally:
        app.dependency_overrides.pop(get_storage, None)
        app.dependency_overrides.pop(get_indexer, None)
