"""Integration isolation tests — cross-chat document sharing lifecycle.

CLAUDE.md §2 isolation layers covered
--------------------------------------
- **API authorization scope + Service layer**: associate endpoint validates
  source ownership before creating the association.
- **Relational DB query**: ChatDocument association drives visibility; removing
  one chat's association must not affect the other chat's visibility.
- **Vespa spy (mock)**: delete_by_document must carry a non-None chat_id
  and must be called exactly once when the last association is removed.

Scenarios
---------
(f) Cross-chat sharing happy-path:
      A uploads X → POST /chats/{B}/documents/{X}/associate (source=A)
      → Chat A list contains X; Chat B list also contains X.
(g) Partial deletion:
      A deletes X (B still shares it) → Chat A list excludes X; Chat B still has X;
      indexer.delete NOT called; storage file still present.
(h) Full deletion:
      B also deletes X → indexer.delete called exactly once; storage file gone;
      Document ORM row gone.
(i) Source chat does not own the doc → associate raises/returns 404.
(m) Vespa spy: the chat_id passed to delete_by_document is not None and is
    the caller's chat_id (i.e. the last remover's chat_id), never omitted.
"""

from __future__ import annotations

import io
import uuid
from pathlib import Path

import pytest
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.orm import Document
from app.services.vespa_indexer import VespaIndexer
from tests._helpers import SpyIndexer, make_chat

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _override_storage(tmp_path: Path):  # type: ignore[no-untyped-def]
    from app.storage.local import LocalBlobStorage

    def _get() -> LocalBlobStorage:
        return LocalBlobStorage(root=tmp_path)

    return _get


def _override_indexer_factory(spy: SpyIndexer):  # type: ignore[no-untyped-def]
    def _get() -> VespaIndexer:
        return spy  # type: ignore[return-value]

    return _get


def _pdf(content: bytes = b"%PDF-1.4 cross-chat-sharing-test", filename: str = "shared.pdf"):  # type: ignore[no-untyped-def]
    return ("file", (filename, io.BytesIO(content), "application/pdf"))


async def _upload(api_client: AsyncClient, chat_id: uuid.UUID, filename: str = "doc.pdf") -> str:
    resp = await api_client.post(
        f"/chats/{chat_id}/documents",
        files=[_pdf(filename=filename)],
    )
    assert resp.status_code == 201, f"Upload to {chat_id} failed: {resp.text}"
    return resp.json()["id"]


async def _associate(
    api_client: AsyncClient,
    target_chat_id: uuid.UUID,
    doc_id: str,
    source_chat_id: uuid.UUID,
) -> int:
    resp = await api_client.post(
        f"/chats/{target_chat_id}/documents/{doc_id}/associate",
        json={"source_chat_id": str(source_chat_id)},
    )
    return resp.status_code


async def _list_doc_ids(api_client: AsyncClient, chat_id: uuid.UUID) -> set[str]:
    resp = await api_client.get(f"/chats/{chat_id}/documents")
    assert resp.status_code == 200
    return {d["id"] for d in resp.json()}


# ---------------------------------------------------------------------------
# (f) Happy-path: both chats can see the shared document after associate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_associate_both_chats_can_see_document(
    api_client: AsyncClient,
    db_session: AsyncSession,
    tmp_path: Path,
) -> None:
    """(f) A uploads X; B associates X; both A and B can list/get X.

    Layer: API authorization scope + Service layer (ChatDocument association).
    """
    from app.api.documents import get_indexer, get_storage
    from app.main import app

    spy = SpyIndexer()
    app.dependency_overrides[get_storage] = _override_storage(tmp_path)
    app.dependency_overrides[get_indexer] = _override_indexer_factory(spy)
    try:
        chat_a = await make_chat(db_session, "f-assoc-A")
        chat_b = await make_chat(db_session, "f-assoc-B")

        doc_id = await _upload(api_client, chat_a.id, "shared-f.pdf")

        status = await _associate(api_client, chat_b.id, doc_id, chat_a.id)
        assert status == 201, f"Associate should return 201, got {status}"

        ids_a = await _list_doc_ids(api_client, chat_a.id)
        ids_b = await _list_doc_ids(api_client, chat_b.id)

        assert doc_id in ids_a, "Chat A must still see the document after sharing"
        assert doc_id in ids_b, "Chat B must see the document after association"

        # Both chats can GET the document directly
        get_a = await api_client.get(f"/chats/{chat_a.id}/documents/{doc_id}")
        assert get_a.status_code == 200
        get_b = await api_client.get(f"/chats/{chat_b.id}/documents/{doc_id}")
        assert get_b.status_code == 200
    finally:
        app.dependency_overrides.pop(get_storage, None)
        app.dependency_overrides.pop(get_indexer, None)


# ---------------------------------------------------------------------------
# (g) Partial deletion: A removes → only A loses visibility; indexer NOT called
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_partial_deletion_a_removes_b_still_has_document(
    api_client: AsyncClient,
    db_session: AsyncSession,
    tmp_path: Path,
) -> None:
    """(g) A uploads X; B associates; A deletes X.
    → Chat A list excludes X; Chat B still has X; indexer NOT called; file still exists.

    Layer: API authorization scope + Service layer cascade logic.
    """
    from app.api.documents import get_indexer, get_storage
    from app.main import app
    from app.storage.local import LocalBlobStorage

    spy = SpyIndexer()
    storage = LocalBlobStorage(root=tmp_path)
    app.dependency_overrides[get_storage] = _override_storage(tmp_path)
    app.dependency_overrides[get_indexer] = _override_indexer_factory(spy)
    try:
        chat_a = await make_chat(db_session, "g-partial-A")
        chat_b = await make_chat(db_session, "g-partial-B")

        doc_id = await _upload(api_client, chat_a.id, "partial-g.pdf")
        await _associate(api_client, chat_b.id, doc_id, chat_a.id)

        # Chat A deletes the document from its scope
        del_resp = await api_client.delete(f"/chats/{chat_a.id}/documents/{doc_id}")
        assert del_resp.status_code == 204, f"Delete should return 204, got {del_resp.status_code}"

        # Chat A list must NOT contain the document
        ids_a = await _list_doc_ids(api_client, chat_a.id)
        assert doc_id not in ids_a, "Chat A must not see the document after deleting it"

        # Chat B must STILL see the document
        ids_b = await _list_doc_ids(api_client, chat_b.id)
        assert doc_id in ids_b, "Chat B must still see the document when it is the only remaining reference"

        # indexer.delete_by_document must NOT have been called
        assert spy.call_count == 0, (
            f"indexer.delete_by_document must not be called while Chat B still holds the document; "
            f"got {spy.call_count} call(s): {spy.calls}"
        )

        # Storage file must still exist
        # Get the doc storage path from the DB to verify on disk
        doc_row = await db_session.scalar(select(Document).where(Document.id == uuid.UUID(doc_id)))
        assert doc_row is not None, "Document ORM row must still exist after partial delete"
        file_on_disk = await storage.exists(
            doc_row.chat_id, doc_row.id, doc_row.original_filename
        )
        assert file_on_disk, "Storage file must still exist when another chat still references the document"
    finally:
        app.dependency_overrides.pop(get_storage, None)
        app.dependency_overrides.pop(get_indexer, None)


# ---------------------------------------------------------------------------
# (h) Full deletion: B also removes → indexer called once; file gone; ORM row gone
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_full_deletion_last_owner_triggers_indexer_and_removes_file(
    api_client: AsyncClient,
    db_session: AsyncSession,
    tmp_path: Path,
) -> None:
    """(h) A uploads X; B associates; A deletes; B deletes.
    → indexer.delete called exactly once; storage file gone; Document ORM row gone.

    Layer: API authorization scope + Service layer cascade + Vespa spy (m).
    """
    from app.api.documents import get_indexer, get_storage
    from app.main import app
    from app.storage.local import LocalBlobStorage

    spy = SpyIndexer()
    storage = LocalBlobStorage(root=tmp_path)
    app.dependency_overrides[get_storage] = _override_storage(tmp_path)
    app.dependency_overrides[get_indexer] = _override_indexer_factory(spy)
    try:
        chat_a = await make_chat(db_session, "h-full-A")
        chat_b = await make_chat(db_session, "h-full-B")

        doc_id = await _upload(api_client, chat_a.id, "full-h.pdf")
        await _associate(api_client, chat_b.id, doc_id, chat_a.id)

        # Step 1: Chat A removes its association
        del_a = await api_client.delete(f"/chats/{chat_a.id}/documents/{doc_id}")
        assert del_a.status_code == 204
        assert spy.call_count == 0, "Indexer must not be called after first partial delete"

        # Step 2: Chat B removes its association → last reference → full delete
        del_b = await api_client.delete(f"/chats/{chat_b.id}/documents/{doc_id}")
        assert del_b.status_code == 204

        # Indexer must have been called exactly once
        assert spy.call_count == 1, (
            f"indexer.delete_by_document must be called exactly once on full delete; "
            f"got {spy.call_count} call(s)"
        )

        # (m) Vespa spy: the chat_id passed must not be None
        called_chat_id, called_doc_id = spy.calls[0]
        assert called_chat_id is not None, (
            "chat_id passed to indexer.delete_by_document must never be None"
        )
        assert called_doc_id == uuid.UUID(doc_id), (
            f"document_id passed to indexer must match the deleted document; "
            f"expected {doc_id}, got {called_doc_id}"
        )
        # The caller is Chat B (the last remover), so chat_id should be chat_b.id
        assert called_chat_id == chat_b.id, (
            f"chat_id passed to indexer must be the caller's chat_id (chat_b.id={chat_b.id}); "
            f"got {called_chat_id}"
        )

        # Document ORM row must be gone
        doc_row = await db_session.scalar(
            select(Document).where(Document.id == uuid.UUID(doc_id))
        )
        assert doc_row is None, "Document ORM row must be deleted after all associations are removed"

        # Storage file must be gone
        file_gone = not await storage.exists(
            chat_b.id, uuid.UUID(doc_id), "full-h.pdf"
        )
        assert file_gone, "Storage file must be deleted after all associations are removed"
    finally:
        app.dependency_overrides.pop(get_storage, None)
        app.dependency_overrides.pop(get_indexer, None)


# ---------------------------------------------------------------------------
# (i) Source chat does not own the document → associate returns 404
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_associate_source_chat_does_not_own_doc_returns_404(
    api_client: AsyncClient,
    db_session: AsyncSession,
    tmp_path: Path,
) -> None:
    """(i) Chat C tries to associate a document that Chat A owns, claiming Chat B is the source.
    Since Chat B does not own the document, the associate endpoint must return 404.

    Layer: Service layer validation (source_chat ownership check).
    """
    from app.api.documents import get_indexer, get_storage
    from app.main import app

    spy = SpyIndexer()
    app.dependency_overrides[get_storage] = _override_storage(tmp_path)
    app.dependency_overrides[get_indexer] = _override_indexer_factory(spy)
    try:
        chat_a = await make_chat(db_session, "i-src-A")
        chat_b = await make_chat(db_session, "i-src-B")
        chat_c = await make_chat(db_session, "i-src-C")

        doc_id = await _upload(api_client, chat_a.id, "owned-by-a.pdf")

        # Chat C attempts to associate using Chat B as source (B doesn't own the doc)
        resp = await api_client.post(
            f"/chats/{chat_c.id}/documents/{doc_id}/associate",
            json={"source_chat_id": str(chat_b.id)},
        )
        assert resp.status_code == 404, (
            f"Expected 404 when source chat does not own the document, got {resp.status_code}"
        )

        # Document must NOT be visible in Chat C
        ids_c = await _list_doc_ids(api_client, chat_c.id)
        assert doc_id not in ids_c, "Document must not be shared with Chat C on failed associate"
    finally:
        app.dependency_overrides.pop(get_storage, None)
        app.dependency_overrides.pop(get_indexer, None)


# ---------------------------------------------------------------------------
# (m) Vespa spy: standalone verification that chat_id is never None on delete
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_vespa_spy_delete_carries_non_none_chat_id(
    api_client: AsyncClient,
    db_session: AsyncSession,
    tmp_path: Path,
) -> None:
    """(m) When delete_document triggers indexer.delete_by_document,
    the chat_id argument must never be None or a sentinel value.

    Layer: Vespa filter — the chat_id injected into Vespa operations must always
    be a real UUID so Vespa can enforce its own chat-scoped filter.
    """
    from app.api.documents import get_indexer, get_storage
    from app.main import app

    spy = SpyIndexer()
    app.dependency_overrides[get_storage] = _override_storage(tmp_path)
    app.dependency_overrides[get_indexer] = _override_indexer_factory(spy)
    try:
        chat_a = await make_chat(db_session, "m-spy-A")

        doc_id = await _upload(api_client, chat_a.id, "spy-m.pdf")

        del_resp = await api_client.delete(f"/chats/{chat_a.id}/documents/{doc_id}")
        assert del_resp.status_code == 204

        assert spy.call_count == 1, "indexer.delete_by_document must be called once"
        called_chat_id, called_doc_id = spy.calls[0]

        # Core assertion: chat_id must never be None or a zero UUID
        assert called_chat_id is not None, (
            "chat_id passed to indexer.delete_by_document must not be None"
        )
        assert called_chat_id != uuid.UUID(int=0), (
            "chat_id must not be the zero/nil UUID"
        )
        assert isinstance(called_chat_id, uuid.UUID), (
            f"chat_id must be a uuid.UUID instance, got {type(called_chat_id)}"
        )
        assert called_chat_id == chat_a.id, (
            f"chat_id must equal the caller's chat_id ({chat_a.id}), got {called_chat_id}"
        )
    finally:
        app.dependency_overrides.pop(get_storage, None)
        app.dependency_overrides.pop(get_indexer, None)
