"""Unit tests for app.services.document_service.

These tests run against a real PostgreSQL instance (via Docker Compose) but
use SAVEPOINT-based rollback (see ``tests/conftest.py``) so each test starts
from a clean state without schema teardown.

Isolation coverage (CLAUDE.md §2 mandatory gates)
--------------------------------------------------
- ``list_documents`` for Chat A must NOT return documents from Chat B.
- ``get_document(chat_id=A, doc_id=<B doc>)`` must raise ``DocumentNotFound``.
- ``delete_document`` cascade: sole-owner → full delete (indexer + storage + ORM);
  shared document → only association removed.
- ``associate_document``: builds cross-chat sharing; duplicate raises 409.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.errors import (
    ChatDocumentAlreadyAssociated,
    ChatNotFound,
    DocumentAlreadyExists,
    DocumentNotFound,
    InvalidUpload,
)
from app.models.domain import DocumentRead
from app.models.orm import Chat, ChatDocument, Document
from app.services import document_service
from app.services.vespa_indexer import NullVespaIndexer
from app.storage.local import LocalBlobStorage

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FAKE_PDF = b"%PDF-1.4 fake content for testing"
_FAKE_PDF2 = b"%PDF-1.4 different content for second doc"


async def _make_chat(db: AsyncSession, name: str = "Test Chat") -> Chat:
    chat = Chat(name=name)
    db.add(chat)
    await db.flush()
    await db.refresh(chat)
    return chat


def _make_storage(tmp_path: Path) -> LocalBlobStorage:
    return LocalBlobStorage(root=tmp_path)


def _make_spy_indexer() -> AsyncMock:
    """Return an AsyncMock that satisfies the VespaIndexer Protocol."""
    indexer = AsyncMock()
    indexer.delete_by_document = AsyncMock(return_value=None)
    return indexer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def chat_a(db_session: AsyncSession) -> Chat:
    return await _make_chat(db_session, "Chat A")


@pytest_asyncio.fixture
async def chat_b(db_session: AsyncSession) -> Chat:
    return await _make_chat(db_session, "Chat B")


# ---------------------------------------------------------------------------
# upload_document — happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_upload_document_happy_path(
    db_session: AsyncSession, chat_a: Chat, tmp_path: Path
) -> None:
    """upload_document stores the file, creates ORM rows, returns DocumentRead."""
    storage = _make_storage(tmp_path)
    indexer = NullVespaIndexer()

    result = await document_service.upload_document(
        db_session,
        chat_a.id,
        file_bytes=_FAKE_PDF,
        filename="paper.pdf",
        mime_type="application/pdf",
        storage=storage,
        indexer=indexer,
    )

    assert isinstance(result, DocumentRead)
    assert result.chat_id == chat_a.id
    assert result.original_filename == "paper.pdf"
    assert result.status == "uploaded"
    assert result.mime_type == "application/pdf"
    assert result.page_count is None  # Phase 4 will fill this.

    # Verify ORM row exists.
    doc_row = await db_session.scalar(
        select(Document).where(Document.id == result.id)
    )
    assert doc_row is not None
    assert doc_row.checksum_sha256 == result.checksum_sha256

    # Verify ChatDocument association exists.
    assoc = await db_session.scalar(
        select(ChatDocument).where(
            ChatDocument.chat_id == chat_a.id,
            ChatDocument.document_id == result.id,
        )
    )
    assert assoc is not None

    # Verify file on disk.
    assert await storage.exists(chat_a.id, result.id, "paper.pdf")


# ---------------------------------------------------------------------------
# upload_document — chat not found
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_upload_document_chat_not_found(
    db_session: AsyncSession, tmp_path: Path
) -> None:
    """upload_document raises ChatNotFound for a non-existent chat."""
    storage = _make_storage(tmp_path)
    indexer = NullVespaIndexer()

    with pytest.raises(ChatNotFound):
        await document_service.upload_document(
            db_session,
            uuid.uuid4(),
            file_bytes=_FAKE_PDF,
            filename="paper.pdf",
            mime_type="application/pdf",
            storage=storage,
            indexer=indexer,
        )


# ---------------------------------------------------------------------------
# upload_document — duplicate checksum
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_upload_document_duplicate_checksum(
    db_session: AsyncSession, chat_a: Chat, tmp_path: Path
) -> None:
    """Uploading the same file twice to the same chat raises DocumentAlreadyExists."""
    storage = _make_storage(tmp_path)
    indexer = NullVespaIndexer()

    first = await document_service.upload_document(
        db_session,
        chat_a.id,
        file_bytes=_FAKE_PDF,
        filename="paper.pdf",
        mime_type="application/pdf",
        storage=storage,
        indexer=indexer,
    )

    with pytest.raises(DocumentAlreadyExists) as exc_info:
        await document_service.upload_document(
            db_session,
            chat_a.id,
            file_bytes=_FAKE_PDF,  # same bytes → same checksum
            filename="paper_copy.pdf",
            mime_type="application/pdf",
            storage=storage,
            indexer=indexer,
        )

    assert exc_info.value.document_id == first.id
    assert exc_info.value.chat_id == chat_a.id


# ---------------------------------------------------------------------------
# upload_document — invalid MIME type
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_upload_document_invalid_mime(
    db_session: AsyncSession, chat_a: Chat, tmp_path: Path
) -> None:
    """Uploading a non-PDF file raises InvalidUpload."""
    storage = _make_storage(tmp_path)
    indexer = NullVespaIndexer()

    with pytest.raises(InvalidUpload):
        await document_service.upload_document(
            db_session,
            chat_a.id,
            file_bytes=b"<html>not a pdf</html>",
            filename="page.html",
            mime_type="text/html",
            storage=storage,
            indexer=indexer,
        )


# ---------------------------------------------------------------------------
# list_documents — isolation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_documents_isolation(
    db_session: AsyncSession, chat_a: Chat, chat_b: Chat, tmp_path: Path
) -> None:
    """list_documents(chat_a) must not return documents from chat_b."""
    storage = _make_storage(tmp_path)
    indexer = NullVespaIndexer()

    await document_service.upload_document(
        db_session,
        chat_a.id,
        file_bytes=_FAKE_PDF,
        filename="a.pdf",
        mime_type="application/pdf",
        storage=storage,
        indexer=indexer,
    )
    await document_service.upload_document(
        db_session,
        chat_b.id,
        file_bytes=_FAKE_PDF2,
        filename="b.pdf",
        mime_type="application/pdf",
        storage=storage,
        indexer=indexer,
    )

    docs_a = await document_service.list_documents(db_session, chat_a.id)
    assert len(docs_a) == 1
    assert docs_a[0].chat_id == chat_a.id
    assert docs_a[0].original_filename == "a.pdf"


# ---------------------------------------------------------------------------
# get_document — cross-chat returns DocumentNotFound
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_document_cross_chat_raises(
    db_session: AsyncSession, chat_a: Chat, chat_b: Chat, tmp_path: Path
) -> None:
    """get_document(chat_a, doc_b) raises DocumentNotFound (no cross-chat leak)."""
    storage = _make_storage(tmp_path)
    indexer = NullVespaIndexer()

    doc_b = await document_service.upload_document(
        db_session,
        chat_b.id,
        file_bytes=_FAKE_PDF,
        filename="b.pdf",
        mime_type="application/pdf",
        storage=storage,
        indexer=indexer,
    )

    with pytest.raises(DocumentNotFound):
        await document_service.get_document(db_session, chat_a.id, doc_b.id)


# ---------------------------------------------------------------------------
# delete_document — sole owner → full delete
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delete_document_sole_owner(
    db_session: AsyncSession, chat_a: Chat, tmp_path: Path
) -> None:
    """Deleting a solely-owned document removes ORM row, file, and calls indexer."""
    storage = _make_storage(tmp_path)
    spy_indexer = _make_spy_indexer()

    doc = await document_service.upload_document(
        db_session,
        chat_a.id,
        file_bytes=_FAKE_PDF,
        filename="to_delete.pdf",
        mime_type="application/pdf",
        storage=storage,
        indexer=spy_indexer,
    )

    assert await storage.exists(chat_a.id, doc.id, "to_delete.pdf")

    await document_service.delete_document(
        db_session, chat_a.id, doc.id, storage=storage, indexer=spy_indexer
    )

    # Indexer was called exactly once.
    spy_indexer.delete_by_document.assert_awaited_once_with(chat_a.id, doc.id)

    # File no longer on disk.
    assert not await storage.exists(chat_a.id, doc.id, "to_delete.pdf")

    # ORM row gone.
    db_doc = await db_session.scalar(select(Document).where(Document.id == doc.id))
    assert db_doc is None

    # Association gone.
    assoc = await db_session.scalar(
        select(ChatDocument).where(
            ChatDocument.chat_id == chat_a.id,
            ChatDocument.document_id == doc.id,
        )
    )
    assert assoc is None


# ---------------------------------------------------------------------------
# delete_document — shared → only association removed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delete_document_shared(
    db_session: AsyncSession, chat_a: Chat, chat_b: Chat, tmp_path: Path
) -> None:
    """Deleting a shared document from chat_a only removes the association."""
    storage = _make_storage(tmp_path)
    spy_indexer = _make_spy_indexer()

    # Upload to chat_a (owner).
    doc = await document_service.upload_document(
        db_session,
        chat_a.id,
        file_bytes=_FAKE_PDF,
        filename="shared.pdf",
        mime_type="application/pdf",
        storage=storage,
        indexer=spy_indexer,
    )

    # Share with chat_b.
    await document_service.associate_document(
        db_session, chat_b.id, doc.id, source_chat_id=chat_a.id
    )

    # Now delete from chat_a's scope.
    await document_service.delete_document(
        db_session, chat_a.id, doc.id, storage=storage, indexer=spy_indexer
    )

    # Indexer must NOT have been called (document still referenced by chat_b).
    spy_indexer.delete_by_document.assert_not_awaited()

    # File still on disk.
    assert await storage.exists(chat_a.id, doc.id, "shared.pdf")

    # ORM row still exists.
    db_doc = await db_session.scalar(select(Document).where(Document.id == doc.id))
    assert db_doc is not None

    # chat_a's association gone.
    assoc_a = await db_session.scalar(
        select(ChatDocument).where(
            ChatDocument.chat_id == chat_a.id,
            ChatDocument.document_id == doc.id,
        )
    )
    assert assoc_a is None

    # chat_b's association still present.
    assoc_b = await db_session.scalar(
        select(ChatDocument).where(
            ChatDocument.chat_id == chat_b.id,
            ChatDocument.document_id == doc.id,
        )
    )
    assert assoc_b is not None


# ---------------------------------------------------------------------------
# associate_document
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_associate_document_success(
    db_session: AsyncSession, chat_a: Chat, chat_b: Chat, tmp_path: Path
) -> None:
    """associate_document creates the ChatDocument association row."""
    storage = _make_storage(tmp_path)
    indexer = NullVespaIndexer()

    doc = await document_service.upload_document(
        db_session,
        chat_a.id,
        file_bytes=_FAKE_PDF,
        filename="assoc.pdf",
        mime_type="application/pdf",
        storage=storage,
        indexer=indexer,
    )

    result = await document_service.associate_document(
        db_session, chat_b.id, doc.id, source_chat_id=chat_a.id
    )

    assert result.chat_id == chat_b.id
    assert result.document_id == doc.id

    # Verify in DB.
    assoc = await db_session.scalar(
        select(ChatDocument).where(
            ChatDocument.chat_id == chat_b.id,
            ChatDocument.document_id == doc.id,
        )
    )
    assert assoc is not None


@pytest.mark.asyncio
async def test_associate_document_duplicate_raises(
    db_session: AsyncSession, chat_a: Chat, chat_b: Chat, tmp_path: Path
) -> None:
    """associate_document raises ChatDocumentAlreadyAssociated on duplicate."""
    storage = _make_storage(tmp_path)
    indexer = NullVespaIndexer()

    doc = await document_service.upload_document(
        db_session,
        chat_a.id,
        file_bytes=_FAKE_PDF,
        filename="dup_assoc.pdf",
        mime_type="application/pdf",
        storage=storage,
        indexer=indexer,
    )

    await document_service.associate_document(
        db_session, chat_b.id, doc.id, source_chat_id=chat_a.id
    )

    with pytest.raises(ChatDocumentAlreadyAssociated):
        await document_service.associate_document(
            db_session, chat_b.id, doc.id, source_chat_id=chat_a.id
        )
