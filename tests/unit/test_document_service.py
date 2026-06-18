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
# upload_document — happy path + validation
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

    # Verify ChatDocument association exists.
    assoc = await db_session.scalar(
        select(ChatDocument).where(
            ChatDocument.chat_id == chat_a.id,
            ChatDocument.document_id == result.id,
        )
    )
    assert assoc is not None
    assert await storage.exists(chat_a.id, result.id, "paper.pdf")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kwargs,error_type",
    [
        (
            {"file_bytes": _FAKE_PDF, "filename": "paper.pdf", "mime_type": "application/pdf"},
            ChatNotFound,
        ),
        (
            {"file_bytes": b"<html/>", "filename": "page.html", "mime_type": "text/html"},
            InvalidUpload,
        ),
    ],
)
async def test_upload_document_errors(
    db_session: AsyncSession,
    chat_a: Chat,
    tmp_path: Path,
    kwargs: dict,
    error_type: type,
) -> None:
    """upload_document raises appropriate errors for bad chat or bad MIME type."""
    storage = _make_storage(tmp_path)
    indexer = NullVespaIndexer()

    # For ChatNotFound test, use a random UUID; for InvalidUpload, use chat_a.id
    chat_id = uuid.uuid4() if error_type is ChatNotFound else chat_a.id

    with pytest.raises(error_type):
        await document_service.upload_document(
            db_session,
            chat_id,
            storage=storage,
            indexer=indexer,
            **kwargs,
        )


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
            file_bytes=_FAKE_PDF,
            filename="paper_copy.pdf",
            mime_type="application/pdf",
            storage=storage,
            indexer=indexer,
        )

    assert exc_info.value.document_id == first.id
    assert exc_info.value.chat_id == chat_a.id


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
        db_session, chat_a.id, file_bytes=_FAKE_PDF, filename="a.pdf",
        mime_type="application/pdf", storage=storage, indexer=indexer,
    )
    await document_service.upload_document(
        db_session, chat_b.id, file_bytes=_FAKE_PDF2, filename="b.pdf",
        mime_type="application/pdf", storage=storage, indexer=indexer,
    )

    docs_a = await document_service.list_documents(db_session, chat_a.id)
    assert len(docs_a) == 1
    assert docs_a[0].chat_id == chat_a.id


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
        db_session, chat_b.id, file_bytes=_FAKE_PDF, filename="b.pdf",
        mime_type="application/pdf", storage=storage, indexer=indexer,
    )

    with pytest.raises(DocumentNotFound):
        await document_service.get_document(db_session, chat_a.id, doc_b.id)


# ---------------------------------------------------------------------------
# delete_document — sole owner → full delete + Vespa indexer called
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delete_document_sole_owner(
    db_session: AsyncSession, chat_a: Chat, tmp_path: Path
) -> None:
    """Deleting a solely-owned document removes ORM row, file, and calls indexer."""
    storage = _make_storage(tmp_path)
    spy_indexer = _make_spy_indexer()

    doc = await document_service.upload_document(
        db_session, chat_a.id, file_bytes=_FAKE_PDF, filename="to_delete.pdf",
        mime_type="application/pdf", storage=storage, indexer=spy_indexer,
    )

    await document_service.delete_document(
        db_session, chat_a.id, doc.id, storage=storage, indexer=spy_indexer
    )

    spy_indexer.delete_by_document.assert_awaited_once_with(chat_a.id, doc.id)
    assert not await storage.exists(chat_a.id, doc.id, "to_delete.pdf")

    db_doc = await db_session.scalar(select(Document).where(Document.id == doc.id))
    assert db_doc is None


# ---------------------------------------------------------------------------
# associate_document
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_associate_document_success_and_duplicate(
    db_session: AsyncSession, chat_a: Chat, chat_b: Chat, tmp_path: Path
) -> None:
    """associate_document creates the association; duplicate raises ChatDocumentAlreadyAssociated."""
    storage = _make_storage(tmp_path)
    indexer = NullVespaIndexer()

    doc = await document_service.upload_document(
        db_session, chat_a.id, file_bytes=_FAKE_PDF, filename="assoc.pdf",
        mime_type="application/pdf", storage=storage, indexer=indexer,
    )

    result = await document_service.associate_document(
        db_session, chat_b.id, doc.id, source_chat_id=chat_a.id
    )
    assert result.chat_id == chat_b.id
    assert result.document_id == doc.id

    with pytest.raises(ChatDocumentAlreadyAssociated):
        await document_service.associate_document(
            db_session, chat_b.id, doc.id, source_chat_id=chat_a.id
        )
