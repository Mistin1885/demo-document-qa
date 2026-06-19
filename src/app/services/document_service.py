"""Document service — pure async domain logic for the Document resource.

Design contract (CLAUDE.md §12)
--------------------------------
- No FastAPI imports; no HTTP-specific types.
- All public functions are async and accept an ``AsyncSession``.
- All inputs/outputs are domain models (``DocumentRead`` etc.) or primitives.
- ORM queries use SQLAlchemy Core/ORM expressions — no raw SQL strings.
- **Every query that retrieves documents is scoped to ``chat_id`` via a
  ``chat_documents.chat_id = :chat_id`` join** — this is the service-layer
  isolation enforcement point mandated by CLAUDE.md §2.
- ``chat_id`` is NEVER accepted from an LLM/agent caller; it is always
  injected from the URL path or AgentState.

Isolation layers implemented here (CLAUDE.md §2)
-------------------------------------------------
1. All list / get / delete operations join ``chat_documents`` and filter by
   ``chat_documents.chat_id = current_chat_id``.
2. ``upload_document`` creates *both* a ``Document`` row (owner chat) *and*
   a ``ChatDocument`` association row so the document appears in the chat's
   scope immediately.
3. Cross-chat association (``associate_document``) requires the source chat to
   already have the document in its scope (via its own ``ChatDocument`` row).
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.errors import (
    ChatDocumentAlreadyAssociated,
    ChatNotFound,
    DocumentAlreadyExists,
    DocumentNotFound,
    DocumentStorageError,
    InvalidUpload,
)
from app.models.domain import ChatDocumentRead, DocumentRead
from app.models.orm import Chat, ChatDocument, Document, IngestionJob

if TYPE_CHECKING:
    from app.services.vespa_indexer import VespaIndexer
    from app.storage.local import BlobStorage

# ---------------------------------------------------------------------------
# Allowed MIME types for upload validation
# ---------------------------------------------------------------------------

_ALLOWED_MIME_TYPES: frozenset[str] = frozenset({"application/pdf"})


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _sha256(data: bytes) -> str:
    """Return the hex-encoded SHA-256 checksum of ``data``."""
    return hashlib.sha256(data).hexdigest()


def _now() -> datetime:
    """Return current UTC time as a naive ``datetime`` (ORM convention)."""
    return datetime.now(UTC).replace(tzinfo=None)


async def _assert_chat_exists(session: AsyncSession, chat_id: uuid.UUID) -> None:
    """Raise ``ChatNotFound`` if ``chat_id`` does not correspond to a real Chat."""
    exists = await session.scalar(
        select(func.count()).select_from(Chat).where(Chat.id == chat_id)
    )
    if not exists:
        raise ChatNotFound(chat_id)


async def _get_document_in_chat(
    session: AsyncSession,
    chat_id: uuid.UUID,
    document_id: uuid.UUID,
) -> Document:
    """Return the ``Document`` ORM row that is visible in ``chat_id``'s scope.

    Raises:
        DocumentNotFound: When the document does not exist **or** is not
            associated with ``chat_id`` (CLAUDE.md §2 — never reveal existence
            outside the caller's scope).
    """
    # Join through chat_documents to enforce chat-scope isolation.
    stmt = (
        select(Document)
        .join(ChatDocument, ChatDocument.document_id == Document.id)
        .where(
            ChatDocument.chat_id == chat_id,
            Document.id == document_id,
        )
    )
    row = await session.scalar(stmt)
    if row is None:
        raise DocumentNotFound(document_id, chat_id)
    return row


# ---------------------------------------------------------------------------
# Public service functions
# ---------------------------------------------------------------------------


async def upload_document(
    session: AsyncSession,
    chat_id: uuid.UUID,
    *,
    file_bytes: bytes,
    filename: str,
    mime_type: str,
    description: str | None = None,
    storage: BlobStorage,
    indexer: VespaIndexer,
) -> DocumentRead:
    """Validate, store, and register a new document under ``chat_id``.

    Steps:
    1. Verify the chat exists.
    2. Validate MIME type.
    3. Compute SHA-256; check for duplicate within the same chat.
    4. Save file bytes via ``storage``.
    5. Write ``Document`` ORM row (owner = ``chat_id``).
    6. Write ``ChatDocument`` association row.

    Args:
        session: Active async DB session.
        chat_id: Owner chat UUID.
        file_bytes: Raw uploaded bytes.
        filename: Original file name (e.g. ``"paper.pdf"``).
        mime_type: MIME type reported by the client (e.g. ``"application/pdf"``).
        description: Optional human-readable description (unused until Phase 4).
        storage: Injected blob storage adapter.
        indexer: Injected Vespa indexer (unused at upload — only for delete path).

    Returns:
        ``DocumentRead`` of the newly created document.

    Raises:
        ChatNotFound: When ``chat_id`` does not exist.
        InvalidUpload: When ``mime_type`` is not in ``_ALLOWED_MIME_TYPES``.
        DocumentAlreadyExists: When a document with the same SHA-256 already
            exists in this chat (idempotent re-upload returns existing doc id).
        DocumentStorageError: When file persistence fails.
    """
    await _assert_chat_exists(session, chat_id)

    # --- MIME validation (simple; PoC scope) ---
    if mime_type not in _ALLOWED_MIME_TYPES:
        # Also reject by extension when MIME is generic octet-stream.
        if not (
            mime_type == "application/octet-stream"
            and filename.lower().endswith(".pdf")
        ):
            raise InvalidUpload(
                f"Unsupported file type '{mime_type}'. Only PDF uploads are accepted."
            )

    # --- Duplicate detection (same checksum within the same chat) ---
    checksum = _sha256(file_bytes)
    existing_stmt = (
        select(Document)
        .join(ChatDocument, ChatDocument.document_id == Document.id)
        .where(
            ChatDocument.chat_id == chat_id,
            Document.checksum_sha256 == checksum,
        )
    )
    existing = await session.scalar(existing_stmt)
    if existing is not None:
        raise DocumentAlreadyExists(existing.id, chat_id)

    # --- Persist file ---
    document_id = uuid.uuid4()
    try:
        storage_path = await storage.save(chat_id, document_id, filename, file_bytes)
    except Exception as exc:  # noqa: BLE001
        raise DocumentStorageError(f"Failed to save file '{filename}': {exc}") from exc

    # --- Write ORM rows ---
    now = _now()
    doc = Document(
        id=document_id,
        chat_id=chat_id,
        source_type="upload",
        original_filename=filename,
        storage_path=storage_path,
        mime_type=mime_type,
        page_count=None,  # Phase 4 will fill this after MinerU parsing.
        status="uploaded",
        checksum_sha256=checksum,
        created_at=now,
        updated_at=now,
    )
    session.add(doc)
    await session.flush()

    association = ChatDocument(chat_id=chat_id, document_id=document_id)
    session.add(association)
    await session.flush()
    await session.refresh(doc)

    return DocumentRead.model_validate(doc)


async def list_documents(
    session: AsyncSession,
    chat_id: uuid.UUID,
    *,
    limit: int = 50,
    offset: int = 0,
) -> list[DocumentRead]:
    """Return documents visible in ``chat_id``'s scope.

    Includes both owned documents and cross-chat shared documents (those added
    via ``associate_document``).  The query enforces chat-scope isolation via
    the ``chat_documents`` join.

    Raises:
        ChatNotFound: When ``chat_id`` does not exist.
    """
    await _assert_chat_exists(session, chat_id)

    stmt = (
        select(Document)
        .join(ChatDocument, ChatDocument.document_id == Document.id)
        .where(ChatDocument.chat_id == chat_id)
        .order_by(Document.updated_at.desc(), Document.id)
        .limit(limit)
        .offset(offset)
    )
    rows = await session.scalars(stmt)
    return [DocumentRead.model_validate(row) for row in rows.all()]


async def get_document(
    session: AsyncSession,
    chat_id: uuid.UUID,
    document_id: uuid.UUID,
) -> DocumentRead:
    """Return a single document visible in ``chat_id``'s scope.

    Raises:
        DocumentNotFound: When the document is absent or not in this chat's scope.
    """
    doc = await _get_document_in_chat(session, chat_id, document_id)
    return DocumentRead.model_validate(doc)


async def delete_document(
    session: AsyncSession,
    chat_id: uuid.UUID,
    document_id: uuid.UUID,
    *,
    storage: BlobStorage,
    indexer: VespaIndexer,
) -> None:
    """Remove a document from ``chat_id``'s scope with full cascade logic.

    Behaviour:
    - Verify the document is in ``chat_id``'s scope; raise ``DocumentNotFound``
      otherwise.
    - Remove the ``ChatDocument(chat_id, document_id)`` association.
    - If **no other chat** references the document after removal:
        1. Call ``indexer.delete_by_document(chat_id, document_id)``
           (Vespa cleanup — no-op until Phase 6).
        2. Delete the file from ``storage``.
        3. Delete the ``Document`` ORM row (cascades to all child rows).
    - If other chats still reference the document, only the association is
      removed; the document survives.

    Raises:
        DocumentNotFound: When the document is not in this chat's scope.
    """
    # Verify scope; raises DocumentNotFound if absent.
    doc = await _get_document_in_chat(session, chat_id, document_id)

    # Remove this chat's association.
    assoc_stmt = select(ChatDocument).where(
        ChatDocument.chat_id == chat_id,
        ChatDocument.document_id == document_id,
    )
    assoc = await session.scalar(assoc_stmt)
    if assoc is not None:
        await session.delete(assoc)
        await session.flush()

    # Check how many associations remain.
    remaining_count = await session.scalar(
        select(func.count())
        .select_from(ChatDocument)
        .where(ChatDocument.document_id == document_id)
    )

    if not remaining_count:
        # Full delete: Vespa, storage, ORM row.
        await indexer.delete_by_document(chat_id, document_id)

        try:
            await storage.delete(chat_id, document_id, doc.original_filename)
        except Exception as exc:  # noqa: BLE001
            raise DocumentStorageError(
                f"Failed to delete file for document {document_id}: {exc}"
            ) from exc

        await session.delete(doc)
        await session.flush()
    # else: other chats still reference the document; leave it intact.


async def associate_document(
    session: AsyncSession,
    target_chat_id: uuid.UUID,
    document_id: uuid.UUID,
    *,
    source_chat_id: uuid.UUID,
) -> ChatDocumentRead:
    """Create a ``ChatDocument`` association so ``target_chat_id`` can access the document.

    Validation:
    - ``target_chat_id`` must exist.
    - ``source_chat_id`` must exist and the document must be in its scope.
    - The association ``(target_chat_id, document_id)`` must not already exist.

    Args:
        session: Active async DB session.
        target_chat_id: The chat that wants to share the document.
        document_id: UUID of the document to associate.
        source_chat_id: The chat that currently owns / has the document.

    Returns:
        ``ChatDocumentRead`` of the newly created association.

    Raises:
        ChatNotFound: When ``target_chat_id`` or ``source_chat_id`` does not exist.
        DocumentNotFound: When the document is not in ``source_chat_id``'s scope.
        ChatDocumentAlreadyAssociated: When the association already exists.
    """
    # Validate target chat.
    await _assert_chat_exists(session, target_chat_id)

    # Validate source chat and document scope.
    await _assert_chat_exists(session, source_chat_id)
    await _get_document_in_chat(session, source_chat_id, document_id)

    # Check for duplicate association.
    existing = await session.scalar(
        select(ChatDocument).where(
            ChatDocument.chat_id == target_chat_id,
            ChatDocument.document_id == document_id,
        )
    )
    if existing is not None:
        raise ChatDocumentAlreadyAssociated(document_id, target_chat_id)

    assoc = ChatDocument(chat_id=target_chat_id, document_id=document_id)
    session.add(assoc)
    await session.flush()

    return ChatDocumentRead(chat_id=target_chat_id, document_id=document_id)


async def trigger_ingestion(
    session: AsyncSession,
    chat_id: uuid.UUID,
    document_id: uuid.UUID,
) -> None:
    """Entry point for Phase 4 to kick off MinerU parsing.

    Currently a no-op placeholder.  Phase 4 will:
    1. Update ``Document.status`` to ``"parsing"``.
    2. Create an ``IngestionJob`` row.
    3. Enqueue the parsing task to the background worker.

    Args:
        session: Active async DB session.
        chat_id: Owner chat (used to verify scope).
        document_id: Document to ingest.

    Raises:
        DocumentNotFound: When the document is not in ``chat_id``'s scope.
    """
    # Verify scope without doing anything.
    await _get_document_in_chat(session, chat_id, document_id)
    # TODO(Phase 4): create IngestionJob, dispatch background task.


async def enqueue_ingestion(
    session: AsyncSession,
    chat_id: uuid.UUID,
    document_id: uuid.UUID,
) -> uuid.UUID:
    """Mark a document as queued for MinerU ingestion and create an ingestion job.

    This function does **not** run MinerU itself. The API layer schedules a
    background worker after calling this function.
    """
    doc = await _get_document_in_chat(session, chat_id, document_id)
    now = _now()

    doc.status = "parsing"
    doc.updated_at = now

    job = IngestionJob(
        chat_id=chat_id,
        document_id=document_id,
        state="pending",
        attempt=0,
        created_at=now,
    )
    session.add(job)
    await session.flush()

    return job.id


__all__ = [
    "upload_document",
    "list_documents",
    "get_document",
    "delete_document",
    "associate_document",
    "trigger_ingestion",
    "enqueue_ingestion",
]
