"""FastAPI router for the /chats/{chat_id}/documents resource.

This is a **thin layer** — all domain logic lives in
``app.services.document_service``.  The router is responsible only for:
  - Request / response serialisation (Pydantic models).
  - Dependency injection (DB session, storage, indexer via ``Depends``).
  - HTTP status codes and error mapping.

CLAUDE.md §2 isolation is enforced inside ``document_service``:
every query is scoped to the ``chat_id`` from the URL path.
``chat_id`` is NEVER sourced from request body or LLM output.
"""

from __future__ import annotations

import uuid

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    Form,
    HTTPException,
    Query,
    UploadFile,
)
from fastapi import status as http_status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db import get_session
from app.errors import (
    ChatDocumentAlreadyAssociated,
    ChatNotFound,
    DocumentAlreadyExists,
    DocumentNotFound,
    InvalidUpload,
)
from app.models.domain import ChatDocumentRead, DocumentRead
from app.services import document_service
from app.services.ingestion_worker import run_mineru_ingestion
from app.services.vespa_indexer import NullVespaIndexer, VespaIndexer
from app.storage import LocalBlobStorage
from app.storage.local import BlobStorage
from app.vespa.feed import VespaFeedClient

router = APIRouter()


# ---------------------------------------------------------------------------
# Dependency providers
# ---------------------------------------------------------------------------


def get_storage() -> BlobStorage:
    """Return the default local blob storage adapter."""
    return LocalBlobStorage()


def get_indexer() -> VespaIndexer:
    """Return the appropriate Vespa indexer based on environment / settings.

    - In ``test`` environments (``app_env == 'test'``) or when Vespa is
      disabled (``vespa_enabled=False``), return the no-op ``NullVespaIndexer``
      so unit tests never require a running Vespa instance.
    - In all other environments, construct a ``VespaFeedClient`` from settings.

    Tests can override this dependency via ``app.dependency_overrides``.
    """
    settings = get_settings()
    if settings.app_env == "test" or not getattr(settings, "vespa_enabled", True):
        return NullVespaIndexer()
    return VespaFeedClient(
        endpoint=settings.vespa_endpoint,
        embedding_dim=settings.embedding_dim,
    )


def _not_found(detail: str) -> HTTPException:
    return HTTPException(status_code=http_status.HTTP_404_NOT_FOUND, detail=detail)


# ---------------------------------------------------------------------------
# POST /chats/{chat_id}/documents  — upload a new document
# ---------------------------------------------------------------------------


@router.post(
    "",
    response_model=DocumentRead,
    status_code=http_status.HTTP_201_CREATED,
    summary="Upload a document to a chat",
)
async def upload_document(
    chat_id: uuid.UUID,
    file: UploadFile,
    background_tasks: BackgroundTasks,
    description: str | None = Form(default=None),
    session: AsyncSession = Depends(get_session),
    storage: BlobStorage = Depends(get_storage),
    indexer: VespaIndexer = Depends(get_indexer),
) -> DocumentRead:
    """Upload a PDF document to the chat.

    - Validates MIME type (only ``application/pdf`` accepted).
    - Computes SHA-256; returns 409 if the same file was already uploaded to
      this chat (idempotent).
    - Persists the file and registers the document in the DB.
    """
    file_bytes = await file.read()
    mime_type = file.content_type or "application/octet-stream"
    filename = file.filename or "upload.pdf"

    try:
        doc = await document_service.upload_document(
            session,
            chat_id,
            file_bytes=file_bytes,
            filename=filename,
            mime_type=mime_type,
            description=description,
            storage=storage,
            indexer=indexer,
        )
        settings = get_settings()
        if settings.auto_ingest_uploads and settings.app_env != "test":
            job_id = await document_service.enqueue_ingestion(session, chat_id, doc.id)
            doc = await document_service.get_document(session, chat_id, doc.id)
            # Ensure the background task can read the document/job even if it
            # starts before FastAPI's dependency teardown performs its commit.
            await session.commit()
            background_tasks.add_task(run_mineru_ingestion, chat_id, doc.id, job_id)
        return doc
    except ChatNotFound as exc:
        raise _not_found(str(exc)) from exc
    except InvalidUpload as exc:
        raise HTTPException(
            status_code=http_status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc
    except DocumentAlreadyExists as exc:
        raise HTTPException(
            status_code=http_status.HTTP_409_CONFLICT,
            detail={"detail": str(exc), "document_id": str(exc.document_id)},
        ) from exc


# ---------------------------------------------------------------------------
# GET /chats/{chat_id}/documents  — list documents in this chat's scope
# ---------------------------------------------------------------------------


@router.get(
    "",
    response_model=list[DocumentRead],
    summary="List documents in a chat",
)
async def list_documents(
    chat_id: uuid.UUID,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    session: AsyncSession = Depends(get_session),
) -> list[DocumentRead]:
    """List all documents visible in this chat (owned + shared).

    Ordered by ``updated_at`` descending, then ``id`` for tiebreak.
    """
    try:
        return await document_service.list_documents(
            session, chat_id, limit=limit, offset=offset
        )
    except ChatNotFound as exc:
        raise _not_found(str(exc)) from exc


# ---------------------------------------------------------------------------
# GET /chats/{chat_id}/documents/{doc_id}  — get a single document
# ---------------------------------------------------------------------------


@router.get(
    "/{doc_id}",
    response_model=DocumentRead,
    summary="Get a single document",
)
async def get_document(
    chat_id: uuid.UUID,
    doc_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
) -> DocumentRead:
    """Retrieve a document by ID.  Returns 404 if not in this chat's scope."""
    try:
        return await document_service.get_document(session, chat_id, doc_id)
    except DocumentNotFound as exc:
        raise _not_found(str(exc)) from exc


# ---------------------------------------------------------------------------
# DELETE /chats/{chat_id}/documents/{doc_id}  — delete / disassociate
# ---------------------------------------------------------------------------


@router.delete(
    "/{doc_id}",
    status_code=http_status.HTTP_204_NO_CONTENT,
    summary="Delete a document from a chat",
)
async def delete_document(
    chat_id: uuid.UUID,
    doc_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
    storage: BlobStorage = Depends(get_storage),
    indexer: VespaIndexer = Depends(get_indexer),
) -> None:
    """Remove a document from this chat's scope.

    If no other chat references the document, performs a full delete
    (Vespa index cleanup + file removal + DB row deletion).
    If other chats still reference it, only the association is removed.
    """
    try:
        await document_service.delete_document(
            session, chat_id, doc_id, storage=storage, indexer=indexer
        )
    except DocumentNotFound as exc:
        raise _not_found(str(exc)) from exc


# ---------------------------------------------------------------------------
# POST /chats/{chat_id}/documents/{doc_id}/associate  — cross-chat sharing
# ---------------------------------------------------------------------------


class AssociateRequest(BaseModel):
    """Body for POST .../associate."""

    source_chat_id: uuid.UUID


@router.post(
    "/{doc_id}/associate",
    response_model=ChatDocumentRead,
    status_code=http_status.HTTP_201_CREATED,
    summary="Share a document from another chat",
)
async def associate_document(
    chat_id: uuid.UUID,
    doc_id: uuid.UUID,
    body: AssociateRequest,
    session: AsyncSession = Depends(get_session),
) -> ChatDocumentRead:
    """Share an existing document (from ``source_chat_id``) with this chat.

    Validation:
    - ``source_chat_id`` must exist and already have the document in scope.
    - Returns 409 if the document is already associated with this chat.
    """
    try:
        return await document_service.associate_document(
            session,
            chat_id,
            doc_id,
            source_chat_id=body.source_chat_id,
        )
    except (ChatNotFound, DocumentNotFound) as exc:
        raise _not_found(str(exc)) from exc
    except ChatDocumentAlreadyAssociated as exc:
        raise HTTPException(
            status_code=http_status.HTTP_409_CONFLICT,
            detail={"detail": str(exc), "document_id": str(exc.document_id)},
        ) from exc


__all__ = ["router"]
