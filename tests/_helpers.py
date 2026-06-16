"""Shared test helpers for isolation test suites.

Provides lightweight factory functions for creating Chats, Sessions, and
Documents directly via the service layer.  These are intentionally NOT
pytest fixtures so that isolation test files can control exactly when
and how objects are created within each test function.

Usage::

    from tests._helpers import make_chat, make_session, make_document_upload

Design constraints
------------------
- All helpers accept an ``AsyncSession`` and perform work inside its current
  transaction (SAVEPOINT-safe).
- No HTTP/FastAPI concerns here — keep it service-layer only.
- ``SpyIndexer`` records ``delete_by_document`` calls for Vespa-spy assertions.
"""

from __future__ import annotations

import uuid
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.domain import DocumentRead, SessionCreate, SessionRead
from app.models.orm import Chat
from app.services import document_service, session_service
from app.storage.local import LocalBlobStorage

_FAKE_PDF = b"%PDF-1.4 fake content isolation helper"


# ---------------------------------------------------------------------------
# Chat factory
# ---------------------------------------------------------------------------


async def make_chat(db: AsyncSession, name: str = "Test Chat") -> Chat:
    """Create a Chat ORM row directly and return the ORM instance.

    Returns the ORM ``Chat`` rather than ``ChatRead`` so tests can access
    ``chat.id`` immediately without a second query.
    """
    chat = Chat(name=name)
    db.add(chat)
    await db.flush()
    await db.refresh(chat)
    return chat


# ---------------------------------------------------------------------------
# Session factory
# ---------------------------------------------------------------------------


async def make_session(
    db: AsyncSession,
    chat: Chat,
    name: str = "Test Session",
) -> SessionRead:
    """Create a Session under ``chat`` and return a ``SessionRead``."""
    return await session_service.create_session(
        db,
        chat_id=chat.id,
        data=SessionCreate(chat_id=chat.id, name=name),
    )


# ---------------------------------------------------------------------------
# Document factory
# ---------------------------------------------------------------------------


async def make_document(
    db: AsyncSession,
    chat: Chat,
    *,
    tmp_path: Path,
    filename: str = "test.pdf",
    file_bytes: bytes = _FAKE_PDF,
) -> DocumentRead:
    """Upload a Document under ``chat`` using ``LocalBlobStorage`` backed by ``tmp_path``."""
    storage = LocalBlobStorage(root=tmp_path)
    indexer = NullSpyIndexer()
    return await document_service.upload_document(
        db,
        chat.id,
        file_bytes=file_bytes,
        filename=filename,
        mime_type="application/pdf",
        storage=storage,
        indexer=indexer,
    )


# ---------------------------------------------------------------------------
# Spy indexer
# ---------------------------------------------------------------------------


class SpyIndexer:
    """Minimal async spy that records every ``delete_by_document`` call.

    Deliberately **not** using ``unittest.mock.AsyncMock`` to keep the
    behaviour explicit and avoid accidental mock autocreation.

    Attributes
    ----------
    calls:
        List of ``(chat_id, document_id)`` tuples recorded in order.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[uuid.UUID, uuid.UUID]] = []

    async def delete_by_document(
        self,
        chat_id: uuid.UUID,
        document_id: uuid.UUID,
    ) -> None:
        self.calls.append((chat_id, document_id))

    @property
    def call_count(self) -> int:
        return len(self.calls)


class NullSpyIndexer(SpyIndexer):
    """SpyIndexer that is also safe as a no-op default — useful in ``make_document``."""


__all__ = ["make_chat", "make_session", "make_document", "SpyIndexer", "NullSpyIndexer"]
