"""Application-level domain errors.

Conventions
-----------
- Each error class maps to exactly one HTTP status code via the exception
  handlers registered in ``app.main``.
- No FastAPI / Starlette types are imported here — this module is pure Python
  so that service-layer code can raise these without depending on the HTTP
  framework.
"""

from __future__ import annotations

import uuid


class AppError(Exception):
    """Base class for all application domain errors."""


# ---------------------------------------------------------------------------
# Chat errors
# ---------------------------------------------------------------------------


class ChatNotFound(AppError):
    """Raised when a Chat row cannot be found for the given ``chat_id``."""

    def __init__(self, chat_id: uuid.UUID) -> None:
        self.chat_id = chat_id
        super().__init__(f"Chat {chat_id} not found.")


# ---------------------------------------------------------------------------
# Session errors
# ---------------------------------------------------------------------------


class SessionNotFound(AppError):
    """Raised when a Session cannot be found within the given ``chat_id`` scope.

    CLAUDE.md §2 isolation: the error deliberately omits whether the session
    exists under a *different* chat — callers only see "not found in this chat".
    """

    def __init__(self, session_id: uuid.UUID, chat_id: uuid.UUID) -> None:
        self.session_id = session_id
        self.chat_id = chat_id
        super().__init__(f"Session {session_id} not found in chat {chat_id}.")


# ---------------------------------------------------------------------------
# Document errors
# ---------------------------------------------------------------------------


class DocumentNotFound(AppError):
    """Raised when a Document cannot be found within the given ``chat_id`` scope.

    CLAUDE.md §2 isolation: the error deliberately omits whether the document
    exists under a *different* chat — callers only see "not found in this chat".
    """

    def __init__(self, document_id: uuid.UUID, chat_id: uuid.UUID) -> None:
        self.document_id = document_id
        self.chat_id = chat_id
        super().__init__(f"Document {document_id} not found in chat {chat_id}.")


class DocumentStorageError(AppError):
    """Raised when file storage operations fail (save / delete / read)."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


class InvalidUpload(AppError):
    """Raised when an uploaded file fails validation (wrong MIME type, etc.)."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


class DocumentAlreadyExists(AppError):
    """Raised on duplicate upload (same checksum within the same chat)."""

    def __init__(self, document_id: uuid.UUID, chat_id: uuid.UUID) -> None:
        self.document_id = document_id
        self.chat_id = chat_id
        super().__init__(
            f"Document with the same checksum already exists in chat {chat_id}: {document_id}."
        )


class ChatDocumentAlreadyAssociated(AppError):
    """Raised when a ChatDocument association already exists."""

    def __init__(self, document_id: uuid.UUID, chat_id: uuid.UUID) -> None:
        self.document_id = document_id
        self.chat_id = chat_id
        super().__init__(f"Document {document_id} is already associated with chat {chat_id}.")


# ---------------------------------------------------------------------------
# Structured fact errors (Phase 5.3)
# ---------------------------------------------------------------------------


class FactNotFound(AppError):
    """Raised when a StructuredFact cannot be found within the given ``chat_id`` scope.

    CLAUDE.md §2 isolation: the error deliberately omits whether the fact
    exists under a *different* chat — callers only see "not found in this chat".
    """

    def __init__(self, fact_id: uuid.UUID, chat_id: uuid.UUID) -> None:
        self.fact_id = fact_id
        self.chat_id = chat_id
        super().__init__(f"Fact {fact_id} not found in chat {chat_id}.")


class InvalidFactFilter(AppError):
    """Raised when a caller supplies a fact filter that violates the restricted schema.

    This error is raised by the service layer before any SQL is executed so that
    the LLM / agent cannot inject arbitrary SQL predicates.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)


__all__ = [
    "AppError",
    "ChatNotFound",
    "SessionNotFound",
    "DocumentNotFound",
    "DocumentStorageError",
    "InvalidUpload",
    "DocumentAlreadyExists",
    "ChatDocumentAlreadyAssociated",
    "FactNotFound",
    "InvalidFactFilter",
]
