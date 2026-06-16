"""Local filesystem blob storage adapter.

Design rules (CLAUDE.md §12)
------------------------------
- ``BlobStorage`` is a ``typing.Protocol`` — no FastAPI types, no ORM imports.
- ``LocalBlobStorage`` wraps synchronous ``pathlib`` operations in
  ``anyio.to_thread.run_sync`` to stay non-blocking in the async event loop.
- Directories are created lazily on first ``save``; never at import time.
- The storage root is configurable via ``Settings.app_data_root``; defaults to
  ``"data"`` → ``data/storage/<chat_id>/<document_id>/<filename>``.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Protocol, runtime_checkable

import anyio


@runtime_checkable
class BlobStorage(Protocol):
    """Interface for storing and retrieving opaque file blobs.

    All methods are async to allow swapping in an object-storage backend
    (e.g. S3 via aioboto3) without changing callers.
    """

    async def save(
        self,
        chat_id: uuid.UUID,
        document_id: uuid.UUID,
        filename: str,
        data: bytes,
    ) -> str:
        """Persist ``data`` and return the storage path as a string.

        Args:
            chat_id: Owner chat UUID (used to partition storage).
            document_id: Document UUID (unique sub-directory).
            filename: Original filename (used as the leaf file name).
            data: Raw bytes to write.

        Returns:
            Relative storage path string (e.g.
            ``"data/storage/<chat_id>/<document_id>/<filename>"``).
        """
        ...

    async def open_for_read(
        self,
        chat_id: uuid.UUID,
        document_id: uuid.UUID,
        filename: str,
    ) -> AsyncIterator[bytes]:
        """Yield the raw bytes of the stored file in chunks.

        Raises:
            DocumentStorageError: When the file cannot be found or opened.
        """
        ...

    async def delete(
        self,
        chat_id: uuid.UUID,
        document_id: uuid.UUID,
        filename: str,
    ) -> None:
        """Remove the stored file (and the document sub-directory if empty).

        No-op if the file does not exist.
        """
        ...

    async def exists(
        self,
        chat_id: uuid.UUID,
        document_id: uuid.UUID,
        filename: str,
    ) -> bool:
        """Return ``True`` if the file is present in storage."""
        ...


class LocalBlobStorage:
    """``BlobStorage`` implementation backed by the local filesystem.

    Storage layout::

        {root}/storage/<chat_id>/<document_id>/<original_filename>

    where ``root`` defaults to ``"data"`` (relative to the process CWD).
    """

    def __init__(self, root: str | Path | None = None) -> None:
        if root is None:
            # Lazy import to avoid importing config at module load time.
            from app.config import get_settings

            root = get_settings().app_data_root
        self._storage_root = Path(root) / "storage"

    def _path(
        self, chat_id: uuid.UUID, document_id: uuid.UUID, filename: str
    ) -> Path:
        """Resolve the full filesystem path for a stored file."""
        return self._storage_root / str(chat_id) / str(document_id) / filename

    async def save(
        self,
        chat_id: uuid.UUID,
        document_id: uuid.UUID,
        filename: str,
        data: bytes,
    ) -> str:
        """Write ``data`` to disk and return the relative storage path."""
        target = self._path(chat_id, document_id, filename)

        def _write() -> None:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)

        await anyio.to_thread.run_sync(_write)
        # Return the path relative to CWD (portable string for the ORM row).
        return str(target)

    async def open_for_read(
        self,
        chat_id: uuid.UUID,
        document_id: uuid.UUID,
        filename: str,
    ) -> AsyncIterator[bytes]:
        """Yield the file contents as a single bytes chunk.

        Raises:
            DocumentStorageError: When the file is not found.
        """
        from app.errors import DocumentStorageError

        target = self._path(chat_id, document_id, filename)

        def _read() -> bytes:
            if not target.exists():
                raise DocumentStorageError(
                    f"Storage file not found: {target}"
                )
            return target.read_bytes()

        data = await anyio.to_thread.run_sync(_read)

        async def _iter() -> AsyncIterator[bytes]:
            yield data

        return _iter()

    async def delete(
        self,
        chat_id: uuid.UUID,
        document_id: uuid.UUID,
        filename: str,
    ) -> None:
        """Remove the stored file; silently ignore missing files."""
        target = self._path(chat_id, document_id, filename)

        def _remove() -> None:
            if target.exists():
                target.unlink()
            # Remove the document sub-directory if it's now empty.
            doc_dir = target.parent
            if doc_dir.exists() and not any(doc_dir.iterdir()):
                doc_dir.rmdir()

        await anyio.to_thread.run_sync(_remove)

    async def exists(
        self,
        chat_id: uuid.UUID,
        document_id: uuid.UUID,
        filename: str,
    ) -> bool:
        """Return ``True`` if the stored file exists on disk."""
        target = self._path(chat_id, document_id, filename)

        def _check() -> bool:
            return target.exists()

        return await anyio.to_thread.run_sync(_check)


__all__ = ["BlobStorage", "LocalBlobStorage"]
