"""File storage abstraction layer.

The :class:`BlobStorage` Protocol defines the interface; :class:`LocalBlobStorage`
implements it using the local filesystem under ``data/storage/``.

Usage::

    from app.storage import LocalBlobStorage

    storage = LocalBlobStorage()
    path = await storage.save(chat_id, document_id, "paper.pdf", file_bytes)
"""

from app.storage.local import BlobStorage, LocalBlobStorage

__all__ = ["BlobStorage", "LocalBlobStorage"]
