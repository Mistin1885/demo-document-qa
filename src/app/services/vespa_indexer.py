"""Vespa indexer Protocol and no-op default implementation.

Phase 6 will replace ``NullVespaIndexer`` with a real Vespa client adapter.
Until then, ``document_service`` injects this stub so that:

1. The delete cascade path can call ``indexer.delete_by_document(...)``
   without importing a non-existent Vespa module.
2. Tests can inject a spy/mock to assert deletion was requested.

CLAUDE.md §12 compliance:
- No FastAPI imports.
- Uses ``typing.Protocol`` (structural subtyping) so the real adapter needs
  only satisfy the interface, not inherit.
"""

from __future__ import annotations

import uuid
from typing import Protocol, runtime_checkable


@runtime_checkable
class VespaIndexer(Protocol):
    """Interface for Vespa index operations scoped to a single document."""

    async def delete_by_document(
        self,
        chat_id: uuid.UUID,
        document_id: uuid.UUID,
    ) -> None:
        """Remove all Vespa document chunks that belong to ``document_id``
        in the context of ``chat_id``.

        Must be idempotent — calling it for an already-deleted document must
        not raise.
        """
        ...


class NullVespaIndexer:
    """No-op ``VespaIndexer`` used until Phase 6 implements the real client.

    All operations succeed silently.  Use this as the production default until
    the Vespa adapter is wired in.
    """

    async def delete_by_document(
        self,
        chat_id: uuid.UUID,
        document_id: uuid.UUID,
    ) -> None:
        """No-op: Vespa not yet available.  Phase 6 will replace this."""
        return


__all__ = ["VespaIndexer", "NullVespaIndexer"]
