"""Manifest service — Phase 5.4.

Thin service layer that delegates to
:func:`app.enrichment.manifest.build_chat_manifest` while enforcing the
isolation contract (CLAUDE.md §2).

Design rules (CLAUDE.md §12)
------------------------------
- No FastAPI imports.
- No ``dict[str, Any]``.
- All public functions are async and fully type-annotated.
- ``chat_id`` is always taken from the ``current_chat_id`` parameter — never
  from user-supplied input or an LLM call.

Isolation contract (CLAUDE.md §2)
----------------------------------
- :func:`build_chat_manifest` raises :exc:`app.errors.ChatNotFound` when the
  chat does not exist, so the API layer can return a 404.
- All SQL queries within the builder are scoped to ``current_chat_id``.
"""

from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.enrichment.manifest import build_chat_manifest as _build_manifest
from app.enrichment.models import ChatManifest
from app.errors import ChatNotFound
from app.models.orm import Chat


async def get_chat_manifest(
    session: AsyncSession,
    *,
    current_chat_id: uuid.UUID,
) -> ChatManifest:
    """Build and return the chat-level manifest for *current_chat_id*.

    Parameters
    ----------
    session:
        Active async DB session.
    current_chat_id:
        The authoritative chat scope injected by the route handler from the
        URL path parameter.  Callers cannot override this via any user-supplied
        body or query parameter.

    Returns
    -------
    ChatManifest
        Read-time computed manifest for the chat.

    Raises
    ------
    ChatNotFound
        When no ``Chat`` row exists for ``current_chat_id``.
    """
    # Verify the chat exists — raise ChatNotFound so the API layer returns 404.
    chat_result = await session.execute(
        select(Chat).where(Chat.id == current_chat_id)
    )
    chat = chat_result.scalar_one_or_none()
    if chat is None:
        raise ChatNotFound(current_chat_id)

    # Delegate to the core builder (which enforces isolation internally).
    return await _build_manifest(session, current_chat_id=current_chat_id)


# Alias for acceptance-criteria import: build_chat_manifest is the public name.
build_chat_manifest = get_chat_manifest

__all__ = ["build_chat_manifest", "get_chat_manifest"]
