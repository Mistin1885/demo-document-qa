"""FastAPI router for the /chats/{chat_id}/manifest resource (Phase 5.4).

Design rules (CLAUDE.md §2, §6, §12)
----------------------------------------
- ``chat_id`` is always taken from the URL path parameter; no body or query
  param can override the isolation boundary.
- The endpoint is read-only (``GET``); it performs a structural / deterministic
  fetch of **all** documents belonging to the chat (CLAUDE.md §6 retrieval
  routing: whole-chat summary → fetch-all, never top-k).
- No domain logic lives here — all work is delegated to
  ``app.services.manifest_service.get_chat_manifest``.
- ``Cache-Control: max-age=10`` header is set to allow lightweight re-fetches
  by the frontend while bounding staleness.

HTTP surface
------------
``GET /chats/{chat_id}/manifest``
    - 200 + :class:`~app.enrichment.models.ChatManifest` JSON on success.
    - 404 when the chat does not exist (``ChatNotFound`` → global handler).
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends
from fastapi.responses import Response
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_session
from app.enrichment.models import ChatManifest
from app.services.manifest_service import get_chat_manifest

router = APIRouter()


@router.get(
    "",
    response_model=ChatManifest,
    summary="Get the chat-level manifest",
    description=(
        "Returns a lightweight manifest for all documents in the chat: "
        "titles, authors, abstract summaries, topics, token estimates, "
        "source types, and ingestion status. "
        "Scoped strictly to ``chat_id`` from the URL path — callers cannot "
        "override the isolation boundary."
    ),
)
async def get_manifest(
    chat_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
) -> Response:
    """Build and return the chat-level manifest.

    The ``chat_id`` path parameter is the **only** isolation boundary accepted.
    The response is set with ``Cache-Control: max-age=10`` to allow lightweight
    re-fetches from the frontend without hitting the database on every render.

    Returns 404 when the chat does not exist.
    """
    manifest = await get_chat_manifest(session, current_chat_id=chat_id)
    return Response(
        content=manifest.model_dump_json(),
        media_type="application/json",
        headers={"Cache-Control": "max-age=10"},
    )


__all__ = ["router"]
