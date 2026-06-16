"""FastAPI router for Session endpoints nested under /chats/{chat_id}/sessions.

Design rules (CLAUDE.md §12)
------------------------------
- Router is a **thin** layer: no domain logic here, only HTTP wiring.
- ``chat_id`` comes from the **URL path only**; it must never be accepted
  from the request body (CLAUDE.md §3 isolation model).
- Exception mapping: ``ChatNotFound`` → 404, ``SessionNotFound`` → 404.
- All parameters are typed; the router depends on FastAPI only (service layer
  is framework-free).
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_session
from app.errors import ChatNotFound, SessionNotFound
from app.models.domain import SessionCreate, SessionRead, SessionUpdate
from app.services import session_service

router = APIRouter()


def _not_found(detail: str) -> HTTPException:
    return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=detail)


# ---------------------------------------------------------------------------
# POST /chats/{chat_id}/sessions
# ---------------------------------------------------------------------------


@router.post(
    "",
    response_model=SessionRead,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new session under a chat",
)
async def create_session(
    chat_id: uuid.UUID,
    body: SessionCreate,
    db: AsyncSession = Depends(get_session),
) -> SessionRead:
    """Create a Session scoped to ``chat_id``.

    ``chat_id`` is injected from the URL path; the body must not supply it.
    """
    try:
        return await session_service.create_session(db, chat_id=chat_id, data=body)
    except ChatNotFound as exc:
        raise _not_found(str(exc)) from exc


# ---------------------------------------------------------------------------
# GET /chats/{chat_id}/sessions
# ---------------------------------------------------------------------------


@router.get(
    "",
    response_model=list[SessionRead],
    summary="List sessions under a chat",
)
async def list_sessions(
    chat_id: uuid.UUID,
    limit: int = 50,
    offset: int = 0,
    db: AsyncSession = Depends(get_session),
) -> list[SessionRead]:
    """Return all Sessions belonging to ``chat_id``, newest first."""
    try:
        return await session_service.list_sessions(
            db, chat_id=chat_id, limit=limit, offset=offset
        )
    except ChatNotFound as exc:
        raise _not_found(str(exc)) from exc


# ---------------------------------------------------------------------------
# GET /chats/{chat_id}/sessions/{session_id}
# ---------------------------------------------------------------------------


@router.get(
    "/{session_id}",
    response_model=SessionRead,
    summary="Get a single session",
)
async def get_session_endpoint(
    chat_id: uuid.UUID,
    session_id: uuid.UUID,
    db: AsyncSession = Depends(get_session),
) -> SessionRead:
    """Fetch a Session by ID, scoped to ``chat_id``.

    Returns 404 whether the session does not exist or belongs to a different
    chat — callers must not infer cross-chat existence from error responses.
    """
    try:
        return await session_service.get_session_by_id(
            db, chat_id=chat_id, session_id=session_id
        )
    except SessionNotFound as exc:
        raise _not_found(str(exc)) from exc


# ---------------------------------------------------------------------------
# PATCH /chats/{chat_id}/sessions/{session_id}
# ---------------------------------------------------------------------------


@router.patch(
    "/{session_id}",
    response_model=SessionRead,
    summary="Partially update a session",
)
async def patch_session(
    chat_id: uuid.UUID,
    session_id: uuid.UUID,
    body: SessionUpdate,
    db: AsyncSession = Depends(get_session),
) -> SessionRead:
    """Update mutable fields of a Session.  ``chat_id`` is enforced from the URL."""
    try:
        return await session_service.update_session(
            db, chat_id=chat_id, session_id=session_id, data=body
        )
    except SessionNotFound as exc:
        raise _not_found(str(exc)) from exc


# ---------------------------------------------------------------------------
# DELETE /chats/{chat_id}/sessions/{session_id}
# ---------------------------------------------------------------------------


@router.delete(
    "/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a session",
)
async def delete_session(
    chat_id: uuid.UUID,
    session_id: uuid.UUID,
    db: AsyncSession = Depends(get_session),
) -> None:
    """Delete a Session and cascade-delete its messages."""
    try:
        await session_service.delete_session(
            db, chat_id=chat_id, session_id=session_id
        )
    except SessionNotFound as exc:
        raise _not_found(str(exc)) from exc
