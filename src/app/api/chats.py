"""FastAPI router for the /chats resource.

This is a **thin layer** — all domain logic lives in ``app.services.chat_service``.
The router is responsible only for:
  - Request / response serialisation (Pydantic models).
  - Dependency injection (DB session via ``Depends``).
  - HTTP status codes.

``ChatNotFound`` is handled globally via the exception handler registered in
``app.main``; individual endpoints do not need try/except blocks.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, Query
from fastapi import status as http_status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_session
from app.models.domain import ChatCreate, ChatRead, ChatUpdate
from app.services import chat_service

router = APIRouter()


@router.post(
    "/",
    response_model=ChatRead,
    status_code=http_status.HTTP_201_CREATED,
    include_in_schema=False,
)
@router.post("", response_model=ChatRead, status_code=http_status.HTTP_201_CREATED)
async def create_chat(
    data: ChatCreate,
    session: AsyncSession = Depends(get_session),
) -> ChatRead:
    """Create a new Chat (document-isolation boundary)."""
    return await chat_service.create_chat(session, data)


@router.get("/", response_model=list[ChatRead], include_in_schema=False)
@router.get("", response_model=list[ChatRead])
async def list_chats(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    session: AsyncSession = Depends(get_session),
) -> list[ChatRead]:
    """List Chats ordered by ``updated_at`` descending (newest first)."""
    return await chat_service.list_chats(session, limit=limit, offset=offset)


@router.get("/{chat_id}", response_model=ChatRead)
async def get_chat(
    chat_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
) -> ChatRead:
    """Retrieve a single Chat by its UUID.  Returns 404 if not found."""
    return await chat_service.get_chat(session, chat_id)


@router.patch("/{chat_id}", response_model=ChatRead)
async def update_chat(
    chat_id: uuid.UUID,
    data: ChatUpdate,
    session: AsyncSession = Depends(get_session),
) -> ChatRead:
    """Partially update a Chat.  Returns 404 if not found."""
    return await chat_service.update_chat(session, chat_id, data)


@router.delete("/{chat_id}", status_code=http_status.HTTP_204_NO_CONTENT)
async def delete_chat(
    chat_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
) -> None:
    """Delete a Chat and all its cascade-dependent resources.  Returns 404 if not found."""
    await chat_service.delete_chat(session, chat_id)


__all__ = ["router"]
