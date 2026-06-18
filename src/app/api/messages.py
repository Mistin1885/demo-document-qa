"""FastAPI router for the QA messages endpoint.

POST /chats/{chat_id}/sessions/{session_id}/messages

Supports:
  - ``stream=true``  → SSE streaming (text/event-stream)
  - ``stream=false`` → JSON QAResponse

Stop generation:
  Implemented via disconnect-aware SSE: if the client disconnects the
  ``asyncio.Event`` is set and the QAService.stream generator exits early.

Isolation contract (CLAUDE.md §2):
  - ``chat_id`` and ``session_id`` come from the URL path only.
  - The route layer verifies session ownership via session_service.
  - All QA service calls receive ``chat_id`` from the path.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_session
from app.errors import ChatNotFound, SessionNotFound
from app.providers.mock import MockChatProvider
from app.services.qa_service import QAService

router = APIRouter()


# ---------------------------------------------------------------------------
# Request body
# ---------------------------------------------------------------------------


class MessageRequest(BaseModel):
    """Request body for POST /messages.

    ``extra="forbid"`` ensures unknown fields return HTTP 422.
    """

    model_config = ConfigDict(extra="forbid")

    question: str
    stream: bool = True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _not_found(detail: str) -> HTTPException:
    return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=detail)


def _sse_line(kind: str, data: object) -> str:
    """Format a single SSE event line pair."""
    payload = json.dumps(data, default=str)
    return f"event: {kind}\ndata: {payload}\n\n"


# ---------------------------------------------------------------------------
# POST /chats/{chat_id}/sessions/{session_id}/messages
# ---------------------------------------------------------------------------


@router.post(
    "",
    summary="Send a question to the agent and get an answer",
    status_code=status.HTTP_200_OK,
    response_model=None,
)
async def post_message(
    chat_id: uuid.UUID,
    session_id: uuid.UUID,
    body: MessageRequest,
    request: Request,
    db: AsyncSession = Depends(get_session),
) -> StreamingResponse | JSONResponse:
    """Send a question and receive an answer.

    Route layer verifies that ``session_id`` belongs to ``chat_id``.
    ``chat_id`` is never taken from the body.

    - ``stream=true``  → ``text/event-stream`` SSE response.
    - ``stream=false`` → JSON ``QAResponse``.
    """
    from app.services import session_service  # noqa: PLC0415

    # Verify session ownership — raises SessionNotFound/ChatNotFound on failure
    try:
        await session_service.get_session_by_id(db, chat_id=chat_id, session_id=session_id)
    except SessionNotFound as exc:
        raise _not_found(str(exc)) from exc
    except ChatNotFound as exc:
        raise _not_found(str(exc)) from exc

    # Build service + provider
    from app.db import _get_sessionmaker  # noqa: PLC0415
    from app.vespa.mock import NullRetrievalService  # noqa: PLC0415

    session_factory = _get_sessionmaker()
    retrieval_service = NullRetrievalService()
    qa_service = QAService(session_factory=session_factory, retrieval_service=retrieval_service)
    chat_provider = MockChatProvider()

    # ------------------------------------------------------------------
    # Streaming path
    # ------------------------------------------------------------------
    if body.stream:
        stop_event = asyncio.Event()

        async def _event_generator() -> AsyncGenerator[str, None]:
            async for evt in qa_service.stream(
                chat_id,
                session_id,
                body.question,
                chat_provider=chat_provider,
                stop_event=stop_event,
            ):
                # Check client disconnect between events
                if await request.is_disconnected():
                    stop_event.set()
                    yield _sse_line("error", {"code": "STOPPED", "detail": "client disconnected"})
                    return
                yield _sse_line(evt.kind, evt.data)

        return StreamingResponse(
            _event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    # ------------------------------------------------------------------
    # Non-streaming path
    # ------------------------------------------------------------------
    try:
        response = await qa_service.run(
            chat_id,
            session_id,
            body.question,
            chat_provider=chat_provider,
        )
    except SessionNotFound as exc:
        raise _not_found(str(exc)) from exc
    except ChatNotFound as exc:
        raise _not_found(str(exc)) from exc

    return JSONResponse(
        content=json.loads(response.model_dump_json()),
        status_code=status.HTTP_200_OK,
    )
