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

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.agent.state import GenerationConfig
from app.config import get_settings
from app.db import get_session
from app.errors import ChatNotFound, SessionNotFound
from app.models.domain import MessageRead
from app.models.orm import Chat as ChatORM
from app.models.orm import ProviderProfile as ProviderProfileORM
from app.providers.base import ChatProvider
from app.providers.extractive import ExtractiveEvidenceChatProvider
from app.providers.openai_compat import OpenAICompatChatProvider
from app.providers.registry import build_chat_provider
from app.retrieval.service import RetrievalService
from app.services.qa_service import QAService
from app.vespa.mock import NullRetrievalService

router = APIRouter()


# ---------------------------------------------------------------------------
# Request body
# ---------------------------------------------------------------------------


class MessageRequest(BaseModel):
    """Request body for POST /messages.

    ``extra="forbid"`` ensures unknown fields return HTTP 422.

    Generation overrides (all optional — fall back to env-level ``LLM_*``
    settings when omitted):
      - ``max_answer_tokens``: cap on output tokens (use to avoid truncated
        long summaries; provider hard ceiling still applies).
      - ``temperature``: sampling temperature for the answer.
      - ``context_window``: total input budget; resizes the
        ``ContextBudgetManager`` proportionally so more evidence can fit.
    """

    model_config = ConfigDict(extra="forbid")

    question: str
    stream: bool = True
    max_answer_tokens: int | None = Field(default=None, ge=1, le=32_768)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    context_window: int | None = Field(default=None, ge=1_000, le=200_000)
    deep_qa_mode: bool = False


def _generation_config_from_request(body: MessageRequest) -> GenerationConfig:
    """Build a GenerationConfig from request body + env-level fallbacks."""
    settings = get_settings()
    return GenerationConfig(
        max_answer_tokens=body.max_answer_tokens
        if body.max_answer_tokens is not None
        else (32_768 if body.deep_qa_mode else settings.llm_max_tokens),
        temperature=body.temperature
        if body.temperature is not None
        else settings.llm_temperature,
        context_window=body.context_window
        if body.context_window is not None
        else (200_000 if body.deep_qa_mode else settings.llm_context_window),
        deep_qa_mode=body.deep_qa_mode,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _not_found(detail: str) -> HTTPException:
    return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=detail)


# ---------------------------------------------------------------------------
# Provider helpers — load real chat provider; retrieval uses Vespa-native E5.
# ---------------------------------------------------------------------------


def _build_env_chat_provider() -> ChatProvider | None:
    """Build a ChatProvider from env-level ``LLM_*`` settings, if configured.

    Returns ``None`` when ``LLM_PROVIDER=mock`` (the default) or required fields
    are missing, so the caller can fall back to an evidence-grounded extractive
    provider instead of a mock answer.

    ``LLM_API_URL`` is normalised: a trailing ``/chat/completions`` is stripped
    because the OpenAI SDK appends that path itself.
    """
    settings = get_settings()
    if settings.llm_provider == "mock":
        return None
    if settings.llm_provider != "openai_compatible":
        # OpenAI native / Gemini native via env is not wired here yet; use DB
        # provider_profiles for those.
        return None
    if not settings.llm_api_url or not settings.llm_model:
        return None

    base_url = settings.llm_api_url
    for suffix in ("/chat/completions", "/chat/completions/"):
        if base_url.endswith(suffix):
            base_url = base_url[: -len(suffix)]
            break
    base_url = base_url.rstrip("/")

    api_key = (
        settings.llm_api_key.get_secret_value() if settings.llm_api_key is not None else ""
    )
    return OpenAICompatChatProvider(
        api_key=api_key,
        base_url=base_url,
        model=settings.llm_model,
        provider_name="env_openai_compat",
    )


def _profile_as_like(p: ProviderProfileORM, embedding_dim: int) -> object:
    """Shim ORM ProviderProfile to the ProviderProfileLike structural protocol.

    The registry protocol uses ``model_name`` / ``provider_name`` / ``api_key_plain``
    / ``embedding_dim``; the ORM uses ``model`` / ``name`` / no plain key.
    """
    from types import SimpleNamespace  # noqa: PLC0415
    return SimpleNamespace(
        provider_type=p.provider_type,
        model_name=p.model,
        api_key_encrypted=p.api_key_encrypted,
        api_key_plain=None,
        base_url=p.base_url,
        context_window=p.context_window,
        embedding_dim=embedding_dim,
        provider_name=p.name,
    )


async def _load_chat(db: AsyncSession, chat_id: uuid.UUID) -> ChatORM | None:
    """Load Chat with the default chat profile relationship eagerly."""
    stmt = (
        select(ChatORM)
        .where(ChatORM.id == chat_id)
        .options(
            selectinload(ChatORM.default_chat_profile),
        )
    )
    return (await db.scalars(stmt)).first()


async def _build_providers(
    db: AsyncSession, chat_id: uuid.UUID
) -> tuple[ChatProvider, RetrievalService | NullRetrievalService]:
    """Return (chat_provider, retrieval_service) for this chat.

    Falls back to ExtractiveEvidenceChatProvider / NullRetrievalService when:
    - the chat has no default provider profile configured, OR
    - ``vespa_enabled=False`` in settings (for the retrieval service).

    Retrieval intentionally does not read ``default_embedding_profile`` or
    ``default_reranker_profile``: embeddings and native rerank are computed by
    the bundled Vespa E5 embedder/rank profiles.
    """
    settings = get_settings()
    chat_orm = await _load_chat(db, chat_id)

    # --- chat (LLM) provider ---
    # Resolution order:
    #   1. Chat's default_chat_profile (DB-stored, encrypted key) — production path.
    #   2. Env-level LLM_* settings — convenience for dev/demo without a DB row.
    #   3. ExtractiveEvidenceChatProvider — evidence-only fallback when the
    #      deployment has no external chat LLM configured.
    chat_prov: ChatProvider
    if chat_orm is not None and chat_orm.default_chat_profile is not None:
        try:
            profile_like = _profile_as_like(chat_orm.default_chat_profile, settings.embedding_dim)
            chat_prov = build_chat_provider(profile_like)  # type: ignore[arg-type]
        except Exception:
            chat_prov = _build_env_chat_provider() or ExtractiveEvidenceChatProvider()
    else:
        chat_prov = _build_env_chat_provider() or ExtractiveEvidenceChatProvider()

    # --- retrieval service (Vespa-native embedding + native rerank) ---
    retrieval_svc: RetrievalService | NullRetrievalService
    if settings.vespa_enabled:
        retrieval_svc = RetrievalService(
            endpoint=settings.vespa_endpoint,
            embedding_provider=None,
            embedding_dim=settings.embedding_dim,
        )
    else:
        retrieval_svc = NullRetrievalService()

    return chat_prov, retrieval_svc


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

    # Build providers from chat/env config. Without a chat LLM, use the
    # evidence-grounded extractive fallback; retrieval still uses Vespa native
    # embedding/rerank.
    from app.db import _get_sessionmaker  # noqa: PLC0415

    chat_provider, retrieval_service = await _build_providers(db, chat_id)
    session_factory = _get_sessionmaker()
    qa_service = QAService(session_factory=session_factory, retrieval_service=retrieval_service)
    gen_cfg = _generation_config_from_request(body)

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
                generation_config=gen_cfg,
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
            generation_config=gen_cfg,
        )
    except SessionNotFound as exc:
        raise _not_found(str(exc)) from exc
    except ChatNotFound as exc:
        raise _not_found(str(exc)) from exc

    return JSONResponse(
        content=json.loads(response.model_dump_json()),
        status_code=status.HTTP_200_OK,
    )


# ---------------------------------------------------------------------------
# GET /chats/{chat_id}/sessions/{session_id}/messages
# ---------------------------------------------------------------------------


@router.get(
    "",
    response_model=list[MessageRead],
    summary="List persisted messages for a session",
)
async def list_messages(
    chat_id: uuid.UUID,
    session_id: uuid.UUID,
    limit: int = Query(default=200, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    db: AsyncSession = Depends(get_session),
) -> list[MessageRead]:
    """Return persisted user/assistant messages ordered by ``created_at``.

    Session scope is enforced inside ``session_service.list_messages`` — a
    session belonging to a different chat raises ``SessionNotFound`` and the
    client receives 404 (no cross-chat leakage in the error shape).
    """
    from app.services import session_service  # noqa: PLC0415

    try:
        return await session_service.list_messages(
            db, chat_id=chat_id, session_id=session_id, limit=limit, offset=offset
        )
    except SessionNotFound as exc:
        raise _not_found(str(exc)) from exc
    except ChatNotFound as exc:
        raise _not_found(str(exc)) from exc
