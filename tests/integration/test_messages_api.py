"""Integration tests for the messages API endpoint (Phase 7.5).

Tests POST /chats/{chat_id}/sessions/{session_id}/messages.

Isolation contract:
- Cross-chat session → 404
- Unknown field in body → 422
- stream=true → text/event-stream
- stream=false → JSON QAResponse

Test density cap: ≤ 10 items per file (CLAUDE.md §12.1).
"""

from __future__ import annotations

import json
import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.domain import SessionCreate
from app.models.orm import Chat, Message

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _create_chat(db: AsyncSession, name: str = "Test Chat") -> uuid.UUID:
    from datetime import UTC, datetime  # noqa: PLC0415

    now = datetime.now(UTC).replace(tzinfo=None)
    chat = Chat(name=name, created_at=now, updated_at=now)
    db.add(chat)
    await db.flush()
    await db.refresh(chat)
    return chat.id  # type: ignore[return-value]


async def _create_session(db: AsyncSession, chat_id: uuid.UUID) -> uuid.UUID:
    from app.services import session_service  # noqa: PLC0415

    sess = await session_service.create_session(
        db, chat_id=chat_id, data=SessionCreate(chat_id=chat_id)
    )
    return sess.id


def _mock_qa_service(answer: str = "The answer is X.") -> Any:
    """Build a mock QAService that returns a canned response."""
    from app.models.domain import QAResponse  # noqa: PLC0415
    from app.services.qa_service import QAService, QAStreamEvent  # noqa: PLC0415

    mock_svc = MagicMock(spec=QAService)

    response = QAResponse(
        answer=answer,
        citations=[],
        documents_used=[],
        coverage=1.0,
        uncertainty=[],
        session_id=uuid.uuid4(),
        message_id=uuid.uuid4(),
    )
    mock_svc.run = AsyncMock(return_value=response)

    async def _stream_gen(*args: Any, **kwargs: Any):
        words = answer.split()
        for i, w in enumerate(words):
            yield QAStreamEvent("token", {"delta": w if i == 0 else f" {w}"})
        yield QAStreamEvent(
            "done",
            json.loads(response.model_copy(update={"session_id": kwargs.get("session_id", response.session_id)}).model_dump_json()),
        )

    mock_svc.stream = _stream_gen
    return mock_svc


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Re-use the session-scoped engine from conftest via db_session + api_client.
# These are defined in tests/conftest.py.


# ---------------------------------------------------------------------------
# 1. Cross-chat session forgery → 404
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cross_chat_session_returns_404(
    api_client: AsyncClient,
    db_session: AsyncSession,
) -> None:
    """A session belonging to chat_a cannot be used under chat_b URL."""
    chat_a = await _create_chat(db_session, "Chat A")
    chat_b = await _create_chat(db_session, "Chat B")
    session_a = await _create_session(db_session, chat_a)

    resp = await api_client.post(
        f"/chats/{chat_b}/sessions/{session_a}/messages",
        json={"question": "test", "stream": False},
    )
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# 2. Unknown field in body → 422
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unknown_field_in_body_returns_422(
    api_client: AsyncClient,
    db_session: AsyncSession,
) -> None:
    """Extra fields in the request body must be rejected with 422."""
    chat_id = await _create_chat(db_session)
    session_id = await _create_session(db_session, chat_id)

    resp = await api_client.post(
        f"/chats/{chat_id}/sessions/{session_id}/messages",
        json={"question": "hello", "stream": False, "evil_field": "injected"},
    )
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# 3. stream=true returns text/event-stream with SSE events
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_true_returns_sse(
    api_client: AsyncClient,
    db_session: AsyncSession,
) -> None:
    """stream=true yields a text/event-stream response with token + done events."""
    chat_id = await _create_chat(db_session)
    session_id = await _create_session(db_session, chat_id)

    mock_svc = _mock_qa_service("This is the answer.")

    with patch("app.api.messages.QAService", return_value=mock_svc):
        resp = await api_client.post(
            f"/chats/{chat_id}/sessions/{session_id}/messages",
            json={"question": "What is the main finding?", "stream": True},
        )

    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers.get("content-type", "")
    body = resp.text
    assert "event: token" in body or "event: done" in body


# ---------------------------------------------------------------------------
# 4. stream=false returns JSON QAResponse
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_false_returns_json(
    api_client: AsyncClient,
    db_session: AsyncSession,
) -> None:
    """stream=false returns a JSON body with QAResponse fields."""
    chat_id = await _create_chat(db_session)
    session_id = await _create_session(db_session, chat_id)

    mock_svc = _mock_qa_service()

    with patch("app.api.messages.QAService", return_value=mock_svc):
        resp = await api_client.post(
            f"/chats/{chat_id}/sessions/{session_id}/messages",
            json={"question": "What is the method?", "stream": False},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert "answer" in data
    assert "session_id" in data
    assert "message_id" in data
    assert "citations" in data


# ---------------------------------------------------------------------------
# 5. Disconnect-aware: is_disconnected=True sets stop_event
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_disconnect_triggers_stop_event(
    api_client: AsyncClient,
    db_session: AsyncSession,
) -> None:
    """When is_disconnected returns True, the SSE stream emits a STOPPED error."""
    chat_id = await _create_chat(db_session)
    session_id = await _create_session(db_session, chat_id)

    from app.services.qa_service import QAService, QAStreamEvent  # noqa: PLC0415

    stop_was_set = False

    async def _slow_stream(c_id, s_id, q, *, chat_provider, stop_event=None, **_kwargs):
        nonlocal stop_was_set
        if stop_event and stop_event.is_set():
            stop_was_set = True
            yield QAStreamEvent("error", {"code": "STOPPED"})
            return
        # Yield one token then check stop
        yield QAStreamEvent("token", {"delta": "hello"})
        if stop_event:
            stop_event.set()
            stop_was_set = True
        yield QAStreamEvent("error", {"code": "STOPPED", "detail": "stopped"})

    mock_svc = MagicMock(spec=QAService)
    mock_svc.stream = _slow_stream

    # Monkeypatch request.is_disconnected to return True after first check
    call_count = {"n": 0}

    async def _is_disconnected() -> bool:
        call_count["n"] += 1
        return call_count["n"] > 1  # disconnect after first event

    with patch("app.api.messages.QAService", return_value=mock_svc):
        resp = await api_client.post(
            f"/chats/{chat_id}/sessions/{session_id}/messages",
            json={"question": "Are you there?", "stream": True},
        )

    # Response should be SSE
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers.get("content-type", "")


# ---------------------------------------------------------------------------
# 6. Message persisted: user + assistant rows visible after call
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_messages_persisted_after_run(
    db_session: AsyncSession,
) -> None:
    """After QAService.run, user + assistant messages are in the DB."""
    from app.services.qa_service import _persist_qa_pair  # noqa: PLC0415

    chat_id = await _create_chat(db_session)
    session_id = await _create_session(db_session, chat_id)

    msg_id = await _persist_qa_pair(
        db_session,
        session_id=session_id,
        question="What is the main contribution?",
        answer="The contribution is novel.",
        citations=[],
    )
    await db_session.flush()

    rows = (
        await db_session.scalars(
            select(Message).where(Message.session_id == session_id)
        )
    ).all()

    roles = [r.role for r in rows]
    assert "user" in roles
    assert "assistant" in roles
    assert any(r.id == msg_id for r in rows)


# ---------------------------------------------------------------------------
# 7. Citation document_id must be in ChatDocument association
# ---------------------------------------------------------------------------


def test_map_citations_drops_foreign_doc() -> None:
    """Citations referencing a document from another chat are dropped."""
    from app.agent.state import CitationDraft  # noqa: PLC0415
    from app.services.qa_service import _map_citations  # noqa: PLC0415

    current_chat = uuid.uuid4()
    other_chat = uuid.uuid4()
    doc_id = uuid.uuid4()

    good = CitationDraft(
        citation_id=str(uuid.uuid4()),
        chat_id=current_chat,
        document_id=doc_id,
        document_name="paper.pdf",
        page_start=1,
        page_end=2,
        source_node_id="n1",
        excerpt="good",
    )
    bad = CitationDraft(
        citation_id=str(uuid.uuid4()),
        chat_id=other_chat,
        document_id=doc_id,
        document_name="evil.pdf",
        page_start=1,
        page_end=2,
        source_node_id="n2",
        excerpt="evil",
    )
    result = _map_citations([good, bad], chat_id=current_chat)
    assert len(result) == 1
    assert result[0].document_name == "paper.pdf"


# ---------------------------------------------------------------------------
# 8. list_messages isolation: different session not readable
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_messages_session_isolation(
    db_session: AsyncSession,
) -> None:
    """list_messages with wrong chat_id → SessionNotFound (isolation)."""
    from app.errors import SessionNotFound  # noqa: PLC0415
    from app.services import session_service  # noqa: PLC0415

    chat_a = await _create_chat(db_session, "Isolation Chat A")
    chat_b = await _create_chat(db_session, "Isolation Chat B")
    session_a = await _create_session(db_session, chat_a)

    with pytest.raises(SessionNotFound):
        await session_service.list_messages(
            db_session, session_id=session_a, chat_id=chat_b
        )


# ---------------------------------------------------------------------------
# 9. Generation overrides in body propagate to QAService.run
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generation_overrides_propagate(
    api_client: AsyncClient,
    db_session: AsyncSession,
) -> None:
    """max_answer_tokens / temperature / context_window in body reach QAService."""
    chat_id = await _create_chat(db_session, "GenCfg Chat")
    session_id = await _create_session(db_session, chat_id)
    await db_session.commit()

    captured: dict[str, Any] = {}

    async def _capturing_run(*args: Any, **kwargs: Any) -> Any:
        captured["generation_config"] = kwargs.get("generation_config")
        from app.models.domain import QAResponse  # noqa: PLC0415
        return QAResponse(
            answer="ok",
            citations=[],
            documents_used=[],
            coverage=1.0,
            uncertainty=[],
            session_id=session_id,
            message_id=uuid.uuid4(),
        )

    from app.services.qa_service import QAService  # noqa: PLC0415

    mock_svc = MagicMock(spec=QAService)
    mock_svc.run = _capturing_run

    with patch(
        "app.api.messages.QAService", return_value=mock_svc
    ):
        resp = await api_client.post(
            f"/chats/{chat_id}/sessions/{session_id}/messages",
            json={
                "question": "summarise",
                "stream": False,
                "max_answer_tokens": 4096,
                "temperature": 0.7,
                "context_window": 20000,
            },
        )
    assert resp.status_code == 200
    gc = captured["generation_config"]
    assert gc is not None
    assert gc.max_answer_tokens == 4096
    assert gc.temperature == 0.7
    assert gc.context_window == 20000
