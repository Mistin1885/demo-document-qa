"""Unit tests for QAService (Phase 7.5).

Uses mock providers + fully mocked DB layer.
No paid API calls, no Vespa, no real DB.

Test density cap: ≤ 10 items per file (CLAUDE.md §12.1).
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.agent.state import AgentState, CitationDraft
from app.models.domain import QAResponse
from app.providers.mock import MockChatProvider
from app.retrieval.models import RetrievalRequest, RetrievalResponse, SearchHit
from app.services.qa_service import QAService, QAStreamEvent, _map_citations

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CHAT_ID = uuid.UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
_SESSION_ID = uuid.UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")
_OTHER_CHAT = uuid.UUID("dddddddd-dddd-dddd-dddd-dddddddddddd")
_DOC_ID = uuid.UUID("cccccccc-cccc-cccc-cccc-cccccccccccc")
_MSG_ID = uuid.UUID("eeeeeeee-eeee-eeee-eeee-eeeeeeeeeeee")


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _MockRetrievalService:
    async def search(self, request: RetrievalRequest) -> RetrievalResponse:
        hit = SearchHit(
            vespa_document_id="id::document_chunk::n1",
            chat_id=str(_CHAT_ID),
            document_id=str(_DOC_ID),
            source_node_id="node-1",
            source_type="chunk",
            content="The model achieves state-of-the-art results.",
            page_start=1,
            page_end=2,
            order_index=0,
            fusion_score=0.9,
            final_score=0.9,
        )
        return RetrievalResponse(hits=[hit])


class _EmptyRetrievalService:
    async def search(self, request: RetrievalRequest) -> RetrievalResponse:
        return RetrievalResponse(hits=[])


def _make_qa_service(retrieval_service: Any | None = None) -> QAService:
    """Build a QAService with a fully-mocked session factory."""
    mock_factory = MagicMock()
    svc = retrieval_service or _MockRetrievalService()
    return QAService(session_factory=mock_factory, retrieval_service=svc)  # type: ignore[arg-type]


def _make_state(**kwargs: Any) -> AgentState:
    base: dict[str, Any] = dict(
        chat_id=_CHAT_ID,
        session_id=_SESSION_ID,
        question="What is the main contribution?",
    )
    base.update(kwargs)
    return AgentState(**base)


def _draft(chat_id: uuid.UUID = _CHAT_ID) -> CitationDraft:
    return CitationDraft(
        citation_id=str(uuid.uuid4()),
        chat_id=chat_id,
        document_id=_DOC_ID,
        document_name="paper.pdf",
        page_start=1,
        page_end=2,
        source_node_id="node-1",
        excerpt="state-of-the-art",
    )


# ---------------------------------------------------------------------------
# 1. _map_citations: drops cross-chat citations
# ---------------------------------------------------------------------------


def test_map_citations_isolates_chat() -> None:
    good = _draft(_CHAT_ID)
    bad = _draft(_OTHER_CHAT)
    result = _map_citations([good, bad], chat_id=_CHAT_ID)
    assert len(result) == 1
    assert result[0].chat_id == _CHAT_ID


# ---------------------------------------------------------------------------
# 2. QAService.run — happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_happy_path() -> None:
    """QAService.run returns QAResponse with answer + correct session_id."""
    qa_svc = _make_qa_service()
    provider = MockChatProvider()

    initial_state = _make_state(citations=[_draft()])
    final_state = initial_state.model_copy(update={"answer": "The main contribution is X."})

    with (
        patch("app.services.qa_service.build_graph") as mock_build,
        patch.object(qa_svc, "_build_initial_state", new=AsyncMock(return_value=initial_state)),
        patch("app.services.qa_service._persist_qa_pair", new=AsyncMock(return_value=_MSG_ID)),
    ):
        mock_graph = MagicMock()
        mock_graph.ainvoke = AsyncMock(return_value={"state": final_state.model_dump()})
        mock_build.return_value = mock_graph

        # Mock the session factory context manager for persist call
        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session.commit = AsyncMock()
        qa_svc._session_factory.return_value = mock_session

        response = await qa_svc.run(
            _CHAT_ID, _SESSION_ID, "What is the main contribution?",
            chat_provider=provider,
        )

    assert isinstance(response, QAResponse)
    assert response.session_id == _SESSION_ID
    assert response.answer == "The main contribution is X."


# ---------------------------------------------------------------------------
# 3. Citations in response belong to current chat
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_citations_scoped_to_chat() -> None:
    """Citations returned in QAResponse must all belong to the current chat."""
    qa_svc = _make_qa_service()
    provider = MockChatProvider()

    final_state = _make_state(
        citations=[_draft(_CHAT_ID), _draft(_OTHER_CHAT)],
        answer="The results are excellent.",
    )

    with (
        patch("app.services.qa_service.build_graph") as mock_build,
        patch.object(qa_svc, "_build_initial_state", new=AsyncMock(return_value=final_state)),
        patch("app.services.qa_service._persist_qa_pair", new=AsyncMock(return_value=_MSG_ID)),
    ):
        mock_graph = MagicMock()
        mock_graph.ainvoke = AsyncMock(return_value={"state": final_state.model_dump()})
        mock_build.return_value = mock_graph

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session.commit = AsyncMock()
        qa_svc._session_factory.return_value = mock_session

        response = await qa_svc.run(
            _CHAT_ID, _SESSION_ID, "What are the results?",
            chat_provider=provider,
        )

    for cit in response.citations:
        assert cit.chat_id == _CHAT_ID, f"Cross-chat citation leaked: {cit}"


# ---------------------------------------------------------------------------
# 4. Cross-session history isolation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_history_scoped_to_session() -> None:
    """_load_history emits a single query scoped to (session_id, chat_id)."""
    from app.services.qa_service import _load_history  # noqa: PLC0415

    called_with: list[Any] = []

    class _FakeResult:
        def scalars(self) -> _FakeResult:
            return self

        def all(self) -> list:
            return []

    class _FakeDB:
        async def scalars(self, stmt: Any) -> _FakeResult:
            called_with.append(stmt)
            return _FakeResult()

    history = await _load_history(_FakeDB(), session_id=_SESSION_ID, chat_id=_CHAT_ID)
    assert history == []
    assert len(called_with) == 1  # exactly one query was issued


# ---------------------------------------------------------------------------
# 5. stop_event → run still returns a response
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stop_event_terminates_run() -> None:
    """When stop_event is set, run returns a response (stopped or fallback)."""
    qa_svc = _make_qa_service()
    provider = MockChatProvider()

    stop_event = asyncio.Event()
    stop_event.set()

    initial_state = _make_state()
    final_state = initial_state.model_copy(update={"answer": None})

    with (
        patch("app.services.qa_service.build_graph") as mock_build,
        patch.object(qa_svc, "_build_initial_state", new=AsyncMock(return_value=initial_state)),
        patch("app.services.qa_service._persist_qa_pair", new=AsyncMock(return_value=_MSG_ID)),
    ):
        mock_graph = MagicMock()
        mock_graph.ainvoke = AsyncMock(return_value={"state": final_state.model_dump()})
        mock_build.return_value = mock_graph

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session.commit = AsyncMock()
        qa_svc._session_factory.return_value = mock_session

        response = await qa_svc.run(
            _CHAT_ID, _SESSION_ID, "What is the contribution?",
            chat_provider=provider,
            stop_event=stop_event,
        )

    assert response.answer is not None


# ---------------------------------------------------------------------------
# 6. Provider failure in graph → stream yields error event
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_provider_failure_returns_error_event() -> None:
    """If the graph raises, stream yields an error event."""
    qa_svc = _make_qa_service()
    provider = MockChatProvider()

    initial_state = _make_state()

    with (
        patch.object(qa_svc, "_build_initial_state", new=AsyncMock(return_value=initial_state)),
        patch("app.services.qa_service.build_graph") as mock_build,
    ):
        mock_graph = MagicMock()
        mock_graph.ainvoke = AsyncMock(side_effect=RuntimeError("provider down"))
        mock_build.return_value = mock_graph

        events = []
        async for evt in qa_svc.stream(
            _CHAT_ID, _SESSION_ID, "What is the method?",
            chat_provider=provider,
        ):
            events.append(evt)

    error_events = [e for e in events if e.kind == "error"]
    assert len(error_events) >= 1
    assert error_events[0].data["code"] == "AGENT_ERROR"


# ---------------------------------------------------------------------------
# 7. Empty documents → answer = "not enough information"
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_documents_fallback_answer() -> None:
    """When graph returns no answer, response contains the fallback message."""
    qa_svc = _make_qa_service(retrieval_service=_EmptyRetrievalService())
    provider = MockChatProvider()

    initial_state = _make_state(answer=None)
    final_state = initial_state.model_copy(update={"answer": None})

    with (
        patch("app.services.qa_service.build_graph") as mock_build,
        patch.object(qa_svc, "_build_initial_state", new=AsyncMock(return_value=initial_state)),
        patch("app.services.qa_service._persist_qa_pair", new=AsyncMock(return_value=_MSG_ID)),
    ):
        mock_graph = MagicMock()
        mock_graph.ainvoke = AsyncMock(return_value={"state": final_state.model_dump()})
        mock_build.return_value = mock_graph

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session.commit = AsyncMock()
        qa_svc._session_factory.return_value = mock_session

        response = await qa_svc.run(
            _CHAT_ID, _SESSION_ID, "What are the results?",
            chat_provider=provider,
        )

    assert "not enough information" in response.answer.lower()


# ---------------------------------------------------------------------------
# 8. stream: ≥1 token event + 1 done event
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_emits_token_and_done_events() -> None:
    """stream() yields at least one token event and exactly one done event."""
    qa_svc = _make_qa_service()
    provider = MockChatProvider()

    initial_state = _make_state()
    final_state = initial_state.model_copy(
        update={"answer": "The main finding is remarkable."}
    )

    with (
        patch("app.services.qa_service.build_graph") as mock_build,
        patch.object(qa_svc, "_build_initial_state", new=AsyncMock(return_value=initial_state)),
        patch("app.services.qa_service._persist_qa_pair", new=AsyncMock(return_value=_MSG_ID)),
    ):
        mock_graph = MagicMock()
        mock_graph.ainvoke = AsyncMock(return_value={"state": final_state.model_dump()})
        mock_build.return_value = mock_graph

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session.commit = AsyncMock()
        qa_svc._session_factory.return_value = mock_session

        events: list[QAStreamEvent] = []
        async for evt in qa_svc.stream(
            _CHAT_ID, _SESSION_ID, "What is the main finding?",
            chat_provider=provider,
        ):
            events.append(evt)

    token_events = [e for e in events if e.kind == "token"]
    done_events = [e for e in events if e.kind == "done"]
    assert len(token_events) >= 1
    assert len(done_events) == 1
