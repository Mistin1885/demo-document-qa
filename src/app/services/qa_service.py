"""QAService — orchestrate the LangGraph agent for QA over chat documents.

Design contract (CLAUDE.md §2, §8, §12, §13)
----------------------------------------------
- ``chat_id`` and ``session_id`` are injected from the URL path; never from
  the LLM or agent.
- All tool results are scoped to ``chat_id`` by ``RetrievalService`` and
  ``AgentState``.
- Conversation history is loaded from the same session only (isolation).
- Messages are persisted after the graph completes (user + assistant).
- ``QAResponse`` citations are asserted to belong to ``chat_id`` before return.
- SSE streaming uses ``asyncio.Event`` for stop-signal support.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.agent.budget import ContextBudgetManager
from app.agent.graph import build_graph
from app.agent.state import AgentState, CitationDraft, ConversationTurn, GenerationConfig
from app.errors import ChatNotFound, SessionNotFound
from app.models.domain import Citation, QAResponse, ToolTrace, ToolTraceStep
from app.models.orm import Message
from app.models.orm import Session as SessionORM
from app.providers.base import ChatProvider

# ---------------------------------------------------------------------------
# QA stream event dataclass
# ---------------------------------------------------------------------------


class QAStreamEvent:
    """One SSE event emitted by QAService.stream."""

    __slots__ = ("kind", "data")

    def __init__(self, kind: str, data: Any) -> None:
        self.kind = kind
        self.data = data


# ---------------------------------------------------------------------------
# DBMessageStore — Phase 7.5 concrete MessageStore
# ---------------------------------------------------------------------------


class DBMessageStore:
    """Persists messages to the ``messages`` table via a session factory.

    Implements the ``MessageStore`` protocol defined in
    ``app.agent.nodes.persist_messages``.  The store writes both the user
    question and the assistant answer; each write uses a fresh session from
    the factory so it does not interfere with the request-scoped session used
    by the API layer.

    The ``save_message`` method is intentionally a no-op in the agent's
    ``persist_messages`` node when using QAService — the service itself
    handles persistence after ``graph.ainvoke``.  This store is used ONLY
    when the graph is invoked standalone (e.g. integration tests that want DB
    persistence from the graph itself).
    """

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        expected_chat_id: uuid.UUID | None = None,
    ) -> None:
        self._factory = session_factory
        self._expected_chat_id = expected_chat_id

    async def save_message(
        self,
        *,
        chat_id: uuid.UUID,
        session_id: uuid.UUID,
        role: str,
        content: str,
    ) -> None:
        if self._expected_chat_id is not None and chat_id != self._expected_chat_id:
            raise ValueError(
                f"DBMessageStore: chat_id mismatch: "
                f"expected {self._expected_chat_id}, got {chat_id}"
            )
        async with self._factory() as db:
            msg = Message(
                id=uuid.uuid4(),
                session_id=session_id,
                role=role,
                content=content,
            )
            db.add(msg)
            await db.commit()


# ---------------------------------------------------------------------------
# Helper: load conversation history scoped to (session_id, chat_id)
# ---------------------------------------------------------------------------


async def _load_history(
    db: AsyncSession,
    *,
    session_id: uuid.UUID,
    chat_id: uuid.UUID,
) -> list[ConversationTurn]:
    """Return messages for *session_id* under *chat_id*, oldest-first.

    Enforces dual-scope isolation: the query joins through Session to verify
    ``Session.chat_id == chat_id`` so a caller cannot read history from a
    session in a different chat.
    """
    stmt = (
        select(Message)
        .join(SessionORM, SessionORM.id == Message.session_id)
        .where(
            Message.session_id == session_id,
            SessionORM.chat_id == chat_id,
        )
        .order_by(Message.created_at)
    )
    rows = (await db.scalars(stmt)).all()
    turns: list[ConversationTurn] = []
    for row in rows:
        if row.role in ("user", "assistant"):
            turns.append(ConversationTurn(role=row.role, content=row.content))  # type: ignore[arg-type]
    return turns


# ---------------------------------------------------------------------------
# Helper: persist user + assistant messages, return assistant message_id
# ---------------------------------------------------------------------------


async def _persist_qa_pair(
    db: AsyncSession,
    *,
    session_id: uuid.UUID,
    question: str,
    answer: str,
    citations: list[Citation],
) -> uuid.UUID:
    """Write user question + assistant answer; return the assistant message id."""
    user_msg = Message(
        id=uuid.uuid4(),
        session_id=session_id,
        role="user",
        content=question,
    )
    db.add(user_msg)

    assistant_id = uuid.uuid4()
    assistant_msg = Message(
        id=assistant_id,
        session_id=session_id,
        role="assistant",
        content=answer,
        citations=[c.model_dump(mode="json") for c in citations] if citations else None,
    )
    db.add(assistant_msg)
    await db.flush()
    return assistant_id


# ---------------------------------------------------------------------------
# Helper: map AgentState citations → domain Citations (enforcing chat scope)
# ---------------------------------------------------------------------------


def _map_citations(
    drafts: list[CitationDraft],
    *,
    chat_id: uuid.UUID,
) -> list[Citation]:
    """Convert CitationDraft list → Citation list, dropping cross-chat entries."""
    result: list[Citation] = []
    for d in drafts:
        if d.chat_id != chat_id:
            continue  # enforce isolation
        result.append(
            Citation(
                citation_id=uuid.UUID(d.citation_id) if len(d.citation_id) == 36 else uuid.uuid4(),
                chat_id=d.chat_id,
                document_id=d.document_id,
                document_name=d.document_name,
                page_start=d.page_start,
                page_end=d.page_end,
                section_title=d.section_title,
                source_node_id=None,
                excerpt=d.excerpt,
            )
        )
    return result


# ---------------------------------------------------------------------------
# Helper: build ToolTrace from AgentState for debug_trace
# ---------------------------------------------------------------------------


def _build_tool_trace(state: AgentState) -> ToolTrace:
    steps = [
        ToolTraceStep(
            tool_name=r.tool_name,
            status=r.status if r.status in ("ok", "overflow", "error") else "ok",
            token_estimate=r.token_estimate,
            note=r.error,
        )
        for r in state.tool_calls
    ]
    return ToolTrace(
        steps=steps,
        total_rounds=state.iteration_count,
        token_count_is_estimate=state.token_count_is_estimate,
    )


# ---------------------------------------------------------------------------
# Helper: compute coverage float from AgentState
# ---------------------------------------------------------------------------


def _compute_coverage(state: AgentState) -> float:
    reqs = state.coverage_requirements
    if not reqs:
        return 1.0 if state.answer else 0.0
    satisfied = sum(1 for r in reqs if r.satisfied)
    return satisfied / len(reqs)


# ---------------------------------------------------------------------------
# Helper: serialise an object to JSON-safe dict / primitive
# ---------------------------------------------------------------------------


def _to_json_safe(obj: Any) -> Any:
    if isinstance(obj, uuid.UUID):
        return str(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_json_safe(v) for v in obj]
    return obj


# ---------------------------------------------------------------------------
# QAService
# ---------------------------------------------------------------------------


class QAService:
    """Service orchestrating the LangGraph agent for QA endpoints.

    Usage::

        service = QAService(session_factory=factory, retrieval_service=svc)
        response = await service.run(chat_id, session_id, "What is …?",
                                     chat_provider=provider,
                                     budget_manager=bm)

    Both ``run`` and ``stream`` inject ``chat_id``/``session_id`` from the
    URL path — the LLM never sees or controls these values.
    """

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        retrieval_service: Any,  # RetrievalService — typed as Any to avoid circular
    ) -> None:
        self._session_factory = session_factory
        self._retrieval_service = retrieval_service

    # ------------------------------------------------------------------
    # _build_initial_state
    # ------------------------------------------------------------------

    async def _build_initial_state(
        self,
        db: AsyncSession,
        *,
        chat_id: uuid.UUID,
        session_id: uuid.UUID,
        question: str,
        generation_config: GenerationConfig | None = None,
    ) -> AgentState:
        """Validate ownership, load history, and build AgentState."""
        # Validate session belongs to chat (raises SessionNotFound if not)
        from app.services import session_service  # noqa: PLC0415

        await session_service.get_session_by_id(db, chat_id=chat_id, session_id=session_id)

        history = await _load_history(db, session_id=session_id, chat_id=chat_id)

        return AgentState(
            chat_id=chat_id,
            session_id=session_id,
            question=question,
            conversation_history=history,
            generation_config=generation_config or GenerationConfig(),
        )

    # ------------------------------------------------------------------
    # run (non-streaming)
    # ------------------------------------------------------------------

    async def run(
        self,
        chat_id: uuid.UUID,
        session_id: uuid.UUID,
        question: str,
        *,
        chat_provider: ChatProvider,
        budget_manager: ContextBudgetManager | None = None,
        stop_event: asyncio.Event | None = None,
        include_debug_trace: bool = False,
        generation_config: GenerationConfig | None = None,
    ) -> QAResponse:
        """Run the full agent graph synchronously and return a QAResponse.

        Parameters
        ----------
        chat_id:
            Injected from the URL path; never from the request body.
        session_id:
            Injected from the URL path; scoped to chat_id.
        question:
            The user's question text.
        chat_provider:
            Chat LLM provider for the generate_answer node.
        budget_manager:
            Optional ContextBudgetManager; defaults to ContextBudgetManager().
        stop_event:
            Optional asyncio.Event; when set the graph aborts early.
        include_debug_trace:
            If True, include tool trace in the QAResponse.
        """
        from app.agent.tools._invocation import ToolDeps  # noqa: PLC0415

        async with self._session_factory() as db:
            initial_state = await self._build_initial_state(
                db,
                chat_id=chat_id,
                session_id=session_id,
                question=question,
                generation_config=generation_config,
            )

        # Build ToolDeps
        @asynccontextmanager
        async def _session_ctx():
            async with self._session_factory() as s:
                yield s

        deps = ToolDeps(
            retrieval_service=self._retrieval_service,
            chat_provider=chat_provider,
            session_factory=_session_ctx,
        )

        # Budget manager: caller may pass an explicit one, otherwise honour
        # generation_config.context_window if present, else default 10k.
        if budget_manager is not None:
            bm = budget_manager
        elif generation_config is not None and generation_config.context_window is not None:
            bm = ContextBudgetManager(default_context_window=generation_config.context_window)
        else:
            bm = ContextBudgetManager()

        # Use InMemoryMessageStore in graph — we persist ourselves after
        from app.agent.nodes.persist_messages import InMemoryMessageStore  # noqa: PLC0415

        store = InMemoryMessageStore()
        graph = build_graph(deps=deps, chat_provider=chat_provider, budget_manager=bm, message_store=store)

        # Run graph
        result_container = await graph.ainvoke({"state": initial_state.model_dump()})
        final_state = AgentState.model_validate(result_container["state"])

        # Check stop_event
        if stop_event is not None and stop_event.is_set():
            final_state.answer = final_state.answer or "(stopped)"

        # Map citations (enforce chat scope)
        citations = _map_citations(final_state.citations, chat_id=chat_id)

        answer_text = final_state.answer or (
            "There is not enough information in the current chat's documents "
            "to answer this question."
        )

        # Persist messages
        async with self._session_factory() as db:
            message_id = await _persist_qa_pair(
                db,
                session_id=session_id,
                question=question,
                answer=answer_text,
                citations=citations,
            )
            await db.commit()

        documents_used = list(
            {c.document_id for c in citations}
            | {ev.document_id for ev in final_state.evidence_items}
        )
        uncertainty = [e.detail for e in final_state.errors]
        coverage = _compute_coverage(final_state)
        debug = _build_tool_trace(final_state) if include_debug_trace else None

        return QAResponse(
            answer=answer_text,
            citations=citations,
            documents_used=documents_used,
            coverage=coverage,
            uncertainty=uncertainty,
            session_id=session_id,
            message_id=message_id,
            debug_trace=debug,
        )

    # ------------------------------------------------------------------
    # stream (SSE)
    # ------------------------------------------------------------------

    async def stream(
        self,
        chat_id: uuid.UUID,
        session_id: uuid.UUID,
        question: str,
        *,
        chat_provider: ChatProvider,
        budget_manager: ContextBudgetManager | None = None,
        stop_event: asyncio.Event | None = None,
        generation_config: GenerationConfig | None = None,
    ) -> AsyncIterator[QAStreamEvent]:
        """Run the agent and yield SSE events.

        Events:
            token   — partial text chunks (from the final answer)
            citation — one finalised Citation
            done    — full QAResponse
            error   — {"code": ..., "detail": ...}
        """
        from app.agent.tools._invocation import ToolDeps  # noqa: PLC0415

        stop_ev = stop_event or asyncio.Event()

        try:
            async with self._session_factory() as db:
                initial_state = await self._build_initial_state(
                    db,
                    chat_id=chat_id,
                    session_id=session_id,
                    question=question,
                    generation_config=generation_config,
                )
        except (ChatNotFound, SessionNotFound) as exc:
            yield QAStreamEvent(
                "error", {"code": "NOT_FOUND", "detail": str(exc)}
            )
            return
        except Exception as exc:  # noqa: BLE001
            yield QAStreamEvent(
                "error", {"code": "INTERNAL", "detail": str(exc)}
            )
            return

        @asynccontextmanager
        async def _session_ctx():
            async with self._session_factory() as s:
                yield s

        deps = ToolDeps(
            retrieval_service=self._retrieval_service,
            chat_provider=chat_provider,
            session_factory=_session_ctx,
        )
        if budget_manager is not None:
            bm = budget_manager
        elif generation_config is not None and generation_config.context_window is not None:
            bm = ContextBudgetManager(default_context_window=generation_config.context_window)
        else:
            bm = ContextBudgetManager()

        from app.agent.nodes.persist_messages import InMemoryMessageStore  # noqa: PLC0415

        store = InMemoryMessageStore()
        graph = build_graph(deps=deps, chat_provider=chat_provider, budget_manager=bm, message_store=store)

        try:
            result_container = await graph.ainvoke({"state": initial_state.model_dump()})
        except Exception as exc:  # noqa: BLE001
            yield QAStreamEvent("error", {"code": "AGENT_ERROR", "detail": str(exc)})
            return

        final_state = AgentState.model_validate(result_container["state"])

        if stop_ev.is_set():
            yield QAStreamEvent("error", {"code": "STOPPED", "detail": "generation stopped"})
            return

        answer_text = final_state.answer or (
            "There is not enough information in the current chat's documents "
            "to answer this question."
        )

        # Emit token events (word-by-word for the answer)
        words = answer_text.split()
        for i, word in enumerate(words):
            if stop_ev.is_set():
                yield QAStreamEvent("error", {"code": "STOPPED", "detail": "generation stopped"})
                return
            chunk = word if i == 0 else f" {word}"
            yield QAStreamEvent("token", {"delta": chunk})

        # Map citations
        citations = _map_citations(final_state.citations, chat_id=chat_id)
        for cit in citations:
            yield QAStreamEvent(
                "citation",
                _to_json_safe(cit.model_dump()),
            )

        # Persist messages
        try:
            async with self._session_factory() as db:
                message_id = await _persist_qa_pair(
                    db,
                    session_id=session_id,
                    question=question,
                    answer=answer_text,
                    citations=citations,
                )
                await db.commit()
        except Exception as exc:  # noqa: BLE001
            yield QAStreamEvent("error", {"code": "PERSIST_ERROR", "detail": str(exc)})
            return

        documents_used = list(
            {c.document_id for c in citations}
            | {ev.document_id for ev in final_state.evidence_items}
        )
        coverage = _compute_coverage(final_state)
        uncertainty = [e.detail for e in final_state.errors]

        response = QAResponse(
            answer=answer_text,
            citations=citations,
            documents_used=documents_used,
            coverage=coverage,
            uncertainty=uncertainty,
            session_id=session_id,
            message_id=message_id,
        )
        yield QAStreamEvent("done", _to_json_safe(response.model_dump()))


__all__ = ["QAService", "QAStreamEvent", "DBMessageStore"]
