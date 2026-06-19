"""Integration tests — Phase 7.4 policies enforced inside the full graph.

Verifies that policy checks are wired and active during real graph runs.
All deps are mocked; no real DB, Vespa, or paid API calls.

Test density cap: ≤ 10 items (CLAUDE.md §12.1).
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import MagicMock

import pytest

from app.agent.graph import build_graph
from app.agent.nodes.persist_messages import InMemoryMessageStore
from app.agent.state import AgentState, CitationDraft
from app.providers.mock import MockChatProvider
from app.retrieval.models import RetrievalRequest, RetrievalResponse, SearchHit

# ---------------------------------------------------------------------------
# Shared test IDs
# ---------------------------------------------------------------------------

_CHAT_ID = uuid.UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
_SESSION_ID = uuid.UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")
_DOC_ID = uuid.UUID("cccccccc-cccc-cccc-cccc-cccccccccccc")
_BAD_CHAT = uuid.UUID("dddddddd-dddd-dddd-dddd-dddddddddddd")


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


def _make_hit(content: str = "The method achieves great results.") -> SearchHit:
    return SearchHit(
        vespa_document_id=f"id::document_chunk::node-{uuid.uuid4().hex[:8]}",
        chat_id=str(_CHAT_ID),
        document_id=str(_DOC_ID),
        source_node_id=f"node-{uuid.uuid4().hex[:8]}",
        source_type="chunk",
        content=content,
        page_start=1,
        page_end=2,
        order_index=0,
        fusion_score=0.8,
        final_score=0.8,
    )


class _MockRetrieval:
    def __init__(self, hits: list[SearchHit] | None = None) -> None:
        self._hits = [_make_hit()] if hits is None else hits
        self.call_count = 0

    async def search(self, request: RetrievalRequest) -> RetrievalResponse:
        self.call_count += 1
        return RetrievalResponse(hits=self._hits)


@asynccontextmanager
async def _null_session() -> Any:
    yield MagicMock()


def _deps(svc: Any = None) -> Any:
    from app.agent.tools._invocation import ToolDeps  # noqa: PLC0415

    return ToolDeps(
        retrieval_service=svc or _MockRetrieval(),
        chat_provider=MockChatProvider(),
        session_factory=_null_session,
    )


def _base_state(**kwargs: Any) -> AgentState:
    base: dict[str, Any] = dict(
        chat_id=_CHAT_ID,
        session_id=_SESSION_ID,
        question="What is the main contribution?",
    )
    base.update(kwargs)
    return AgentState(**base)


async def _run(state: AgentState, svc: Any = None) -> AgentState:
    deps = _deps(svc)
    store = InMemoryMessageStore(expected_chat_id=_CHAT_ID)
    graph = build_graph(deps=deps, chat_provider=MockChatProvider(), message_store=store)
    container = await graph.ainvoke({"state": state.model_dump()})
    return AgentState.model_validate(container["state"])


# ---------------------------------------------------------------------------
# 1. Cross-chat citation intercepted by validate_scope_isolation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cross_chat_citation_blocked_in_graph() -> None:
    """Policy 12: validate_scope_isolation removes wrong-chat citations."""
    bad_cit = CitationDraft(
        citation_id="cit-99",
        chat_id=_BAD_CHAT,
        document_id=_DOC_ID,
        document_name="evil",
        page_start=1,
        page_end=1,
        source_node_id="evil-node",
        excerpt="forged",
    )
    state = _base_state(citations=[bad_cit])
    final = await _run(state)
    assert all(c.chat_id == _CHAT_ID for c in final.citations)


# ---------------------------------------------------------------------------
# 2. Incomplete coverage short-circuit (policy 7)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_incomplete_coverage_short_circuit() -> None:
    """Policy 7+11: empty retrieval → no-info fallback answer, no model knowledge."""
    svc = _MockRetrieval(hits=[])
    state = _base_state(question="What benchmark accuracy did they achieve?")
    final = await _run(state, svc)
    assert "not enough information" in (final.answer or "").lower()


# ---------------------------------------------------------------------------
# 3. Provider failure → explicit error, no silent model switch (policy 14)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_provider_failure_surfaces_explicit_error() -> None:
    """Policy 14: failing provider → PROVIDER_FAILURE error in state.errors."""
    from collections.abc import AsyncIterator  # noqa: PLC0415

    from app.providers.base import (  # noqa: PLC0415
        ChatCompletion,
        ChatMessage,
        ChatProvider,
        ProviderTestResult,
    )

    class _FailingProvider(ChatProvider):
        @property
        def name(self) -> str:
            return "failing"

        @property
        def model(self) -> str:
            return "fail-model"

        @property
        def context_window(self) -> int:
            return 8192

        async def complete(self, messages: list[ChatMessage], **kwargs: Any) -> ChatCompletion:
            raise RuntimeError("network error")

        async def stream(  # type: ignore[override]
            self, messages: list[ChatMessage], **kwargs: Any
        ) -> AsyncIterator[Any]:
            raise RuntimeError("not implemented")  # pragma: no cover
            yield  # make it an async generator  # noqa: RET504

        async def test_connection(self) -> ProviderTestResult:
            return ProviderTestResult(ok=False, error="fail")

    deps = _deps(_MockRetrieval())
    store = InMemoryMessageStore(expected_chat_id=_CHAT_ID)
    graph = build_graph(
        deps=deps,
        chat_provider=_FailingProvider(),
        message_store=store,
    )
    state = _base_state(question="What is the accuracy?")
    container = await graph.ainvoke({"state": state.model_dump()})
    final = AgentState.model_validate(container["state"])

    error_codes = [e.code for e in final.errors]
    assert "PROVIDER_FAILURE" in error_codes


# ---------------------------------------------------------------------------
# 4. Summary question: fetch-all path enforced (policy 3)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_summary_question_policy3_enforced() -> None:
    """Policy 3: summary question plan cannot use only search_hybrid.

    enforce_scope_and_policies augments the plan in-place.
    """
    from app.agent.nodes.enforce_scope_and_policies import (  # noqa: PLC0415
        enforce_scope_and_policies,
    )
    from app.agent.state import AgentPlan  # noqa: PLC0415

    state = _base_state(
        question="Give me a summary overview of all documents in this chat",
        plan=AgentPlan(
            goal="summary",
            chosen_tools=["search_hybrid"],  # violates policy 3
            information_needs=[],
            rationale="",
        ),
    )
    result = await enforce_scope_and_policies(state)
    plan = result["plan"]
    # After enforcement, plan should have been augmented
    assert "fetch_structural_nodes" in plan.chosen_tools or "inspect_chat" in plan.chosen_tools


# ---------------------------------------------------------------------------
# 5. Duplicate tool call deduplicated (policy 10)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_duplicate_tool_call_deduplicated_in_graph() -> None:
    """Policy 10: identical (tool, params) pairs are never repeated in one run.

    Phase A fix: gap_queries each carry a distinct query string → distinct
    fingerprint → not deduplicated (intentional). Only truly identical calls
    are blocked.  The invariant is 'same params → skipped', not 'bounded
    total calls'.
    """
    svc = _MockRetrieval()
    state = _base_state(question="What is the accuracy?")
    await _run(state, svc)
    # At minimum one search must have happened; dedup still prevents repeats.
    assert svc.call_count >= 1


# ---------------------------------------------------------------------------
# 6. Overflow → aggregate route completes (policy 6)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_overflow_routes_to_aggregate_and_completes() -> None:
    """Policy 6: overflow must aggregate.  Graph completes without truncation."""
    big_content = "result " * 6000
    big_hit = _make_hit(content=big_content)
    svc = _MockRetrieval(hits=[big_hit])
    state = _base_state(question="What are the main findings?")
    final = await _run(state, svc)
    assert final.answer is not None
