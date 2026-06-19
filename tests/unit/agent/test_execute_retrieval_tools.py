"""Unit tests for execute_retrieval_tools — gap_queries expansion and inspect_document fanout.

Test density cap: ≤ 10 items per file (CLAUDE.md §12.1).
"""

from __future__ import annotations

import uuid

from app.agent.nodes.execute_retrieval_tools import _plan_to_invocations
from app.agent.state import AgentPlan, AgentState, DocumentManifest
from app.retrieval.models import RetrievalRequest, RetrievalResponse, SearchHit

_CHAT_ID = uuid.UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
_SESSION_ID = uuid.UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")
_DOC_A = uuid.UUID("11111111-1111-1111-1111-111111111111")
_DOC_B = uuid.UUID("22222222-2222-2222-2222-222222222222")


def _make_hit(chat_id: uuid.UUID = _CHAT_ID, doc_id: uuid.UUID = _DOC_A) -> SearchHit:
    return SearchHit(
        vespa_document_id=f"id::document_chunk::n-{uuid.uuid4().hex[:8]}",
        chat_id=str(chat_id),
        document_id=str(doc_id),
        source_node_id=f"node-{uuid.uuid4().hex[:8]}",
        source_type="chunk",
        content="relevant content",
        page_start=1,
        page_end=2,
        order_index=0,
        fusion_score=0.7,
        final_score=0.7,
    )


class _CountingRetrievalService:
    def __init__(self) -> None:
        self.queries: list[str] = []

    async def search(self, request: RetrievalRequest) -> RetrievalResponse:
        self.queries.append(request.query)
        return RetrievalResponse(hits=[_make_hit(request.chat_id)])


def _make_state(
    *,
    chosen_tools: list[str],
    gap_queries: list[str] | None = None,
    rationale: str = "general question: hybrid retrieval",
    doc_ids: list[uuid.UUID] | None = None,
    question: str = "What is the contribution?",
) -> AgentState:
    plan = AgentPlan(
        goal=question,
        information_needs=["test need"],
        chosen_tools=chosen_tools,
        rationale=rationale,
        gap_queries=gap_queries or [],
    )
    doc_manifests = [
        DocumentManifest(document_id=d) for d in (doc_ids or [])
    ]
    return AgentState(
        chat_id=_CHAT_ID,
        session_id=_SESSION_ID,
        question=question,
        plan=plan,
        document_manifests=doc_manifests,
    )


# 1. Two gap_queries → two distinct search_hybrid invocations (different query strings)
def test_gap_queries_produce_distinct_invocations() -> None:
    state = _make_state(
        chosen_tools=["search_hybrid"],
        gap_queries=["method A details", "dataset B statistics"],
    )
    invocations = _plan_to_invocations(state)
    search_invocations = [(n, p) for n, p in invocations if n == "search_hybrid"]
    queries = [p.query for _, p in search_invocations]
    assert "method A details" in queries
    assert "dataset B statistics" in queries
    # queries must be distinct → fingerprints will differ
    assert len(set(queries)) == len(queries)


# 2. Summary path → inspect_document called for ALL documents in manifests
def test_summary_path_inspect_document_fanout() -> None:
    state = _make_state(
        chosen_tools=["inspect_document"],
        rationale="summary/overview question: using structural fetch-all path",
        doc_ids=[_DOC_A, _DOC_B],
    )
    invocations = _plan_to_invocations(state)
    inspect = [(n, p) for n, p in invocations if n == "inspect_document"]
    doc_ids_called = {p.document_id for _, p in inspect}
    assert _DOC_A in doc_ids_called
    assert _DOC_B in doc_ids_called
    assert len(inspect) == 2


# 3. Non-summary path → inspect_document only once (first doc)
def test_non_summary_inspect_document_single() -> None:
    state = _make_state(
        chosen_tools=["inspect_document"],
        rationale="general question: hybrid retrieval",
        doc_ids=[_DOC_A, _DOC_B],
    )
    invocations = _plan_to_invocations(state)
    inspect = [(n, p) for n, p in invocations if n == "inspect_document"]
    assert len(inspect) == 1


# 4. No gap_queries → one search_hybrid with state.question
def test_no_gap_queries_uses_question() -> None:
    state = _make_state(chosen_tools=["search_hybrid"], gap_queries=[])
    invocations = _plan_to_invocations(state)
    search = [(n, p) for n, p in invocations if n == "search_hybrid"]
    assert len(search) == 1
    assert search[0][1].query == state.question


# 5. Summary path search_hybrid gets broad preset
def test_summary_path_search_hybrid_broad_preset() -> None:
    state = _make_state(
        chosen_tools=["search_hybrid"],
        rationale="summary/overview question: using structural fetch-all path",
    )
    invocations = _plan_to_invocations(state)
    search = [(n, p) for n, p in invocations if n == "search_hybrid"]
    assert all(p.preset == "broad" for _, p in search)
