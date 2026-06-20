"""Agent document scope routing tests (≤10 tests)."""

from __future__ import annotations

import uuid

from app.agent.nodes.execute_retrieval_tools import _plan_to_invocations
from app.agent.state import AgentPlan, AgentState


def test_search_hybrid_invocations_include_session_document_scope() -> None:
    doc_id = uuid.uuid4()
    state = AgentState(
        chat_id=uuid.uuid4(),
        session_id=uuid.uuid4(),
        question="compare methods",
        scoped_document_ids=[doc_id],
        plan=AgentPlan(
            goal="compare methods",
            chosen_tools=["search_hybrid"],
            gap_queries=["LightRAG GraphRAG"],
        ),
    )

    invocations = _plan_to_invocations(state)

    assert len(invocations) == 1
    _tool_name, params = invocations[0]
    assert params.document_ids == [doc_id]
