"""Phase B decomposition tests (≤10 tests)."""

from __future__ import annotations

import uuid

import pytest

from app.agent.nodes.execute_retrieval_tools import _plan_to_invocations
from app.agent.nodes.plan_information_needs import plan_information_needs
from app.agent.state import AgentState

_CHAT_ID = uuid.UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
_SESSION_ID = uuid.UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")


def _state(question: str) -> AgentState:
    return AgentState(chat_id=_CHAT_ID, session_id=_SESSION_ID, question=question)


@pytest.mark.asyncio
async def test_comparison_question_decomposes_into_subqueries() -> None:
    result = await plan_information_needs(_state("Compare LightRAG with GraphRAG."))
    plan = result["plan"]
    assert plan.chosen_tools == ["search_hybrid"]
    assert {"LightRAG", "GraphRAG"}.issubset(set(plan.gap_queries))
    assert "LightRAG GraphRAG differences" in plan.gap_queries


@pytest.mark.asyncio
async def test_ablation_question_sets_structured_fact_filter_hints() -> None:
    result = await plan_information_needs(
        _state("Performance of ablated versions of LightRAG")
    )
    plan = result["plan"]
    assert "query_structured_facts" in plan.chosen_tools
    assert "search_hybrid" in plan.chosen_tools
    assert "ablation" in plan.fact_filter_hints.kinds
    assert plan.gap_queries


@pytest.mark.asyncio
async def test_fact_filter_hints_flow_to_query_structured_facts_params() -> None:
    state = _state("Performance of ablated versions of LightRAG")
    result = await plan_information_needs(state)
    planned = state.model_copy(update={"plan": result["plan"]})
    invocations = _plan_to_invocations(planned)
    facts = [(name, params) for name, params in invocations if name == "query_structured_facts"]
    assert len(facts) == 1
    assert facts[0][1].kinds == ["ablation", "metric"]
    assert facts[0][1].keys == ["ablation"]
