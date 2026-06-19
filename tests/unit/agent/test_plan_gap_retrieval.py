"""Unit tests for plan_gap_retrieval node.

Test density cap: ≤ 10 items per file (CLAUDE.md §12.1).
"""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock

import pytest

from app.agent.nodes.plan_gap_retrieval import plan_gap_retrieval
from app.agent.state import AgentPlan, AgentState, CoverageRequirement

_CHAT_ID = uuid.UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
_SESSION_ID = uuid.UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")


def _make_state(*, unsatisfied_descs: list[str], chosen_tools: list[str] | None = None) -> AgentState:
    reqs = [
        CoverageRequirement(requirement_id=f"req-{i}", description=d, satisfied=False)
        for i, d in enumerate(unsatisfied_descs)
    ]
    plan = AgentPlan(
        goal="test",
        information_needs=["need-1"],
        chosen_tools=chosen_tools or ["search_hybrid"],
        rationale="general question: hybrid retrieval",
    )
    return AgentState(
        chat_id=_CHAT_ID,
        session_id=_SESSION_ID,
        question="What is the main contribution?",
        plan=plan,
        coverage_requirements=reqs,
        iteration_count=1,
    )


# 1. Happy path: 2 unsatisfied → gap_queries populated, iteration +1, rationale is plain string
@pytest.mark.asyncio
async def test_gap_queries_populated() -> None:
    state = _make_state(unsatisfied_descs=["method details", "experimental results"])
    result = await plan_gap_retrieval(state, MagicMock())
    plan = result["plan"]
    assert plan.gap_queries == ["method details", "experimental results"]
    assert result["iteration_count"] == 2
    # rationale must not contain semicolons or structured gap_queries= payload
    assert "gap_queries=" not in plan.rationale
    assert "gap_retrieval_round=2" in plan.rationale


# 2. No-op: unsatisfied=0 → plan unchanged (no new plan key), iteration still +1
@pytest.mark.asyncio
async def test_no_op_when_all_satisfied() -> None:
    state = _make_state(unsatisfied_descs=[])
    # All reqs are satisfied (empty list), so plan is returned as-is
    result = await plan_gap_retrieval(state, MagicMock())
    assert "plan" not in result
    assert result["iteration_count"] == 2


# 3. search_hybrid already in chosen_tools → not appended again
@pytest.mark.asyncio
async def test_search_hybrid_not_duplicated_in_tools() -> None:
    state = _make_state(
        unsatisfied_descs=["architecture overview"],
        chosen_tools=["search_hybrid"],
    )
    result = await plan_gap_retrieval(state, MagicMock())
    assert result["plan"].chosen_tools.count("search_hybrid") == 1


# 4. gap_queries appended to information_needs for debug traceability
@pytest.mark.asyncio
async def test_gap_descs_appended_to_information_needs() -> None:
    state = _make_state(unsatisfied_descs=["dataset description"])
    result = await plan_gap_retrieval(state, MagicMock())
    assert "dataset description" in result["plan"].information_needs
