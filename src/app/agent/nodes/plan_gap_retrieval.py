"""Node: plan_gap_retrieval — add gap-filling search_hybrid calls for unsatisfied requirements.

For each unsatisfied CoverageRequirement, writes its description into
plan.gap_queries so execute_retrieval_tools can issue distinct search_hybrid
calls (different query strings → different fingerprints → not deduplicated).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from app.agent.state import AgentPlan, AgentState

if TYPE_CHECKING:
    pass


async def plan_gap_retrieval(state: AgentState, deps: Any) -> dict[str, Any]:
    """Plan gap-filling retrieval for unsatisfied coverage requirements."""
    state.record_event("node_enter", "plan_gap_retrieval")

    unsatisfied = [r for r in state.coverage_requirements if not r.satisfied]

    if not unsatisfied or state.plan is None:
        state.record_event("node_exit", "plan_gap_retrieval", skipped=True)
        return {
            "iteration_count": state.iteration_count + 1,
            "debug_trace": state.debug_trace,
        }

    gap_queries = [r.description for r in unsatisfied]
    updated_needs = list(state.plan.information_needs) + gap_queries

    updated_tools = list(state.plan.chosen_tools)
    if "search_hybrid" not in updated_tools:
        updated_tools.append("search_hybrid")

    new_iteration = state.iteration_count + 1
    updated_rationale = f"{state.plan.rationale} | gap_retrieval_round={new_iteration}"

    new_plan = AgentPlan(
        goal=state.plan.goal,
        information_needs=updated_needs,
        chosen_tools=updated_tools,
        rationale=updated_rationale,
        gap_queries=gap_queries,
    )

    state.record_event(
        "node_exit",
        "plan_gap_retrieval",
        gap_count=len(unsatisfied),
        iteration_count=new_iteration,
    )

    return {
        "plan": new_plan,
        "iteration_count": new_iteration,
        "debug_trace": state.debug_trace,
    }


__all__ = ["plan_gap_retrieval"]
