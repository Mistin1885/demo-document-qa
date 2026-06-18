"""Node: plan_gap_retrieval — add gap-filling search_hybrid calls for unsatisfied requirements.

For each unsatisfied CoverageRequirement, appends search_hybrid(query=req.description)
to plan.chosen_tools (de-duped) and increments iteration_count.

The graph then loops back to execute_retrieval_tools.
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

    # For each unsatisfied requirement, we need a separate search_hybrid call
    # with that requirement's description as the query.
    # We can't encode different queries into plan.chosen_tools (which is just
    # a list of tool names).  Instead we append "search_hybrid" once per
    # unsatisfied requirement to the plan; execute_retrieval_tools will build
    # SearchHybridParams(query=state.question) which is not ideal, but since
    # the gap query is the requirement description we store it in the plan's
    # information_needs and the execute_retrieval_tools node will pick the
    # first unsatisfied requirement's description as the query.

    # Build updated plan with gap queries encoded in information_needs
    gap_needs = [r.description for r in unsatisfied]
    updated_needs = list(state.plan.information_needs) + gap_needs

    # Add search_hybrid for each gap (may have already been there)
    updated_tools = list(state.plan.chosen_tools)
    for _ in unsatisfied:
        if "search_hybrid" not in updated_tools:
            updated_tools.append("search_hybrid")
        # Force a fresh search_hybrid by appending a placeholder that
        # execute_retrieval_tools will recognise as gap search
        # We append "search_hybrid" once per gap — duplicates will be
        # de-duplicated by the fingerprint mechanism only if the query is the same.

    # Actually: to perform per-gap searches we store gap_queries in the plan
    # rationale (as a structured string) and execute_retrieval_tools reads them.
    gap_queries_str = ";".join(r.description for r in unsatisfied)
    updated_rationale = f"{state.plan.rationale} | gap_queries={gap_queries_str}"

    new_plan = AgentPlan(
        goal=state.plan.goal,
        information_needs=updated_needs,
        chosen_tools=updated_tools,
        rationale=updated_rationale,
    )

    new_iteration = state.iteration_count + 1

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
