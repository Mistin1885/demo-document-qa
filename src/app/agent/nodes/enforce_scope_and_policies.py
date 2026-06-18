"""Node: enforce_scope_and_policies — run pre-retrieval policies (Phase 7.4).

Calls PolicyEngine.enforce_pre_retrieval(state, plan) which covers:
  - Policy 1: chat_id must be present and non-nil.
  - Policy 2: session history isolation check (trace-level).
  - Policy 3: summary questions must use fetch-all, not top-k only.
  - Policy 4: numeric questions must include query_structured_facts.
  - Policy 9: iteration cap.
  - Policy 10: duplicate-call intent logged.

On PolicyViolation (policies 1, 9) the node records the error and returns
without raising — the graph continues but the error is surfaced in
state.errors and the debug trace.  The plan may be mutated in-place by
policies 3 and 4.
"""

from __future__ import annotations

from typing import Any

from app.agent.policies import PolicyEngine, PolicyViolation
from app.agent.state import AgentState

_engine = PolicyEngine()


async def enforce_scope_and_policies(state: AgentState) -> dict[str, Any]:
    """Validate scope and enforce pre-retrieval policies; record trace."""
    state.record_event("node_enter", "enforce_scope_and_policies")

    if state.plan is None:
        # No plan yet — skip policy enforcement (will run after plan_information_needs)
        state.record_event(
            "node_exit",
            "enforce_scope_and_policies",
            status="skipped_no_plan",
        )
        return {"debug_trace": state.debug_trace}

    try:
        _engine.enforce_pre_retrieval(state, state.plan)
        state.record_event(
            "node_exit",
            "enforce_scope_and_policies",
            status="ok",
            chosen_tools=state.plan.chosen_tools,
        )
    except PolicyViolation as exc:
        state.record_event(
            "node_exit",
            "enforce_scope_and_policies",
            status="violation",
            policy_id=exc.policy_id,
            code=exc.code,
            detail=exc.detail,
        )

    return {
        "plan": state.plan,
        "errors": list(state.errors),
        "debug_trace": state.debug_trace,
    }


__all__ = ["enforce_scope_and_policies"]
