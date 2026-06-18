"""Node: check_context_budget — detect overflow and route accordingly.

Uses ContextBudgetManager.detect_overflow(state).  The routing edge in graph.py
reads state.remaining_budget == 0 as overflow signal (or we use a sentinel field).

We use a separate field approach: we record whether overflow occurred in
state.errors is NOT a good proxy; instead we set a sentinel in the trace
and the conditional edge reads from a small helper.

Approach: the node updates state.context_token_estimate and sets
  state.token_count_is_estimate if applicable.  The graph's conditional edge
  calls _is_overflow(state) which re-runs detect_overflow.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from app.agent.budget import ContextBudgetManager
from app.agent.state import AgentState

if TYPE_CHECKING:
    pass


async def check_context_budget(
    state: AgentState,
    budget_manager: ContextBudgetManager,
) -> dict[str, Any]:
    """Estimate token usage and check overflow."""
    state.record_event("node_enter", "check_context_budget")

    evidence_tokens = sum(budget_manager.count_tokens(ev.content) for ev in state.evidence_items)
    history_tokens = budget_manager.estimate_conversation(state.conversation_history)
    plan_tokens = (
        budget_manager.count_tokens(state.plan.goal)
        + budget_manager.count_tokens(state.plan.rationale)
        if state.plan is not None
        else 0
    )
    total = evidence_tokens + history_tokens + plan_tokens
    overflow = budget_manager.detect_overflow(state)

    state.record_event(
        "budget_check",
        "check_context_budget",
        evidence_tokens=evidence_tokens,
        history_tokens=history_tokens,
        plan_tokens=plan_tokens,
        total=total,
        overflow=overflow,
    )
    state.record_event("node_exit", "check_context_budget", overflow=overflow)

    return {
        "context_token_estimate": total,
        "token_count_is_estimate": budget_manager.last_was_estimate,
        "remaining_budget": max(0, budget_manager.default_context_window - total),
        "debug_trace": state.debug_trace,
    }


def is_overflow(state: AgentState, budget_manager: ContextBudgetManager) -> bool:
    """Conditional edge helper: return True if we should route to aggregate_sources."""
    return budget_manager.detect_overflow(state)


__all__ = ["check_context_budget", "is_overflow"]
