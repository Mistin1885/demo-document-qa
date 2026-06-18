"""Node: check_coverage — assess whether information needs are satisfied.

For each CoverageRequirement with evidence_ids == []:
  - Compute simple token overlap between requirement.description and
    evidence content.
  - If at least one evidence item has non-zero overlap, mark satisfied=True
    and record matching evidence_ids.

Routing:
  - If any unsatisfied requirements remain AND iteration_count < 2 → "incomplete"
    so graph routes to plan_gap_retrieval.
  - Otherwise → "complete" (we've done our best).

Policy 7 (Phase 7.4):
  - When coverage_state would be "incomplete" AND iteration_count >= 2
    (the cap), we set coverage_state="complete" so the graph proceeds to
    verify_critical_claims.  The generate_answer node enforces policy 7
    (incomplete at cap → return no-info answer) via PolicyEngine.enforce_pre_answer.
"""

from __future__ import annotations

import re
from typing import Any

from app.agent.state import AgentState, CoverageRequirement


def _token_overlap(text_a: str, text_b: str) -> int:
    """Return count of shared lowercase tokens between two strings."""
    tokens_a = set(re.sub(r"[^\w]", " ", text_a.lower()).split())
    tokens_b = set(re.sub(r"[^\w]", " ", text_b.lower()).split())
    # Ignore very short stop-words
    stop = {"the", "a", "an", "of", "in", "for", "and", "or", "to", "is", "are", "that"}
    return len((tokens_a - stop) & (tokens_b - stop))


async def check_coverage(state: AgentState) -> dict[str, Any]:
    """Mark coverage requirements satisfied/unsatisfied based on evidence."""
    state.record_event("node_enter", "check_coverage")

    updated_reqs: list[CoverageRequirement] = []

    for req in state.coverage_requirements:
        if req.satisfied:
            updated_reqs.append(req)
            continue

        # Find evidence that overlaps with this requirement
        matched_ids: list[str] = []
        for ev in state.evidence_items:
            if _token_overlap(req.description, ev.content) > 0:
                matched_ids.append(ev.evidence_id)

        if matched_ids:
            updated_reqs.append(
                CoverageRequirement(
                    requirement_id=req.requirement_id,
                    description=req.description,
                    evidence_ids=matched_ids[:5],  # cap to 5 per requirement
                    satisfied=True,
                )
            )
        else:
            updated_reqs.append(req)

    # Determine new coverage state
    has_unsatisfied = any(not r.satisfied for r in updated_reqs)

    # Route to gap retrieval only if we have budget (iteration_count < 2)
    if has_unsatisfied and state.iteration_count < 2:
        new_coverage_state = "incomplete"
    else:
        # At or past cap: force "complete" so graph proceeds.
        # Policy 7 in generate_answer will enforce the no-info fallback
        # if coverage is still insufficient.
        new_coverage_state = "complete"

    state.record_event(
        "node_exit",
        "check_coverage",
        satisfied=sum(1 for r in updated_reqs if r.satisfied),
        unsatisfied=sum(1 for r in updated_reqs if not r.satisfied),
        coverage_state=new_coverage_state,
    )

    return {
        "coverage_requirements": updated_reqs,
        "coverage_state": new_coverage_state,  # type: ignore[dict-item]
        "debug_trace": state.debug_trace,
    }


__all__ = ["check_coverage"]
