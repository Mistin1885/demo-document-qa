"""Node: aggregate_sources_node — compact evidence workspace on overflow.

Calls the aggregate_sources tool, then:
  1. Adds returned aggregated EvidenceItems to state.
  2. Removes original items that belong to the same (document_id, section_title)
     groups as the aggregated items (since they are now represented compactly).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from app.agent.state import AgentState, EvidenceItem
from app.agent.tools._models import AggregateSourcesParams

if TYPE_CHECKING:
    from app.agent.tools._invocation import ToolDeps


async def aggregate_sources_node(state: AgentState, deps: ToolDeps) -> dict[str, Any]:
    """Call aggregate_sources tool and replace per-group originals with aggregated items."""
    state.record_event("node_enter", "aggregate_sources_node")

    from app.agent.tools.aggregate_sources import aggregate_sources  # noqa: PLC0415

    params = AggregateSourcesParams(strategy="per_section")
    invocation = await aggregate_sources(state, params, deps=deps)

    # Identify groups covered by the aggregated items
    covered_groups: set[tuple[str, str | None]] = set()
    for agg_ev in invocation.evidence:
        covered_groups.add((str(agg_ev.document_id), agg_ev.section_title))

    # Remove original items belonging to covered groups
    kept_originals: list[EvidenceItem] = []
    for ev in state.evidence_items:
        group_key = (str(ev.document_id), ev.section_title)
        if group_key not in covered_groups or ev.source_type == "aggregated_evidence":
            kept_originals.append(ev)

    # Build new evidence list: kept originals + aggregated items
    agg_ids = {ev.evidence_id for ev in invocation.evidence}
    kept_non_agg = [ev for ev in kept_originals if ev.evidence_id not in agg_ids]
    new_evidence = kept_non_agg + list(invocation.evidence)

    new_tool_calls = list(state.tool_calls) + [invocation.record]
    new_errors = list(state.errors) + list(invocation.errors)

    state.record_event(
        "node_exit",
        "aggregate_sources_node",
        original_count=len(state.evidence_items),
        after_count=len(new_evidence),
        aggregated_groups=len(covered_groups),
    )

    return {
        "evidence_items": new_evidence,
        "tool_calls": new_tool_calls,
        "errors": new_errors,
        "debug_trace": state.debug_trace,
    }


__all__ = ["aggregate_sources_node"]
