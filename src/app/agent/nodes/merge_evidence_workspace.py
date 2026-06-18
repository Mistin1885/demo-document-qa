"""Node: merge_evidence_workspace — de-duplicate and sort evidence items.

After execute_retrieval_tools, the evidence workspace may contain items
from multiple tools.  This node:
  1. De-duplicates by evidence_id (add_evidence already does this, but we
     re-run on the full list for safety after graph-level merging).
  2. Re-sorts evidence by (document_id, page_start asc, score desc).

No tool calls; no network I/O.
"""

from __future__ import annotations

from typing import Any

from app.agent.state import AgentState, EvidenceItem


def _sort_key(ev: EvidenceItem) -> tuple[str, int, float]:
    return (str(ev.document_id), ev.page_start, -(ev.score or 0.0))


async def merge_evidence_workspace(state: AgentState) -> dict[str, Any]:
    """De-duplicate and sort state.evidence_items."""
    state.record_event("node_enter", "merge_evidence_workspace")

    seen: set[str] = set()
    unique: list[EvidenceItem] = []
    for ev in state.evidence_items:
        if ev.evidence_id not in seen:
            seen.add(ev.evidence_id)
            unique.append(ev)

    sorted_evidence = sorted(unique, key=_sort_key)

    state.record_event(
        "node_exit",
        "merge_evidence_workspace",
        before=len(state.evidence_items),
        after=len(sorted_evidence),
    )

    return {
        "evidence_items": sorted_evidence,
        "debug_trace": state.debug_trace,
    }


__all__ = ["merge_evidence_workspace"]
