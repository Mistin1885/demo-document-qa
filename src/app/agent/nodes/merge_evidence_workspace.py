"""Node: merge_evidence_workspace — de-duplicate and sort evidence items.

After execute_retrieval_tools, the evidence workspace may contain items
from multiple tools.  This node:
  1. De-duplicates by evidence_id (add_evidence already does this, but we
     re-run on the full list for safety after graph-level merging).
  2. Re-sorts evidence by (document_id, page_start asc, score desc).

No tool calls; no network I/O.
"""

from __future__ import annotations

import re
from typing import Any

from app.agent.state import AgentState, EvidenceItem


def _label_priority(question: str, ev: EvidenceItem) -> int:
    """Prioritise evidence with the exact Figure/Table label from the question."""
    label_match = re.search(r"\b(figure|table)\s+(\d+)\b", question, flags=re.IGNORECASE)
    if label_match is None:
        return 0
    wanted = f"{label_match.group(1)} {label_match.group(2)}".lower()
    content = ev.content.lower()
    if wanted in content:
        return -3
    if label_match.group(1).lower() in content:
        # Same modality but wrong label (e.g. Figure 1 for a Figure 2 question)
        # should not appear before an exact-label direct chunk.
        return 3
    return 1


def _origin_priority(ev: EvidenceItem) -> int:
    if ev.origin_tool == "grep_document_chunks":
        return 0
    if ev.source_type in {"table", "figure", "equation", "table_record", "figure_caption"}:
        return 1
    return 2


def _sort_key(ev: EvidenceItem, question: str) -> tuple[int, int, str, int, float]:
    return (
        _label_priority(question, ev),
        _origin_priority(ev),
        str(ev.document_id),
        ev.page_start,
        -(ev.score or 0.0),
    )


async def merge_evidence_workspace(state: AgentState) -> dict[str, Any]:
    """De-duplicate and sort state.evidence_items."""
    state.record_event("node_enter", "merge_evidence_workspace")

    seen: set[str] = set()
    unique: list[EvidenceItem] = []
    for ev in state.evidence_items:
        if ev.evidence_id not in seen:
            seen.add(ev.evidence_id)
            unique.append(ev)

    sorted_evidence = sorted(unique, key=lambda ev: _sort_key(ev, state.question))

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
