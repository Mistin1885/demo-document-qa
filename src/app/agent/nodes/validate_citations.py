"""Node: validate_citations — extract inline markers, build CitationDraft list,
and run PolicyEngine.enforce_answer + enforce_citations (Phase 7.4).

Steps:
  1. Scan state.answer for [c<idx>] markers → build CitationDraft list.
  2. Call PolicyEngine.enforce_answer(state, answer_draft):
       - Policy 11: override answer with no-info fallback if no evidence.
       - Policy 8: mark unverified numeric tokens.
  3. Call PolicyEngine.enforce_citations(state, citations):
       - Policy 12: remove citations with wrong chat_id.
       - Policy 13: remove citations whose document_id is not in
         chat_document_ids (when provided in state metadata).

Rules:
  - citation_id = f"cit-{idx}" (1-based, matching marker numbering).
  - citation.chat_id = state.chat_id (never from LLM output).
  - document_name defaults to section_title or str(document_id).
"""

from __future__ import annotations

import re
from typing import Any

from app.agent.policies import PolicyEngine
from app.agent.state import AgentState, CitationDraft

_MARKER_RE = re.compile(r"\[c(\d+)\]")
_engine = PolicyEngine()


async def validate_citations(state: AgentState) -> dict[str, Any]:
    """Extract citation markers from answer and map to evidence items."""
    state.record_event("node_enter", "validate_citations")

    answer = state.answer

    if answer is None:
        state.record_event("node_exit", "validate_citations", citations_built=0)
        return {"citations": [], "debug_trace": state.debug_trace}

    # --- Step 1: Parse [c<idx>] markers ---
    raw_citations: list[CitationDraft] = []
    seen_indices: set[int] = set()

    for match in _MARKER_RE.finditer(answer):
        idx = int(match.group(1))  # 1-based
        if idx in seen_indices:
            continue
        seen_indices.add(idx)

        evidence_index = idx - 1  # 0-based
        if evidence_index < 0 or evidence_index >= len(state.evidence_items):
            state.record_error(
                code="INVALID_CITATION_MARKER",
                detail=(
                    f"Marker [c{idx}] has no corresponding evidence item "
                    f"(only {len(state.evidence_items)} items)"
                ),
            )
            continue

        ev = state.evidence_items[evidence_index]
        raw_citations.append(
            CitationDraft(
                citation_id=f"cit-{idx}",
                # chat_id is ALWAYS from state — never from LLM output
                chat_id=state.chat_id,
                document_id=ev.document_id,
                document_name=ev.section_title or str(ev.document_id),
                page_start=ev.page_start,
                page_end=ev.page_end,
                section_title=ev.section_title,
                source_node_id=ev.source_node_id,
                excerpt=ev.content[:500],  # cap excerpt length
            )
        )

    # --- Step 2: enforce_answer (policies 8 + 11) ---
    validated_answer = _engine.enforce_answer(state, answer)

    # --- Step 3: enforce_citations (policies 12 + 13) ---
    # chat_document_ids is not available here (no DB I/O in nodes),
    # so we only run policy 12 (chat_id check); policy 13 runs in
    # validate_scope_isolation which receives chat_document_ids from the caller.
    clean_citations = _engine.enforce_citations(state, raw_citations, chat_document_ids=None)

    state.record_event(
        "node_exit",
        "validate_citations",
        citations_built=len(clean_citations),
        answer_overridden=validated_answer != answer,
    )

    return {
        "answer": validated_answer,
        "citations": clean_citations,
        "errors": list(state.errors),
        "debug_trace": state.debug_trace,
    }


__all__ = ["validate_citations"]
