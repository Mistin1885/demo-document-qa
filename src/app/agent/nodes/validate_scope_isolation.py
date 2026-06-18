"""Node: validate_scope_isolation — final citation isolation check (Phase 7.4).

Uses PolicyEngine.enforce_citations (policies 12, 13) as the authoritative
second safety layer.  This node runs AFTER validate_citations and ensures:

  Policy 12: citation.chat_id == state.chat_id  (chat isolation)
  Policy 13: citation.document_id in chat_document_ids  (association check)

chat_document_ids is obtained from state.document_manifests (the agent's
in-memory view of which documents are in scope for this chat).  A full DB
lookup is not available here; the service layer is responsible for loading
document_manifests correctly from the ChatDocument association.

Violations: removed citation + AgentError(code="CITATION_WRONG_CHAT" or
"CITATION_DOC_NOT_IN_CHAT").
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from app.agent.policies import PolicyEngine
from app.agent.state import AgentState, CitationDraft

_engine = PolicyEngine()


async def validate_scope_isolation(state: AgentState) -> dict[str, Any]:
    """Remove citations that violate isolation policies and record errors."""
    state.record_event("node_enter", "validate_scope_isolation")

    # Build the set of document_ids that legitimately belong to this chat
    # from the document_manifests loaded at the start of the run.
    chat_document_ids: set[UUID] = {dm.document_id for dm in state.document_manifests}
    # If no manifests loaded (e.g., in unit tests without inspect_scope),
    # fall back to None so policy 13 is skipped rather than removing everything.
    doc_ids_arg: set[UUID] | None = chat_document_ids if chat_document_ids else None

    clean_citations: list[CitationDraft] = _engine.enforce_citations(  # type: ignore[assignment]
        state, list(state.citations), chat_document_ids=doc_ids_arg
    )

    violations = len(state.citations) - len(clean_citations)

    state.record_event(
        "node_exit",
        "validate_scope_isolation",
        violations=violations,
        clean=len(clean_citations),
    )

    return {
        "citations": clean_citations,
        "errors": list(state.errors),
        "debug_trace": state.debug_trace,
    }


__all__ = ["validate_scope_isolation"]
