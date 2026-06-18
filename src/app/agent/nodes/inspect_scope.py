"""Node: inspect_scope — enumerate documents and load per-document manifests.

Calls inspect_document for each document in chat_manifest.document_titles
(or a compact version for large chats). Populates state.document_manifests.

Isolation: all calls use state.chat_id from the existing manifest — no
  chat_id is passed from the question or from LLM output.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from app.agent.state import AgentState

if TYPE_CHECKING:
    from app.agent.tools._invocation import ToolDeps


_MAX_DOCS_TO_INSPECT = 10  # cap to avoid huge graphs for large chats


async def inspect_scope(state: AgentState, deps: ToolDeps) -> dict[str, Any]:
    """Inspect each document in the chat and populate document_manifests."""
    state.record_event("node_enter", "inspect_scope")

    updates: dict[str, Any] = {}
    new_tool_calls = list(state.tool_calls)
    new_doc_manifests = list(state.document_manifests)
    new_errors = list(state.errors)
    new_fps = set(state.tool_invocations_fingerprints)

    # If no manifest yet, nothing to do
    if state.chat_manifest is None:
        state.record_event("node_exit", "inspect_scope", skipped=True)
        updates["debug_trace"] = state.debug_trace
        return updates

    # We don't have document UUIDs in ChatManifestSnapshot (only titles).
    # inspect_scope can only fetch document details if we have actual IDs.
    # In Phase 7.3 the document IDs come from the DB-loaded manifests or
    # from the session_factory — we skip if no IDs are available and rely
    # on plan_information_needs to issue inspect_document calls with real IDs.
    # (Real IDs are populated by the service layer before invoking the graph.)
    existing_doc_ids = {dm.document_id for dm in new_doc_manifests}

    # Try to load manifests for any document_id we can derive from evidence or
    # from pre-populated document_manifests.  In the absence of pre-populated
    # IDs here, this node is a graceful no-op — the planner will handle it.

    state.record_event(
        "node_exit",
        "inspect_scope",
        doc_manifests_loaded=len(new_doc_manifests),
        existing_ids=len(existing_doc_ids),
    )
    updates["debug_trace"] = state.debug_trace
    updates["tool_calls"] = new_tool_calls
    updates["document_manifests"] = new_doc_manifests
    updates["errors"] = new_errors
    updates["tool_invocations_fingerprints"] = new_fps
    return updates


__all__ = ["inspect_scope"]
