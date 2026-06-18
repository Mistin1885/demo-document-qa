"""Node: execute_retrieval_tools — run chosen tools from plan.chosen_tools.

For each tool in state.plan.chosen_tools:
  1. Build default params for the tool.
  2. Use PolicyEngine.check_duplicate_tool_call() (policy 10) to skip
     already-invoked (tool_name, params) pairs.
  3. Call the tool.
  4. Merge evidence / facts / errors / manifest into state.

Isolation: all tools receive state (which carries chat_id) and deps.
  They inject chat_id into queries — the LLM cannot alter it.

Policy 5 (rerank required) is enforced POST-execution via
PolicyEngine.enforce_post_retrieval in this node.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

from app.agent.policies import PolicyEngine
from app.agent.state import AgentState
from app.agent.tools import (
    TOOL_REGISTRY,
    AggregateSourcesParams,
    FetchStructuralNodesParams,
    InspectChatParams,
    InspectDocumentParams,
    QueryStructuredFactsParams,
    SearchHybridParams,
)

if TYPE_CHECKING:
    from app.agent.tools._invocation import ToolDeps

_engine = PolicyEngine()


def _build_default_params(tool_name: str, state: AgentState) -> Any:
    """Return a default params instance for *tool_name*."""
    if tool_name == "inspect_chat":
        return InspectChatParams()
    if tool_name == "inspect_document":
        # Use first unvisited document from manifests
        visited = {dm.document_id for dm in state.document_manifests}
        unvisited = [dm for dm in state.document_manifests if dm.document_id not in visited]
        doc_id = unvisited[0].document_id if unvisited else uuid.UUID(int=0)
        return InspectDocumentParams(document_id=doc_id)
    if tool_name == "fetch_structural_nodes":
        return FetchStructuralNodesParams(
            source_types=["document_overview", "chapter_summary", "compact_chapter_summary"],
        )
    if tool_name == "search_hybrid":
        return SearchHybridParams(query=state.question)
    if tool_name == "query_structured_facts":
        return QueryStructuredFactsParams()
    if tool_name == "aggregate_sources":
        return AggregateSourcesParams(strategy="per_section")
    if tool_name == "expand_evidence":
        # No sensible default without an evidence_id; skip
        return None
    return None


async def execute_retrieval_tools(state: AgentState, deps: ToolDeps) -> dict[str, Any]:
    """Execute tools from plan.chosen_tools; merge results into state."""
    state.record_event("node_enter", "execute_retrieval_tools")

    if state.plan is None:
        state.record_event("node_exit", "execute_retrieval_tools", skipped=True)
        return {"debug_trace": state.debug_trace}

    new_tool_calls = list(state.tool_calls)
    new_evidence = list(state.evidence_items)
    new_facts = list(state.structured_facts)
    new_errors = list(state.errors)
    new_fps = set(state.tool_invocations_fingerprints)
    new_chat_manifest = state.chat_manifest
    new_doc_manifests = list(state.document_manifests)

    existing_evidence_ids = {ev.evidence_id for ev in new_evidence}

    for tool_name in state.plan.chosen_tools:
        spec = TOOL_REGISTRY.get(tool_name)
        if spec is None:
            continue

        params = _build_default_params(tool_name, state)
        if params is None:
            continue

        params_dict = params.model_dump()

        # Policy 10: skip duplicate tool calls (via PolicyEngine)
        if _engine.check_duplicate_tool_call(state, tool_name, params_dict):
            continue

        # Register fingerprint before calling to prevent races
        fp = AgentState._fingerprint(tool_name, params_dict)  # type: ignore[attr-defined]
        new_fps.add(fp)

        state.record_event("tool_call", tool_name, status="calling", params=params_dict)

        invocation = await spec.callable(state, params, deps=deps)

        new_tool_calls.append(invocation.record)

        # Merge evidence (de-dup by evidence_id)
        for ev in invocation.evidence:
            if ev.evidence_id not in existing_evidence_ids:
                new_evidence.append(ev)
                existing_evidence_ids.add(ev.evidence_id)

        # Merge facts
        existing_fact_ids = {str(f.fact_id) for f in new_facts}
        for fact in invocation.facts:
            if str(fact.fact_id) not in existing_fact_ids:
                new_facts.append(fact)
                existing_fact_ids.add(str(fact.fact_id))

        # Merge errors
        new_errors.extend(invocation.errors)

        # Update manifest if provided
        if invocation.chat_manifest is not None:
            new_chat_manifest = invocation.chat_manifest

        if invocation.document_manifest is not None:
            known = {dm.document_id for dm in new_doc_manifests}
            if invocation.document_manifest.document_id not in known:
                new_doc_manifests.append(invocation.document_manifest)

    # Policy 5: enforce rerank_mode != "none" on all search_hybrid calls
    _engine.enforce_post_retrieval(state, new_tool_calls)

    state.record_event("node_exit", "execute_retrieval_tools", tools_run=len(new_tool_calls))

    return {
        "tool_calls": new_tool_calls,
        "evidence_items": new_evidence,
        "structured_facts": new_facts,
        "errors": new_errors,
        "tool_invocations_fingerprints": new_fps,
        "chat_manifest": new_chat_manifest,
        "document_manifests": new_doc_manifests,
        "debug_trace": state.debug_trace,
    }


__all__ = ["execute_retrieval_tools"]
