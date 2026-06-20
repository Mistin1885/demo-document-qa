"""Node: execute_retrieval_tools — run chosen tools from plan.chosen_tools.

For each tool in state.plan.chosen_tools:
  1. Build params via _plan_to_invocations (which expands gap_queries into
     individual search_hybrid calls and handles summary-path inspect_document).
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

_SUMMARY_MARKER = "summary/overview question"


def _is_summary_path(state: AgentState) -> bool:
    """Return True when the current plan is on the summary/overview path."""
    if state.plan is None:
        return False
    return _SUMMARY_MARKER in state.plan.rationale or "fetch_structural_nodes" in state.plan.chosen_tools


def _search_params(state: AgentState, query: str, *, preset: str) -> SearchHybridParams:
    """Build search params, widening retrieval when deep QA is enabled."""
    if state.generation_config.deep_qa_mode:
        return SearchHybridParams(query=query, preset="broad", top_k=20, max_tokens=12_000)
    return SearchHybridParams(query=query, preset=preset)


def _plan_to_invocations(state: AgentState) -> list[tuple[str, Any]]:
    """Expand plan.chosen_tools into a flat list of (tool_name, params) pairs.

    Key behaviours:
    - search_hybrid: consume plan.gap_queries one-per-invocation (distinct
      query strings → distinct fingerprints → not deduplicated by policy 10).
      Once gap_queries are exhausted, fall back to state.question.
    - inspect_document on summary path: produce one invocation per document
      in state.document_manifests (not just the first one).
    - All other tools: single default-params invocation.
    """
    if state.replan_tool_calls:
        replan_invocations: list[tuple[str, Any]] = []
        for request in state.replan_tool_calls:
            if request.tool == "search_hybrid" and request.query:
                replan_invocations.append((
                    request.tool,
                    _search_params(state, request.query, preset="broad"),
                ))
            elif request.tool == "fetch_structural_nodes":
                replan_invocations.append((
                    request.tool,
                    FetchStructuralNodesParams(
                        source_types=request.source_types or None,
                    ),
                ))
            elif request.tool == "query_structured_facts":
                replan_invocations.append((
                    request.tool,
                    QueryStructuredFactsParams(
                        kinds=request.fact_kinds or None,  # type: ignore[arg-type]
                        keys=request.fact_keys or None,
                    ),
                ))
            elif request.tool == "inspect_document" and request.document_id is not None:
                replan_invocations.append((
                    request.tool,
                    InspectDocumentParams(document_id=request.document_id),
                ))
        return replan_invocations

    if state.plan is None:
        return []

    summary_path = _is_summary_path(state)
    preset: str = "broad" if summary_path else "default"

    # Work through gap_queries in order; we copy the list so state is not mutated.
    remaining_gaps = list(state.plan.gap_queries)
    search_hybrid_seen = False

    invocations: list[tuple[str, Any]] = []

    for tool_name in state.plan.chosen_tools:
        if tool_name == "inspect_chat":
            invocations.append((tool_name, InspectChatParams()))

        elif tool_name == "inspect_document":
            if summary_path and state.document_manifests:
                for dm in state.document_manifests:
                    invocations.append((tool_name, InspectDocumentParams(document_id=dm.document_id)))
            else:
                doc_id = (
                    state.document_manifests[0].document_id
                    if state.document_manifests
                    else uuid.UUID(int=0)
                )
                invocations.append((tool_name, InspectDocumentParams(document_id=doc_id)))

        elif tool_name == "fetch_structural_nodes":
            invocations.append((
                tool_name,
                FetchStructuralNodesParams(
                    source_types=["document_overview", "chapter_summary", "compact_chapter_summary"],
                ),
            ))

        elif tool_name == "search_hybrid":
            if not search_hybrid_seen:
                search_hybrid_seen = True
                # Emit one invocation per gap query first.
                for gap_q in remaining_gaps:
                    invocations.append((
                        tool_name,
                        _search_params(state, gap_q, preset=preset),
                    ))
                # Always append a default question-based search as well,
                # but only when there are no gap queries (first round) — if
                # gap_queries are present this is a gap-fill round and the
                # question-based search was already done in round 1 (will be
                # deduped by fingerprint anyway, but we skip it to keep the
                # invocation list clean).
                if not remaining_gaps:
                    invocations.append((
                        tool_name,
                        _search_params(state, state.question, preset=preset),
                    ))

        elif tool_name == "query_structured_facts":
            hints = state.plan.fact_filter_hints
            invocations.append(
                (
                    tool_name,
                    QueryStructuredFactsParams(
                        kinds=hints.kinds or None,  # type: ignore[arg-type]
                        keys=hints.keys or None,
                    ),
                )
            )

        elif tool_name == "aggregate_sources":
            invocations.append((tool_name, AggregateSourcesParams(strategy="per_section")))

        # expand_evidence has no sensible default without an evidence_id; skip.

    return invocations


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

    for tool_name, params in _plan_to_invocations(state):
        spec = TOOL_REGISTRY.get(tool_name)
        if spec is None:
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
        "replan_tool_calls": [],
    }


__all__ = ["execute_retrieval_tools"]
