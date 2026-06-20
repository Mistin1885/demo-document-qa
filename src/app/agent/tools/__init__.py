"""Agent tools package — Phase 7.2.

Exports:
- All 7 tool async functions.
- TOOL_REGISTRY: dict[str, ToolSpec] for Phase 7.3 graph and 7.4 policy.

TOOL_REGISTRY maps tool name → ToolSpec(name, params_model, description, callable).
Phase 7.3 nodes and 7.4 policy use the registry rather than scattered imports.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from app.agent.tools._invocation import ToolDeps, ToolInvocation
from app.agent.tools._models import (
    AggregateSourcesParams,
    ExpandEvidenceParams,
    FetchStructuralNodesParams,
    GrepDocumentChunksParams,
    InspectChatParams,
    InspectDocumentParams,
    QueryStructuredFactsParams,
    SearchHybridParams,
)
from app.agent.tools.aggregate_sources import aggregate_sources
from app.agent.tools.expand_evidence import expand_evidence
from app.agent.tools.fetch_structural_nodes import fetch_structural_nodes
from app.agent.tools.grep_document_chunks import grep_document_chunks
from app.agent.tools.inspect_chat import inspect_chat
from app.agent.tools.inspect_document import inspect_document
from app.agent.tools.query_structured_facts import query_structured_facts
from app.agent.tools.search_hybrid import search_hybrid


@dataclass
class ToolSpec:
    """Registry entry for one agent tool.

    Attributes
    ----------
    name:
        Canonical tool name (matches the function name).
    params_model:
        Pydantic v2 ``*Params`` class for this tool.
    description:
        One-sentence description used by the Phase 7.3/7.4 planner.
    callable:
        The async tool function:
        ``async def <name>(state, params, *, deps) -> ToolInvocation``.
    """

    name: str
    params_model: type[BaseModel]
    description: str
    callable: Callable[..., Any]


TOOL_REGISTRY: dict[str, ToolSpec] = {
    "inspect_chat": ToolSpec(
        name="inspect_chat",
        params_model=InspectChatParams,
        description=(
            "Read the chat-level manifest: document count, titles, main topics, "
            "source types, and token estimates. Use first to understand chat scope."
        ),
        callable=inspect_chat,
    ),
    "inspect_document": ToolSpec(
        name="inspect_document",
        params_model=InspectDocumentParams,
        description=(
            "Read metadata, abstract, and section tree for one document. "
            "Use to understand a single document's structure before retrieval."
        ),
        callable=inspect_document,
    ),
    "fetch_structural_nodes": ToolSpec(
        name="fetch_structural_nodes",
        params_model=FetchStructuralNodesParams,
        description=(
            "Deterministically fetch DocumentNode and Summary rows from PostgreSQL "
            "(whole-doc / whole-chat / specific section). Never replaced by top-k."
        ),
        callable=fetch_structural_nodes,
    ),
    "grep_document_chunks": ToolSpec(
        name="grep_document_chunks",
        params_model=GrepDocumentChunksParams,
        description=(
            "Deterministically scan chat-scoped stored document chunks, HTML tables, "
            "figure captions, and equations with lexical grep. Use alongside "
            "search_hybrid when exact Figure/Table labels or literal formulas matter."
        ),
        callable=grep_document_chunks,
    ),
    "search_hybrid": ToolSpec(
        name="search_hybrid",
        params_model=SearchHybridParams,
        description=(
            "Vespa hybrid BM25 + ANN retrieval with RRF fusion and optional rerank. "
            "Use for technical details, concepts, definitions, claims, and methods."
        ),
        callable=search_hybrid,
    ),
    "query_structured_facts": ToolSpec(
        name="query_structured_facts",
        params_model=QueryStructuredFactsParams,
        description=(
            "Query PostgreSQL structured_facts for numbers, benchmarks, datasets, "
            "and metrics. Prefer this over search_hybrid for quantitative queries."
        ),
        callable=query_structured_facts,
    ),
    "aggregate_sources": ToolSpec(
        name="aggregate_sources",
        params_model=AggregateSourcesParams,
        description=(
            "Compact state.evidence_items into per-document or per-section summaries "
            "when context budget overflows. Deterministic placeholder (Phase 7.2); "
            "Phase 9 replaces with LLM-based aggregation."
        ),
        callable=aggregate_sources,
    ),
    "expand_evidence": ToolSpec(
        name="expand_evidence",
        params_model=ExpandEvidenceParams,
        description=(
            "Expand a specific EvidenceItem to its neighboring sections, page nodes, "
            "or adjacent paragraphs from the database."
        ),
        callable=expand_evidence,
    ),
}

__all__ = [
    "TOOL_REGISTRY",
    "ToolDeps",
    "ToolInvocation",
    "ToolSpec",
    # Tool functions
    "aggregate_sources",
    "expand_evidence",
    "fetch_structural_nodes",
    "grep_document_chunks",
    "inspect_chat",
    "inspect_document",
    "query_structured_facts",
    "search_hybrid",
    # Params models
    "AggregateSourcesParams",
    "ExpandEvidenceParams",
    "FetchStructuralNodesParams",
    "GrepDocumentChunksParams",
    "InspectChatParams",
    "InspectDocumentParams",
    "QueryStructuredFactsParams",
    "SearchHybridParams",
]
