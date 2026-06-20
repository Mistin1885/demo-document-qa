"""Tool: grep_document_chunks — deterministic lexical scan over stored document chunks.

This complements ``search_hybrid``.  It is intentionally not a top-k vector
search: it scans chat-scoped ``DocumentNode`` rows, scores lexical / exact-label
matches in Python, and returns the best raw chunks, HTML tables, figure captions,
and equation blocks.  It is useful when hybrid search returns a nearby but wrong
modality (for example Figure 1 evidence for a Figure 2 question) or when a table
is best answered from its literal HTML chunk.
"""

from __future__ import annotations

import re
import uuid

from sqlalchemy import select

from app.agent.budget import ContextBudgetManager
from app.agent.state import AgentError, AgentState, EvidenceItem, ToolCallRecord, make_evidence_id
from app.agent.tools._invocation import ToolDeps, ToolInvocation
from app.agent.tools._models import GrepDocumentChunksParams
from app.models.orm import DocumentNode

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{1,}|[0-9]+(?:[,][0-9]{3})*(?:\.[0-9]+)?%?")
_STOPWORDS = frozenset({
    "the", "and", "for", "that", "this", "with", "from", "what", "which", "does",
    "into", "about", "using", "used", "are", "was", "were", "has", "have", "had",
    "how", "why", "who", "when", "where", "please", "include", "create", "give",
    "show", "shown", "report", "explain", "describe", "based", "paper",
})
_LABEL_RE = re.compile(r"\b(?:Figure|Table)\s+\d+\b", re.IGNORECASE)


def _terms(text: str) -> set[str]:
    return {m.group(0).lower() for m in _WORD_RE.finditer(text) if m.group(0).lower() not in _STOPWORDS}


def _labels(text: str) -> set[str]:
    return {m.group(0).lower() for m in _LABEL_RE.finditer(text)}


def _wants_html(query: str, params: GrepDocumentChunksParams) -> bool:
    q = query.lower()
    return params.include_html or any(marker in q for marker in ("table", "<table", "html", "rows", "columns"))


def _score_node(node: DocumentNode, params: GrepDocumentChunksParams) -> int:
    metadata_html = ""
    if isinstance(node.metadata_, dict):
        raw_html = node.metadata_.get("html_body")
        metadata_html = raw_html if isinstance(raw_html, str) else ""
    haystack = "\n".join(
        part for part in (node.title or "", node.content or "", metadata_html) if part
    ).lower()
    query_terms = _terms(params.query)
    score = len(query_terms & _terms(haystack))

    # Exact figure/table labels must dominate generic modality words so Figure 2
    # questions do not get Figure 1 evidence just because both mention figures.
    query_labels = _labels(params.query)
    node_labels = _labels(f"{node.title or ''}\n{node.content or ''}\n{metadata_html}")
    if query_labels:
        if query_labels & node_labels:
            score += 25
        elif any(label.split()[0] in {nl.split()[0] for nl in node_labels} for label in query_labels):
            score -= 12

    lower = params.query.lower()
    node_type = node.node_type.lower()
    content = f"{node.content or ''}\n{metadata_html}".lower()
    if "figure" in lower and node_type == "figure":
        score += 8
    if "table" in lower and (node_type == "table" or "<table" in content):
        score += 10
    if _wants_html(params.query, params) and "<table" in content:
        score += 12
    if any(marker in lower for marker in ("formula", "expression", "equation", "×", "token")):
        if node_type == "equation" or re.search(r"\d+[\s,]*(?:×|x)\s*\d+", content):
            score += 8
    for term in params.required_terms:
        if term.lower() in haystack:
            score += 6
        else:
            score -= 4
    return score


async def grep_document_chunks(
    state: AgentState,
    params: GrepDocumentChunksParams,
    *,
    deps: ToolDeps,
) -> ToolInvocation:
    """Scan chat-scoped DocumentNode rows and return literal matching chunks."""
    call_id = str(uuid.uuid4())
    budget_mgr = ContextBudgetManager()
    errors: list[AgentError] = []
    evidence: list[EvidenceItem] = []

    try:
        async with deps.session_factory() as session:
            stmt = select(DocumentNode).where(DocumentNode.chat_id == state.chat_id)
            document_ids = params.document_ids or (list(state.scoped_document_ids) or None)
            if document_ids:
                stmt = stmt.where(DocumentNode.document_id.in_(document_ids))
            if params.source_types:
                # Node type naming is not identical to Vespa source_type naming;
                # accept both raw node types and common indexed source aliases.
                aliases = set(params.source_types)
                mapped: set[str] = set()
                for source_type in aliases:
                    if source_type in {"chunk", "raw_block"}:
                        mapped.update({"paragraph", "section", "subsection", "document"})
                    elif source_type == "table_record":
                        mapped.add("table")
                    elif source_type == "figure_caption":
                        mapped.add("figure")
                    elif source_type == "performance_fact":
                        mapped.update({"table", "paragraph"})
                    else:
                        mapped.add(source_type)
                stmt = stmt.where(DocumentNode.node_type.in_(sorted(mapped)))
            stmt = stmt.order_by(DocumentNode.document_id, DocumentNode.order_index).limit(params.scan_limit)
            result = await session.execute(stmt)
            nodes = list(result.scalars().all())

        scored = [(_score_node(node, params), node.order_index, node) for node in nodes]
        # Keep positive matches; if all are weak, still return the best few literal
        # chunks so coverage can decide whether the alternate path helped.
        positives = [item for item in scored if item[0] > 0]
        ranked = sorted(positives or scored, key=lambda item: (item[0], -item[1]), reverse=True)

        token_est = 0
        for score, _order, node in ranked[: params.limit]:
            metadata_html = ""
            if isinstance(node.metadata_, dict):
                raw_html = node.metadata_.get("html_body")
                metadata_html = raw_html if isinstance(raw_html, str) else ""
            content = node.content or ""
            if metadata_html and (params.include_html or "table" in params.query.lower()):
                content = "\n".join(part for part in (content, metadata_html) if part)
            if not params.include_html and "<table" in content.lower() and "table" not in params.query.lower():
                # Avoid accidental huge tables unless explicitly useful.
                content = re.sub(r"<[^>]+>", " ", content)
                content = " ".join(content.split())
            ev_id = make_evidence_id("grep_document_chunks", str(node.id), str(node.document_id))
            evidence.append(EvidenceItem(
                evidence_id=ev_id,
                source_type=node.node_type,
                document_id=node.document_id,
                source_node_id=str(node.id),
                page_start=node.page_start,
                page_end=node.page_end,
                content=content,
                score=float(score),
                vector_score=None,
                section_title=node.title,
                heading_path=None,
                origin_tool="grep_document_chunks",
            ))
            token_est += budget_mgr.count_tokens(content)
            if token_est >= params.max_tokens:
                break

        status = "empty" if not evidence else ("overflow" if token_est > params.max_tokens else "ok")
        record = ToolCallRecord(
            call_id=call_id,
            tool_name="grep_document_chunks",
            params=params.model_dump(),
            status=status,  # type: ignore[arg-type]
            token_estimate=token_est,
            source_count=len(evidence),
        )
    except Exception as exc:
        errors.append(AgentError(code="grep_document_chunks_error", detail=str(exc), tool_name="grep_document_chunks"))
        record = ToolCallRecord(
            call_id=call_id,
            tool_name="grep_document_chunks",
            params=params.model_dump(),
            status="error",
            token_estimate=0,
            source_count=0,
            error=str(exc),
        )

    return ToolInvocation(record=record, evidence=evidence, facts=[], errors=errors)


__all__ = ["grep_document_chunks"]
