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


def _is_formula_query(query: str) -> bool:
    q = query.lower()
    return any(
        marker in q
        for marker in (
            "formula",
            "equation",
            "expression",
            "formally represent",
            "formal representation",
            "公式",
            "方程",
            "數學式",
            "数学式",
            "算式",
        )
    )


def _node_text(node: DocumentNode, *, include_html: bool, query: str) -> str:
    metadata_html = ""
    if isinstance(node.metadata_, dict):
        raw_html = node.metadata_.get("html_body")
        metadata_html = raw_html if isinstance(raw_html, str) else ""
    content = node.content or ""
    if metadata_html and (include_html or "table" in query.lower()):
        content = "\n".join(part for part in (content, metadata_html) if part)
    return content


def _searchable_text(node: DocumentNode) -> str:
    metadata_html = ""
    if isinstance(node.metadata_, dict):
        raw_html = node.metadata_.get("html_body")
        metadata_html = raw_html if isinstance(raw_html, str) else ""
    return "\n".join(
        part for part in (node.title or "", node.content or "", metadata_html) if part
    )


def _score_node(node: DocumentNode, params: GrepDocumentChunksParams) -> int:
    haystack = _searchable_text(node).lower()
    query_terms = _terms(params.query)
    score = len(query_terms & _terms(haystack))

    # Exact figure/table labels must dominate generic modality words so Figure 2
    # questions do not get Figure 1 evidence just because both mention figures.
    query_labels = _labels(params.query)
    node_labels = _labels(_searchable_text(node))
    if query_labels:
        if query_labels & node_labels:
            score += 25
        elif any(label.split()[0] in {nl.split()[0] for nl in node_labels} for label in query_labels):
            score -= 12

    lower = params.query.lower()
    node_type = node.node_type.lower()
    content = _searchable_text(node).lower()
    if "figure" in lower and node_type == "figure":
        score += 8
    if "table" in lower and (node_type == "table" or "<table" in content):
        score += 10
    if _wants_html(params.query, params) and "<table" in content:
        score += 12
    if _is_formula_query(params.query) or any(marker in lower for marker in ("×", "token")):
        if node_type == "equation" or re.search(r"\d+[\s,]*(?:×|x)\s*\d+", content):
            score += 8
        if re.search(r"\\hat|operatorname|\\mathcal|=", content):
            score += 8
    for term in params.required_terms:
        if term.lower() in haystack:
            score += 6
        else:
            score -= 4
    return score


def _expanded_context(
    center: DocumentNode,
    nodes: list[DocumentNode],
    params: GrepDocumentChunksParams,
) -> tuple[str, int, int]:
    """Return center text plus ordered nearby nodes within context_chars.

    This is the important grep behaviour for formulas: if a query matches the
    lead-in sentence ("graph generation module as follows"), the actual formula
    is often the next equation node.  Returning a small ordered neighborhood is
    much more useful than returning the single matching sentence.
    """
    center_text = _node_text(center, include_html=params.include_html, query=params.query)
    if not params.include_context or params.context_chars <= 0:
        return center_text, center.page_start, center.page_end

    ordered = sorted(nodes, key=lambda n: (str(n.document_id), n.order_index))
    try:
        center_idx = next(i for i, node in enumerate(ordered) if node.id == center.id)
    except StopIteration:
        return center_text, center.page_start, center.page_end

    budget = max(params.context_chars, len(center_text))
    selected: dict[int, DocumentNode] = {center_idx: center}
    total = len(center_text)
    query_label_match = re.search(r"\b(figure|table)\s+(\d+)\b", params.query, flags=re.IGNORECASE)
    wanted_label = (
        f"{query_label_match.group(1)} {query_label_match.group(2)}".lower()
        if query_label_match
        else None
    )
    same_modality_label_re = (
        re.compile(rf"\b{query_label_match.group(1)}\s+\d+\b", flags=re.IGNORECASE)
        if query_label_match
        else None
    )

    # Formula / "as follows" hits usually need the following equation; include
    # next nodes first, then previous context.
    prefer_next_first = _is_formula_query(params.query) or "as follows" in center_text.lower()
    offsets = []
    for distance in range(1, 6):
        if prefer_next_first:
            offsets.extend([distance, -distance])
        else:
            offsets.extend([-distance, distance])

    for offset in offsets:
        idx = center_idx + offset
        if idx < 0 or idx >= len(ordered):
            continue
        candidate = ordered[idx]
        if candidate.document_id != center.document_id:
            continue
        # Stay local to the same parent section when possible, but allow one
        # adjacent equation/table/figure even when parent_id is absent or differs.
        if (
            center.parent_id is not None
            and candidate.parent_id is not None
            and candidate.parent_id != center.parent_id
            and candidate.node_type not in {"equation", "table", "figure"}
        ):
            continue
        candidate_text = _node_text(candidate, include_html=params.include_html, query=params.query)
        if not candidate_text:
            continue
        if (
            wanted_label is not None
            and same_modality_label_re is not None
            and same_modality_label_re.search(candidate_text)
            and wanted_label not in candidate_text.lower()
        ):
            continue
        projected = total + len(candidate_text) + 32
        if projected > budget and selected:
            # Always allow a formula/equation neighbor for formula questions.
            if not (_is_formula_query(params.query) and candidate.node_type == "equation"):
                continue
        selected[idx] = candidate
        total = projected
        if total >= budget:
            break

    parts: list[str] = []
    page_start = center.page_start
    page_end = center.page_end
    for idx in sorted(selected):
        node = ordered[idx]
        text = _node_text(node, include_html=params.include_html, query=params.query)
        if not text:
            continue
        label = node.title or node.node_type
        parts.append(f"[{label}]\n{text}" if label else text)
        page_start = min(page_start, node.page_start)
        page_end = max(page_end, node.page_end)

    return "\n\n".join(parts), page_start, page_end


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
            content, page_start, page_end = _expanded_context(node, nodes, params)
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
                page_start=page_start,
                page_end=page_end,
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
