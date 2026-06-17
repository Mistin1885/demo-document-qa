"""Document-level enrichment — LLM JSON-parsing path (Phase 5.2).

Public API
----------
- :func:`enrich_document_overview` — async: ``HierarchyResult +
  list[SectionEnrichment] → DocumentOverview``.

Algorithm
---------
All section compact summaries + aggregated entities / methods / limitations /
performance_facts are assembled into a single prompt.  The LLM is asked to
return **one JSON object** conforming to ``DocumentOverview`` schema.  On
parse failure the call is retried up to 2 times; if all retries fail,
``EnrichmentParseError`` is raised.

When the LLM omits ``source_section_ids``, the fallback is the set of all
abstract + section + subsection section enrichment node IDs, guaranteeing
provenance traceability.

Design constraints (CLAUDE.md §3 / §12)
-----------------------------------------
- No FastAPI / SQLAlchemy imports.
- ``chat_id`` is always taken from ``hierarchy.chat_id`` — never from caller.
- No ``isinstance(provider, MockChatProvider)`` branches.
- No ``dict[str, Any]`` in domain fields.
- All public functions have full type hints.
"""

from __future__ import annotations

import json
import uuid
from typing import Any
from uuid import UUID

from app.enrichment.models import (
    ConclusionItem,
    Contribution,
    DatasetMention,
    DocumentOverview,
    FindingItem,
    LimitationMention,
    MethodMention,
    MetricMention,
    SectionEnrichment,
)
from app.enrichment.prompts import build_document_system_prompt, build_document_user_message
from app.enrichment.section import EnrichmentParseError
from app.parsing.models import BlockType, HierarchyResult, NodeType, ParsedBlock
from app.providers.base import ChatMessage, ChatProvider

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_RETRIES = 2

_ENRICHABLE_FOR_FALLBACK: frozenset[NodeType] = frozenset(
    [NodeType.section, NodeType.subsection, NodeType.abstract]
)


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------


def _estimate_tokens(text: str) -> int:
    """Approximate token count: tiktoken cl100k_base if available, else len/4."""
    try:
        import tiktoken  # type: ignore[import-untyped]

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Aggregation helpers (pure functions)
# ---------------------------------------------------------------------------


def _top_entities(section_enrichments: list[SectionEnrichment], top_n: int = 20) -> list[str]:
    """Deduplicated top-N entities across all section enrichments."""
    seen: dict[str, None] = {}
    for se in section_enrichments:
        for ent in se.entities:
            stripped = ent.strip()
            if stripped:
                seen[stripped] = None
    return sorted(seen.keys())[:top_n]


def _top_methods(section_enrichments: list[SectionEnrichment], top_n: int = 10) -> list[str]:
    """Deduplicated top-N method names across all section enrichments."""
    seen: dict[str, None] = {}
    for se in section_enrichments:
        for m in se.methods:
            stripped = m.name.strip()
            if stripped:
                seen[stripped] = None
    return sorted(seen.keys())[:top_n]


def _all_limitations(section_enrichments: list[SectionEnrichment]) -> list[str]:
    """All limitation texts across all section enrichments (deduped)."""
    seen: dict[str, None] = {}
    for se in section_enrichments:
        for lim in se.limitations:
            stripped = lim.text.strip()
            if stripped:
                seen[stripped] = None
    return list(seen.keys())


def _all_performance_facts(section_enrichments: list[SectionEnrichment]) -> list[str]:
    """All performance facts formatted as '<metric>=<value> (<context>)'."""
    items: list[str] = []
    for se in section_enrichments:
        for pf in se.performance_facts:
            entry = f"{pf.metric}={pf.value}"
            if pf.context:
                entry += f" ({pf.context[:60]})"
            items.append(entry)
    return items


def _gather_abstract_text(
    hierarchy: HierarchyResult,
    block_map: dict[UUID, ParsedBlock],
) -> str:
    """Return joined abstract text or empty string if no abstract node."""
    for node in hierarchy.nodes:
        if node.node_type == NodeType.abstract:
            parts: list[str] = []
            for bid in node.source_block_ids:
                blk = block_map.get(bid)
                if blk is not None and blk.block_type in (
                    BlockType.paragraph,
                    BlockType.text,
                ):
                    parts.append(blk.text.strip())
            return " ".join(p for p in parts if p)
    return ""


def _get_doc_title(
    hierarchy: HierarchyResult,
    doc_title_override: str | None,
) -> str | None:
    """Return doc title from override or document root node."""
    if doc_title_override:
        return doc_title_override
    for node in hierarchy.nodes:
        if node.node_type == NodeType.document and node.title:
            return node.title
    return None


def _get_page_count(
    hierarchy: HierarchyResult,
    page_count_override: int | None,
) -> int:
    """Return page count from override or max(page_end) across nodes."""
    if page_count_override is not None:
        return page_count_override
    max_page = 1
    for node in hierarchy.nodes:
        if node.page_end > max_page:
            max_page = node.page_end
    return max_page


def _fallback_source_section_ids(
    section_enrichments: list[SectionEnrichment],
) -> list[UUID]:
    """Fallback when LLM omits source_section_ids.

    Returns all abstract + section + subsection enrichment node IDs.
    """
    return [
        se.source_node_id
        for se in section_enrichments
        if se.node_type in ("abstract", "section", "subsection")
    ]


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------


def _coerce_uuid_list(raw: list[object]) -> list[UUID]:
    """Coerce a list of strings / UUIDs into UUID objects, silently ignoring failures."""
    result: list[UUID] = []
    for item in raw:
        try:
            result.append(uuid.UUID(str(item)))
        except (ValueError, AttributeError):
            pass
    return result


def _try_parse_document_overview(  # noqa: PLR0912
    raw: str,
    hierarchy: HierarchyResult,
    section_enrichments: list[SectionEnrichment],
    doc_title: str | None,
    authors_excerpt: str | None,
    page_count: int,
    model_name: str,
) -> DocumentOverview:
    """Parse *raw* LLM text into a ``DocumentOverview``.

    Raises
    ------
    ValueError
        When the JSON cannot be decoded or is structurally invalid.
    """
    stripped = raw.strip()
    # Strip optional markdown fences
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        stripped = "\n".join(lines[1:] if len(lines) > 2 else lines)
        stripped = stripped.rstrip("`").strip()

    data: Any = json.loads(stripped)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object, got {type(data).__name__}")

    # ---- overview ----
    overview = str(data.get("overview", "")).strip()
    if not overview:
        raise ValueError("Missing 'overview' field in LLM response")

    # ---- source_section_ids: global fallback when LLM omits entirely ----
    top_level_ids_raw = data.get("source_section_ids")
    fallback_ids = _fallback_source_section_ids(section_enrichments)

    def _section_ids_from_item(item: dict[str, Any]) -> list[UUID]:
        raw_ids = item.get("source_section_ids")
        if not raw_ids:
            return []
        return _coerce_uuid_list(raw_ids)

    # ---- contributions ----
    contributions: list[Contribution] = []
    for c in data.get("contributions", []):
        if not isinstance(c, dict):
            continue
        title = str(c.get("title", "")).strip()
        summary = str(c.get("summary", "")).strip()
        if title:
            sec_ids = _section_ids_from_item(c) or fallback_ids
            contributions.append(
                Contribution(title=title, summary=summary, source_section_ids=sec_ids)
            )

    # ---- methods ----
    methods: list[MethodMention] = []
    for m in data.get("methods", []):
        if not isinstance(m, dict):
            continue
        name = str(m.get("name", "")).strip()
        desc = str(m.get("description", "")).strip()
        if name:
            sec_ids = _section_ids_from_item(m)
            methods.append(MethodMention(name=name, description=desc, source_section_ids=sec_ids))

    # ---- findings ----
    findings: list[FindingItem] = []
    for f in data.get("findings", []):
        if not isinstance(f, dict):
            continue
        stmt = str(f.get("statement", "")).strip()
        if stmt:
            sec_ids = _section_ids_from_item(f) or fallback_ids
            findings.append(
                FindingItem(
                    statement=stmt,
                    evidence=f.get("evidence") or None,
                    source_section_ids=sec_ids,
                )
            )

    # ---- limitations ----
    limitations: list[LimitationMention] = []
    for lim in data.get("limitations", []):
        if not isinstance(lim, dict):
            continue
        text = str(lim.get("text", "")).strip()
        if text:
            sec_ids = _section_ids_from_item(lim)
            limitations.append(LimitationMention(text=text, source_section_ids=sec_ids))

    # ---- datasets ----
    datasets: list[DatasetMention] = []
    _valid_roles = {"training", "evaluation", "benchmark", "ablation", "other"}
    for ds in data.get("datasets", []):
        if not isinstance(ds, dict):
            continue
        name = str(ds.get("name", "")).strip()
        if name:
            role_raw = str(ds.get("role", "other")).lower()
            role = role_raw if role_raw in _valid_roles else "other"
            sec_ids = _section_ids_from_item(ds) or fallback_ids
            datasets.append(
                DatasetMention(
                    name=name,
                    role=role,  # type: ignore[arg-type]
                    size_hint=ds.get("size_hint") or None,
                    source_section_ids=sec_ids,
                )
            )

    # ---- metrics ----
    metrics: list[MetricMention] = []
    for met in data.get("metrics", []):
        if not isinstance(met, dict):
            continue
        name = str(met.get("name", "")).strip()
        if name:
            best_val = met.get("best_value")
            base_val = met.get("baseline_value")
            sec_ids = _section_ids_from_item(met) or fallback_ids
            metrics.append(
                MetricMention(
                    name=name,
                    best_value=float(best_val) if best_val is not None else None,
                    baseline_value=float(base_val) if base_val is not None else None,
                    improvement=met.get("improvement") or None,
                    source_section_ids=sec_ids,
                )
            )

    # ---- conclusions ----
    conclusions: list[ConclusionItem] = []
    _valid_cats = {"finding", "future_work", "caveat", "claim"}
    for con in data.get("conclusions", []):
        if not isinstance(con, dict):
            continue
        stmt = str(con.get("statement", "")).strip()
        if stmt:
            cat_raw = str(con.get("category", "finding")).lower()
            cat = cat_raw if cat_raw in _valid_cats else "finding"
            sec_ids = _section_ids_from_item(con) or fallback_ids
            conclusions.append(
                ConclusionItem(
                    statement=stmt,
                    category=cat,  # type: ignore[arg-type]
                    source_section_ids=sec_ids,
                )
            )

    # ---- source_section_ids (document-level) ----
    if top_level_ids_raw and isinstance(top_level_ids_raw, list):
        source_section_ids = _coerce_uuid_list(top_level_ids_raw)
    else:
        source_section_ids = fallback_ids

    if not source_section_ids:
        source_section_ids = fallback_ids

    token_count_estimate = _estimate_tokens(overview)

    return DocumentOverview(
        chat_id=hierarchy.chat_id,
        document_id=hierarchy.document_id,
        doc_title=doc_title,
        authors_excerpt=authors_excerpt,
        page_count=page_count,
        overview=overview,
        contributions=contributions,
        methods=methods,
        findings=findings,
        limitations=limitations,
        datasets=datasets,
        metrics=metrics,
        conclusions=conclusions,
        source_section_ids=source_section_ids,
        token_count_estimate=token_count_estimate,
        model_used=model_name,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def enrich_document_overview(
    hierarchy: HierarchyResult,
    section_enrichments: list[SectionEnrichment],
    *,
    chat_provider: ChatProvider,
    doc_title: str | None = None,
    authors_excerpt: str | None = None,
    page_count: int | None = None,
    blocks: list[ParsedBlock] | None = None,
) -> DocumentOverview:
    """Produce a paper-wide ``DocumentOverview`` from prior section enrichments.

    Parameters
    ----------
    hierarchy:
        Full hierarchy produced by ``derive_hierarchy`` (Phase 4.3).
    section_enrichments:
        Output of ``enrich_document_sections()`` (Phase 5.1).
    chat_provider:
        Any ``ChatProvider`` implementation.  Called at most 2 times per
        invocation (initial attempt + 1 retry on parse failure).
    doc_title:
        Paper title override.  When ``None``, derived from the document-root
        node in *hierarchy*.
    authors_excerpt:
        Authors string override (first 200 chars).  When ``None``, the
        hierarchy's abstract/authors text is used if available.
    page_count:
        Total page count override.  When ``None``, ``max(node.page_end)``
        across all hierarchy nodes is used.
    blocks:
        Optional flat list of ``ParsedBlock`` instances (used to extract
        abstract text for the prompt).  When ``None``, abstract text is
        omitted from the prompt.

    Returns
    -------
    DocumentOverview

    Raises
    ------
    EnrichmentParseError
        When all retries are exhausted and the LLM response cannot be parsed.

    Notes
    -----
    - ``chat_id`` is always taken from ``hierarchy.chat_id``.
    - ``document_id`` is taken from ``hierarchy.document_id``.
    - This function never imports FastAPI, ORM, or database types.
    """
    resolved_doc_title = _get_doc_title(hierarchy, doc_title)
    resolved_page_count = _get_page_count(hierarchy, page_count)

    # Build block lookup if blocks were provided
    block_map: dict[UUID, ParsedBlock] = {}
    if blocks:
        block_map = {b.block_id: b for b in blocks}

    abstract_text = _gather_abstract_text(hierarchy, block_map) if block_map else ""

    # Build prompt inputs
    sections: list[tuple[str, str, int, int, str]] = [
        (
            str(se.source_node_id),
            se.title or "(untitled)",
            se.page_start,
            se.page_end,
            se.compact_summary,
        )
        for se in section_enrichments
    ]

    entities_top = _top_entities(section_enrichments)
    methods_top = _top_methods(section_enrichments)
    limitations_all = _all_limitations(section_enrichments)
    performance_facts_all = _all_performance_facts(section_enrichments)

    messages: list[ChatMessage] = [
        ChatMessage(role="system", content=build_document_system_prompt()),
        ChatMessage(
            role="user",
            content=build_document_user_message(
                sections=sections,
                abstract_text=abstract_text or None,
                entities_top=entities_top,
                methods_top=methods_top,
                limitations_all=limitations_all,
                performance_facts_all=performance_facts_all,
                doc_title=resolved_doc_title,
                authors_excerpt=authors_excerpt,
            ),
        ),
    ]

    last_exc: Exception = EnrichmentParseError("No attempts made")
    for _attempt in range(_MAX_RETRIES):
        completion = await chat_provider.complete(
            messages,
            temperature=0.0,
            max_tokens=2048,
        )
        try:
            return _try_parse_document_overview(
                completion.content,
                hierarchy,
                section_enrichments,
                resolved_doc_title,
                authors_excerpt,
                resolved_page_count,
                chat_provider.model,
            )
        except (json.JSONDecodeError, ValueError, Exception) as exc:
            last_exc = exc

    raise EnrichmentParseError(
        f"Failed to parse DocumentOverview for document {hierarchy.document_id} "
        f"after {_MAX_RETRIES} retries: {last_exc}"
    )


__all__ = ["enrich_document_overview"]
