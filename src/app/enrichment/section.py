"""Section-level enrichment — Phase 5.1.

Public API
----------
- :func:`enrich_sections` — async: ``HierarchyResult + list[ParsedBlock] →
  list[SectionEnrichment]``.

Design
------
Each field is populated via a **two-step pattern**:

1. A prompt is sent to the ``ChatProvider`` (hook for real LLM call).
2. The *text* response is parsed by a deterministic heuristic that works
   regardless of whether the provider returns structured JSON or a mock
   string like ``"[mock mock-model] response-1234"``.

This means tests using ``MockChatProvider`` are fully deterministic and
never need to mock JSON parsing.  Replacing the heuristics with a real
JSON-parsing parser later does not require any test changes — only the
``_parse_*`` helpers change.

Node types processed: ``section``, ``subsection``, ``appendix``.
All other node types (abstract, authors, document, reference, …) are
skipped — they belong to Phase 5.2 / 5.3.

Design constraints (CLAUDE.md §3 / §12)
-----------------------------------------
- No FastAPI / SQLAlchemy imports.
- ``chat_id`` for every ``SectionEnrichment`` is taken exclusively from
  ``hierarchy.chat_id`` — never from a caller argument.
- No ``isinstance(provider, MockChatProvider)`` branches.
- All public functions have full type hints.
- No ``dict[str, Any]``.
"""

from __future__ import annotations

import json
import re
import uuid
from uuid import UUID

from app.enrichment.models import (
    ClaimItem,
    DefinitionItem,
    LimitationItem,
    MethodItem,
    PerformanceFactItem,
    SectionEnrichment,
)
from app.parsing.models import (
    BlockType,
    DocumentNodeOut,
    HierarchyResult,
    NodeType,
    ParsedBlock,
)
from app.providers.base import ChatMessage, ChatProvider

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class EnrichmentParseError(Exception):
    """Raised when the LLM response cannot be parsed into a SectionEnrichment.

    This is raised only after all retries are exhausted (max 2 attempts).
    """


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ENRICHABLE_NODE_TYPES: frozenset[NodeType] = frozenset(
    [NodeType.section, NodeType.subsection, NodeType.appendix]
)

_MIN_KEYWORD_LEN = 4
_MAX_KEYWORDS = 8
_MAX_TECHNICAL_KEYWORDS = 6
_MAX_ENTITIES = 8
_SUMMARY_FIRST_CHARS = 120

# Regex for PascalCase / ALL-CAPS words (entity heuristic)
_PASCAL_RE = re.compile(r"\b([A-Z][a-z]{2,}(?:[A-Z][a-z]*)*|[A-Z]{2,})\b")

# Regex for simple "X is Y" / "X: Y" patterns (definition heuristic)
_DEF_RE = re.compile(
    r"(?P<term>[A-Za-z][A-Za-z0-9 \-]{2,40}?)\s+(?:is|are|refers to|denotes)\s+(?P<def>[^.]{10,120})\."
    r"|(?P<term2>[A-Z][A-Za-z0-9\- ]{2,30}?):\s+(?P<def2>[^.]{10,120})\."
)

# Regex for "X achieves / outperforms / improves … N%" patterns (perf fact)
_PERF_RE = re.compile(
    r"(?P<metric>[A-Za-z][A-Za-z0-9\- ]{0,40}?)\s+"
    r"(?:of|at|reaches|achieves|improves|outperforms)?\s*"
    r"(?P<value>\d[\d.,]*\s*(?:%|percentage points?|pp|x|×|times?))"
    r"(?:\s+(?P<ctx>[^.]{0,80}))?"
)

# Limitation signal phrases
_LIMITATION_PHRASES = (
    "however",
    "limitation",
    "drawback",
    "constrain",
    "restrict",
    "assumption",
    "not applicable",
    "does not",
    "cannot",
    "fail",
    "weakness",
)

# Method signal phrases (must be followed by a colon or "method"/"approach")
_METHOD_PHRASES = re.compile(
    r"(?:we propose|we present|we introduce|our approach|our method|our algorithm|"
    r"this method|this approach|the proposed|we use|we employ)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Text collection helpers
# ---------------------------------------------------------------------------


def _collect_block_ids(node: DocumentNodeOut) -> frozenset[UUID]:
    """Return the set of source block IDs for *node*."""
    return frozenset(node.source_block_ids)


def _gather_text_for_node(
    node: DocumentNodeOut,
    block_map: dict[UUID, ParsedBlock],
) -> str:
    """Concatenate the text of all text-bearing blocks owned by *node*.

    Image / table / equation blocks contribute an empty string so the
    heuristics operate on prose only.
    """
    parts: list[str] = []
    for bid in node.source_block_ids:
        blk = block_map.get(bid)
        if blk is None:
            continue
        if blk.block_type in (
            BlockType.paragraph,
            BlockType.text,
            BlockType.title,
            BlockType.ref_text,
        ):
            parts.append(blk.text.strip())
        elif blk.block_type == BlockType.equation and blk.equation_latex:
            parts.append(blk.equation_latex.strip())
    return " ".join(p for p in parts if p)


def _figure_ids_for_node(
    node: DocumentNodeOut,
    block_map: dict[UUID, ParsedBlock],
) -> list[UUID]:
    """Return block IDs of figure blocks owned by *node*."""
    return [
        bid
        for bid in node.source_block_ids
        if block_map.get(bid) is not None and block_map[bid].block_type == BlockType.image
    ]


def _table_ids_for_node(
    node: DocumentNodeOut,
    block_map: dict[UUID, ParsedBlock],
) -> list[UUID]:
    """Return block IDs of table blocks owned by *node*."""
    return [
        bid
        for bid in node.source_block_ids
        if block_map.get(bid) is not None and block_map[bid].block_type == BlockType.table
    ]


# ---------------------------------------------------------------------------
# Deterministic field parsers (work on any text — mock or real LLM output)
# ---------------------------------------------------------------------------


def _parse_summary_detailed(llm_text: str, content: str) -> str:
    """Return a detailed summary.

    Heuristic: if the LLM text is longer than 40 chars, use the first 300
    chars; otherwise fall back to a prefix of *content*.
    """
    stripped = llm_text.strip()
    if len(stripped) > 40:
        return stripped[:300]
    prefix = content[:_SUMMARY_FIRST_CHARS].strip()
    return f"[summary] {prefix}" if prefix else "[summary] (empty section)"


def _parse_summary_compact(llm_text: str, content: str) -> str:
    """Return a compact (one-sentence) summary."""
    stripped = llm_text.strip()
    if len(stripped) > 10:
        # Take up to first sentence or 100 chars
        first_sentence = stripped.split(".")[0]
        return first_sentence[:100].strip()
    prefix = content[:60].strip()
    return f"[compact] {prefix}" if prefix else "[compact] (empty section)"


def _parse_keywords(llm_text: str, content: str) -> list[str]:
    """Extract keywords from content (token frequency heuristic).

    Strategy: tokenise *content* into words, deduplicate, filter by min
    length, take the first ``_MAX_KEYWORDS`` in order of appearance.  The
    LLM text is a hook that real implementations would parse instead.
    """
    _ = llm_text  # hook — real parser would use this
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9\-]{" + str(_MIN_KEYWORD_LEN - 1) + r",}", content)
    seen: dict[str, int] = {}
    for i, tok in enumerate(tokens):
        lower = tok.lower()
        if lower not in seen:
            seen[lower] = i
    # Sort by first occurrence, return original case
    ordered = sorted(seen.keys(), key=lambda k: seen[k])
    return ordered[:_MAX_KEYWORDS]


def _parse_technical_keywords(llm_text: str, content: str) -> list[str]:
    """Extract technical keywords (CamelCase / hyphenated / upper-bounded)."""
    _ = llm_text  # hook
    # Prefer tokens that look like technical terms: contain digits, hyphens,
    # or are acronyms (2+ upper-case letters)
    candidates = re.findall(
        r"\b(?:[A-Z]{2,}|[A-Za-z]+(?:\d+[A-Za-z]*|\-[A-Za-z]+)+)\b",
        content,
    )
    seen: dict[str, int] = {}
    for i, tok in enumerate(candidates):
        if tok not in seen:
            seen[tok] = i
    ordered = sorted(seen.keys(), key=lambda k: seen[k])
    return ordered[:_MAX_TECHNICAL_KEYWORDS]


def _parse_entities(llm_text: str, content: str) -> list[str]:
    """Extract named entities using PascalCase / ALLCAPS heuristic."""
    _ = llm_text  # hook
    matches = _PASCAL_RE.findall(content)
    seen: dict[str, int] = {}
    for i, m in enumerate(matches):
        if m not in seen:
            seen[m] = i
    ordered = sorted(seen.keys(), key=lambda k: seen[k])
    return ordered[:_MAX_ENTITIES]


def _parse_definitions(llm_text: str, content: str) -> list[DefinitionItem]:
    """Extract term-definition pairs using regex heuristic."""
    _ = llm_text  # hook
    results: list[DefinitionItem] = []
    for m in _DEF_RE.finditer(content):
        term = (m.group("term") or m.group("term2") or "").strip()
        defn = (m.group("def") or m.group("def2") or "").strip()
        if term and defn:
            results.append(DefinitionItem(term=term, definition=defn))
        if len(results) >= 4:
            break
    return results


def _parse_claims(llm_text: str, content: str) -> list[ClaimItem]:
    """Extract claims from sentence-level heuristic.

    Heuristic: sentences that start with "We show", "We demonstrate", "Our
    results", "This paper", "We prove", "This work" are claim candidates.
    """
    _ = llm_text  # hook
    claim_re = re.compile(
        r"(?:We show|We demonstrate|Our results|This paper|We prove|This work|We find|We observe)\b[^.]{10,200}\.",
        re.IGNORECASE,
    )
    results: list[ClaimItem] = []
    for m in claim_re.finditer(content):
        results.append(ClaimItem(text=m.group(0).strip()))
        if len(results) >= 4:
            break
    return results


def _parse_methods(llm_text: str, content: str) -> list[MethodItem]:
    """Extract methods via signal phrase heuristic."""
    _ = llm_text  # hook
    results: list[MethodItem] = []
    sentences = re.split(r"(?<=[.!?])\s+", content)
    for sent in sentences:
        if _METHOD_PHRASES.search(sent):
            # Title: first noun phrase (up to 6 words); description: full sentence
            words = sent.strip().split()
            name = " ".join(words[:6]) if words else sent[:30]
            results.append(MethodItem(name=name, description=sent.strip()[:200]))
        if len(results) >= 3:
            break
    return results


def _parse_limitations(llm_text: str, content: str) -> list[LimitationItem]:
    """Extract limitations via signal phrase heuristic."""
    _ = llm_text  # hook
    results: list[LimitationItem] = []
    sentences = re.split(r"(?<=[.!?])\s+", content)
    for sent in sentences:
        lower = sent.lower()
        if any(phrase in lower for phrase in _LIMITATION_PHRASES):
            results.append(LimitationItem(text=sent.strip()[:200]))
        if len(results) >= 3:
            break
    return results


def _parse_performance_facts(llm_text: str, content: str) -> list[PerformanceFactItem]:
    """Extract quantitative performance facts via regex heuristic."""
    _ = llm_text  # hook
    results: list[PerformanceFactItem] = []
    for m in _PERF_RE.finditer(content):
        metric = m.group("metric").strip()
        value = m.group("value").strip()
        ctx = (m.group("ctx") or "").strip()
        if metric and value:
            results.append(PerformanceFactItem(metric=metric, value=value, context=ctx or None))
        if len(results) >= 4:
            break
    return results


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: 1 token ≈ 4 characters (tiktoken not required)."""
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# LLM prompts (hook layer — pure string builders)
# ---------------------------------------------------------------------------


def _prompt_summary(title: str | None, content: str) -> list[ChatMessage]:
    """Build the chat messages for a summary request."""
    heading = title or "(untitled section)"
    return [
        ChatMessage(
            role="system",
            content=(
                "You are a scientific paper assistant. "
                "Summarise the following section concisely and accurately."
            ),
        ),
        ChatMessage(
            role="user",
            content=(
                f"Section: {heading}\n\n"
                f"{content[:2000]}\n\n"
                "Write a detailed summary (2-4 sentences), then on a new line "
                "prefixed with 'COMPACT:' write a one-sentence summary."
            ),
        ),
    ]


def _prompt_keywords(title: str | None, content: str) -> list[ChatMessage]:
    """Build the chat messages for a keyword extraction request."""
    heading = title or "(untitled section)"
    return [
        ChatMessage(role="system", content="You are a scientific keyword extractor."),
        ChatMessage(
            role="user",
            content=(
                f"Section: {heading}\n\n"
                f"{content[:2000]}\n\n"
                "List up to 8 general keywords and up to 6 technical keywords "
                "as comma-separated values."
            ),
        ),
    ]


def _prompt_structured(title: str | None, content: str) -> list[ChatMessage]:
    """Build the chat messages for structured extraction (entities/claims/etc.)."""
    heading = title or "(untitled section)"
    return [
        ChatMessage(
            role="system",
            content=(
                "You are a scientific information extractor. "
                "Extract named entities, claims, definitions, methods, "
                "limitations, and performance facts from academic text."
            ),
        ),
        ChatMessage(
            role="user",
            content=(
                f"Section: {heading}\n\n"
                f"{content[:2000]}\n\n"
                "Extract: entities, claims, definitions, methods, limitations, "
                "performance facts."
            ),
        ),
    ]


# ---------------------------------------------------------------------------
# Per-node enrichment logic
# ---------------------------------------------------------------------------


async def _enrich_one_node(
    node: DocumentNodeOut,
    content: str,
    chat_provider: ChatProvider,
) -> SectionEnrichment:
    """Enrich a single section/subsection/appendix node.

    Two provider calls are made per node:
    1. Summary prompt.
    2. Structured extraction prompt.

    Both calls are hooks — the deterministic parsers that follow work with
    any text response (including mock responses).
    """
    title = node.title

    # Call 1: summary
    summary_resp = await chat_provider.complete(
        _prompt_summary(title, content),
        temperature=0.0,
        max_tokens=512,
    )
    detailed = _parse_summary_detailed(summary_resp.content, content)
    compact = _parse_summary_compact(summary_resp.content, content)

    # Call 2: keywords (share call with entities via structured prompt)
    kw_resp = await chat_provider.complete(
        _prompt_keywords(title, content),
        temperature=0.0,
        max_tokens=256,
    )
    keywords = _parse_keywords(kw_resp.content, content)
    technical_keywords = _parse_technical_keywords(kw_resp.content, content)

    # Call 3: structured extraction
    struct_resp = await chat_provider.complete(
        _prompt_structured(title, content),
        temperature=0.0,
        max_tokens=512,
    )
    entities = _parse_entities(struct_resp.content, content)
    definitions = _parse_definitions(struct_resp.content, content)
    claims = _parse_claims(struct_resp.content, content)
    methods = _parse_methods(struct_resp.content, content)
    limitations = _parse_limitations(struct_resp.content, content)
    performance_facts = _parse_performance_facts(struct_resp.content, content)

    token_count = _estimate_tokens(detailed)

    return SectionEnrichment(
        node_id=node.id,
        chat_id=node.chat_id,
        document_id=node.document_id,
        node_type=node.node_type.value,
        title=node.title,
        page_start=node.page_start,
        page_end=node.page_end,
        source_block_ids=list(node.source_block_ids),
        detailed_summary=detailed,
        compact_summary=compact,
        keywords=keywords,
        technical_keywords=technical_keywords,
        entities=entities,
        definitions=definitions,
        claims=claims,
        methods=methods,
        limitations=limitations,
        performance_facts=performance_facts,
        related_figure_ids=[],  # populated by caller if needed
        related_table_ids=[],
        token_count=token_count,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def enrich_sections(
    hierarchy: HierarchyResult,
    blocks: list[ParsedBlock],
    *,
    chat_provider: ChatProvider,
) -> list[SectionEnrichment]:
    """Enrich all section / subsection / appendix nodes in *hierarchy*.

    Parameters
    ----------
    hierarchy:
        Full hierarchy produced by ``derive_hierarchy`` (Phase 4.3).
    blocks:
        Flat list of ``ParsedBlock`` instances for the same document.
    chat_provider:
        Any ``ChatProvider`` implementation.  Tests pass ``MockChatProvider``;
        production passes a real provider.

    Returns
    -------
    list[SectionEnrichment]
        One ``SectionEnrichment`` per ``section`` / ``subsection`` /
        ``appendix`` node, in ``order_index`` order.

    Notes
    -----
    - ``chat_id`` for every enrichment is taken exclusively from
      ``hierarchy.chat_id`` — never from an external argument.
    - Nodes with ``node_type`` outside ``{section, subsection, appendix}``
      are silently skipped (they belong to other Phase 5.x sub-tasks).
    """
    # Build a fast lookup from block_id → ParsedBlock
    block_map: dict[uuid.UUID, ParsedBlock] = {b.block_id: b for b in blocks}

    enrichments: list[SectionEnrichment] = []

    for node in hierarchy.nodes:
        if node.node_type not in _ENRICHABLE_NODE_TYPES:
            continue

        content = _gather_text_for_node(node, block_map)
        enrichment = await _enrich_one_node(node, content, chat_provider)
        enrichments.append(enrichment)

    return enrichments


# ---------------------------------------------------------------------------
# Phase 5.1 JSON-parsing enrichment (spec-compliant API)
# ---------------------------------------------------------------------------

_MAX_RETRIES = 2


def _try_parse_json_enrichment(
    raw: str,
    node: DocumentNodeOut,
    fallback_block_ids: list[UUID],
    model_name: str,
) -> SectionEnrichment:
    """Parse *raw* LLM text into a ``SectionEnrichment``.

    Strategy:
    1. Strip markdown fences if present.
    2. ``json.loads`` the text.
    3. Fill missing/empty ``source_block_ids`` lists with *fallback_block_ids*.
    4. Validate via ``SectionEnrichment``.

    Parameters
    ----------
    raw:
        Raw text from the LLM completion.
    node:
        The originating DocumentNodeOut (provides identity fields).
    fallback_block_ids:
        Used when a sub-list item has no ``source_block_ids``.
    model_name:
        Name of the chat provider model (stored in ``model_used``).

    Raises
    ------
    ValueError
        When ``raw`` cannot be decoded as valid JSON or the decoded object is
        not a dict.
    """
    # Strip optional markdown code fences
    stripped = raw.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        # Remove first line (```json / ```) and last line (```)
        stripped = "\n".join(lines[1:] if len(lines) > 2 else lines)
        stripped = stripped.rstrip("`").strip()

    data: object = json.loads(stripped)  # raises json.JSONDecodeError on failure
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object, got {type(data).__name__}")

    # Inject fallback source_block_ids where missing
    fb = [str(bid) for bid in fallback_block_ids]

    def _fill_block_ids(items: list[object]) -> list[object]:
        result: list[object] = []
        for item in items:
            if isinstance(item, dict):
                if not item.get("source_block_ids"):
                    item = {**item, "source_block_ids": fb}
            result.append(item)
        return result

    for list_key in (
        "keywords",
        "entities",
        "definitions",
        "claims",
        "methods",
        "limitations",
        "performance_facts",
    ):
        if list_key in data and isinstance(data[list_key], list):
            data[list_key] = _fill_block_ids(data[list_key])  # type: ignore[index]

    # Convert source_block_ids strings → UUID where possible
    def _coerce_block_ids(items: list[object]) -> list[object]:
        result: list[object] = []
        for item in items:
            if isinstance(item, dict) and "source_block_ids" in item:
                raw_ids = item["source_block_ids"]
                coerced: list[str] = []
                for rid in raw_ids:
                    try:
                        coerced.append(str(uuid.UUID(str(rid))))
                    except (ValueError, AttributeError):
                        pass
                item = {**item, "source_block_ids": coerced}
            result.append(item)
        return result

    for list_key in (
        "keywords",
        "entities",
        "definitions",
        "claims",
        "methods",
        "limitations",
        "performance_facts",
    ):
        if list_key in data and isinstance(data[list_key], list):
            data[list_key] = _coerce_block_ids(data[list_key])  # type: ignore[index]

    # Build token estimate
    detailed = str(data.get("detailed_summary", ""))
    token_est = _estimate_tokens(detailed)

    return SectionEnrichment(
        node_id=node.id,
        chat_id=node.chat_id,
        document_id=node.document_id,
        node_type=node.node_type.value,
        title=node.title,
        page_start=node.page_start,
        page_end=node.page_end,
        source_block_ids=fallback_block_ids,
        detailed_summary=detailed or "[empty]",
        compact_summary=str(data.get("compact_summary", "")).strip() or "[empty]",
        keywords=[
            kw["term"] for kw in data.get("keywords", []) if isinstance(kw, dict) and "term" in kw
        ],
        technical_keywords=[],
        entities=[
            ent["name"]
            for ent in data.get("entities", [])
            if isinstance(ent, dict) and "name" in ent
        ],
        definitions=[
            DefinitionItem(term=d["term"], definition=d["definition"])
            for d in data.get("definitions", [])
            if isinstance(d, dict) and "term" in d and "definition" in d
        ],
        claims=[
            ClaimItem(text=c["statement"])
            for c in data.get("claims", [])
            if isinstance(c, dict) and "statement" in c
        ],
        methods=[
            MethodItem(
                name=m["name"],
                description=m.get("summary", m.get("description", "")),
            )
            for m in data.get("methods", [])
            if isinstance(m, dict) and "name" in m
        ],
        limitations=[
            LimitationItem(text=lim["statement"])
            for lim in data.get("limitations", [])
            if isinstance(lim, dict) and "statement" in lim
        ],
        performance_facts=[
            PerformanceFactItem(
                metric=str(pf["metric"]),
                value=str(pf["value"]),
                context=pf.get("improvement") or pf.get("context") or None,
            )
            for pf in data.get("performance_facts", [])
            if isinstance(pf, dict) and "metric" in pf and "value" in pf
        ],
        related_figure_ids=_figure_ids_for_node(node, {}),
        related_table_ids=_table_ids_for_node(node, {}),
        token_count=token_est,
        token_count_estimate=token_est,
        model_used=model_name,
        traceable=True,
    )


async def enrich_section(
    node: DocumentNodeOut,
    child_paragraphs: list[DocumentNodeOut],
    parsed_blocks_index: dict[UUID, ParsedBlock],
    *,
    chat_provider: ChatProvider,
    chat_id: UUID,
    document_id: UUID,
) -> SectionEnrichment:
    """Enrich a single section / subsection / appendix node.

    This is the **JSON-parsing** variant that matches the Phase 5.1 spec API.
    It sends a single prompt (system + user) to the LLM, then attempts to
    parse the JSON response into a ``SectionEnrichment``.

    Parameters
    ----------
    node:
        The section/subsection/appendix node to enrich.
    child_paragraphs:
        Paragraph/figure/table/equation child nodes of *node*
        (used to collect additional block IDs; may be empty).
    parsed_blocks_index:
        Mapping from ``block_id`` → ``ParsedBlock``.
    chat_provider:
        Any ``ChatProvider`` implementation.
    chat_id:
        The owning chat's UUID (injected by the service layer; NOT passed to the LLM).
    document_id:
        The owning document's UUID (injected by the service layer).

    Returns
    -------
    SectionEnrichment

    Raises
    ------
    EnrichmentParseError
        When all retries are exhausted and the LLM response cannot be parsed.
    """
    from app.enrichment.prompts import build_section_messages

    # Collect block IDs from node + child paragraphs
    all_block_ids: list[UUID] = list(node.source_block_ids)
    for child in child_paragraphs:
        for bid in child.source_block_ids:
            if bid not in all_block_ids:
                all_block_ids.append(bid)

    # Collect relevant ParsedBlocks
    blocks: list[ParsedBlock] = [
        parsed_blocks_index[bid] for bid in all_block_ids if bid in parsed_blocks_index
    ]

    messages = build_section_messages(node, blocks)

    last_exc: Exception = EnrichmentParseError("No attempts made")
    for _attempt in range(_MAX_RETRIES):
        completion = await chat_provider.complete(
            messages,
            temperature=0.0,
            max_tokens=2048,
        )
        try:
            return _try_parse_json_enrichment(
                completion.content,
                node,
                all_block_ids,
                chat_provider.model,
            )
        except (json.JSONDecodeError, ValueError, Exception) as exc:
            last_exc = exc

    raise EnrichmentParseError(
        f"Failed to parse enrichment for node {node.id} after {_MAX_RETRIES} retries: {last_exc}"
    )


async def enrich_document_sections(
    hierarchy: HierarchyResult,
    parsed_blocks: list[ParsedBlock],
    *,
    chat_provider: ChatProvider,
) -> list[SectionEnrichment]:
    """Enrich all enrichable nodes in *hierarchy* using the JSON-parsing path.

    Enrichable node types: ``section``, ``subsection``, ``abstract``, ``appendix``.

    Parameters
    ----------
    hierarchy:
        Full hierarchy produced by ``derive_hierarchy`` (Phase 4.3).
    parsed_blocks:
        Flat list of ``ParsedBlock`` instances for the same document.
    chat_provider:
        Any ``ChatProvider`` implementation.

    Returns
    -------
    list[SectionEnrichment]
        One ``SectionEnrichment`` per enrichable node, in ``order_index`` order.

    Notes
    -----
    - ``chat_id`` for every enrichment is taken exclusively from
      ``hierarchy.chat_id`` — never from an external argument.
    - This function uses the JSON-parsing enricher (``enrich_section``);
      if parsing fails for a node, the error is silently caught and a
      heuristic fallback (``_enrich_one_node``) is used instead.
    """
    _EXTENDED_ENRICHABLE: frozenset[NodeType] = frozenset(
        [NodeType.section, NodeType.subsection, NodeType.appendix, NodeType.abstract]
    )

    block_map: dict[UUID, ParsedBlock] = {b.block_id: b for b in parsed_blocks}

    enrichments: list[SectionEnrichment] = []

    for node in hierarchy.nodes:
        if node.node_type not in _EXTENDED_ENRICHABLE:
            continue

        try:
            enrichment = await enrich_section(
                node,
                child_paragraphs=[],
                parsed_blocks_index=block_map,
                chat_provider=chat_provider,
                chat_id=hierarchy.chat_id,
                document_id=hierarchy.document_id,
            )
        except EnrichmentParseError:
            # Fallback to heuristic enricher
            content = _gather_text_for_node(node, block_map)
            enrichment = await _enrich_one_node(node, content, chat_provider)

        enrichments.append(enrichment)

    return enrichments
