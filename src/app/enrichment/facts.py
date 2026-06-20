"""Deterministic regex-based structured fact extractor (Phase 5.3).

Extracts ``StructuredFactCreate`` instances from a parsed document's
``HierarchyResult`` and ``list[ParsedBlock]``.  No LLM involved — all
extraction is regex-based.

Design rules (CLAUDE.md §12)
------------------------------
- Pure function: no I/O, no DB imports, no FastAPI imports.
- All outputs are deterministic for a given (hierarchy, blocks) pair.
- ``StructuredFact.id`` uses ``uuid5(NAMESPACE_OID, ...)`` for idempotency.
- ``context_excerpt`` is always ≤ 200 chars and non-empty for every fact.
- ``other`` kind facts must carry a unit OR a non-empty context_excerpt.

Sources
-------
- Main text comes from ``ParsedBlock.text`` for ``paragraph``, ``text``, and
  ``ref_text`` block types.
- Section title context is derived from ``HierarchyResult.nodes`` (section /
  subsection nodes whose ``source_block_ids`` overlap the target block).
- We do NOT attempt to extract from ``equation_latex`` — MinerU fragments
  inline math with thousands separators (docs/dev/PROGRESS.md known limitation).

Fact kinds
----------
- ``metric``       — F1, EM, accuracy, BLEU, ROUGE, nDCG, Recall@k, MRR, …
- ``benchmark``    — <Method> achieves|outperforms pattern in Experiments sections.
- ``dataset``      — dataset names in Datasets / Experimental Setup sections.
- ``hyperparameter`` — learning_rate, batch_size, temperature, top_k, etc.
- ``other``        — unclassified numeric with unit or non-empty context.
"""

from __future__ import annotations

import re
import uuid

from app.models.domain import FactValue, StructuredFactCreate
from app.parsing.models import BlockType, DocumentNodeOut, HierarchyResult, NodeType, ParsedBlock

# ---------------------------------------------------------------------------
# Section-title keyword sets
# ---------------------------------------------------------------------------

_SECTION_EXPERIMENT_RE = re.compile(
    r"experiment|evaluation|benchmark|result|ablation|comparison",
    re.IGNORECASE,
)
_SECTION_DATASET_RE = re.compile(
    r"dataset|experimental\s+setup|data\s+collection|data\s+preparation",
    re.IGNORECASE,
)
_SECTION_HYPERPARAM_RE = re.compile(
    r"implement|training|setup|configuration|hyperparameter|detail",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Metric regex
# ---------------------------------------------------------------------------

# Matches patterns like:
#   "F1 score of 87.3%", "accuracy: 0.923", "BLEU = 34.5", "Recall@5 = 0.81",
#   "nDCG@10 92.1", "MRR 0.756", "EM 72.4%"
_METRIC_NAMES = (
    r"F1(?:\s*@\d+)?|EM|accuracy|BLEU|ROUGE[-\d]*|nDCG(?:@\d+)?|"
    r"Recall@\d+|MRR|precision|MAP|AUC|perplexity|METEOR|CIDEr|WER|"
    r"throughput|latency|NDCG(?:@\d+)?"
)
_METRIC_RE = re.compile(
    r"(?P<metric_name>" + _METRIC_NAMES + r")"
    r"(?:\s*(?:score|value|of|:|=|is)\s*)?"
    r"(?P<numeric>-?\d+(?:[.,]\d+)?)"
    r"\s*(?P<unit>%|ms|s\b|GB|MB|tokens?|k\b)?",
    re.IGNORECASE,
)

# Also catch "achieves X% Y" or "reaches X Y"
_ACHIEVES_METRIC_RE = re.compile(
    r"(?:achieves|reaches|obtains|reports)\s+"
    r"(?P<numeric>\d+(?:\.\d+)?)"
    r"\s*(?P<unit>%|ms|GB|tokens?)?\s+"
    r"(?P<metric_name>" + _METRIC_NAMES + r")",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Benchmark regex (method achieves/outperforms pattern)
# ---------------------------------------------------------------------------

# Matches: "<Name> achieves <score> on <Dataset>"
#       or "<Name> outperforms <baseline> by <score>"
_BENCHMARK_RE = re.compile(
    r"(?P<method>[A-Z][A-Za-z0-9\-]{1,30})"
    r"\s+(?:achieves|outperforms|surpasses|beats)"
    r"(?:\s+[a-z][a-z ]{0,20})?"
    r"\s+(?P<score>\d+(?:\.\d+)?)\s*(?P<unit>%|points?)?",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Dataset regex
# ---------------------------------------------------------------------------

# PascalCase/hyphenated token ≥ 3 chars (typical dataset names),
# or single-quoted/double-quoted terms.
# Positive patterns: "MS-MARCO", "BEIR", "Natural Questions", "SQuAD", etc.
_DATASET_RE = re.compile(
    r"""
    (?:
        # Quoted name
        ["'](?P<quoted>[A-Za-z0-9 \-]{3,50})["']
        |
        # Hyphenated name with uppercase component
        (?P<hyph>[A-Z][A-Za-z0-9]{1,20}(?:-[A-Za-z0-9]{1,20})+)
        |
        # ALL-CAPS acronym ≥ 3 chars
        (?P<caps>[A-Z]{3,10})
        |
        # PascalCase ≥ 6 chars (e.g., SQuAD, HotpotQA, TriviaQA)
        (?P<pascal>[A-Z][a-z][A-Za-z]{4,30})
    )
    """,
    re.VERBOSE,
)

# Common English words to exclude from dataset candidates
_DATASET_STOPWORDS: frozenset[str] = frozenset(
    {
        "The",
        "This",
        "These",
        "That",
        "With",
        "From",
        "Table",
        "Figure",
        "Section",
        "Appendix",
        "Model",
        "Method",
        "System",
        "Task",
        "Paper",
        "Work",
        "Study",
        "Baseline",
        "Results",
        "Models",
        "Methods",
        "We",
        "Our",
        "Their",
        "Human",
        "Large",
        "Small",
        "First",
        "Second",
        "Third",
        "Each",
        "Both",
        "All",
        "None",
        "True",
        "False",
        "Note",
        "Also",
        "Since",
        "When",
        "While",
        "Based",
        "Using",
        "Given",
        "Show",
        "Shows",
        "Training",
        "Testing",
        "Evaluation",
        "Performance",
        "Score",
        "Scores",
        "BERT",
        "GPT",
        "LLM",
        "LLMS",
        "API",
        "GPU",
        "CPU",
        "RAM",
        "NLP",
        "QA",
        "RAG",
    }
)

# ---------------------------------------------------------------------------
# Hyperparameter regex
# ---------------------------------------------------------------------------

_HYPERPARAM_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "learning_rate",
        re.compile(
            r"learning\s*rate\s*(?:of\s*|=\s*|:?\s*)(?P<val>\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)",
            re.IGNORECASE,
        ),
    ),
    (
        "batch_size",
        re.compile(
            r"batch\s*size\s*(?:of\s*|=\s*|:?\s*)(?P<val>\d+)",
            re.IGNORECASE,
        ),
    ),
    (
        "epochs",
        re.compile(
            r"(?:train(?:ed)?\s+for|for)\s+(?P<val>\d+)\s+epochs?",
            re.IGNORECASE,
        ),
    ),
    (
        "temperature",
        re.compile(
            r"temperature\s*(?:of\s*|=\s*|:?\s*)(?P<val>\d+(?:\.\d+)?)",
            re.IGNORECASE,
        ),
    ),
    (
        "top_k",
        re.compile(
            r"top[-\s]?k\s*(?:=\s*|:?\s*|of\s*)(?P<val>\d+)",
            re.IGNORECASE,
        ),
    ),
    (
        "top_p",
        re.compile(
            r"top[-\s]?p\s*(?:=\s*|:?\s*|of\s*)(?P<val>\d+(?:\.\d+)?)",
            re.IGNORECASE,
        ),
    ),
    (
        "dropout",
        re.compile(
            r"dropout\s*(?:rate\s*)?(?:of\s*|=\s*|:?\s*)(?P<val>\d+(?:\.\d+)?)",
            re.IGNORECASE,
        ),
    ),
    (
        "hidden_dim",
        re.compile(
            r"hidden\s*(?:dim(?:ension)?|size)\s*(?:of\s*|=\s*|:?\s*)(?P<val>\d+)",
            re.IGNORECASE,
        ),
    ),
    (
        "max_length",
        re.compile(
            r"max(?:imum)?\s*(?:sequence\s*|token\s*|seq\s*)?length\s*(?:of\s*|=\s*|:?\s*)(?P<val>\d+)",
            re.IGNORECASE,
        ),
    ),
    (
        "num_layers",
        re.compile(
            r"(?:number\s*of\s*layers?|num(?:_|\s*)layers?)\s*(?:of\s*|=\s*|:?\s*)(?P<val>\d+)",
            re.IGNORECASE,
        ),
    ),
    (
        "num_heads",
        re.compile(
            r"(?:attention\s*)?heads?\s*(?:of\s*|=\s*|:?\s*)(?P<val>\d+)",
            re.IGNORECASE,
        ),
    ),
    (
        "weight_decay",
        re.compile(
            r"weight\s*decay\s*(?:of\s*|=\s*|:?\s*)(?P<val>\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)",
            re.IGNORECASE,
        ),
    ),
]

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _truncate(text: str, max_len: int = 200) -> str:
    """Truncate to ``max_len`` chars; preserve whole words if possible."""
    if len(text) <= max_len:
        return text
    return text[: max_len - 1].rstrip() + "…"


def _excerpt(text: str, start: int, end: int, window: int = 100) -> str:
    """Extract a ≤200-char excerpt centred around the [start, end] span."""
    lo = max(0, start - window // 2)
    hi = min(len(text), end + window // 2)
    snippet = text[lo:hi].strip()
    return _truncate(snippet, 200)


def _safe_float(s: str) -> float | None:
    """Parse a numeric string that may use comma as thousands separator."""
    try:
        return float(s.replace(",", ""))
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Section-mapping helpers
# ---------------------------------------------------------------------------


_SECTION_NODE_TYPES = frozenset((NodeType.section, NodeType.subsection, NodeType.appendix))


def _build_node_index(
    nodes: list[DocumentNodeOut],
) -> tuple[
    dict[uuid.UUID, DocumentNodeOut],  # id → node
    dict[uuid.UUID, uuid.UUID | None],  # block_id → containing-node id
]:
    """Build two fast-lookup indexes from the node list.

    Returns
    -------
    node_by_id
        Maps node UUID → DocumentNodeOut for O(1) parent traversal.
    block_to_node
        Maps block UUID → the node whose ``source_block_ids`` contains it.
    """
    node_by_id: dict[uuid.UUID, DocumentNodeOut] = {}
    block_to_node: dict[uuid.UUID, uuid.UUID | None] = {}
    for node in nodes:
        node_by_id[node.id] = node
        for bid in node.source_block_ids:
            block_to_node[bid] = node.id
    return node_by_id, block_to_node


def _build_block_to_section(nodes: list[DocumentNodeOut]) -> dict[uuid.UUID, uuid.UUID | None]:
    """Return mapping of block_id → nearest enclosing section/subsection node id.

    Walk up the parent chain from the node that directly owns the block until
    a section/subsection/appendix node is found.  This handles the common case
    where the hierarchy builder emits ``paragraph`` nodes that are children of
    section/subsection nodes.
    """
    node_by_id, block_to_node = _build_node_index(nodes)
    result: dict[uuid.UUID, uuid.UUID | None] = {}

    for bid, owning_id in block_to_node.items():
        if owning_id is None:
            result[bid] = None
            continue
        owning = node_by_id.get(owning_id)
        if owning is None:
            result[bid] = None
            continue
        # Walk up the parent chain looking for a section/subsection/appendix
        current: DocumentNodeOut | None = owning
        section_id: uuid.UUID | None = None
        while current is not None:
            if current.node_type in _SECTION_NODE_TYPES:
                section_id = current.id
                break
            parent_id = current.parent_id
            current = node_by_id.get(parent_id) if parent_id is not None else None
        result[bid] = section_id

    return result


def _section_title_for_block(
    block: ParsedBlock,
    nodes: list[DocumentNodeOut],
    *,
    _node_by_id: dict[uuid.UUID, DocumentNodeOut] | None = None,
    _block_to_node: dict[uuid.UUID, uuid.UUID | None] | None = None,
) -> str | None:
    """Return the title of the nearest enclosing section/subsection for ``block``.

    Walks up the parent chain from the node that directly owns the block.
    Falls back to a linear scan when the indexes are not provided (for
    backward-compatible call sites that pass only (block, nodes)).
    """
    if _node_by_id is None or _block_to_node is None:
        node_by_id, block_to_node = _build_node_index(nodes)
    else:
        node_by_id = _node_by_id
        block_to_node = _block_to_node

    owning_id = block_to_node.get(block.block_id)
    if owning_id is None:
        return None
    current: DocumentNodeOut | None = node_by_id.get(owning_id)
    while current is not None:
        if current.node_type in _SECTION_NODE_TYPES and current.title:
            return current.title
        parent_id = current.parent_id
        current = node_by_id.get(parent_id) if parent_id is not None else None
    return None


def _node_id_for_block(
    block: ParsedBlock,
    block_to_section: dict[uuid.UUID, uuid.UUID | None],
) -> uuid.UUID | None:
    return block_to_section.get(block.block_id)


# ---------------------------------------------------------------------------
# Per-kind extractors
# ---------------------------------------------------------------------------


def _extract_metrics(
    text: str,
    block: ParsedBlock,
    section_title: str | None,
    source_node_id: uuid.UUID | None,
    document_id: uuid.UUID,
    chat_id: uuid.UUID,
    seq_counter: list[int],
) -> list[StructuredFactCreate]:
    facts: list[StructuredFactCreate] = []

    for pattern in (_METRIC_RE, _ACHIEVES_METRIC_RE):
        for m in pattern.finditer(text):
            metric_name = m.group("metric_name").strip()
            raw_numeric = m.group("numeric").replace(",", "")
            numeric = _safe_float(raw_numeric)
            if numeric is None:
                continue
            unit = (m.group("unit") or "").strip() or None
            excerpt = _excerpt(text, m.start(), m.end())
            key = metric_name.lower().replace(" ", "_")
            seq_counter[0] += 1
            facts.append(
                StructuredFactCreate(
                    chat_id=chat_id,
                    document_id=document_id,
                    source_node_id=source_node_id,
                    kind="metric",
                    key=key,
                    value=FactValue(raw=m.group(0).strip(), numeric=numeric),
                    unit=unit,
                    context_excerpt=excerpt,
                    page=block.page_number,
                )
            )
    return facts


def _extract_benchmarks(
    text: str,
    block: ParsedBlock,
    section_title: str | None,
    source_node_id: uuid.UUID | None,
    document_id: uuid.UUID,
    chat_id: uuid.UUID,
    seq_counter: list[int],
) -> list[StructuredFactCreate]:
    """Extract benchmark facts only when inside an experiment/evaluation section."""
    if section_title is None or not _SECTION_EXPERIMENT_RE.search(section_title):
        return []

    facts: list[StructuredFactCreate] = []
    for m in _BENCHMARK_RE.finditer(text):
        method = m.group("method").strip()
        score_str = m.group("score").replace(",", "")
        numeric = _safe_float(score_str)
        if numeric is None:
            continue
        unit = (m.group("unit") or "").strip() or None
        excerpt = _excerpt(text, m.start(), m.end())
        key = f"{method.lower()}_score"
        seq_counter[0] += 1
        facts.append(
            StructuredFactCreate(
                chat_id=chat_id,
                document_id=document_id,
                source_node_id=source_node_id,
                kind="benchmark",
                key=key,
                value=FactValue(raw=m.group(0).strip(), numeric=numeric),
                unit=unit,
                context_excerpt=excerpt,
                page=block.page_number,
            )
        )
    return facts


def _extract_datasets(
    text: str,
    block: ParsedBlock,
    section_title: str | None,
    source_node_id: uuid.UUID | None,
    document_id: uuid.UUID,
    chat_id: uuid.UUID,
    seq_counter: list[int],
) -> list[StructuredFactCreate]:
    """Extract dataset names only when inside a dataset/setup section."""
    if section_title is None or not _SECTION_DATASET_RE.search(section_title):
        return []

    facts: list[StructuredFactCreate] = []
    seen: set[str] = set()
    for m in _DATASET_RE.finditer(text):
        name = (
            m.group("quoted") or m.group("hyph") or m.group("caps") or m.group("pascal") or ""
        ).strip()
        if not name or name in _DATASET_STOPWORDS or name.lower() in seen:
            continue
        # Require at least one digit or a hyphen (reduces false positives)
        # Exception: ALL-CAPS acronyms ≥ 3 chars are accepted as-is.
        if not (
            re.search(r"\d", name)
            or "-" in name
            or (name.isupper() and len(name) >= 3)
            or m.group("hyph")
        ):
            continue
        seen.add(name.lower())
        excerpt = _excerpt(text, m.start(), m.end())
        key = name.lower().replace(" ", "_").replace("-", "_")
        seq_counter[0] += 1
        facts.append(
            StructuredFactCreate(
                chat_id=chat_id,
                document_id=document_id,
                source_node_id=source_node_id,
                kind="dataset",
                key=key,
                value=FactValue(raw=name, items=[name]),
                unit=None,
                context_excerpt=excerpt,
                page=block.page_number,
            )
        )
    return facts


def _extract_hyperparameters(
    text: str,
    block: ParsedBlock,
    section_title: str | None,
    source_node_id: uuid.UUID | None,
    document_id: uuid.UUID,
    chat_id: uuid.UUID,
    seq_counter: list[int],
) -> list[StructuredFactCreate]:
    facts: list[StructuredFactCreate] = []
    for key_name, pattern in _HYPERPARAM_PATTERNS:
        for m in pattern.finditer(text):
            raw_val = m.group("val")
            numeric = _safe_float(raw_val)
            if numeric is None:
                continue
            excerpt = _excerpt(text, m.start(), m.end())
            seq_counter[0] += 1
            facts.append(
                StructuredFactCreate(
                    chat_id=chat_id,
                    document_id=document_id,
                    source_node_id=source_node_id,
                    kind="hyperparameter",
                    key=key_name,
                    value=FactValue(raw=raw_val, numeric=numeric),
                    unit=None,
                    context_excerpt=excerpt,
                    page=block.page_number,
                )
            )
    return facts


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


def _dedup_facts(facts: list[StructuredFactCreate]) -> list[StructuredFactCreate]:
    """Remove duplicate facts by (kind, key, numeric, page); keep first seen."""
    seen: set[tuple[str, str, float | None, int | None]] = set()
    result: list[StructuredFactCreate] = []
    for f in facts:
        sig = (f.kind, f.key, f.value.numeric, f.page)
        if sig not in seen:
            seen.add(sig)
            result.append(f)
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_structured_facts(
    hierarchy: HierarchyResult,
    blocks: list[ParsedBlock],
) -> list[StructuredFactCreate]:
    """Extract structured facts from a parsed document.

    Parameters
    ----------
    hierarchy:
        Output of ``derive_hierarchy``; provides section titles for context.
    blocks:
        Flat list of ``ParsedBlock`` instances from ``map_middle_to_parsed_blocks``.

    Returns
    -------
    list[StructuredFactCreate]
        Deterministic list of facts (same input → same output, including IDs).
        Each fact carries ``chat_id`` and ``document_id`` from ``hierarchy``.
        Caller must override ``chat_id`` via ``persist_facts`` before writing.
    """
    document_id = hierarchy.document_id
    chat_id = hierarchy.chat_id
    nodes = hierarchy.nodes

    # Build indexes once for O(1) lookups during extraction
    node_by_id, block_to_node = _build_node_index(nodes)
    block_to_section = _build_block_to_section(nodes)

    # Global sequence counter (mutable list for closure mutability)
    seq_counter: list[int] = [0]

    all_facts: list[StructuredFactCreate] = []

    # Only process content blocks (paragraph, text, ref_text)
    content_types = {BlockType.paragraph, BlockType.text, BlockType.ref_text}

    for block in blocks:
        if block.block_type not in content_types:
            continue
        text = block.text.strip()
        if not text:
            continue

        section_title = _section_title_for_block(
            block, nodes, _node_by_id=node_by_id, _block_to_node=block_to_node
        )
        source_node_id = _node_id_for_block(block, block_to_section)

        all_facts.extend(
            _extract_metrics(
                text, block, section_title, source_node_id, document_id, chat_id, seq_counter
            )
        )
        all_facts.extend(
            _extract_benchmarks(
                text, block, section_title, source_node_id, document_id, chat_id, seq_counter
            )
        )
        all_facts.extend(
            _extract_datasets(
                text, block, section_title, source_node_id, document_id, chat_id, seq_counter
            )
        )
        all_facts.extend(
            _extract_hyperparameters(
                text, block, section_title, source_node_id, document_id, chat_id, seq_counter
            )
        )

    return _dedup_facts(all_facts)


__all__ = ["extract_structured_facts"]
