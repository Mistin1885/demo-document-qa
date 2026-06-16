"""Parser evaluation harness for Phase 4.4.

This module implements a pure-Python evaluation pipeline that:
1. Loads a ``GoldenAnnotation`` from a JSON file.
2. Invokes ``MinerUClient.parse_pdf`` (idempotent by default).
3. Runs ``map_middle_to_parsed_blocks`` and ``derive_hierarchy``.
4. Computes a suite of structural and content-fidelity metrics.
5. Produces ``PaperEvalResult`` for a single paper and ``CorpusReport``
   for a collection, writing ``artifacts/evaluation/parser-report.{json,md}``.

Design rules (CLAUDE.md §12)
-----------------------------
- Pure Python / no FastAPI / no raw SQL / no secrets logged.
- Full type hints on every public and private function.
- ``GoldenAnnotation`` and ``PaperEvalResult`` are Pydantic v2 models.
- ``evaluate_paper`` calls ``map_middle_to_parsed_blocks`` + ``derive_hierarchy``
  from the application parsing layer; it does NOT hard-code expected values.

See: CLAUDE.md §6.4; GUIDE §24 (Parser Evaluation).
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

from app.parsing.hierarchy import derive_hierarchy
from app.parsing.mapping import map_middle_to_parsed_blocks
from app.parsing.mineru_client import MinerUClient, MinerUParseResult
from app.parsing.models import DocumentNodeOut, HierarchyResult, NodeType

# ---------------------------------------------------------------------------
# Gate thresholds
# ---------------------------------------------------------------------------

# A paper passes the gate if heading F1 is at or above this value.
_HEADING_F1_GATE: float = 0.3
# Inline math recall gate (0.0 = no inline math expected is OK)
_INLINE_MATH_RECALL_GATE: float = 0.5


# ---------------------------------------------------------------------------
# GoldenAnnotation
# ---------------------------------------------------------------------------


class GoldenAnnotation(BaseModel):
    """Minimum oracle facts for one paper.

    These are hand-written conservative annotations — they catch major
    regressions, not pixel-perfect layout differences.
    """

    paper_id: str
    pdf_filename: str
    doc_title: str
    expected_section_headings: list[str]
    has_abstract: bool
    has_appendix: bool
    expected_reference_count_at_least: int
    expected_inline_math_substring: list[str]
    expected_minimum_figures: int
    expected_minimum_tables: int
    expected_minimum_equations: int
    page_count_hint: int | None = None
    notes: str = ""


def load_golden(path: Path) -> GoldenAnnotation:
    """Read and parse a GoldenAnnotation from a JSON file."""
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return GoldenAnnotation(**data)


# ---------------------------------------------------------------------------
# PaperEvalResult
# ---------------------------------------------------------------------------


class PaperEvalResult(BaseModel):
    """Evaluation result for a single paper.

    All fields are computed by ``evaluate_paper``; ``gate_passed`` summarises
    whether the paper meets the minimum quality bar for pipeline use.
    """

    paper_id: str

    # --- Schema validity --------------------------------------------------- #
    schema_validity: bool
    """mapping + hierarchy produced no exception; parent_ids resolve; order_index dense."""

    page_marker_consistency: bool
    """<Page N> open/close tags in .md equal the page count."""

    block_ownership_unique: bool
    """No ParsedBlock owned by more than one hierarchy node (discarded excluded)."""

    # --- Content fidelity -------------------------------------------------- #
    doc_title_recall: bool
    """Hierarchy document root title matches golden (case-insensitive, normalised)."""

    heading_precision: float
    heading_recall: float
    heading_f1: float
    """Heading precision/recall/F1 against ``expected_section_headings``."""

    abstract_detected: bool
    """True iff golden.has_abstract matches hierarchy having a NodeType.abstract node."""

    references_detected: bool
    """references_start_index is not None."""

    reference_count_ge: bool
    """Reference node count >= golden.expected_reference_count_at_least."""

    appendix_detected: bool
    """Golden.has_appendix matches appendix_start_index presence."""

    no_duplicate_paragraphs: bool
    """All NodeType.paragraph content values are unique (after normalisation)."""

    page_count_matches: bool
    """max page_end across nodes == MinerUParseResult.pages (or hint if no parse)."""

    inline_math_recall: float
    """Fraction of golden.expected_inline_math_substring found in any paragraph."""

    # --- Counts ------------------------------------------------------------ #
    figure_count: int
    table_count: int
    equation_count: int
    reference_count: int

    figure_count_ok: bool
    table_count_ok: bool
    equation_count_ok: bool

    # --- Performance ------------------------------------------------------- #
    duration_seconds: float
    """From MinerUParseResult.duration_seconds (0.0 when idempotent cache reused)."""

    mineru_version: str
    mineru_backend: str

    # --- Gate -------------------------------------------------------------- #
    gate_passed: bool
    failure_reasons: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# CorpusReport
# ---------------------------------------------------------------------------


class CorpusReport(BaseModel):
    """Aggregated evaluation report over a collection of papers."""

    papers: list[PaperEvalResult]
    total_papers: int
    passed: int
    failed: int
    pass_rate: float
    avg_heading_f1: float
    avg_inline_math_recall: float


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _normalise(text: str) -> str:
    """Lowercase, collapse whitespace, strip, remove diacritics."""
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", errors="ignore").decode("ascii")
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _count_page_markers(markdown_text: str) -> tuple[int, int]:
    """Return (open_count, close_count) for <Page N> markers."""
    open_count = len(re.findall(r"<Page\s+\d+>", markdown_text))
    close_count = len(re.findall(r"</Page\s+\d+>", markdown_text))
    return open_count, close_count


def _check_schema_validity(
    nodes: list[DocumentNodeOut],
) -> tuple[bool, list[str]]:
    """Validate node structure: parent_ids resolve, order_index dense + unique."""
    reasons: list[str] = []
    node_ids: set[UUID] = {n.id for n in nodes}

    for n in nodes:
        if n.parent_id is not None and n.parent_id not in node_ids:
            reasons.append(
                f"Node {n.id} (order={n.order_index}) has unresolvable "
                f"parent_id={n.parent_id}"
            )

    order_indices = [n.order_index for n in nodes]
    if len(order_indices) != len(set(order_indices)):
        reasons.append("order_index values are not unique")
    else:
        expected = list(range(len(nodes)))
        if sorted(order_indices) != expected:
            reasons.append(
                f"order_index is not dense 0..{len(nodes)-1}: "
                f"got {sorted(order_indices)[:5]}..."
            )

    return len(reasons) == 0, reasons


def _check_block_ownership_unique(
    hierarchy: HierarchyResult,
) -> tuple[bool, int]:
    """Return (unique_ok, duplicate_count).

    Discarded blocks are NOT in any node's source_block_ids so they don't
    need to be tracked here.
    """
    seen: dict[UUID, int] = {}
    for node in hierarchy.nodes:
        for bid in node.source_block_ids:
            seen[bid] = seen.get(bid, 0) + 1

    duplicates = sum(1 for v in seen.values() if v > 1)
    return duplicates == 0, duplicates


def _compute_heading_metrics(
    nodes: list[DocumentNodeOut],
    golden: GoldenAnnotation,
) -> tuple[float, float, float]:
    """Compute precision / recall / F1 for section/subsection headings.

    Both predicted and expected headings are normalised before matching.
    An abstract heading is excluded since it is not a section node in the
    hierarchy (it maps to NodeType.abstract, not section/subsection).
    """
    predicted_headings = {
        _normalise(n.title)
        for n in nodes
        if n.node_type in (NodeType.section, NodeType.subsection)
        and n.title is not None
    }
    # Filter out boundary names that are structural artefacts in the hierarchy
    structural = {"references", "bibliography"}
    predicted_headings -= structural

    expected_headings = {
        _normalise(h)
        for h in golden.expected_section_headings
        if _normalise(h) != "abstract"
    }

    if not expected_headings and not predicted_headings:
        return 1.0, 1.0, 1.0

    tp = len(predicted_headings & expected_headings)
    fp = len(predicted_headings - expected_headings)
    fn = len(expected_headings - predicted_headings)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return precision, recall, f1


def _compute_inline_math_recall(
    nodes: list[DocumentNodeOut],
    golden: GoldenAnnotation,
) -> float:
    """Fraction of expected inline math substrings found in any node's content."""
    if not golden.expected_inline_math_substring:
        return 1.0

    # Collect all text content from all nodes
    all_content = "\n".join(
        n.content for n in nodes if n.content
    )

    found = sum(
        1
        for substr in golden.expected_inline_math_substring
        if substr in all_content
    )
    return found / len(golden.expected_inline_math_substring)


def _check_duplicate_paragraphs(nodes: list[DocumentNodeOut]) -> tuple[bool, int]:
    """Return (no_duplicates, duplicate_count) for paragraph nodes."""
    para_contents: list[str] = [
        _normalise(n.content)
        for n in nodes
        if n.node_type == NodeType.paragraph and n.content.strip()
    ]
    seen: dict[str, int] = {}
    for c in para_contents:
        seen[c] = seen.get(c, 0) + 1
    duplicates = sum(1 for v in seen.values() if v > 1)
    return duplicates == 0, duplicates


# ---------------------------------------------------------------------------
# Core evaluate_paper
# ---------------------------------------------------------------------------


async def evaluate_paper(
    pdf_path: Path,
    golden: GoldenAnnotation,
    *,
    client: MinerUClient,
    chat_id: UUID,
    document_id: UUID,
    force_reparse: bool = False,
) -> PaperEvalResult:
    """Evaluate the parser pipeline on one paper against a GoldenAnnotation.

    Parameters
    ----------
    pdf_path:
        Absolute path to the source PDF (may be absent if cache exists and
        ``force_reparse=False``).
    golden:
        GoldenAnnotation loaded from ``data/fixtures/golden/<id>.json``.
    client:
        Configured ``MinerUClient`` instance.
    chat_id:
        Injected isolation identifier (not derived from file metadata).
    document_id:
        Injected document identifier (deterministic per paper for eval runs).
    force_reparse:
        When True, ignore any existing parse cache and re-run MinerU.

    Returns
    -------
    PaperEvalResult
        All metrics plus ``gate_passed`` and ``failure_reasons``.
    """
    failure_reasons: list[str] = []

    # ------------------------------------------------------------------ #
    # Step 1: Parse (idempotent reuse by default)
    # ------------------------------------------------------------------ #
    parse_result: MinerUParseResult = await client.parse_pdf(
        pdf_path, force=force_reparse
    )

    # ------------------------------------------------------------------ #
    # Step 2: Mapping + hierarchy
    # ------------------------------------------------------------------ #
    schema_valid = True
    try:
        middle_data = json.loads(
            parse_result.middle_json_path.read_text(encoding="utf-8")
        )
        parsed_blocks = map_middle_to_parsed_blocks(
            middle_data, chat_id=chat_id, document_id=document_id
        )
        hierarchy = derive_hierarchy(
            parsed_blocks, document_id=document_id, chat_id=chat_id
        )
    except Exception as exc:  # noqa: BLE001
        schema_valid = False
        failure_reasons.append(f"mapping/hierarchy raised: {exc}")
        # Return a minimal failing result
        return PaperEvalResult(
            paper_id=golden.paper_id,
            schema_validity=False,
            page_marker_consistency=False,
            block_ownership_unique=False,
            doc_title_recall=False,
            heading_precision=0.0,
            heading_recall=0.0,
            heading_f1=0.0,
            abstract_detected=False,
            references_detected=False,
            reference_count_ge=False,
            appendix_detected=False,
            no_duplicate_paragraphs=False,
            page_count_matches=False,
            inline_math_recall=0.0,
            figure_count=0,
            table_count=0,
            equation_count=0,
            reference_count=0,
            figure_count_ok=False,
            table_count_ok=False,
            equation_count_ok=False,
            duration_seconds=parse_result.duration_seconds,
            mineru_version=parse_result.mineru_version,
            mineru_backend=parse_result.mineru_backend,
            gate_passed=False,
            failure_reasons=failure_reasons,
        )

    nodes = hierarchy.nodes

    # ------------------------------------------------------------------ #
    # Step 3: Schema validity
    # ------------------------------------------------------------------ #
    struct_ok, struct_reasons = _check_schema_validity(nodes)
    if not struct_ok:
        schema_valid = False
        failure_reasons.extend(struct_reasons)

    # ------------------------------------------------------------------ #
    # Step 4: Page marker consistency
    # ------------------------------------------------------------------ #
    markdown_text = parse_result.markdown_path.read_text(encoding="utf-8")
    open_count, close_count = _count_page_markers(markdown_text)
    page_marker_ok = open_count == parse_result.pages and close_count == parse_result.pages
    if not page_marker_ok:
        failure_reasons.append(
            f"page_marker_consistency failed: open={open_count}, "
            f"close={close_count}, pages={parse_result.pages}"
        )

    # ------------------------------------------------------------------ #
    # Step 5: Block ownership uniqueness
    # ------------------------------------------------------------------ #
    ownership_ok, dup_block_count = _check_block_ownership_unique(hierarchy)
    if not ownership_ok:
        failure_reasons.append(
            f"block_ownership_unique failed: {dup_block_count} blocks claimed twice"
        )

    # ------------------------------------------------------------------ #
    # Step 6: Doc title recall
    # ------------------------------------------------------------------ #
    doc_root_nodes = [n for n in nodes if n.node_type == NodeType.document]
    doc_title_ok = False
    if doc_root_nodes and doc_root_nodes[0].title:
        predicted_title = _normalise(doc_root_nodes[0].title)
        expected_title = _normalise(golden.doc_title)
        # Allow partial match: expected title words found in predicted title
        expected_words = set(expected_title.split())
        predicted_words = set(predicted_title.split())
        overlap = len(expected_words & predicted_words)
        doc_title_ok = overlap >= max(1, len(expected_words) // 2)
    if not doc_title_ok:
        failure_reasons.append(
            f"doc_title_recall failed: predicted="
            f"{doc_root_nodes[0].title if doc_root_nodes else None!r}, "
            f"expected={golden.doc_title!r}"
        )

    # ------------------------------------------------------------------ #
    # Step 7: Heading metrics
    # ------------------------------------------------------------------ #
    heading_precision, heading_recall, heading_f1 = _compute_heading_metrics(
        nodes, golden
    )

    # ------------------------------------------------------------------ #
    # Step 8: Abstract detected
    # ------------------------------------------------------------------ #
    has_abstract_node = any(n.node_type == NodeType.abstract for n in nodes)
    abstract_ok = has_abstract_node == golden.has_abstract
    if not abstract_ok:
        failure_reasons.append(
            f"abstract_detected mismatch: found={has_abstract_node}, "
            f"expected={golden.has_abstract}"
        )

    # ------------------------------------------------------------------ #
    # Step 9: References detected + count
    # ------------------------------------------------------------------ #
    references_ok = hierarchy.references_start_index is not None
    ref_nodes = [n for n in nodes if n.node_type == NodeType.reference]
    reference_count = len(ref_nodes)
    ref_count_ok = reference_count >= golden.expected_reference_count_at_least
    if not references_ok:
        failure_reasons.append("references_detected failed: references_start_index is None")
    if not ref_count_ok:
        failure_reasons.append(
            f"reference_count_ge failed: found={reference_count}, "
            f"expected>={golden.expected_reference_count_at_least}"
        )

    # ------------------------------------------------------------------ #
    # Step 10: Appendix detected
    # ------------------------------------------------------------------ #
    has_appendix_in_hier = hierarchy.appendix_start_index is not None
    appendix_ok = has_appendix_in_hier == golden.has_appendix
    if not appendix_ok:
        failure_reasons.append(
            f"appendix_detected mismatch: found={has_appendix_in_hier}, "
            f"expected={golden.has_appendix}"
        )

    # ------------------------------------------------------------------ #
    # Step 11: No duplicate paragraphs
    # ------------------------------------------------------------------ #
    no_dup_para, dup_para_count = _check_duplicate_paragraphs(nodes)
    if not no_dup_para:
        failure_reasons.append(
            f"no_duplicate_paragraphs failed: {dup_para_count} duplicate paragraphs"
        )

    # ------------------------------------------------------------------ #
    # Step 12: Page count matches
    # ------------------------------------------------------------------ #
    if nodes:
        max_page_end = max(n.page_end for n in nodes)
    else:
        max_page_end = 0
    page_count_ok = max_page_end == parse_result.pages
    if golden.page_count_hint is not None and not page_count_ok:
        # Allow hint-based check as fallback
        page_count_ok = max_page_end == golden.page_count_hint or parse_result.pages == golden.page_count_hint

    # ------------------------------------------------------------------ #
    # Step 13: Inline math recall
    # ------------------------------------------------------------------ #
    inline_math_recall = _compute_inline_math_recall(nodes, golden)

    # ------------------------------------------------------------------ #
    # Step 14: Counts
    # ------------------------------------------------------------------ #
    figure_count = sum(1 for n in nodes if n.node_type == NodeType.figure)
    table_count = sum(1 for n in nodes if n.node_type == NodeType.table)
    equation_count = sum(1 for n in nodes if n.node_type == NodeType.equation)

    figure_count_ok = figure_count >= golden.expected_minimum_figures
    table_count_ok = table_count >= golden.expected_minimum_tables
    equation_count_ok = equation_count >= golden.expected_minimum_equations

    if not figure_count_ok:
        failure_reasons.append(
            f"figure_count_ok failed: found={figure_count}, "
            f"expected>={golden.expected_minimum_figures}"
        )
    if not table_count_ok:
        failure_reasons.append(
            f"table_count_ok failed: found={table_count}, "
            f"expected>={golden.expected_minimum_tables}"
        )
    if not equation_count_ok:
        failure_reasons.append(
            f"equation_count_ok failed: found={equation_count}, "
            f"expected>={golden.expected_minimum_equations}"
        )

    # ------------------------------------------------------------------ #
    # Step 15: Gate pass determination
    # ------------------------------------------------------------------ #
    gate_passed = (
        schema_valid
        and page_marker_ok
        and ownership_ok
        and abstract_ok
        and references_ok
        and appendix_ok
        and no_dup_para
        and heading_f1 >= _HEADING_F1_GATE
        and (
            not golden.expected_inline_math_substring
            or inline_math_recall >= _INLINE_MATH_RECALL_GATE
        )
    )

    return PaperEvalResult(
        paper_id=golden.paper_id,
        schema_validity=schema_valid,
        page_marker_consistency=page_marker_ok,
        block_ownership_unique=ownership_ok,
        doc_title_recall=doc_title_ok,
        heading_precision=round(heading_precision, 4),
        heading_recall=round(heading_recall, 4),
        heading_f1=round(heading_f1, 4),
        abstract_detected=abstract_ok,
        references_detected=references_ok,
        reference_count_ge=ref_count_ok,
        appendix_detected=appendix_ok,
        no_duplicate_paragraphs=no_dup_para,
        page_count_matches=page_count_ok,
        inline_math_recall=round(inline_math_recall, 4),
        figure_count=figure_count,
        table_count=table_count,
        equation_count=equation_count,
        reference_count=reference_count,
        figure_count_ok=figure_count_ok,
        table_count_ok=table_count_ok,
        equation_count_ok=equation_count_ok,
        duration_seconds=parse_result.duration_seconds,
        mineru_version=parse_result.mineru_version,
        mineru_backend=parse_result.mineru_backend,
        gate_passed=gate_passed,
        failure_reasons=failure_reasons,
    )


# ---------------------------------------------------------------------------
# Corpus evaluation
# ---------------------------------------------------------------------------


async def evaluate_corpus(
    pairs: list[tuple[Path, GoldenAnnotation]],
    *,
    client: MinerUClient,
    chat_id: UUID | None = None,
    output_dir: Path = Path("artifacts/evaluation"),
    force_reparse: bool = False,
) -> CorpusReport:
    """Evaluate a collection of papers and write the corpus report.

    Parameters
    ----------
    pairs:
        List of ``(pdf_path, golden_annotation)`` tuples.
    client:
        Configured ``MinerUClient``.
    chat_id:
        Optional chat isolation UUID; a fresh one is generated per call if
        omitted (evaluation mode — no real DB).
    output_dir:
        Directory where ``parser-report.json`` and ``parser-report.md`` are
        written.  Created if absent.
    force_reparse:
        Passed through to each ``evaluate_paper`` call.

    Returns
    -------
    CorpusReport
        Aggregated metrics plus per-paper results.
    """
    import uuid as _uuid

    if chat_id is None:
        chat_id = _uuid.uuid4()

    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[PaperEvalResult] = []

    for pdf_path, golden in pairs:
        document_id = _uuid.uuid5(
            _uuid.NAMESPACE_OID, f"eval:{golden.paper_id}"
        )
        result = await evaluate_paper(
            pdf_path,
            golden,
            client=client,
            chat_id=chat_id,
            document_id=document_id,
            force_reparse=force_reparse,
        )
        results.append(result)

    passed = sum(1 for r in results if r.gate_passed)
    failed = len(results) - passed
    pass_rate = passed / len(results) if results else 0.0
    avg_f1 = (
        sum(r.heading_f1 for r in results) / len(results) if results else 0.0
    )
    avg_math = (
        sum(r.inline_math_recall for r in results) / len(results) if results else 0.0
    )

    report = CorpusReport(
        papers=results,
        total_papers=len(results),
        passed=passed,
        failed=failed,
        pass_rate=round(pass_rate, 4),
        avg_heading_f1=round(avg_f1, 4),
        avg_inline_math_recall=round(avg_math, 4),
    )

    # ------------------------------------------------------------------ #
    # Write JSON report
    # ------------------------------------------------------------------ #
    json_path = output_dir / "parser-report.json"
    json_path.write_text(
        report.model_dump_json(indent=2), encoding="utf-8"
    )

    # ------------------------------------------------------------------ #
    # Write Markdown report
    # ------------------------------------------------------------------ #
    md_path = output_dir / "parser-report.md"
    md_path.write_text(_render_markdown(report), encoding="utf-8")

    return report


# ---------------------------------------------------------------------------
# Markdown renderer
# ---------------------------------------------------------------------------


def _render_markdown(report: CorpusReport) -> str:
    """Render a human-readable Markdown summary table."""
    lines: list[str] = [
        "# Parser Evaluation Report",
        "",
        f"**Total papers:** {report.total_papers}  "
        f"**Passed:** {report.passed}  "
        f"**Failed:** {report.failed}  "
        f"**Pass rate:** {report.pass_rate:.1%}",
        "",
        f"**Avg heading F1:** {report.avg_heading_f1:.3f}  "
        f"**Avg inline-math recall:** {report.avg_inline_math_recall:.3f}",
        "",
        "## Per-Paper Results",
        "",
        "| Paper | Gate | Schema | PageMark | BlockOwn | Title | H-F1 | Abstract | Refs | Appendix | NoDupPara | Math | Figs | Tables | Eqs | Duration |",  # noqa: E501
        "|-------|------|--------|----------|----------|-------|------|----------|------|----------|-----------|------|------|--------|-----|----------|",
    ]

    def _tick(v: bool) -> str:
        return "OK" if v else "FAIL"

    for r in report.papers:
        lines.append(
            f"| {r.paper_id} "
            f"| {'PASS' if r.gate_passed else 'FAIL'} "
            f"| {_tick(r.schema_validity)} "
            f"| {_tick(r.page_marker_consistency)} "
            f"| {_tick(r.block_ownership_unique)} "
            f"| {_tick(r.doc_title_recall)} "
            f"| {r.heading_f1:.3f} "
            f"| {_tick(r.abstract_detected)} "
            f"| {_tick(r.references_detected)} "
            f"| {_tick(r.appendix_detected)} "
            f"| {_tick(r.no_duplicate_paragraphs)} "
            f"| {r.inline_math_recall:.3f} "
            f"| {r.figure_count}({'OK' if r.figure_count_ok else 'FAIL'}) "
            f"| {r.table_count}({'OK' if r.table_count_ok else 'FAIL'}) "
            f"| {r.equation_count}({'OK' if r.equation_count_ok else 'FAIL'}) "
            f"| {r.duration_seconds:.1f}s |"
        )

    # Failure details
    failures = [r for r in report.papers if not r.gate_passed]
    if failures:
        lines += [
            "",
            "## Failure Details",
            "",
        ]
        for r in failures:
            lines.append(f"### {r.paper_id}")
            for reason in r.failure_reasons:
                lines.append(f"- {reason}")
            lines.append("")

    lines.append("")
    lines.append("---")
    lines.append("*Report generated by `run_parser_eval.py`*")
    lines.append("")
    return "\n".join(lines)


__all__ = [
    "GoldenAnnotation",
    "PaperEvalResult",
    "CorpusReport",
    "load_golden",
    "evaluate_paper",
    "evaluate_corpus",
]
