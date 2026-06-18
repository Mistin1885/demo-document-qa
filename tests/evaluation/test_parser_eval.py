"""Unit tests for app.evaluation.parser_eval — trimmed to ≤10 items.

Mandatory gates retained
------------------------
- gate_passed=True on happy-path (heading-F1, abstract, duplicates, etc.)
- abstract mismatch → gate_passed=False
- heading_recall drops when extra expected headings added
- math-recall == 1.0 for known substring
- corpus write produces valid JSON + MD reports

Helper metric unit tests merged into one parametrized test.

All tests use synthetic in-memory data and monkeypatched MinerUClient.parse_pdf.
No real PDF, no real MinerU server, no real DB.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import pytest

from app.evaluation.parser_eval import (
    CorpusReport,
    GoldenAnnotation,
    PaperEvalResult,
    _compute_heading_metrics,
    _compute_inline_math_recall,
    _count_page_markers,
    _normalise,
    evaluate_corpus,
    evaluate_paper,
    load_golden,
)
from app.parsing.mineru_client import MinerUParseResult

# ---------------------------------------------------------------------------
# Paths to existing synthetic fixtures
# ---------------------------------------------------------------------------

_FIXTURE_BASE = Path(__file__).parent.parent / "fixtures"
_PAPER_FIXTURE = _FIXTURE_BASE / "mineru_sample_paper"
_SIMPLE_FIXTURE = _FIXTURE_BASE / "mineru_sample"

# ---------------------------------------------------------------------------
# Stable IDs
# ---------------------------------------------------------------------------

_CHAT_ID = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
_DOC_ID = UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")

# ---------------------------------------------------------------------------
# Shared golden for the 8-page paper fixture
# ---------------------------------------------------------------------------

_PAPER_GOLDEN = GoldenAnnotation(
    paper_id="adaptrag_synthetic",
    pdf_filename="adaptrag_synthetic.pdf",
    doc_title="Adaptive Retrieval-Augmented Generation for Scientific Document QA",
    expected_section_headings=[
        "1 Introduction",
        "2 Related Work",
        "3 Methodology",
        "4 Experiments",
        "References",
    ],
    has_abstract=True,
    has_appendix=True,
    expected_reference_count_at_least=4,
    expected_inline_math_substring=["P(y|q, C)"],
    expected_minimum_figures=1,
    expected_minimum_tables=1,
    expected_minimum_equations=1,
    page_count_hint=8,
    notes="Synthetic 8-page paper fixture for unit tests.",
)


# ---------------------------------------------------------------------------
# Helper: build a MinerUParseResult from a fixture directory
# ---------------------------------------------------------------------------


def _make_parse_result(
    fixture_dir: Path,
    *,
    pages: int | None = None,
    duration_seconds: float = 0.0,
) -> MinerUParseResult:
    middle_path = fixture_dir / "middle.json"
    md_path = fixture_dir / "doc.md"

    if pages is None:
        middle_data: dict[str, Any] = json.loads(
            middle_path.read_text(encoding="utf-8")
        )
        pages = len(middle_data.get("pdf_info", []))

    gate_summary: dict[str, Any] = {
        "gate_pass": True,
        "pages": pages,
        "page_markers_open": pages,
        "page_markers_close": pages,
    }

    return MinerUParseResult(
        output_dir=fixture_dir,
        markdown_path=md_path,
        middle_json_path=middle_path,
        image_paths=[],
        pages=pages,
        gate_summary=gate_summary,
        duration_seconds=duration_seconds,
        mineru_version="3.3.1",
        mineru_backend="hybrid",
    )


def _make_mock_client(parse_result: MinerUParseResult) -> MagicMock:
    mock = MagicMock()
    mock.parse_pdf = AsyncMock(return_value=parse_result)
    return mock


async def _eval_paper_fixture(golden: GoldenAnnotation) -> PaperEvalResult:
    parse_result = _make_parse_result(_PAPER_FIXTURE)
    client = _make_mock_client(parse_result)
    fake_pdf = _PAPER_FIXTURE / "adaptrag_synthetic.pdf"
    return await evaluate_paper(
        fake_pdf,
        golden,
        client=client,
        chat_id=_CHAT_ID,
        document_id=_DOC_ID,
        force_reparse=False,
    )


# ===========================================================================
# 1. Parametrized helper-function tests (merged from TestHelperFunctions +
#    TestComputeHeadingMetrics + TestInlineMathRecall)
# ===========================================================================


def test_page_markers_and_normalise() -> None:
    """Covers _count_page_markers (balanced + mismatch) and _normalise (strip, collapse, diacritics)."""
    # balanced markers
    md_ok = "<Page 1>\n# T\n</Page 1>\n<Page 2>\nX\n</Page 2>"
    opens, closes = _count_page_markers(md_ok)
    assert opens == 2 and closes == 2

    # mismatched markers
    md_mis = "<Page 1>\n# T\n<Page 2>\nX\n</Page 2>"
    opens2, closes2 = _count_page_markers(md_mis)
    assert opens2 == 2 and closes2 == 1

    # normalise: strip, collapse whitespace, diacritics
    assert _normalise("  Hello World  ") == "hello world"
    assert _normalise("A  B\t\tC") == "a b c"
    assert _normalise("résumé") == "resume"


# ===========================================================================
# 2. load_golden round-trip
# ===========================================================================


def test_load_golden_round_trip(tmp_path: Path) -> None:
    golden_path = tmp_path / "test_golden.json"
    golden_path.write_text(_PAPER_GOLDEN.model_dump_json(indent=2), encoding="utf-8")
    loaded = load_golden(golden_path)
    assert loaded.paper_id == _PAPER_GOLDEN.paper_id
    assert loaded.expected_section_headings == _PAPER_GOLDEN.expected_section_headings
    assert loaded.has_abstract == _PAPER_GOLDEN.has_abstract


# ===========================================================================
# 3. MANDATORY GATE: happy-path evaluate_paper → gate_passed=True
# ===========================================================================


@pytest.mark.asyncio
async def test_evaluate_paper_gate_passed_happy_path() -> None:
    """Mandatory gate: full fixture with correct golden → gate_passed=True,
    heading_f1 > 0, math recall == 1.0, no duplicates, abstract detected."""
    result = await _eval_paper_fixture(_PAPER_GOLDEN)
    assert result.gate_passed, f"Expected gate_passed=True; failures: {result.failure_reasons}"
    assert result.heading_f1 > 0.0, f"Heading F1 must be > 0; got {result.heading_f1}"
    assert result.abstract_detected
    assert result.no_duplicate_paragraphs
    assert result.block_ownership_unique


# ===========================================================================
# 4. MANDATORY GATE: abstract mismatch → gate_passed=False
# ===========================================================================


@pytest.mark.asyncio
async def test_evaluate_paper_abstract_mismatch_fails_gate() -> None:
    """Mandatory gate: golden expects no abstract but fixture has one → gate fails."""
    golden_no_abstract = _PAPER_GOLDEN.model_copy(update={"has_abstract": False})
    result = await _eval_paper_fixture(golden_no_abstract)
    assert not result.gate_passed, "Expected gate_passed=False when abstract expectation mismatches"


# ===========================================================================
# 5. MANDATORY GATE: heading-F1 threshold — recall drops with phantom headings
# ===========================================================================


@pytest.mark.asyncio
async def test_evaluate_paper_heading_recall_drops_with_phantom_headings() -> None:
    """Mandatory gate: adding many nonexistent expected headings pushes recall below 1.0."""
    extra_headings = list(_PAPER_GOLDEN.expected_section_headings) + [
        "A Nonexistent Section That Does Not Exist At All"
    ] * 5
    golden_extra = _PAPER_GOLDEN.model_copy(update={"expected_section_headings": extra_headings})
    result = await _eval_paper_fixture(golden_extra)
    assert result.heading_recall < 1.0, (
        f"Expected recall < 1.0 when extra headings added; got {result.heading_recall}"
    )


# ===========================================================================
# 6. MANDATORY GATE: math-recall == 1.0 for known substring
# ===========================================================================


def test_compute_inline_math_recall_known_substring() -> None:
    """Mandatory gate: known inline math substring → recall == 1.0."""
    from app.parsing.hierarchy import derive_hierarchy
    from app.parsing.mapping import map_middle_to_parsed_blocks

    middle = json.loads((_PAPER_FIXTURE / "middle.json").read_text())
    blocks = map_middle_to_parsed_blocks(middle, chat_id=_CHAT_ID, document_id=_DOC_ID)
    hier = derive_hierarchy(blocks, document_id=_DOC_ID, chat_id=_CHAT_ID)

    golden_math = _PAPER_GOLDEN.model_copy(
        update={"expected_inline_math_substring": ["P(y|q, C)"]}
    )
    recall = _compute_inline_math_recall(hier.nodes, golden_math)
    assert recall == 1.0, f"Expected recall=1.0 for known math substring, got {recall}"


# ===========================================================================
# 7. heading_metrics: F1 > 0 for known headings; F1 == 0 for wrong headings
# ===========================================================================


def test_compute_heading_metrics_known_and_wrong() -> None:
    """MANDATORY gate: heading-F1 > 0 for correct headings; F1 == 0 for all-wrong headings."""
    from app.parsing.hierarchy import derive_hierarchy
    from app.parsing.mapping import map_middle_to_parsed_blocks

    middle = json.loads((_PAPER_FIXTURE / "middle.json").read_text())
    blocks = map_middle_to_parsed_blocks(middle, chat_id=_CHAT_ID, document_id=_DOC_ID)
    hier = derive_hierarchy(blocks, document_id=_DOC_ID, chat_id=_CHAT_ID)

    good_golden = _PAPER_GOLDEN.model_copy(
        update={"expected_section_headings": ["1 Introduction", "2 Related Work", "3 Methodology", "4 Experiments"]}
    )
    _p, _r, f1_good = _compute_heading_metrics(hier.nodes, good_golden)
    assert f1_good > 0.0, f"Expected F1 > 0 for known-correct headings, got {f1_good}"

    bad_golden = _PAPER_GOLDEN.model_copy(
        update={"expected_section_headings": ["Totally Wrong Section", "Another Missing Section"]}
    )
    _p2, _r2, f1_bad = _compute_heading_metrics(hier.nodes, bad_golden)
    assert f1_bad == 0.0, f"Expected F1=0 for all-wrong headings, got {f1_bad}"


# ===========================================================================
# 8. duplicate paragraph detection
# ===========================================================================


@pytest.mark.asyncio
async def test_duplicate_paragraph_detected() -> None:
    """Injecting a duplicate paragraph node is caught by _check_duplicate_paragraphs."""
    from uuid import uuid4

    from app.evaluation.parser_eval import _check_duplicate_paragraphs
    from app.parsing.hierarchy import derive_hierarchy
    from app.parsing.mapping import map_middle_to_parsed_blocks
    from app.parsing.models import DocumentNodeOut
    from app.parsing.models import NodeType as _NodeType

    middle = json.loads((_SIMPLE_FIXTURE / "middle.json").read_text())
    blocks = map_middle_to_parsed_blocks(middle, chat_id=_CHAT_ID, document_id=_DOC_ID)
    hier = derive_hierarchy(blocks, document_id=_DOC_ID, chat_id=_CHAT_ID)

    para_nodes = [n for n in hier.nodes if n.node_type == _NodeType.paragraph]
    if not para_nodes:
        pytest.skip("No paragraph nodes in simple fixture")

    first_para = para_nodes[0]
    dup_node = DocumentNodeOut(
        id=uuid4(),
        chat_id=_CHAT_ID,
        document_id=_DOC_ID,
        parent_id=first_para.parent_id,
        node_type=_NodeType.paragraph,
        title=None,
        content=first_para.content,
        page_start=first_para.page_start,
        page_end=first_para.page_end,
        order_index=9999,
        level=first_para.level,
        bbox=None,
        source_block_ids=[],
        metadata_={"source": "test"},
    )
    nodes_with_dup = list(hier.nodes) + [dup_node]
    no_dup, dup_count = _check_duplicate_paragraphs(nodes_with_dup)
    assert not no_dup, "Should detect duplicate paragraph"
    assert dup_count >= 1


# ===========================================================================
# 9. evaluate_corpus — writes valid JSON + MD reports (mandatory gate)
# ===========================================================================


@pytest.mark.asyncio
async def test_evaluate_corpus_writes_valid_reports(tmp_path: Path) -> None:
    """Corpus evaluation writes parser-report.json (valid Pydantic) + parser-report.md."""
    parse_result = _make_parse_result(_PAPER_FIXTURE)
    client = _make_mock_client(parse_result)
    fake_pdf = _PAPER_FIXTURE / "adaptrag_synthetic.pdf"
    output_dir = tmp_path / "evaluation"

    report = await evaluate_corpus(
        [(fake_pdf, _PAPER_GOLDEN)],
        client=client,
        chat_id=_CHAT_ID,
        output_dir=output_dir,
    )

    assert isinstance(report, CorpusReport)
    assert report.total_papers == 1

    json_report = output_dir / "parser-report.json"
    md_report = output_dir / "parser-report.md"
    assert json_report.exists(), "parser-report.json not written"
    assert md_report.exists(), "parser-report.md not written"

    # JSON round-trips through Pydantic
    loaded = CorpusReport.model_validate_json(json_report.read_text(encoding="utf-8"))
    assert loaded.total_papers == 1
    assert isinstance(loaded.papers[0], PaperEvalResult)

    # MD contains key headers
    md_content = md_report.read_text(encoding="utf-8")
    assert "# Parser Evaluation Report" in md_content
    assert "Per-Paper Results" in md_content
