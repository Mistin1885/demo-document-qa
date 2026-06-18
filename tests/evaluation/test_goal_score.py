"""Phase 9.3 — Goal Coverage scorer tests.

Tests cover: category math, mandatory-gate failure semantics, total
threshold, and parsing/retrieval/qa edge inputs.

Test density cap: ≤ 10 items per file (CLAUDE.md §12.1).
"""

from __future__ import annotations

import pytest

from app.evaluation import goal_score
from app.evaluation.goal_score import (
    CATEGORY_MAX,
    MIN_TOTAL_SCORE,
    ScoringInputs,
    compute_goal_score,
    render_markdown,
)


def _all_green_inputs() -> ScoringInputs:
    return ScoringInputs(
        parser_report={"pass_rate": 1.0, "avg_inline_math_recall": 1.0},
        retrieval_report={
            "per_mode": {
                "bm25_only": {"recall_at_10": 1.0, "leakage_count": 0},
                "hybrid": {"recall_at_10": 1.0, "leakage_count": 0},
                "hybrid_cross_encoder": {"recall_at_10": 1.0, "leakage_count": 0},
            }
        },
        qa_report={"pass_rate": 1.0, "total": 7, "passed": 7},
    )


# ---------------------------------------------------------------------------
# 1. Category maxima sum to 100.
# ---------------------------------------------------------------------------


def test_category_max_totals_one_hundred() -> None:
    assert sum(CATEGORY_MAX.values()) == 100


# ---------------------------------------------------------------------------
# 2. All-green inputs → total ≥ MIN_TOTAL_SCORE and pass_overall=True.
# ---------------------------------------------------------------------------


def test_all_green_inputs_pass_overall() -> None:
    report = compute_goal_score(_all_green_inputs())
    assert report.total_score >= MIN_TOTAL_SCORE
    assert report.mandatory_all_passed
    assert report.passed_overall
    assert report.percent >= MIN_TOTAL_SCORE / 100


# ---------------------------------------------------------------------------
# 3. Retrieval leakage triggers zero retrieval points even with full recall.
# ---------------------------------------------------------------------------


def test_retrieval_leakage_zeroes_category() -> None:
    inputs = _all_green_inputs()
    inputs.retrieval_report = {
        "per_mode": {
            "hybrid": {"recall_at_10": 1.0, "leakage_count": 3},
        }
    }
    report = compute_goal_score(inputs)
    retrieval = next(c for c in report.categories if c.name == "vespa_retrieval")
    assert retrieval.earned_points == 0
    assert retrieval.status in ("partial", "fail")


# ---------------------------------------------------------------------------
# 4. Failing a mandatory gate flips passed_overall to False even at 100/100.
# ---------------------------------------------------------------------------


def test_mandatory_gate_failure_blocks_pass_even_at_full_score() -> None:
    inputs = _all_green_inputs()
    inputs.chat_isolation_passed = False
    report = compute_goal_score(inputs)
    assert not report.mandatory_all_passed
    assert not report.passed_overall
    failed = [g.name for g in report.mandatory_gates if not g.passed]
    assert "chat_isolation" in failed


# ---------------------------------------------------------------------------
# 5. Missing artifacts → fail status; total well below threshold.
# ---------------------------------------------------------------------------


def test_missing_evidence_yields_fail_status() -> None:
    inputs = ScoringInputs(
        parser_report=None, retrieval_report=None, qa_report=None,
        provider_abstractions_ready=False, connection_test_supported=False,
        encrypted_key_supported=False, chat_isolation_passed=False,
        session_isolation_passed=False, citation_scope_passed=False,
        frontend_ok=False, backend_ok=False, langgraph_qa_passed=False,
        hybrid_retrieval_passed=False,
    )
    report = compute_goal_score(inputs)
    assert not report.meets_min_score
    assert not report.mandatory_all_passed
    assert not report.passed_overall


# ---------------------------------------------------------------------------
# 6. Partial parser pass_rate maps to a proportional partial score.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("pass_rate", "expected_status"),
    [
        (1.0, "pass"),
        (0.75, "partial"),
        (0.0, "fail"),
    ],
)
def test_parser_partial_pass_rate(pass_rate: float, expected_status: str) -> None:
    inputs = _all_green_inputs()
    inputs.parser_report = {"pass_rate": pass_rate, "avg_inline_math_recall": pass_rate}
    report = compute_goal_score(inputs)
    parser_cat = next(c for c in report.categories if c.name == "parsing")
    assert parser_cat.status == expected_status


# ---------------------------------------------------------------------------
# 7. Markdown renders a usable summary.
# ---------------------------------------------------------------------------


def test_render_markdown_contains_summary_lines() -> None:
    report = compute_goal_score(_all_green_inputs())
    md = render_markdown(report)
    assert "Goal Coverage Report" in md
    assert "Mandatory Gates" in md
    assert "PASS" in md or "FAIL" in md
    for cat in report.categories:
        assert cat.name in md


# ---------------------------------------------------------------------------
# 8. load_json gracefully returns None for missing paths.
# ---------------------------------------------------------------------------


def test_load_json_missing_path_returns_none(tmp_path) -> None:  # type: ignore[no-untyped-def]
    assert goal_score.load_json(tmp_path / "nope.json") is None
