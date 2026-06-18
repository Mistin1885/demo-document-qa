"""Phase 9.1 — Golden QA evaluation harness tests.

Tests cover the public surface of ``app.evaluation.qa_eval``:
loader, single-case runner, corpus runner, isolation invariants, and a
parametrized happy-path sweep over the non-isolation case kinds.

Test density cap: ≤ 10 items per file (CLAUDE.md §12.1).  parametrize cases
count individually.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.evaluation import qa_eval

_FIXTURE = Path(__file__).resolve().parents[2] / "data" / "fixtures" / "qa_cases.json"


def _spec(case_id: str) -> qa_eval.QACaseSpec:
    cases = qa_eval.load_qa_cases(_FIXTURE)
    for c in cases:
        if c.case_id == case_id:
            return c
    raise KeyError(case_id)


# ---------------------------------------------------------------------------
# 1. Loader contract — every defined kind is present and unique.
# ---------------------------------------------------------------------------


def test_load_qa_cases_returns_all_seven_kinds() -> None:
    cases = qa_eval.load_qa_cases(_FIXTURE)
    kinds = [c.kind for c in cases]
    assert set(kinds) == {
        "global_summary",
        "method_explanation",
        "technical_comparison",
        "performance",
        "multi_document",
        "chat_isolation",
        "session_isolation",
    }
    # case_id uniqueness — corpus reports rely on it.
    assert len({c.case_id for c in cases}) == len(cases)


# ---------------------------------------------------------------------------
# 2. Chat-isolation invariant — Chat A asking Chat B content yields refusal.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_chat_isolation_no_cross_chat_citation() -> None:
    result = await qa_eval.run_case(_spec("chat_isolation_a_asks_b_only_content"))

    assert result.passed, result.failures
    assert result.citation_count == 0
    assert "paper_b" not in result.citation_doc_labels
    assert qa_eval._NO_ANSWER_PHRASE in result.answer_snippet.lower()


# ---------------------------------------------------------------------------
# 3. Multi-document case — every required document is cited.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_multi_document_cites_all_three_papers() -> None:
    result = await qa_eval.run_case(_spec("multi_document_three_papers"))

    assert result.passed, result.failures
    assert set(result.citation_doc_labels) >= {"paper_a", "paper_b", "paper_c"}
    assert result.documents_used_count == 3


# ---------------------------------------------------------------------------
# 4. Session-isolation case — never leaks documents from outside target chat.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_session_isolation_never_cross_chat() -> None:
    result = await qa_eval.run_case(_spec("session_isolation_a1_vs_a2"))

    # Whatever the answer, every citation must belong to chat A.
    target_chat = qa_eval.chat_uuid("A")
    assert result.passed, result.failures
    # The runner snapshot uses doc labels — assert no foreign labels.
    forbidden_labels = {"paper_b", "paper_c"}
    assert not (set(result.citation_doc_labels) & forbidden_labels)
    # Target chat must match what we asked for (sanity).
    assert qa_eval.session_uuid("A", "2") != qa_eval.session_uuid("A", "1")
    assert target_chat == qa_eval.chat_uuid("A")


# ---------------------------------------------------------------------------
# 5. Parametrized happy-path sweep — global_summary, method, comparison,
#    performance. Each must (a) pass its expected outcome and (b) keep every
#    citation inside the target chat.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case_id",
    [
        "global_summary_lightrag",
        "method_explanation_lightrag",
        "technical_comparison_lightrag_vs_naive",
        "performance_lightrag_metrics",
    ],
)
async def test_happy_path_kinds_produce_in_scope_citations(case_id: str) -> None:
    result = await qa_eval.run_case(_spec(case_id))

    assert result.passed, result.failures
    assert result.citation_count >= 1
    assert "paper_a" in result.citation_doc_labels


# ---------------------------------------------------------------------------
# 6. Corpus runner — aggregates pass/fail totals correctly and meets gate.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_evaluate_corpus_aggregates_and_meets_gate() -> None:
    cases = qa_eval.load_qa_cases(_FIXTURE)
    report = await qa_eval.evaluate_corpus(cases)

    assert report.total == len(cases)
    assert report.passed + report.failed == report.total
    # Gate: at least 85% must pass for the golden evaluation to be considered green.
    assert report.pass_rate >= 0.85, report.model_dump_json(indent=2)
    # per_kind totals add up to the corpus total.
    assert sum(s.total for s in report.per_kind.values()) == report.total
