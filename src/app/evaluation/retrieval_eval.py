"""Retrieval evaluation harness for Phase 6.5.

Computes Recall@k / MRR / nDCG@k / cross-chat leakage across five retrieval
modes and produces ``artifacts/evaluation/retrieval-report.{json,md}``.

CLAUDE.md §7 constraints
-------------------------
- ``RetrievalService`` is the ONLY Vespa query entry point.
- ``chat_id`` is always injected from ``EvalCase``; never from the LLM.
- No raw SQL; no real OpenAI/Gemini calls; all providers are mocks.

Design (CLAUDE.md §12)
-----------------------
- All public functions have type hints.
- ``EvalCase``, ``ModeMetrics``, ``CaseResult``, ``EvalReport``, ``GateResult``
  are typed Pydantic v2 models (no ``dict[str, Any]``).
- Metric functions are pure Python, independently testable.
- ``evaluate_gates`` is deterministic; never hard-codes pass/fail.

Usage
-----
    from app.evaluation.retrieval_eval import evaluate, evaluate_gates, write_report
    report = await evaluate(service, cases)
    gates = evaluate_gates(report)
    write_report(report, json_path, md_path)
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.retrieval.models import RetrievalRequest
from app.retrieval.service import RetrievalService

# ---------------------------------------------------------------------------
# Supported evaluation modes
# ---------------------------------------------------------------------------

#: All five modes evaluated by the harness.
EVAL_MODES: list[str] = [
    "bm25_only",
    "vector_only",
    "hybrid",
    "hybrid_native_rerank",
    "hybrid_cross_encoder",
]

# ---------------------------------------------------------------------------
# EvalCase
# ---------------------------------------------------------------------------


class EvalCase(BaseModel):
    """One evaluation query with ground-truth relevant document IDs.

    ``relevant_vespa_document_ids`` are Vespa document IDs (strings) that are
    considered relevant for the query.  All other returned IDs are treated as
    non-relevant (binary relevance).
    """

    model_config = ConfigDict(extra="forbid")

    query: str
    chat_id: UUID
    relevant_vespa_document_ids: list[str]
    document_ids: list[UUID] | None = None
    notes: str | None = None


# ---------------------------------------------------------------------------
# Pure metric functions
# ---------------------------------------------------------------------------


def recall_at_k(retrieved_ids: list[str], relevant: set[str], k: int) -> float:
    """Compute Recall@k.

    Fraction of relevant documents found in the top-k retrieved results.

    Parameters
    ----------
    retrieved_ids:
        Ordered list of retrieved Vespa document IDs (first = highest rank).
    relevant:
        Set of relevant Vespa document IDs (ground truth).
    k:
        Cutoff rank.

    Returns
    -------
    float
        Value in [0, 1].  Returns 1.0 when ``relevant`` is empty (vacuously
        satisfied).  Returns 0.0 when ``retrieved_ids`` is empty and
        ``relevant`` is non-empty.
    """
    if not relevant:
        return 1.0
    if not retrieved_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant)
    return hits / len(relevant)


def mean_reciprocal_rank(retrieved_ids: list[str], relevant: set[str]) -> float:
    """Compute Mean Reciprocal Rank (MRR) for a single query.

    Returns the reciprocal of the rank of the first relevant document found.
    If no relevant document is found, returns 0.0.

    Parameters
    ----------
    retrieved_ids:
        Ordered list of retrieved Vespa document IDs (first = rank 1).
    relevant:
        Set of relevant Vespa document IDs (ground truth).

    Returns
    -------
    float
        Value in [0, 1].  Returns 1.0 when ``relevant`` is empty.
    """
    if not relevant:
        return 1.0
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: list[str], relevant: set[str], k: int) -> float:
    """Compute nDCG@k with binary relevance.

    Parameters
    ----------
    retrieved_ids:
        Ordered list of retrieved Vespa document IDs.
    relevant:
        Set of relevant Vespa document IDs.
    k:
        Cutoff rank.

    Returns
    -------
    float
        Normalized Discounted Cumulative Gain at k.  Returns 1.0 when
        ``relevant`` is empty.  Returns 0.0 when no relevant doc is found.
    """
    if not relevant:
        return 1.0

    top_k = retrieved_ids[:k]

    # DCG: binary relevance (gain=1 if relevant)
    dcg = 0.0
    for rank, doc_id in enumerate(top_k, start=1):
        if doc_id in relevant:
            dcg += 1.0 / math.log2(rank + 1)

    # Ideal DCG: place all relevant docs at top
    ideal_len = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_len + 1))

    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def cross_chat_leakage(hits: list[Any], expected_chat_id: UUID) -> int:
    """Count hits whose chat_id does not match expected_chat_id.

    Parameters
    ----------
    hits:
        List of ``SearchHit`` objects returned by ``RetrievalService.search()``.
    expected_chat_id:
        The chat UUID that all hits must belong to.

    Returns
    -------
    int
        Number of hits from a different chat.  Must be 0 for isolation to hold.
    """
    expected_str = str(expected_chat_id)
    leaks = 0
    for hit in hits:
        hit_chat_id = str(hit.chat_id) if hasattr(hit, "chat_id") else ""
        if hit_chat_id != expected_str:
            leaks += 1
    return leaks


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------


class MetricsAtK(BaseModel):
    """Recall, nDCG at a specific k value."""

    model_config = ConfigDict(extra="forbid")

    recall: float = 0.0
    ndcg: float = 0.0


class ModeMetrics(BaseModel):
    """Aggregated metrics for one retrieval mode (macro-average across cases)."""

    model_config = ConfigDict(extra="forbid")

    mode: str
    recall_at_1: float = 0.0
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    mrr: float = 0.0
    ndcg_at_10: float = 0.0
    leakage_count: int = 0


class CaseMetrics(BaseModel):
    """Per-mode metrics for a single eval case."""

    model_config = ConfigDict(extra="forbid")

    mode: str
    retrieved_ids: list[str] = Field(default_factory=list)
    recall_at_1: float = 0.0
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    mrr: float = 0.0
    ndcg_at_10: float = 0.0
    leakage: int = 0


class CaseResult(BaseModel):
    """Evaluation result for a single EvalCase across all modes."""

    model_config = ConfigDict(extra="forbid")

    query: str
    chat_id: str
    relevant_ids: list[str] = Field(default_factory=list)
    notes: str | None = None
    per_mode: list[CaseMetrics] = Field(default_factory=list)


class GateResult(BaseModel):
    """Result of mandatory gate checks."""

    model_config = ConfigDict(extra="forbid")

    passed: bool
    gate_leakage_zero: bool
    gate_hybrid_ge_bm25: bool
    gate_hybrid_ge_vector: bool
    gate_native_rerank_no_regression: bool
    gate_cross_encoder_no_regression: bool
    failure_reasons: list[str] = Field(default_factory=list)


class EvalReport(BaseModel):
    """Full retrieval evaluation report."""

    model_config = ConfigDict(extra="forbid")

    per_mode: dict[str, ModeMetrics] = Field(default_factory=dict)
    per_case: list[CaseResult] = Field(default_factory=list)
    leakage_total: int = 0
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    embedding_dim: int = 0
    notes: str | None = None


# ---------------------------------------------------------------------------
# evaluate()
# ---------------------------------------------------------------------------

#: Mapping from eval mode name → (retrieval_mode, rerank_mode)
_MODE_PARAMS: dict[str, tuple[str, str]] = {
    "bm25_only": ("bm25_only", "none"),
    "vector_only": ("vector_only", "none"),
    "hybrid": ("hybrid", "none"),
    "hybrid_native_rerank": ("hybrid", "native"),
    "hybrid_cross_encoder": ("hybrid", "cross_encoder"),
}


async def evaluate(
    service: RetrievalService,
    cases: list[EvalCase],
    modes: list[str] | None = None,
    k_values: list[int] | None = None,
) -> EvalReport:
    """Run retrieval evaluation across cases and modes.

    Parameters
    ----------
    service:
        Configured ``RetrievalService`` (may use FakeVespa transport).
    cases:
        List of ``EvalCase`` with ground-truth relevant IDs.
    modes:
        Subset of ``EVAL_MODES`` to evaluate.  Defaults to all five.
    k_values:
        List of k values for Recall@k.  Defaults to [1, 5, 10].

    Returns
    -------
    EvalReport
        Per-mode macro-averaged metrics and per-case details.
    """
    if modes is None:
        modes = list(EVAL_MODES)
    if k_values is None:
        k_values = [1, 5, 10]

    # Skip cross_encoder mode if no reranker is configured
    effective_modes = list(modes)
    if "hybrid_cross_encoder" in effective_modes and service._reranker_provider is None:
        effective_modes.remove("hybrid_cross_encoder")

    # Accumulate per-mode totals for macro average
    mode_totals: dict[str, dict[str, float]] = {
        m: {
            "recall_at_1": 0.0,
            "recall_at_5": 0.0,
            "recall_at_10": 0.0,
            "mrr": 0.0,
            "ndcg_at_10": 0.0,
            "leakage_count": 0.0,
        }
        for m in effective_modes
    }

    case_results: list[CaseResult] = []
    leakage_total = 0

    for case in cases:
        relevant = set(case.relevant_vespa_document_ids)
        per_mode_metrics: list[CaseMetrics] = []

        for mode in effective_modes:
            retrieval_mode, rerank_mode = _MODE_PARAMS[mode]  # type: ignore[literal-required]

            req = RetrievalRequest(
                chat_id=case.chat_id,
                query=case.query,
                document_ids=case.document_ids,
                retrieval_mode=retrieval_mode,  # type: ignore[arg-type]
                rerank_mode=rerank_mode,  # type: ignore[arg-type]
                final_top_k=10,
                bm25_top_k=20,
                ann_top_k=20,
                fusion_top_k=20,
                rerank_top_k=15,
            )

            resp = await service.search(req)
            retrieved = [h.vespa_document_id for h in resp.hits]

            r1 = recall_at_k(retrieved, relevant, 1)
            r5 = recall_at_k(retrieved, relevant, 5)
            r10 = recall_at_k(retrieved, relevant, 10)
            mrr = mean_reciprocal_rank(retrieved, relevant)
            ndcg10 = ndcg_at_k(retrieved, relevant, 10)
            leakage = cross_chat_leakage(resp.hits, case.chat_id)

            mode_totals[mode]["recall_at_1"] += r1
            mode_totals[mode]["recall_at_5"] += r5
            mode_totals[mode]["recall_at_10"] += r10
            mode_totals[mode]["mrr"] += mrr
            mode_totals[mode]["ndcg_at_10"] += ndcg10
            mode_totals[mode]["leakage_count"] += leakage
            leakage_total += leakage

            per_mode_metrics.append(
                CaseMetrics(
                    mode=mode,
                    retrieved_ids=retrieved,
                    recall_at_1=round(r1, 4),
                    recall_at_5=round(r5, 4),
                    recall_at_10=round(r10, 4),
                    mrr=round(mrr, 4),
                    ndcg_at_10=round(ndcg10, 4),
                    leakage=leakage,
                )
            )

        case_results.append(
            CaseResult(
                query=case.query,
                chat_id=str(case.chat_id),
                relevant_ids=list(relevant),
                notes=case.notes,
                per_mode=per_mode_metrics,
            )
        )

    # Macro averages
    n = len(cases) if cases else 1
    per_mode_summary: dict[str, ModeMetrics] = {}
    for mode, totals in mode_totals.items():
        per_mode_summary[mode] = ModeMetrics(
            mode=mode,
            recall_at_1=round(totals["recall_at_1"] / n, 4),
            recall_at_5=round(totals["recall_at_5"] / n, 4),
            recall_at_10=round(totals["recall_at_10"] / n, 4),
            mrr=round(totals["mrr"] / n, 4),
            ndcg_at_10=round(totals["ndcg_at_10"] / n, 4),
            leakage_count=int(totals["leakage_count"]),
        )

    return EvalReport(
        per_mode=per_mode_summary,
        per_case=case_results,
        leakage_total=leakage_total,
        generated_at=datetime.now(UTC),
        embedding_dim=service._embedding_dim,
    )


# ---------------------------------------------------------------------------
# evaluate_gates()
# ---------------------------------------------------------------------------


def evaluate_gates(report: EvalReport) -> GateResult:
    """Check mandatory quality gates against the report.

    Gates
    -----
    1. ``leakage == 0`` — chat isolation must be perfect.
    2. ``hybrid.recall@10 >= bm25_only.recall@10`` — hybrid at least as good.
    3. ``hybrid.recall@10 >= vector_only.recall@10`` — hybrid at least as good.
    4. ``hybrid_native_rerank.recall@10 >= hybrid.recall@10 - 0.02``
       and ``hybrid_native_rerank.mrr >= hybrid.mrr - 0.02`` — native rerank
       does not significantly regress.
    5. Same regression check for ``hybrid_cross_encoder`` (skipped if absent).

    Returns
    -------
    GateResult
        ``passed=True`` only when all present gates pass.
    """
    failure_reasons: list[str] = []

    # Gate 1: leakage
    g_leak = report.leakage_total == 0
    if not g_leak:
        failure_reasons.append(
            f"FAIL leakage: {report.leakage_total} cross-chat hits detected (expected 0)"
        )

    def _get(mode: str) -> ModeMetrics | None:
        return report.per_mode.get(mode)

    hybrid = _get("hybrid")
    bm25 = _get("bm25_only")
    vector = _get("vector_only")
    native = _get("hybrid_native_rerank")
    ce = _get("hybrid_cross_encoder")

    # Gate 2: hybrid >= bm25
    g_hybrid_bm25 = True
    if hybrid is not None and bm25 is not None:
        g_hybrid_bm25 = hybrid.recall_at_10 >= bm25.recall_at_10
        if not g_hybrid_bm25:
            failure_reasons.append(
                f"FAIL hybrid>=bm25: hybrid.recall@10={hybrid.recall_at_10:.4f} "
                f"< bm25.recall@10={bm25.recall_at_10:.4f}"
            )

    # Gate 3: hybrid >= vector
    g_hybrid_vec = True
    if hybrid is not None and vector is not None:
        g_hybrid_vec = hybrid.recall_at_10 >= vector.recall_at_10
        if not g_hybrid_vec:
            failure_reasons.append(
                f"FAIL hybrid>=vector: hybrid.recall@10={hybrid.recall_at_10:.4f} "
                f"< vector.recall@10={vector.recall_at_10:.4f}"
            )

    # Gate 4: native rerank no regression
    _RERANK_TOL = 0.02
    g_native = True
    if native is not None and hybrid is not None:
        recall_ok = native.recall_at_10 >= hybrid.recall_at_10 - _RERANK_TOL
        mrr_ok = native.mrr >= hybrid.mrr - _RERANK_TOL
        g_native = recall_ok and mrr_ok
        if not recall_ok:
            failure_reasons.append(
                f"FAIL native_rerank recall regression: "
                f"native.recall@10={native.recall_at_10:.4f} "
                f"< hybrid.recall@10-0.02={hybrid.recall_at_10 - _RERANK_TOL:.4f}"
            )
        if not mrr_ok:
            failure_reasons.append(
                f"FAIL native_rerank MRR regression: "
                f"native.mrr={native.mrr:.4f} "
                f"< hybrid.mrr-0.02={hybrid.mrr - _RERANK_TOL:.4f}"
            )

    # Gate 5: cross-encoder no regression (optional; skipped if mode absent)
    g_ce = True
    if ce is not None and hybrid is not None:
        recall_ok = ce.recall_at_10 >= hybrid.recall_at_10 - _RERANK_TOL
        mrr_ok = ce.mrr >= hybrid.mrr - _RERANK_TOL
        g_ce = recall_ok and mrr_ok
        if not recall_ok:
            failure_reasons.append(
                f"FAIL cross_encoder recall regression: "
                f"ce.recall@10={ce.recall_at_10:.4f} "
                f"< hybrid.recall@10-0.02={hybrid.recall_at_10 - _RERANK_TOL:.4f}"
            )
        if not mrr_ok:
            failure_reasons.append(
                f"FAIL cross_encoder MRR regression: "
                f"ce.mrr={ce.mrr:.4f} "
                f"< hybrid.mrr-0.02={hybrid.mrr - _RERANK_TOL:.4f}"
            )

    passed = g_leak and g_hybrid_bm25 and g_hybrid_vec and g_native and g_ce
    return GateResult(
        passed=passed,
        gate_leakage_zero=g_leak,
        gate_hybrid_ge_bm25=g_hybrid_bm25,
        gate_hybrid_ge_vector=g_hybrid_vec,
        gate_native_rerank_no_regression=g_native,
        gate_cross_encoder_no_regression=g_ce,
        failure_reasons=failure_reasons,
    )


# ---------------------------------------------------------------------------
# write_report()
# ---------------------------------------------------------------------------


def write_report(report: EvalReport, json_path: Path, md_path: Path) -> None:
    """Write EvalReport to JSON and Markdown files.

    Parameters
    ----------
    report:
        The evaluation report to serialize.
    json_path:
        Output path for the JSON file.
    md_path:
        Output path for the Markdown file.
    """
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)

    json_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    md_path.write_text(_render_markdown(report), encoding="utf-8")


def _render_markdown(report: EvalReport) -> str:
    """Render the report as a Markdown document with tables and gate results."""
    gates = evaluate_gates(report)
    gate_icon = "PASS" if gates.passed else "FAIL"

    lines: list[str] = [
        "# Retrieval Evaluation Report",
        "",
        f"**Generated:** {report.generated_at.strftime('%Y-%m-%d %H:%M:%S UTC')}  ",
        f"**Embedding dim:** {report.embedding_dim}  ",
        f"**Cases evaluated:** {len(report.per_case)}  ",
        f"**Cross-chat leakage total:** {report.leakage_total}  ",
        "",
        "## Mode x Metric Summary",
        "",
        "| Mode | Recall@1 | Recall@5 | Recall@10 | MRR | nDCG@10 | Leakage |",
        "|------|----------|----------|-----------|-----|---------|---------|",
    ]

    mode_order = [m for m in EVAL_MODES if m in report.per_mode]
    for mode in mode_order:
        m = report.per_mode[mode]
        lines.append(
            f"| {mode} "
            f"| {m.recall_at_1:.4f} "
            f"| {m.recall_at_5:.4f} "
            f"| {m.recall_at_10:.4f} "
            f"| {m.mrr:.4f} "
            f"| {m.ndcg_at_10:.4f} "
            f"| {m.leakage_count} |"
        )

    lines += [
        "",
        "## Mandatory Gate Results",
        "",
        f"**Overall: {gate_icon}**",
        "",
        "| Gate | Result |",
        "|------|--------|",
        f"| leakage == 0 | {'PASS' if gates.gate_leakage_zero else 'FAIL'} |",
        f"| hybrid >= bm25_only (recall@10) | {'PASS' if gates.gate_hybrid_ge_bm25 else 'FAIL'} |",
        f"| hybrid >= vector_only (recall@10) | {'PASS' if gates.gate_hybrid_ge_vector else 'FAIL'} |",
        f"| native_rerank no regression | {'PASS' if gates.gate_native_rerank_no_regression else 'FAIL'} |",
        f"| cross_encoder no regression | {'PASS' if gates.gate_cross_encoder_no_regression else 'FAIL'} |",
    ]

    if gates.failure_reasons:
        lines += [
            "",
            "### Failure Details",
            "",
        ]
        for reason in gates.failure_reasons:
            lines.append(f"- {reason}")

    lines += [
        "",
        "## Per-Case Results",
        "",
    ]
    for case_result in report.per_case:
        lines += [
            f"### Query: `{case_result.query[:80]}`",
            f"- Chat: `{case_result.chat_id}`",
            f"- Relevant IDs: {case_result.relevant_ids}",
        ]
        if case_result.notes:
            lines.append(f"- Notes: {case_result.notes}")
        lines += [
            "",
            "| Mode | R@1 | R@5 | R@10 | MRR | nDCG@10 | Leakage |",
            "|------|-----|-----|------|-----|---------|---------|",
        ]
        for cm in case_result.per_mode:
            lines.append(
                f"| {cm.mode} "
                f"| {cm.recall_at_1:.3f} "
                f"| {cm.recall_at_5:.3f} "
                f"| {cm.recall_at_10:.3f} "
                f"| {cm.mrr:.3f} "
                f"| {cm.ndcg_at_10:.3f} "
                f"| {cm.leakage} |"
            )
        lines.append("")

    lines += [
        "---",
        "*Report generated by `run_retrieval_eval.py`*",
        "",
    ]
    return "\n".join(lines)


__all__ = [
    "EvalCase",
    "EvalReport",
    "ModeMetrics",
    "CaseResult",
    "CaseMetrics",
    "GateResult",
    "EVAL_MODES",
    "recall_at_k",
    "mean_reciprocal_rank",
    "ndcg_at_k",
    "cross_chat_leakage",
    "evaluate",
    "evaluate_gates",
    "write_report",
]
