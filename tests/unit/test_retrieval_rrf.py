"""Unit tests for Reciprocal Rank Fusion (rrf.py).

Tests verify:
1. RRF formula correctness with known inputs.
2. Multi-ranking fusion score accumulation.
3. Edge cases: empty input, single doc.
4. Result ordering is by fusion_score descending.
5. Fields propagation and score attributes.
"""

from __future__ import annotations

import math

import pytest

from app.retrieval.rrf import RRFInput, reciprocal_rank_fusion


def _hit(doc_id: str, score: float = 1.0, **fields: object) -> RRFInput:
    return RRFInput(vespa_document_id=doc_id, score=score, fields=dict(fields))


def _rrf_score(k: int, rank: int) -> float:
    return 1.0 / (k + rank)


# ===========================================================================
# 1. Empty / trivial
# ===========================================================================


class TestEmptyCases:
    def test_no_rankings_returns_empty(self) -> None:
        assert reciprocal_rank_fusion([]) == []

    def test_single_doc_single_ranking_formula(self) -> None:
        result = reciprocal_rank_fusion([[_hit("doc-1", score=10.0)]])
        assert len(result) == 1
        assert math.isclose(result[0].fusion_score, _rrf_score(k=60, rank=1), rel_tol=1e-9)
        assert result[0].vespa_document_id == "doc-1"


# ===========================================================================
# 2. Known-answer formula correctness
# ===========================================================================


class TestKnownAnswers:
    def test_doc_in_two_rankings_accumulates(self) -> None:
        """Doc at rank 1 in both: score = 2/(k+1)."""
        result = reciprocal_rank_fusion([[_hit("doc-A")], [_hit("doc-A")]], k=60)
        assert len(result) == 1
        assert math.isclose(result[0].fusion_score, 2 * _rrf_score(60, 1), rel_tol=1e-9)

    def test_doc_different_positions_cross_rankings(self) -> None:
        """doc-X rank1/rank2, doc-Y rank2/rank1 → equal scores."""
        r1 = [_hit("doc-X"), _hit("doc-Y")]
        r2 = [_hit("doc-Y"), _hit("doc-X")]
        result = reciprocal_rank_fusion([r1, r2], k=60)
        by_id = {h.vespa_document_id: h for h in result}
        expected_x = _rrf_score(60, 1) + _rrf_score(60, 2)
        expected_y = _rrf_score(60, 2) + _rrf_score(60, 1)
        assert math.isclose(by_id["doc-X"].fusion_score, expected_x, rel_tol=1e-9)
        assert math.isclose(by_id["doc-Y"].fusion_score, expected_y, rel_tol=1e-9)

    def test_doc_only_in_one_ranking(self) -> None:
        r1 = [_hit("doc-A"), _hit("doc-B")]
        r2 = [_hit("doc-A")]
        result = reciprocal_rank_fusion([r1, r2], k=60)
        by_id = {h.vespa_document_id: h for h in result}
        assert math.isclose(by_id["doc-A"].fusion_score, 2 * _rrf_score(60, 1), rel_tol=1e-9)
        assert math.isclose(by_id["doc-B"].fusion_score, _rrf_score(60, 2), rel_tol=1e-9)

    def test_custom_k_value(self) -> None:
        result = reciprocal_rank_fusion([[_hit("doc-A")], [_hit("doc-A")]], k=10)
        assert math.isclose(result[0].fusion_score, 2.0 / (10 + 1), rel_tol=1e-9)


# ===========================================================================
# 3. Ordering
# ===========================================================================


def test_results_sorted_descending() -> None:
    r1 = [_hit("doc-A"), _hit("doc-B")]
    r2 = [_hit("doc-A"), _hit("doc-C")]
    result = reciprocal_rank_fusion([r1, r2], k=60)
    scores = [h.fusion_score for h in result]
    assert scores == sorted(scores, reverse=True)


# ===========================================================================
# 4. Fields propagation + score attributes
# ===========================================================================


class TestFieldsAndScores:
    def test_fields_from_first_ranking_kept(self) -> None:
        r1 = [_hit("doc-1", score=5.0, content="from r1")]
        r2 = [_hit("doc-1", score=3.0, content="from r2")]
        result = reciprocal_rank_fusion([r1, r2])
        assert result[0].fields["content"] == "from r1"

    def test_bm25_vector_scores_settable_default_none(self) -> None:
        result = reciprocal_rank_fusion([[_hit("doc-A")], [_hit("doc-A")]])
        fh = result[0]
        assert fh.bm25_score is None
        assert fh.vector_score is None
        fh.bm25_score = 3.14
        fh.vector_score = 0.97
        assert fh.bm25_score == pytest.approx(3.14)
        assert fh.vector_score == pytest.approx(0.97)
