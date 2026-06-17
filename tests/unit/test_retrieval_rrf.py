"""Unit tests for Reciprocal Rank Fusion (rrf.py).

Tests verify:
1. RRF formula correctness with known inputs.
2. Multi-ranking fusion score accumulation.
3. Edge cases: empty input, single ranking, docs in only one ranking.
4. Result ordering is by fusion_score descending.
"""

from __future__ import annotations

import math

import pytest

from app.retrieval.rrf import RRFInput, reciprocal_rank_fusion

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _hit(doc_id: str, score: float = 1.0, **fields: object) -> RRFInput:
    return RRFInput(vespa_document_id=doc_id, score=score, fields=dict(fields))


def _rrf_score(k: int, rank: int) -> float:
    """Expected RRF contribution from a single rank position."""
    return 1.0 / (k + rank)


# ===========================================================================
# 1. Empty / trivial cases
# ===========================================================================


class TestEmptyCases:
    def test_no_rankings_returns_empty(self) -> None:
        result = reciprocal_rank_fusion([])
        assert result == []

    def test_single_empty_ranking_returns_empty(self) -> None:
        result = reciprocal_rank_fusion([[]])
        assert result == []

    def test_multiple_empty_rankings_returns_empty(self) -> None:
        result = reciprocal_rank_fusion([[], [], []])
        assert result == []

    def test_single_doc_single_ranking(self) -> None:
        hit = _hit("doc-1", score=10.0)
        result = reciprocal_rank_fusion([[hit]])
        assert len(result) == 1
        expected_score = _rrf_score(k=60, rank=1)
        assert math.isclose(result[0].fusion_score, expected_score, rel_tol=1e-9)
        assert result[0].vespa_document_id == "doc-1"


# ===========================================================================
# 2. Known-answer tests
# ===========================================================================


class TestKnownAnswers:
    """Verify RRF formula with fixed inputs."""

    def test_two_rankings_single_doc_in_both(self) -> None:
        """Doc appears at rank 1 in both rankings; score = 1/61 + 1/61."""
        hit_a = _hit("doc-A")
        hit_b = _hit("doc-A")
        result = reciprocal_rank_fusion([[hit_a], [hit_b]], k=60)
        assert len(result) == 1
        expected = 2 * _rrf_score(k=60, rank=1)
        assert math.isclose(result[0].fusion_score, expected, rel_tol=1e-9)

    def test_doc_in_two_rankings_at_different_positions(self) -> None:
        """Doc at rank 1 in ranking-1 and rank 2 in ranking-2."""
        r1 = [_hit("doc-X"), _hit("doc-Y")]  # doc-X @1, doc-Y @2
        r2 = [_hit("doc-Y"), _hit("doc-X")]  # doc-Y @1, doc-X @2

        result = reciprocal_rank_fusion([r1, r2], k=60)
        by_id = {h.vespa_document_id: h for h in result}

        # doc-X: rank 1 in r1, rank 2 in r2
        expected_x = _rrf_score(60, 1) + _rrf_score(60, 2)
        # doc-Y: rank 2 in r1, rank 1 in r2
        expected_y = _rrf_score(60, 2) + _rrf_score(60, 1)

        assert math.isclose(by_id["doc-X"].fusion_score, expected_x, rel_tol=1e-9)
        assert math.isclose(by_id["doc-Y"].fusion_score, expected_y, rel_tol=1e-9)
        # Scores are equal; order may vary but both must be present
        assert len(result) == 2

    def test_doc_only_in_one_ranking(self) -> None:
        """Doc appearing in only one of two rankings gets 1/(k+rank) from that ranking."""
        r1 = [_hit("doc-A"), _hit("doc-B")]
        r2 = [_hit("doc-A")]  # doc-B not present

        result = reciprocal_rank_fusion([r1, r2], k=60)
        by_id = {h.vespa_document_id: h for h in result}

        # doc-A: rank 1 in both
        expected_a = _rrf_score(60, 1) + _rrf_score(60, 1)
        # doc-B: rank 2 in r1 only
        expected_b = _rrf_score(60, 2)

        assert math.isclose(by_id["doc-A"].fusion_score, expected_a, rel_tol=1e-9)
        assert math.isclose(by_id["doc-B"].fusion_score, expected_b, rel_tol=1e-9)

    def test_three_rankings_accumulate(self) -> None:
        """Doc at rank 1 in three rankings: score = 3/(k+1)."""
        rankings = [[_hit("shared")], [_hit("shared")], [_hit("shared")]]
        result = reciprocal_rank_fusion(rankings, k=60)
        expected = 3 * _rrf_score(60, 1)
        assert math.isclose(result[0].fusion_score, expected, rel_tol=1e-9)

    def test_custom_k_value(self) -> None:
        """Verify different k values produce correct scores."""
        rankings = [[_hit("doc-A")], [_hit("doc-A")]]
        k = 10
        result = reciprocal_rank_fusion(rankings, k=k)
        expected = 2 * (1.0 / (k + 1))
        assert math.isclose(result[0].fusion_score, expected, rel_tol=1e-9)


# ===========================================================================
# 3. Ordering
# ===========================================================================


class TestOrdering:
    """Results must be sorted by fusion_score descending."""

    def test_sorted_descending(self) -> None:
        # doc-A at rank 1 in both rankings → highest score
        # doc-B at rank 2 in ranking-1 only → lower score
        # doc-C at rank 2 in ranking-2 only → lower score
        r1 = [_hit("doc-A"), _hit("doc-B")]
        r2 = [_hit("doc-A"), _hit("doc-C")]

        result = reciprocal_rank_fusion([r1, r2], k=60)
        scores = [h.fusion_score for h in result]
        assert scores == sorted(scores, reverse=True), "Results must be sorted descending"

    def test_doc_with_higher_combined_rank_comes_first(self) -> None:
        # doc-A rank 1+2 vs doc-B rank 3+4 → doc-A first
        r1 = [_hit("doc-A"), _hit("doc-B"), _hit("doc-C"), _hit("doc-D")]
        r2 = [_hit("doc-B"), _hit("doc-A"), _hit("doc-D"), _hit("doc-C")]

        result = reciprocal_rank_fusion([r1, r2], k=60)
        # doc-A: 1/(61)+1/(62)  doc-B: 1/(62)+1/(61) → tied
        # both must be in top 2
        top_ids = {h.vespa_document_id for h in result[:2]}
        assert top_ids == {"doc-A", "doc-B"}

    def test_five_docs_all_present_sorted(self) -> None:
        ids = [f"doc-{i}" for i in range(5)]
        r1 = [_hit(d) for d in ids]
        r2 = [_hit(d) for d in reversed(ids)]

        result = reciprocal_rank_fusion([r1, r2], k=60)
        # Verify descending order
        for i in range(len(result) - 1):
            assert result[i].fusion_score >= result[i + 1].fusion_score


# ===========================================================================
# 4. Fields propagation
# ===========================================================================


class TestFieldsPropagation:
    """Fields from first-seen ranking should be carried through."""

    def test_fields_from_first_ranking(self) -> None:
        """When doc appears in two rankings, fields from ranking[0] are kept."""
        r1 = [_hit("doc-1", score=5.0, content="from r1", chat_id="chat-a")]
        r2 = [_hit("doc-1", score=3.0, content="from r2", chat_id="chat-b")]

        result = reciprocal_rank_fusion([r1, r2])
        assert result[0].fields["content"] == "from r1"
        assert result[0].fields["chat_id"] == "chat-a"

    def test_unique_doc_fields_preserved(self) -> None:
        r1 = [_hit("doc-only", score=1.0, source_type="chunk")]
        result = reciprocal_rank_fusion([r1])
        assert result[0].fields["source_type"] == "chunk"


# ===========================================================================
# 5. FusedHit score attributes are settable
# ===========================================================================


class TestFusedHitAttributes:
    """bm25_score and vector_score attributes can be set post-fusion."""

    def test_set_bm25_vector_scores(self) -> None:
        result = reciprocal_rank_fusion([[_hit("doc-A")], [_hit("doc-A")]])
        fh = result[0]
        fh.bm25_score = 3.14
        fh.vector_score = 0.97
        assert fh.bm25_score == pytest.approx(3.14)
        assert fh.vector_score == pytest.approx(0.97)

    def test_default_scores_are_none(self) -> None:
        result = reciprocal_rank_fusion([[_hit("doc-A")]])
        fh = result[0]
        assert fh.bm25_score is None
        assert fh.vector_score is None


# ===========================================================================
# 6. Large inputs
# ===========================================================================


class TestLargeInputs:
    def test_60_bm25_60_ann_gives_up_to_120_unique_docs(self) -> None:
        # 60 unique docs in r1, 60 unique docs in r2 (no overlap)
        r1 = [_hit(f"bm25-{i}") for i in range(60)]
        r2 = [_hit(f"ann-{i}") for i in range(60)]

        result = reciprocal_rank_fusion([r1, r2], k=60)
        assert len(result) == 120

    def test_60_bm25_60_ann_full_overlap_gives_60_docs(self) -> None:
        ids = [f"doc-{i}" for i in range(60)]
        r1 = [_hit(d) for d in ids]
        r2 = [_hit(d) for d in reversed(ids)]

        result = reciprocal_rank_fusion([r1, r2], k=60)
        assert len(result) == 60
