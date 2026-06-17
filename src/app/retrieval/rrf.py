"""Reciprocal Rank Fusion (RRF) for combining multiple ranked lists.

CLAUDE.md §7:
- First-version fusion uses RRF (Reciprocal Rank Fusion).
- Never directly add un-normalized BM25 + vector scores.

Formula: RRF(d) = sum over rankings: 1 / (k + rank(d))
where rank is 1-based.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RRFInput:
    """A single hit from one ranking list.

    Parameters
    ----------
    vespa_document_id:
        Unique document identifier used to align hits across rankings.
    score:
        The original score from this ranking (kept for debug/attribution).
    fields:
        All other fields from the Vespa response (content, chat_id, etc.).
    """

    vespa_document_id: str
    score: float
    fields: dict[str, object] = field(default_factory=dict)


@dataclass
class FusedHit:
    """A hit after RRF fusion with accumulated score and source attribution."""

    vespa_document_id: str
    fusion_score: float
    fields: dict[str, object] = field(default_factory=dict)

    # Per-ranking scores for debug (populated by caller after fusion)
    bm25_score: float | None = None
    vector_score: float | None = None


def reciprocal_rank_fusion(
    rankings: list[list[RRFInput]],
    k: int = 60,
) -> list[FusedHit]:
    """Compute Reciprocal Rank Fusion over multiple ranked lists.

    Each list in ``rankings`` is a ranked sequence of ``RRFInput`` objects.
    Hits are identified by ``vespa_document_id`` across lists.

    Formula
    -------
    For each document ``d`` and each ranking ``r`` in which ``d`` appears at
    position ``rank_r(d)`` (1-based)::

        fusion_score(d) = sum_r [ 1 / (k + rank_r(d)) ]

    The returned list is sorted by ``fusion_score`` descending.

    Parameters
    ----------
    rankings:
        Each inner list is one complete ranking (e.g., BM25 hits, ANN hits).
        Order within each list is rank 1, 2, 3, … (best first).
    k:
        RRF smoothing constant (default 60 per literature).

    Returns
    -------
    list[FusedHit]
        Merged and sorted list with ``fusion_score`` populated.
        An empty input returns an empty list.

    Notes
    -----
    - If a document appears in multiple rankings, its ``fields`` are taken
      from the first ranking where it is encountered (lexicographic by list
      index).  The caller should merge field data if needed.
    - The result is deterministic for identical inputs.
    """
    if not rankings:
        return []

    # Accumulate RRF scores and first-seen fields
    scores: dict[str, float] = {}
    fields_map: dict[str, dict[str, object]] = {}

    for ranking in rankings:
        for rank_idx, hit in enumerate(ranking, start=1):
            doc_id = hit.vespa_document_id
            contribution = 1.0 / (k + rank_idx)
            scores[doc_id] = scores.get(doc_id, 0.0) + contribution
            # Keep fields from first-seen ranking
            if doc_id not in fields_map:
                fields_map[doc_id] = hit.fields

    # Build FusedHit list sorted by fusion_score descending
    fused: list[FusedHit] = []
    for doc_id, fusion_score in scores.items():
        fused.append(
            FusedHit(
                vespa_document_id=doc_id,
                fusion_score=fusion_score,
                fields=fields_map[doc_id],
            )
        )

    fused.sort(key=lambda h: h.fusion_score, reverse=True)
    return fused
