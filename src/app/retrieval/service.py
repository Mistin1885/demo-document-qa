"""RetrievalService — the ONLY Vespa query entry point for the Paper Notebook Agent.

CLAUDE.md §7 contract
----------------------
- ``chat_id`` filter is ALWAYS injected; it cannot be passed or overridden by the LLM.
- BM25 candidates + ANN candidates are merged via Reciprocal Rank Fusion (RRF).
- Never add un-normalized BM25 / vector scores directly.
- Supports three rerank modes: ``none``, ``native`` (Vespa second-phase), ``cross_encoder``.
- YQL construction is centralised in ``_yql_where()`` which enforces the chat_id invariant.
- No SQL; no retry; no sleep; no cache; httpx timeout 8 s.
- Embedding dimension validated against provider.dimension when an external
  provider is injected; production uses Vespa's native embedder instead.

Chat-id injection guarantee
---------------------------
``_yql_where(chat_id_str, ...)`` is a pure function that ALWAYS embeds
``chat_id contains "<chat_id>"`` in the returned WHERE clause.  Unit tests
verify this directly (``test_retrieval_yql.py``).

YQL injection safety
--------------------
- ``chat_id`` values are UUID-validated (regex) before inclusion.
- ``document_id`` values are UUID-validated.
- ``source_type`` values are validated against the 13-entry whitelist from CLAUDE.md §5.2.
- Characters ``"``, ``\\``, newlines in any string cause ``InvalidRetrievalFilter``.
- The ``query`` text is passed as a separate request parameter (``userQuery()``),
  NEVER concatenated into the YQL string.
"""

from __future__ import annotations

import asyncio
import re
import time
from typing import Any

import httpx
from httpx import AsyncBaseTransport

from app.errors import (
    InvalidRetrievalFilter,
    RerankerUnavailable,
    RetrievalError,
    VespaDimensionMismatch,
)
from app.providers.base import EmbeddingProvider, RerankerProvider
from app.retrieval.models import (
    RetrievalDebug,
    RetrievalRequest,
    RetrievalResponse,
    RetrievalTimings,
    SearchHit,
)
from app.retrieval.rrf import FusedHit, RRFInput, reciprocal_rank_fusion
from app.vespa.feed import VALID_SOURCE_TYPES

# ---------------------------------------------------------------------------
# UUID validation regex
# ---------------------------------------------------------------------------

_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# YQL injection-safe helpers
# ---------------------------------------------------------------------------

_FORBIDDEN_CHARS_RE = re.compile(r'["\\\n\r]')


def _safe_str(value: str, context: str) -> str:
    """Validate *value* is injection-safe for Vespa YQL string literals.

    Raises ``InvalidRetrievalFilter`` if ``value`` contains ``"``, ``\\``,
    or newline characters.
    """
    if _FORBIDDEN_CHARS_RE.search(value):
        raise InvalidRetrievalFilter(
            f"Invalid character in {context}: value contains forbidden characters "
            f"(double-quote, backslash, or newline). value={value!r}"
        )
    return value


def _validate_uuid_str(value: str, context: str) -> str:
    """Validate *value* matches UUID format; raise ``InvalidRetrievalFilter`` otherwise."""
    if not _UUID_RE.match(value):
        raise InvalidRetrievalFilter(
            f"Invalid UUID in {context}: {value!r} is not a valid UUID string."
        )
    return value


def _yql_where(
    chat_id_str: str,
    document_id_strs: list[str] | None = None,
    source_type_strs: list[str] | None = None,
) -> str:
    """Build the YQL WHERE clause with mandatory ``chat_id`` filter.

    This function is the SINGLE path for WHERE-clause construction.
    The ``chat_id contains "<id>"`` predicate is ALWAYS present.

    Parameters
    ----------
    chat_id_str:
        UUID string for the chat.  Must pass UUID regex validation.
    document_id_strs:
        Optional list of document UUID strings to restrict the query.
    source_type_strs:
        Optional list of source type strings (must be in VALID_SOURCE_TYPES).

    Returns
    -------
    str
        YQL WHERE clause fragment, e.g.::

            chat_id contains "abc..." and document_id in ("x","y")

    Raises
    ------
    InvalidRetrievalFilter
        If any value fails UUID validation, source_type whitelist check, or
        injection-safety check.
    """
    # Validate and escape chat_id
    _validate_uuid_str(chat_id_str, "chat_id")
    _safe_str(chat_id_str, "chat_id")

    parts: list[str] = [f'chat_id contains "{chat_id_str}"']

    if document_id_strs:
        for doc_id in document_id_strs:
            _validate_uuid_str(doc_id, "document_id")
            _safe_str(doc_id, "document_id")
        id_list = ", ".join(f'"{d}"' for d in document_id_strs)
        parts.append(f"document_id in ({id_list})")

    if source_type_strs:
        for st in source_type_strs:
            _safe_str(st, "source_type")
            if st not in VALID_SOURCE_TYPES:
                raise InvalidRetrievalFilter(
                    f"source_type {st!r} is not in the allowed whitelist. "
                    f"Allowed: {sorted(VALID_SOURCE_TYPES)}"
                )
        st_list = ", ".join(f'"{s}"' for s in source_type_strs)
        parts.append(f"source_type in ({st_list})")

    return " and ".join(parts)


# ---------------------------------------------------------------------------
# RetrievalService
# ---------------------------------------------------------------------------


class RetrievalService:
    """Single public Vespa query entry point.

    Parameters
    ----------
    endpoint:
        Vespa container base URL (e.g. ``"http://localhost:8080"``).
    embedding_provider:
        EmbeddingProvider used to embed the query.
    reranker_provider:
        Optional RerankerProvider for ``rerank_mode="cross_encoder"``.
    application:
        Vespa namespace / application name (default ``"default"``).
    schema:
        Vespa document type name (default ``"document_chunk"``).
    cluster:
        Vespa content cluster name (default ``"documents"``).
    embedding_dim:
        Expected embedding dimension; validated against
        ``embedding_provider.dimension``. Defaults to 384, matching the bundled
        Vespa E5 native embedder schema.
    """

    _SEARCH_TIMEOUT: float = 8.0

    def __init__(
        self,
        endpoint: str,
        embedding_provider: EmbeddingProvider | None = None,
        reranker_provider: RerankerProvider | None = None,
        application: str = "default",
        schema: str = "document_chunk",
        cluster: str = "documents",
        embedding_dim: int = 384,
        http_transport: AsyncBaseTransport | None = None,
    ) -> None:
        self._endpoint = endpoint.rstrip("/")
        self._embedding_provider = embedding_provider
        self._reranker_provider = reranker_provider
        self._application = application
        self._schema = schema
        self._cluster = cluster
        self._embedding_dim = embedding_dim
        self._search_url = f"{self._endpoint}/search/"
        self._http_transport = http_transport

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def search(self, req: RetrievalRequest) -> RetrievalResponse:
        """Execute hybrid BM25 + ANN search with RRF fusion and optional rerank.

        This is the ONLY public method.  ``req.chat_id`` is injected from the
        service layer (AgentState); the LLM cannot supply it.

        Parameters
        ----------
        req:
            Fully-constructed ``RetrievalRequest`` with chat_id already set.

        Returns
        -------
        RetrievalResponse
            ``hits`` list with per-stage scores; ``debug`` if ``req.debug``.

        Raises
        ------
        VespaDimensionMismatch
            If ``embedding_provider.dimension != embedding_dim``.
        RerankerUnavailable
            If ``rerank_mode="cross_encoder"`` but no reranker provider configured.
        RetrievalError
            If Vespa returns a non-200 status.
        InvalidRetrievalFilter
            If any filter value fails validation (UUID, whitelist, injection check).
        """
        t_start = time.monotonic()

        # --- Validate embedding dim ---
        if (
            self._embedding_provider is not None
            and self._embedding_provider.dimension != self._embedding_dim
        ):
            raise VespaDimensionMismatch(
                expected=self._embedding_dim,
                got=self._embedding_provider.dimension,
            )

        # --- Prepare string forms of filters ---
        chat_id_str = str(req.chat_id)
        doc_id_strs = [str(d) for d in req.document_ids] if req.document_ids else None
        src_types = req.source_types or None

        # Build WHERE clause (validates all values; raises InvalidRetrievalFilter)
        base_where = _yql_where(chat_id_str, doc_id_strs, src_types)

        debug_queries: list[str] = []

        # --- Step 1: Prepare query embedding ---
        t_embed_start = time.monotonic()
        qvec: list[float] | None = None
        if self._embedding_provider is not None:
            embeddings = await self._embedding_provider.embed([req.query])
            qvec = embeddings[0]
        embed_ms = (time.monotonic() - t_embed_start) * 1000.0

        # Tensor body for Vespa. If no Python embedding provider is configured,
        # ask Vespa's configured embedder to produce query(qvec) internally.
        tensor_values: dict[str, object] | str
        if qvec is None:
            tensor_values = "embed(e5, @user_query)"
        else:
            tensor_values = {"values": qvec}

        # --- Step 2: BM25 and/or ANN queries based on retrieval_mode ---
        bm25_yql = (
            f"select * from sources {self._schema} "
            f"where userQuery() and {base_where} "
            f"limit {req.bm25_top_k}"
        )
        ann_yql = (
            f"select * from sources {self._schema} "
            f"where {{targetHits:{req.ann_top_k}}}nearestNeighbor(embedding, qvec) "
            f"and {base_where} "
            f"limit {req.ann_top_k}"
        )

        bm25_body: dict[str, object] = {
            "yql": bm25_yql,
            "query": req.query,
            "user_query": req.query,
            "hits": req.bm25_top_k,
            "ranking.profile": "bm25_only",
            "input.query(qvec)": tensor_values,
        }
        ann_body: dict[str, object] = {
            "yql": ann_yql,
            "user_query": req.query,
            "hits": req.ann_top_k,
            "ranking.profile": "semantic_only",
            "input.query(qvec)": tensor_values,
        }

        t_bm25_start = time.monotonic()
        t_ann_start = time.monotonic()

        run_bm25 = req.retrieval_mode in ("hybrid", "bm25_only")
        run_ann = req.retrieval_mode in ("hybrid", "vector_only")

        if req.debug:
            if run_bm25:
                debug_queries.append(bm25_yql)
            if run_ann:
                debug_queries.append(ann_yql)

        _empty_vespa: dict[str, Any] = {"root": {"children": []}}

        async with httpx.AsyncClient(
            timeout=self._SEARCH_TIMEOUT,
            transport=self._http_transport,
        ) as client:
            if run_bm25 and run_ann:
                bm25_task = asyncio.create_task(self._post_search(client, bm25_body))
                ann_task = asyncio.create_task(self._post_search(client, ann_body))
                bm25_raw, ann_raw = await asyncio.gather(bm25_task, ann_task)
            elif run_bm25:
                bm25_raw = await self._post_search(client, bm25_body)
                ann_raw = _empty_vespa
            else:  # vector_only
                bm25_raw = _empty_vespa
                ann_raw = await self._post_search(client, ann_body)

        bm25_ms = (time.monotonic() - t_bm25_start) * 1000.0
        ann_ms = (time.monotonic() - t_ann_start) * 1000.0

        bm25_hits = self._parse_vespa_hits(bm25_raw)
        ann_hits = self._parse_vespa_hits(ann_raw)

        # --- Step 3: RRF fusion (or direct ranking for single-mode) ---
        t_fusion_start = time.monotonic()

        if req.retrieval_mode == "bm25_only":
            # No fusion; use BM25 hits directly, sorted by relevance
            bm25_rrf = [
                RRFInput(
                    vespa_document_id=h["vespa_document_id"],
                    score=h.get("relevance", 0.0),
                    fields=h,
                )
                for h in bm25_hits
            ]
            fused = reciprocal_rank_fusion([bm25_rrf])
            for fh in fused:
                fh.bm25_score = fh.fusion_score
        elif req.retrieval_mode == "vector_only":
            # No fusion; use ANN hits directly
            ann_rrf = [
                RRFInput(
                    vespa_document_id=h["vespa_document_id"],
                    score=h.get("relevance", 0.0),
                    fields=h,
                )
                for h in ann_hits
            ]
            fused = reciprocal_rank_fusion([ann_rrf])
            for fh in fused:
                fh.vector_score = fh.fusion_score
        else:
            # Full hybrid: BM25 + ANN → RRF
            bm25_rrf = [
                RRFInput(
                    vespa_document_id=h["vespa_document_id"],
                    score=h.get("relevance", 0.0),
                    fields=h,
                )
                for h in bm25_hits
            ]
            ann_rrf = [
                RRFInput(
                    vespa_document_id=h["vespa_document_id"],
                    score=h.get("relevance", 0.0),
                    fields=h,
                )
                for h in ann_hits
            ]
            fused = reciprocal_rank_fusion([bm25_rrf, ann_rrf])
            bm25_score_by_id = {
                h["vespa_document_id"]: h.get("relevance", 0.0) for h in bm25_hits
            }
            ann_score_by_id = {
                h["vespa_document_id"]: h.get("relevance", 0.0) for h in ann_hits
            }
            for fh in fused:
                fh.bm25_score = bm25_score_by_id.get(fh.vespa_document_id)
                fh.vector_score = ann_score_by_id.get(fh.vespa_document_id)

        fusion_ms = (time.monotonic() - t_fusion_start) * 1000.0

        # Truncate to fusion_top_k
        fused = fused[: req.fusion_top_k]

        # --- Step 4: Rerank ---
        t_rerank_start = time.monotonic()
        final_hits: list[SearchHit]
        if req.rerank_mode == "none":
            final_hits = self._fused_to_search_hits(fused[: req.final_top_k])
            rerank_ms = 0.0
        elif req.rerank_mode == "native":
            final_hits, debug_queries, rerank_ms = await self._native_rerank(
                fused=fused,
                req=req,
                tensor_values=tensor_values,
                debug_queries=debug_queries,
            )
        elif req.rerank_mode == "cross_encoder":
            if self._reranker_provider is None:
                raise RerankerUnavailable(
                    "rerank_mode='cross_encoder' requires a RerankerProvider, but none was configured."
                )
            final_hits, rerank_ms = await self._cross_encoder_rerank(
                fused=fused,
                req=req,
            )
        else:
            # Should never reach here due to Literal type enforcement
            raise RetrievalError(f"Unknown rerank_mode: {req.rerank_mode!r}")  # pragma: no cover

        rerank_ms = (time.monotonic() - t_rerank_start) * 1000.0
        total_ms = (time.monotonic() - t_start) * 1000.0

        # Assign final_rank (1-based)
        for i, hit in enumerate(final_hits, start=1):
            hit.final_rank = i

        debug: RetrievalDebug | None = None
        if req.debug:
            debug = RetrievalDebug(
                bm25_hits_count=len(bm25_hits),
                ann_hits_count=len(ann_hits),
                fused_hits_count=len(fused),
                after_rerank_count=len(final_hits),
                timings_ms=RetrievalTimings(
                    embed_ms=embed_ms,
                    bm25_ms=bm25_ms,
                    ann_ms=ann_ms,
                    fusion_ms=fusion_ms,
                    rerank_ms=rerank_ms,
                    total_ms=total_ms,
                ),
                queries=debug_queries,
            )

        return RetrievalResponse(hits=final_hits, debug=debug)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _post_search(
        self, client: httpx.AsyncClient, body: dict[str, object]
    ) -> dict[str, Any]:
        """POST a single search request to Vespa; return the parsed JSON body."""
        resp = await client.post(self._search_url, json=body)
        if resp.status_code != 200:
            raise RetrievalError(
                f"Vespa search returned HTTP {resp.status_code}: {resp.text[:300]}"
            )
        return resp.json()  # type: ignore[no-any-return]

    def _parse_vespa_hits(self, response: dict[str, Any]) -> list[dict[str, Any]]:
        """Extract the list of hit field-dicts from a Vespa /search/ response.

        Vespa search response structure::

            {
              "root": {
                "children": [
                  {
                    "id": "id:default:document_chunk::...",
                    "relevance": 12.3,
                    "fields": { "vespa_document_id": "...", "content": "...", ... }
                  },
                  ...
                ]
              }
            }

        Returns a flat list of dicts where each dict is ``fields`` merged with
        ``{"relevance": <float>}``.
        """
        root = response.get("root", {})
        children = root.get("children", [])
        hits: list[dict[str, Any]] = []
        for child in children:
            fields: dict[str, Any] = child.get("fields", {})
            fields["relevance"] = child.get("relevance", 0.0)
            # Ensure vespa_document_id is present (may come from fields or id)
            if "vespa_document_id" not in fields:
                vespa_id = child.get("id", "")
                # Vespa id format: "id:<ns>:<schema>::<docid>"
                if "::" in vespa_id:
                    fields["vespa_document_id"] = vespa_id.split("::")[-1]
            hits.append(fields)
        return hits

    def _fused_to_search_hits(self, fused: list[FusedHit]) -> list[SearchHit]:
        """Convert a list of FusedHit to SearchHit (no rerank scores)."""
        hits: list[SearchHit] = []
        for fh in fused:
            hit = self._fields_to_search_hit(fh.fields)
            hit.bm25_score = fh.bm25_score
            hit.vector_score = fh.vector_score
            hit.fusion_score = fh.fusion_score
            hit.final_score = fh.fusion_score
            hits.append(hit)
        return hits

    def _fields_to_search_hit(self, fields: dict[str, Any]) -> SearchHit:
        """Map a Vespa fields dict to a SearchHit model."""
        return SearchHit(
            vespa_document_id=str(fields.get("vespa_document_id", "")),
            chat_id=str(fields.get("chat_id", "")),
            document_id=str(fields.get("document_id", "")),
            source_node_id=str(fields.get("source_node_id", "")),
            parent_node_id=fields.get("parent_node_id"),
            source_type=str(fields.get("source_type", "")),
            title=fields.get("title") or None,
            heading_path=fields.get("heading_path") or None,
            content=str(fields.get("content", "")),
            page_start=int(fields.get("page_start", 1)),
            page_end=int(fields.get("page_end", 1)),
            order_index=int(fields.get("order_index", 0)),
        )

    async def _native_rerank(
        self,
        fused: list[FusedHit],
        req: RetrievalRequest,
        tensor_values: dict[str, object] | str,
        debug_queries: list[str],
    ) -> tuple[list[SearchHit], list[str], float]:
        """Fire a third Vespa query using ``hybrid_with_native_rerank`` on fused doc IDs.

        Returns (final_hits, updated_debug_queries, rerank_ms).
        """
        rerank_candidates = fused[: req.rerank_top_k]
        if not rerank_candidates:
            return [], debug_queries, 0.0

        # Build a YQL that fetches only these specific vespa_document_ids
        # with the native rerank profile
        id_list = ", ".join(f'"{h.vespa_document_id}"' for h in rerank_candidates)
        chat_id_str = str(req.chat_id)
        _validate_uuid_str(chat_id_str, "chat_id (native rerank)")

        rerank_yql = (
            f"select * from sources {self._schema} "
            f"where vespa_document_id in ({id_list}) "
            f'and chat_id contains "{chat_id_str}" '
            f"limit {req.rerank_top_k}"
        )

        if req.debug:
            debug_queries = debug_queries + [rerank_yql]

        rerank_body: dict[str, object] = {
            "yql": rerank_yql,
            "user_query": req.query,
            "hits": req.rerank_top_k,
            "ranking.profile": "hybrid_with_native_rerank",
            "input.query(qvec)": tensor_values,
        }

        t_start = time.monotonic()
        async with httpx.AsyncClient(
            timeout=self._SEARCH_TIMEOUT,
            transport=self._http_transport,
        ) as client:
            rerank_raw = await self._post_search(client, rerank_body)
        rerank_ms = (time.monotonic() - t_start) * 1000.0

        rerank_hits = self._parse_vespa_hits(rerank_raw)
        native_score_by_id = {h["vespa_document_id"]: h.get("relevance", 0.0) for h in rerank_hits}

        # Also capture pre-existing fused scores indexed by doc id
        fused_by_id = {fh.vespa_document_id: fh for fh in rerank_candidates}

        # Build final hits sorted by native_rerank_score (desc)
        # Use only those that came back from the rerank query
        hits_with_scores: list[tuple[float, FusedHit]] = []
        for doc_id, native_score in native_score_by_id.items():
            if doc_id in fused_by_id:
                hits_with_scores.append((native_score, fused_by_id[doc_id]))

        # Sort by native score descending, then take final_top_k
        hits_with_scores.sort(key=lambda t: t[0], reverse=True)
        hits_with_scores = hits_with_scores[: req.final_top_k]

        final_hits: list[SearchHit] = []
        for native_score, fh in hits_with_scores:
            # Look up the full fields from rerank response
            rerank_fields_map = {h["vespa_document_id"]: h for h in rerank_hits}
            fields = rerank_fields_map.get(fh.vespa_document_id, fh.fields)
            hit = self._fields_to_search_hit(fields)
            hit.bm25_score = fh.bm25_score
            hit.vector_score = fh.vector_score
            hit.fusion_score = fh.fusion_score
            hit.native_rerank_score = native_score
            hit.final_score = native_score
            final_hits.append(hit)

        return final_hits, debug_queries, rerank_ms

    async def _cross_encoder_rerank(
        self,
        fused: list[FusedHit],
        req: RetrievalRequest,
    ) -> tuple[list[SearchHit], float]:
        """Call the external RerankerProvider on the top-rerank_top_k fused hits.

        Returns (final_hits, rerank_ms).
        """
        rerank_candidates = fused[: req.rerank_top_k]
        if not rerank_candidates:
            return [], 0.0

        # Extract content text for the reranker
        contents = [str(fh.fields.get("content", "")) for fh in rerank_candidates]

        assert self._reranker_provider is not None  # caller already checked

        t_start = time.monotonic()
        scores = await self._reranker_provider.score(req.query, contents)
        rerank_ms = (time.monotonic() - t_start) * 1000.0

        # Pair up
        scored: list[tuple[float, FusedHit]] = list(zip(scores, rerank_candidates, strict=True))
        scored.sort(key=lambda t: t[0], reverse=True)
        scored = scored[: req.final_top_k]

        final_hits: list[SearchHit] = []
        for ce_score, fh in scored:
            hit = self._fields_to_search_hit(fh.fields)
            hit.bm25_score = fh.bm25_score
            hit.vector_score = fh.vector_score
            hit.fusion_score = fh.fusion_score
            hit.cross_encoder_score = ce_score
            hit.final_score = ce_score
            final_hits.append(hit)

        return final_hits, rerank_ms
