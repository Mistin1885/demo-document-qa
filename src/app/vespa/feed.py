"""Vespa feed/delete service for the Paper Notebook Agent (Phase 6.2).

All document chunks are fed via the Vespa Document REST API:
  POST /document/v1/{namespace}/{schema}/docid/{vespa_document_id}

Delete-by-document uses the selection visit/delete endpoint:
  DELETE /document/v1/{namespace}/{schema}/docid?selection=...&cluster=documents

Design contract (CLAUDE.md §6.2)
---------------------------------
- ``chat_id`` is NEVER accepted from external callers; it is always injected by
  the service layer.
- Pure async (httpx.AsyncClient); no thread/process pool.
- Embedding tensor encoded as Vespa indexed tensor: ``{"values": [...]}``
  when supplied. Production ingestion omits it so Vespa's native embedder
  computes the tensor during document processing.
- Feed is idempotent: POST to the docid endpoint replaces/upserts by
  ``vespa_document_id``.
- ``VespaFeedClient`` satisfies the ``VespaIndexer`` Protocol via structural
  subtyping (no inheritance needed).
- Dimension mismatch raises ``VespaDimensionMismatch`` — no silent truncation.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime

import httpx
from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.errors import VespaDimensionMismatch

# ---------------------------------------------------------------------------
# Source type literal (14 values — CLAUDE.md §5.2)
# ---------------------------------------------------------------------------

SourceType = str
"""
One of:
  raw_block, chunk, section_summary, compact_section_summary,
  chapter_summary, compact_chapter_summary, document_overview,
  technology_card, claim, definition, performance_fact,
  table_record, figure_caption.
  (Also: abstract_summary — persisted as chapter_summary in Vespa.)
"""

VALID_SOURCE_TYPES: frozenset[str] = frozenset(
    [
        "raw_block",
        "chunk",
        "section_summary",
        "compact_section_summary",
        "chapter_summary",
        "compact_chapter_summary",
        "document_overview",
        "technology_card",
        "claim",
        "definition",
        "performance_fact",
        "table_record",
        "figure_caption",
    ]
)

# ---------------------------------------------------------------------------
# VespaChunk — flat data container for one Vespa document
# ---------------------------------------------------------------------------


class VespaChunk(BaseModel):
    """Flat representation of one Vespa document chunk.

    Maps exactly to the ``document_chunk.sd`` schema fields.  The ``embedding``
    field must have exactly ``embedding_dim`` floats (validated at feed time).

    Design rules
    ------------
    - ``extra="forbid"`` prevents silent data loss.
    - No ``dict[str, Any]`` — every sub-structure is typed.
    - ``vespa_document_id`` is a deterministic UUID-5 (see ``encoders.make_vespa_id``).
    """

    model_config = ConfigDict(extra="forbid")

    vespa_document_id: str
    """Deterministic UUID-5 string: ``uuid5(NAMESPACE_OID, f"{document_id}:{source_type}:{source_node_id}:{order_index}")``.

    Used as the Vespa document ``docid``; guarantees upsert idempotency.
    """

    chat_id: str
    document_id: str
    source_node_id: str
    parent_node_id: str | None = None

    source_type: str
    """One of the 14 ``VALID_SOURCE_TYPES``."""

    title: str = ""
    heading_path: str = ""
    content: str
    embedding_content: str = ""
    """Search/embedding text. Defaults to ``content``.

    For tables, ``content`` intentionally remains the raw HTML evidence shown
    to the answer generator, while ``embedding_content`` blends same-page text,
    caption, and a natural-language table description to make semantic
    retrieval hit table chunks.
    """

    keywords: list[str] = Field(default_factory=list)
    technical_keywords: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)

    page_start: int = Field(ge=1, default=1)
    page_end: int = Field(ge=1, default=1)
    order_index: int = Field(ge=0, default=0)
    token_count: int = Field(ge=0, default=0)

    embedding: list[float] = Field(default_factory=list)
    """Optional precomputed embedding vector.

    When empty, the feed payload omits ``embedding`` so Vespa's configured
    embedder can compute it during document processing.
    """

    created_at: int = Field(default=0)
    """Unix epoch milliseconds (long in Vespa schema)."""

    @model_validator(mode="after")
    def _default_embedding_content(self) -> VespaChunk:
        if not self.embedding_content:
            self.embedding_content = self.content
        return self


# ---------------------------------------------------------------------------
# FeedReport
# ---------------------------------------------------------------------------


@dataclass
class FeedReport:
    """Summary of a ``feed_chunks`` operation."""

    success_count: int = 0
    fail_count: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        return self.success_count + self.fail_count


# ---------------------------------------------------------------------------
# VespaFeedClient
# ---------------------------------------------------------------------------


class VespaFeedClient:
    """Async Vespa feed/delete client using the Document REST API.

    This class satisfies ``VespaIndexer`` via structural subtyping.

    Parameters
    ----------
    endpoint:
        Vespa container URL, e.g. ``http://localhost:8080``.
    application_name:
        Vespa application/namespace name (default ``"default"``).
    schema_name:
        Vespa document type (default ``"document_chunk"``).
    embedding_dim:
        Expected embedding dimension.  Feed will raise
        ``VespaDimensionMismatch`` if any chunk carries a different size.
    cluster:
        Vespa content cluster name used in selection-based delete queries
        (default ``"documents"``).
    feed_timeout:
        Per-request timeout in seconds for feed PUT operations (default 5 s).
    delete_timeout:
        Timeout for selection-based DELETE requests (default 60 s).
    """

    def __init__(
        self,
        endpoint: str = "http://localhost:8080",
        application_name: str = "default",
        schema_name: str = "document_chunk",
        embedding_dim: int = 384,
        cluster: str = "documents",
        feed_timeout: float = 5.0,
        delete_timeout: float = 60.0,
    ) -> None:
        self._endpoint = endpoint.rstrip("/")
        self._namespace = application_name
        self._schema = schema_name
        self._embedding_dim = embedding_dim
        self._cluster = cluster
        self._feed_timeout = feed_timeout
        self._delete_timeout = delete_timeout

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _doc_base_url(self) -> str:
        return f"{self._endpoint}/document/v1/{self._namespace}/{self._schema}/docid"

    def _chunk_to_fields(self, chunk: VespaChunk) -> dict[str, object]:
        """Convert a ``VespaChunk`` to the Vespa ``{"fields": {...}}`` payload.

        When a precomputed embedding exists, its tensor value is encoded as
        ``{"values": [...]}``.
        Production ingestion omits ``embedding`` and lets Vespa's indexing
        expression compute it from ``content`` with the native embedder.
        """
        raw_fields: dict[str, object] = {
            "vespa_document_id": chunk.vespa_document_id,
            "chat_id": chunk.chat_id,
            "document_id": chunk.document_id,
            "source_node_id": chunk.source_node_id,
            "source_type": chunk.source_type,
            "title": chunk.title,
            "heading_path": chunk.heading_path,
            "content": chunk.content,
            "embedding_content": chunk.embedding_content,
            "keywords": chunk.keywords,
            "technical_keywords": chunk.technical_keywords,
            "entities": chunk.entities,
            "page_start": chunk.page_start,
            "page_end": chunk.page_end,
            "order_index": chunk.order_index,
            "token_count": chunk.token_count,
            "created_at": chunk.created_at,
        }
        if chunk.parent_node_id is not None:
            raw_fields["parent_node_id"] = chunk.parent_node_id
        if chunk.embedding:
            raw_fields["embedding"] = {"values": chunk.embedding}
        return {"fields": raw_fields}

    def _validate_embedding_dim(self, chunk: VespaChunk) -> None:
        """Raise ``VespaDimensionMismatch`` if ``chunk.embedding`` length is wrong."""
        got = len(chunk.embedding)
        if got != 0 and got != self._embedding_dim:
            raise VespaDimensionMismatch(expected=self._embedding_dim, got=got)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def feed_chunks(self, chunks: list[VespaChunk]) -> FeedReport:
        """POST each chunk to Vespa.  Validates embedding dims before sending.

        Each POST is a Vespa put keyed on ``vespa_document_id``, so feeding
        the same chunk twice produces exactly one Vespa document.

        Parameters
        ----------
        chunks:
            List of ``VespaChunk`` instances to feed.

        Returns
        -------
        FeedReport
            Counts of successes, failures, and error messages.

        Raises
        ------
        VespaDimensionMismatch
            If any chunk's embedding length differs from ``self._embedding_dim``.
        """
        # Validate all dims up front (fail-fast before any network I/O)
        for chunk in chunks:
            self._validate_embedding_dim(chunk)

        report = FeedReport()
        async with httpx.AsyncClient(timeout=self._feed_timeout) as client:
            for chunk in chunks:
                url = f"{self._doc_base_url()}/{chunk.vespa_document_id}"
                body = self._chunk_to_fields(chunk)
                try:
                    resp = await client.post(url, json=body)
                    if resp.status_code in (200, 201):
                        report.success_count += 1
                    else:
                        report.fail_count += 1
                        report.errors.append(
                            f"POST {chunk.vespa_document_id} → {resp.status_code}: {resp.text[:200]}"
                        )
                except httpx.HTTPError as exc:
                    report.fail_count += 1
                    report.errors.append(f"POST {chunk.vespa_document_id} → error: {exc}")

        return report

    async def delete_by_document(
        self,
        chat_id: uuid.UUID,
        document_id: uuid.UUID,
    ) -> int:
        """Delete all Vespa chunks that match ``chat_id`` AND ``document_id``.

        Uses the Vespa selection visit DELETE endpoint:
          DELETE /document/v1/{ns}/{schema}/docid
            ?selection=document_chunk.chat_id%3D%3D%22...%22+AND+...
            &cluster=documents

        Returns
        -------
        int
            Number of documents reported deleted by Vespa (best-effort; Vespa
            may not always return an exact count for large deletions).

        Notes
        -----
        - Idempotent: calling this for a non-existent document is a no-op.
        - ``chat_id`` is injected by the service layer, never by the caller.
        """
        # Build selection string with explicit field-qualified names.
        # Vespa YQL selection syntax for visitor: use == with quotes.
        chat_id_str = str(chat_id)
        document_id_str = str(document_id)
        selection = (
            f'document_chunk.chat_id=="{chat_id_str}" '
            f'AND document_chunk.document_id=="{document_id_str}"'
        )
        params = {
            "selection": selection,
            "cluster": self._cluster,
        }
        url = self._doc_base_url()
        try:
            async with httpx.AsyncClient(timeout=self._delete_timeout) as client:
                resp = await client.delete(url, params=params)
                if resp.status_code in (200, 204):
                    # Vespa may return {"documentCount": N} or just 204
                    try:
                        data = resp.json()
                        return int(data.get("documentCount", 0))
                    except Exception:
                        return 0
                # 404 is acceptable (no documents existed)
                if resp.status_code == 404:
                    return 0
                # Any other status: log but don't raise (idempotent contract)
                return 0
        except httpx.HTTPError:
            return 0

    async def health_check(self) -> bool:
        """Return ``True`` if Vespa's config/application status endpoint responds 200."""
        url = f"{self._endpoint}/ApplicationStatus"
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(url)
                return resp.status_code == 200
        except httpx.HTTPError:
            return False


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------


def make_feed_client_from_settings() -> VespaFeedClient:
    """Construct a ``VespaFeedClient`` from ``app.config.get_settings()``.

    Import deferred to avoid circular imports at module load time.
    """
    from app.config import get_settings

    s = get_settings()
    return VespaFeedClient(
        endpoint=s.vespa_endpoint,
        embedding_dim=s.embedding_dim,
    )


# ---------------------------------------------------------------------------
# Deterministic vespa_document_id helper (shared with encoders)
# ---------------------------------------------------------------------------


def make_vespa_id(
    document_id: uuid.UUID,
    source_type: str,
    source_node_id: uuid.UUID | str,
    order_index: int,
) -> str:
    """Return a deterministic UUID-5 string suitable for a Vespa ``docid``.

    Namespace: ``uuid.NAMESPACE_OID``
    Name:      ``"{document_id}:{source_type}:{source_node_id}:{order_index}"``

    This guarantees that:
    - The same logical chunk always produces the same ``vespa_document_id``.
    - PUT operations are idempotent upserts.
    """
    name = f"{document_id}:{source_type}:{source_node_id}:{order_index}"
    return str(uuid.uuid5(uuid.NAMESPACE_OID, name))


def _now_epoch_ms() -> int:
    """Return current UTC time as epoch milliseconds (for ``created_at`` long field)."""
    return int(datetime.now(tz=UTC).timestamp() * 1000)


__all__ = [
    "VALID_SOURCE_TYPES",
    "FeedReport",
    "SourceType",
    "VespaChunk",
    "VespaFeedClient",
    "make_feed_client_from_settings",
    "make_vespa_id",
]
