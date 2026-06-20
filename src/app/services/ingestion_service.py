"""Ingestion feed helper: encodes all enrichment outputs and feeds them to Vespa.

Phase 6.2 scope: this module wires encoders + embedding + VespaFeedClient
together.  It does NOT re-implement parsing or enrichment.

Design contract (CLAUDE.md §6, §12)
-------------------------------------
- ``chat_id`` is injected by the service layer; never passed by the LLM.
- Pure async; no thread/process pool.
- Idempotent: always calls ``delete_by_document`` before feed so re-feeding
  a document replaces old chunks with fresh ones.
- Embedding is produced by the ``EmbeddingProvider`` in a single batch call.
- ``VespaDimensionMismatch`` is propagated to the caller (never silently caught).
"""

from __future__ import annotations

import uuid

from app.enrichment.models import DocumentEnrichment, SectionEnrichment
from app.errors import VespaDimensionMismatch
from app.models.orm import StructuredFact
from app.parsing.figure_narrator import FigureNarration
from app.parsing.models import DocumentNodeOut, ParsedBlock
from app.providers.base import EmbeddingProvider
from app.vespa.encoders import (
    encode_chunks_from_section,
    encode_document_overview,
    encode_figure_narrations,
    encode_raw_blocks,
    encode_section_summary,
    encode_structured_facts,
)
from app.vespa.feed import FeedReport, VespaChunk, VespaFeedClient


async def feed_document(
    chat_id: uuid.UUID,
    document_id: uuid.UUID,
    *,
    blocks: list[ParsedBlock],
    hierarchy: list[DocumentNodeOut],
    section_enrichments: list[SectionEnrichment],
    document_enrichment: DocumentEnrichment | None,
    facts: list[StructuredFact],
    figure_narrations: list[FigureNarration] | None = None,
    embedding_provider: EmbeddingProvider | None = None,
    vespa_client: VespaFeedClient,
) -> FeedReport:
    """Encode, embed, and feed all enrichment outputs for one document to Vespa.

    Workflow
    --------
    1. Delete existing Vespa chunks (idempotent re-feed — old chunks replaced).
    2. Encode all content into ``VespaChunk`` lists (no embeddings yet).
    3. If an embedding provider is supplied, batch-embed all chunk texts.
       Otherwise, feed chunks without embeddings and let Vespa's configured
       embedder compute them internally.

    Parameters
    ----------
    chat_id:
        Chat isolation boundary — injected by the service layer.
    document_id:
        Target document UUID.
    blocks:
        Raw ``ParsedBlock`` list from MinerU mapping.
    hierarchy:
        ``DocumentNodeOut`` list from ``derive_hierarchy``.
    section_enrichments:
        ``SectionEnrichment`` list produced by Phase 5.1.
    document_enrichment:
        ``DocumentEnrichment`` produced by Phase 5.2 (``None`` if not yet done).
    facts:
        ``StructuredFact`` ORM rows belonging to this document.
    embedding_provider:
        Optional provider that converts text → float vectors. Leave ``None``
        to use Vespa's built-in embedder.
    vespa_client:
        Configured ``VespaFeedClient``.

    Returns
    -------
    FeedReport
        Aggregate counts of successes / failures from ``feed_chunks``.

    Raises
    ------
    VespaDimensionMismatch
        If the embedding provider returns vectors of a different length than
        the Vespa schema dimension (``vespa_client._embedding_dim``).
    """
    # Step 1: delete stale Vespa chunks (idempotent)
    await vespa_client.delete_by_document(chat_id, document_id)

    # Step 2: encode all content into VespaChunk instances (no embeddings yet)
    all_chunks: list[VespaChunk] = []

    # 2a. Raw blocks
    all_chunks.extend(encode_raw_blocks(blocks, chat_id, document_id))

    # 2b. Semantic chunks from hierarchy nodes (section / subsection / paragraph)
    for node in hierarchy:
        if node.node_type.value in ("section", "subsection", "paragraph", "abstract"):
            node_chunks = encode_chunks_from_section(node, chat_id, document_id)
            all_chunks.extend(node_chunks)

    # 2c. Section summaries
    for section in section_enrichments:
        all_chunks.extend(encode_section_summary(section, chat_id, document_id))

    # 2d. Document overview + abstract chapter summary
    if document_enrichment is not None:
        all_chunks.extend(encode_document_overview(document_enrichment, chat_id, document_id))

    # 2e. Structured facts
    all_chunks.extend(encode_structured_facts(facts))

    # 2f. Figure / table narrations (multimodal — VLM-generated text per
    # image/table block, embedded together with the same-page text).
    if figure_narrations:
        all_chunks.extend(encode_figure_narrations(figure_narrations))

    if not all_chunks:
        return FeedReport()

    if embedding_provider is not None:
        # Step 3: batch-embed all chunk texts
        texts = [chunk.content for chunk in all_chunks]
        vectors = await embedding_provider.embed(texts)

        # Step 4: validate embedding dimension
        schema_dim = vespa_client._embedding_dim
        if vectors and len(vectors[0]) != schema_dim:
            raise VespaDimensionMismatch(expected=schema_dim, got=len(vectors[0]))

        # Step 5: attach embeddings and feed
        for chunk, vec in zip(all_chunks, vectors, strict=True):
            chunk.embedding = vec

    return await vespa_client.feed_chunks(all_chunks)


__all__ = ["feed_document"]
