"""Background MinerU ingestion worker.

This module updates ingestion state, invokes MinerU, persists parsed document
structure/enrichment outputs, and feeds chunks to Vespa.
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import delete, select
from sqlalchemy.orm import selectinload

from app.config import get_settings
from app.db import get_sessionmaker
from app.enrichment.document import enrich_document
from app.enrichment.facts import extract_structured_facts
from app.enrichment.section import enrich_sections
from app.models.orm import (
    Chat,
    Document,
    DocumentNode,
    IngestionJob,
    ProviderProfile,
    StructuredFact,
    Summary,
)
from app.parsing import (
    MinerUClient,
    MinerUServerUnavailable,
    load_middle_json,
    map_middle_to_parsed_blocks,
)
from app.parsing.hierarchy import derive_hierarchy
from app.parsing.models import BBox, HierarchyResult
from app.providers.base import ChatProvider
from app.providers.mock import MockChatProvider
from app.providers.registry import build_chat_provider
from app.services.enrichment_service import (
    persist_document_summaries,
    persist_section_summaries,
)
from app.services.facts_service import persist_facts
from app.services.ingestion_service import feed_document
from app.vespa.feed import VespaFeedClient

logger = logging.getLogger(__name__)


def _now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def _short_error(exc: BaseException) -> str:
    text = f"{exc.__class__.__name__}: {exc}"
    return text[:4000]


def _profile_as_like(profile: ProviderProfile) -> object:
    from types import SimpleNamespace

    return SimpleNamespace(
        provider_type=profile.provider_type,
        model_name=profile.model,
        api_key_encrypted=profile.api_key_encrypted,
        api_key_plain=None,
        base_url=profile.base_url,
        context_window=profile.context_window,
        embedding_dim=None,
        provider_name=profile.name,
    )


async def _load_chat(session: object, chat_id: uuid.UUID) -> Chat | None:
    stmt = (
        select(Chat)
        .where(Chat.id == chat_id)
        .options(
            selectinload(Chat.default_chat_profile),
        )
    )
    return (await session.scalars(stmt)).first()  # type: ignore[attr-defined]


async def _build_chat_provider(
    session: object,
    chat_id: uuid.UUID,
) -> ChatProvider:
    chat = await _load_chat(session, chat_id)

    chat_provider: ChatProvider = MockChatProvider()
    if chat is not None and chat.default_chat_profile is not None:
        try:
            chat_provider = build_chat_provider(  # type: ignore[arg-type]
                _profile_as_like(chat.default_chat_profile)
            )
        except Exception:
            logger.exception("Falling back to mock chat provider for ingestion")

    return chat_provider


def _bbox_to_json(bbox: BBox | None) -> dict[str, float] | None:
    return bbox.model_dump() if bbox is not None else None


async def _replace_document_nodes(
    session: object,
    hierarchy: HierarchyResult,
) -> None:
    await session.execute(  # type: ignore[attr-defined]
        delete(Summary).where(
            Summary.chat_id == hierarchy.chat_id,
            Summary.document_id == hierarchy.document_id,
        )
    )
    await session.execute(  # type: ignore[attr-defined]
        delete(StructuredFact).where(
            StructuredFact.chat_id == hierarchy.chat_id,
            StructuredFact.document_id == hierarchy.document_id,
        )
    )
    await session.execute(  # type: ignore[attr-defined]
        delete(DocumentNode).where(
            DocumentNode.chat_id == hierarchy.chat_id,
            DocumentNode.document_id == hierarchy.document_id,
        )
    )
    await session.flush()  # type: ignore[attr-defined]

    for node in hierarchy.nodes:
        session.add(  # type: ignore[attr-defined]
            DocumentNode(
                id=node.id,
                chat_id=node.chat_id,
                document_id=node.document_id,
                parent_id=node.parent_id,
                node_type=node.node_type.value,
                title=node.title,
                content=node.content,
                page_start=node.page_start,
                page_end=node.page_end,
                order_index=node.order_index,
                level=node.level,
                bbox=_bbox_to_json(node.bbox),
                metadata_=node.metadata_,
            )
        )
    await session.flush()  # type: ignore[attr-defined]


async def _persist_enrichment_and_index(
    *,
    chat_id: uuid.UUID,
    document_id: uuid.UUID,
    middle_json_path: Path,
    page_count: int,
) -> None:
    settings = get_settings()
    sessionmaker = get_sessionmaker()

    middle = load_middle_json(middle_json_path)
    blocks = map_middle_to_parsed_blocks(
        middle,
        chat_id=chat_id,
        document_id=document_id,
    )
    hierarchy = derive_hierarchy(blocks, document_id=document_id, chat_id=chat_id)

    async with sessionmaker() as session:
        doc = await session.get(Document, document_id)
        if doc is not None:
            doc.status = "enriching"
            doc.page_count = page_count
            doc.updated_at = _now()
        await _replace_document_nodes(session, hierarchy)

        chat_provider = await _build_chat_provider(session, chat_id)
        section_enrichments = await enrich_sections(
            hierarchy,
            blocks,
            chat_provider=chat_provider,
        )
        await persist_section_summaries(session, section_enrichments)

        document_enrichment = await enrich_document(
            hierarchy,
            blocks,
            section_enrichments,
            chat_provider=chat_provider,
        )
        abstract_node_id = next(
            (n.id for n in hierarchy.nodes if n.node_type.value == "abstract"),
            None,
        )
        await persist_document_summaries(
            session,
            document_enrichment,
            abstract_node_id=abstract_node_id,
        )

        fact_models = extract_structured_facts(hierarchy, blocks)
        await persist_facts(session, fact_models, current_chat_id=chat_id)
        facts = (
            await session.scalars(
                select(StructuredFact).where(
                    StructuredFact.chat_id == chat_id,
                    StructuredFact.document_id == document_id,
                )
            )
        ).all()

        await session.commit()

    if not settings.vespa_enabled:
        return

    vespa_client = VespaFeedClient(
        endpoint=settings.vespa_endpoint,
        embedding_dim=settings.embedding_dim,
    )
    report = await feed_document(
        chat_id,
        document_id,
        blocks=blocks,
        hierarchy=hierarchy.nodes,
        section_enrichments=section_enrichments,
        document_enrichment=document_enrichment,
        facts=list(facts),
        vespa_client=vespa_client,
    )
    if report.fail_count:
        raise RuntimeError(
            f"Vespa feed failed for {report.fail_count}/{report.total} chunks: "
            f"{'; '.join(report.errors[:3])}"
        )


async def run_mineru_ingestion(
    chat_id: uuid.UUID,
    document_id: uuid.UUID,
    job_id: uuid.UUID,
) -> None:
    """Run MinerU parsing for one uploaded document and update DB state.

    State transitions:
    - ``Document.status``: ``parsing`` → ``parsed`` or ``failed``
    - ``IngestionJob.state``: ``pending`` → ``running`` → ``succeeded`` or ``failed``
    """

    settings = get_settings()
    sessionmaker = get_sessionmaker()

    # Mark the job as running in a short transaction, then release the DB
    # connection while MinerU performs the long-running subprocess work.
    async with sessionmaker() as session:
        job = await session.get(IngestionJob, job_id)
        doc = await session.get(Document, document_id)
        if job is None or doc is None or doc.chat_id != chat_id:
            logger.warning(
                "Skipping MinerU ingestion because job/document is missing",
                extra={
                    "chat_id": str(chat_id),
                    "document_id": str(document_id),
                    "job_id": str(job_id),
                },
            )
            return

        now = _now()
        job.state = "running"
        job.attempt += 1
        job.started_at = now
        job.last_error = None
        doc.status = "parsing"
        doc.updated_at = now
        storage_path = doc.storage_path
        await session.commit()

    try:
        client = MinerUClient(
            server_url=settings.mineru_server_url,
            # Keep MinerU artefacts isolated per document. MinerU appends the
            # PDF basename under this root, so a document-id root avoids cache
            # collisions when different chats upload files with the same name.
            parsed_root=Path(settings.app_data_root) / "parsed" / str(document_id),
        )
        if not await client.health_check():
            raise MinerUServerUnavailable(
                f"MinerU server is unavailable or model is not ready: "
                f"{settings.mineru_server_url}"
            )
        result = await client.parse_pdf(Path(storage_path), document_id=str(document_id))
        await _persist_enrichment_and_index(
            chat_id=chat_id,
            document_id=document_id,
            middle_json_path=result.middle_json_path,
            page_count=result.pages,
        )
    except Exception as exc:  # noqa: BLE001
        error = _short_error(exc)
        logger.exception(
            "MinerU ingestion failed",
            extra={
                "chat_id": str(chat_id),
                "document_id": str(document_id),
                "job_id": str(job_id),
            },
        )
        async with sessionmaker() as session:
            job = await session.get(IngestionJob, job_id)
            doc = await session.get(Document, document_id)
            now = _now()
            if job is not None:
                job.state = "failed"
                job.last_error = error
                job.finished_at = now
            if doc is not None:
                doc.status = "failed"
                doc.updated_at = now
            await session.commit()
        return

    async with sessionmaker() as session:
        job = await session.get(IngestionJob, job_id)
        doc = await session.get(Document, document_id)
        now = _now()
        if job is not None:
            job.state = "succeeded"
            job.last_error = None
            job.finished_at = now
        if doc is not None:
            doc.status = "indexed" if settings.vespa_enabled else "parsed"
            doc.page_count = result.pages
            doc.updated_at = now
        await session.commit()


__all__ = ["run_mineru_ingestion"]
