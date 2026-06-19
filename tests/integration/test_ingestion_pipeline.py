"""Integration tests for the post-MinerU ingestion pipeline."""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest
from sqlalchemy import delete, func, select

from app.db import get_sessionmaker
from app.models.orm import Chat, Document, DocumentNode, StructuredFact, Summary
from app.services import ingestion_worker
from app.vespa.feed import FeedReport


@pytest.mark.asyncio
async def test_post_mineru_pipeline_persists_nodes_enrichment_and_feeds_vespa(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sessionmaker = get_sessionmaker()
    chat_id = uuid.uuid4()
    document_id = uuid.uuid4()
    middle_path = Path("tests/fixtures/mineru_sample_paper/middle.json")

    async with sessionmaker() as session:
        session.add(Chat(id=chat_id, name="pipeline-test", description=None))
        session.add(
            Document(
                id=document_id,
                chat_id=chat_id,
                source_type="upload",
                original_filename="paper.pdf",
                storage_path="data/storage/test/paper.pdf",
                mime_type="application/pdf",
                status="parsing",
                checksum_sha256="0" * 64,
            )
        )
        await session.commit()

    feed_calls: list[dict[str, int]] = []

    async def fake_feed_document(*args, **kwargs):  # type: ignore[no-untyped-def]
        feed_calls.append(
            {
                "blocks": len(kwargs["blocks"]),
                "hierarchy": len(kwargs["hierarchy"]),
                "facts": len(kwargs["facts"]),
            }
        )
        assert "embedding_provider" not in kwargs
        return FeedReport(success_count=1, fail_count=0)

    monkeypatch.setattr(ingestion_worker, "feed_document", fake_feed_document)

    try:
        await ingestion_worker._persist_enrichment_and_index(  # noqa: SLF001
            chat_id=chat_id,
            document_id=document_id,
            middle_json_path=middle_path,
            page_count=8,
        )

        async with sessionmaker() as session:
            node_count = await session.scalar(
                select(func.count()).select_from(DocumentNode).where(
                    DocumentNode.chat_id == chat_id,
                    DocumentNode.document_id == document_id,
                )
            )
            summary_count = await session.scalar(
                select(func.count()).select_from(Summary).where(
                    Summary.chat_id == chat_id,
                    Summary.document_id == document_id,
                )
            )
            fact_count = await session.scalar(
                select(func.count()).select_from(StructuredFact).where(
                    StructuredFact.chat_id == chat_id,
                    StructuredFact.document_id == document_id,
                )
            )
            doc = await session.get(Document, document_id)

        assert node_count is not None and node_count > 0
        assert summary_count is not None and summary_count > 0
        assert fact_count is not None and fact_count >= 5
        assert doc is not None
        assert doc.status == "enriching"
        assert doc.page_count == 8
        assert feed_calls == [{"blocks": 37, "hierarchy": 34, "facts": fact_count}]
    finally:
        async with sessionmaker() as session:
            await session.execute(delete(Chat).where(Chat.id == chat_id))
            await session.commit()
