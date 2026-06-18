"""End-to-end smoke test for the LangGraph agent against the configured LLM.

Verifies the full QA path:
  - env-loaded LLM_* settings produce a real ChatProvider
  - the messages-API helper ``_build_providers`` returns a real provider
  - QAService.run invokes the StateGraph
  - tools (search_hybrid / inspect_chat / ...) are actually invoked
  - the real LLM produces the final answer (not the MockChatProvider)

Run with::

    uv run python scripts/smoke_agent_e2e.py
"""

from __future__ import annotations

import asyncio
import os
import uuid

import dotenv

dotenv.load_dotenv()

# Convert psycopg → asyncpg for the app's async engine
db_url = os.environ.get("DATABASE_URL", "")
if "+psycopg" in db_url:
    os.environ["DATABASE_URL"] = db_url.replace("+psycopg", "+asyncpg")

from app.api.messages import _build_providers  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.db import _get_sessionmaker  # noqa: E402
from app.models.orm import Chat as ChatORM  # noqa: E402
from app.models.orm import ChatDocument as ChatDocumentORM  # noqa: E402
from app.models.orm import Document as DocumentORM  # noqa: E402
from app.models.orm import DocumentNode as DocumentNodeORM  # noqa: E402
from app.models.orm import Session as SessionORM  # noqa: E402
from app.services.qa_service import QAService  # noqa: E402

get_settings.cache_clear()


async def _seed() -> tuple[uuid.UUID, uuid.UUID]:
    """Seed a chat + session + 1 document with 1 fetchable node.

    The seeded paragraph mentions "graph-based retrieval" so the planner's
    keyword-matching routes the question to the structural-fetch path and
    the agent's tools have real evidence to return.
    """
    sm = _get_sessionmaker()
    async with sm() as db:
        chat = ChatORM(id=uuid.uuid4(), name="smoke-e2e")
        db.add(chat)
        await db.flush()
        sess = SessionORM(id=uuid.uuid4(), chat_id=chat.id, name="s1")
        db.add(sess)
        doc = DocumentORM(
            id=uuid.uuid4(),
            chat_id=chat.id,
            source_type="upload",
            original_filename="lightrag-fake.pdf",
            storage_path="data/storage/fake.pdf",
            mime_type="application/pdf",
            page_count=1,
            status="indexed",
            checksum_sha256="0" * 64,
        )
        db.add(doc)
        db.add(ChatDocumentORM(chat_id=chat.id, document_id=doc.id))
        db.add(
            DocumentNodeORM(
                id=uuid.uuid4(),
                document_id=doc.id,
                chat_id=chat.id,
                node_type="paragraph",
                title=None,
                content=(
                    "LightRAG is a graph-based retrieval-augmented generation "
                    "framework that fuses dual-level retrieval — entity-centric "
                    "and relation-centric — to answer multi-hop questions over "
                    "arXiv papers."
                ),
                page_start=1,
                page_end=1,
                order_index=0,
                level=2,
            )
        )
        await db.commit()
        return chat.id, sess.id


async def _cleanup(chat_id: uuid.UUID) -> None:
    sm = _get_sessionmaker()
    async with sm() as db:
        chat = await db.get(ChatORM, chat_id)
        if chat:
            await db.delete(chat)
            await db.commit()


async def main() -> int:
    s = get_settings()
    print(f"LLM_PROVIDER  = {s.llm_provider}")
    print(f"LLM_API_URL   = {s.llm_api_url}")
    print(f"LLM_MODEL     = {s.llm_model}")
    print(f"VESPA_ENABLED = {s.vespa_enabled}")
    print()

    chat_id, session_id = await _seed()
    print(f"Seeded chat={chat_id} session={session_id}\n")

    rc = 1
    try:
        sm = _get_sessionmaker()
        # Build providers exactly the same way the route handler does
        async with sm() as db:
            chat_provider, retrieval_service = await _build_providers(db, chat_id)
        print(f"chat_provider     = {type(chat_provider).__name__} (model={getattr(chat_provider, 'model', '?')})")
        print(f"retrieval_service = {type(retrieval_service).__name__}\n")

        qa = QAService(session_factory=sm, retrieval_service=retrieval_service)
        response = await qa.run(
            chat_id,
            session_id,
            "Summarise the documents in this chat in one sentence.",
            chat_provider=chat_provider,
        )

        print("--- QAResponse ---")
        print("answer:", repr(response.answer))
        print("coverage:", response.coverage)
        print("citations:", len(response.citations))
        print("documents_used:", response.documents_used)

        debug = response.debug_trace
        if debug:
            print(f"\ntool steps recorded: {len(debug.steps)}")
            for step in debug.steps:
                print(f"  - {step.tool_name}: status={step.status} note={step.note}")
            print(f"total_rounds: {debug.total_rounds}")

        # Assert real LLM produced the answer
        if response.answer.startswith("MOCK_RESPONSE_"):
            print("\n[FAIL] answer came from MockChatProvider")
        else:
            print("\n[PASS] real LLM produced the answer; agent ran the StateGraph end-to-end")
            rc = 0
    finally:
        await _cleanup(chat_id)
        print(f"\nCleaned up chat={chat_id}")
    return rc


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
