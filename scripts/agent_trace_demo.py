"""Verbose end-to-end trace of the LangGraph agent.

Same flow as ``smoke_agent_e2e.py`` but instruments the chat provider so every
LLM call is printed, and dumps the full AgentState event log + tool
invocations + evidence after the graph completes.

Run::

    uv run python scripts/agent_trace_demo.py
"""

from __future__ import annotations

import asyncio
import json
import os
import textwrap
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import dotenv

dotenv.load_dotenv()

db_url = os.environ.get("DATABASE_URL", "")
if "+psycopg" in db_url:
    os.environ["DATABASE_URL"] = db_url.replace("+psycopg", "+asyncpg")

from app.agent.budget import ContextBudgetManager  # noqa: E402
from app.agent.graph import build_graph  # noqa: E402
from app.agent.nodes.persist_messages import InMemoryMessageStore  # noqa: E402
from app.agent.state import AgentState  # noqa: E402
from app.agent.tools._invocation import ToolDeps  # noqa: E402
from app.api.messages import _build_providers  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.db import _get_sessionmaker  # noqa: E402
from app.models.orm import Chat as ChatORM  # noqa: E402
from app.models.orm import ChatDocument as ChatDocumentORM  # noqa: E402
from app.models.orm import Document as DocumentORM  # noqa: E402
from app.models.orm import DocumentNode as DocumentNodeORM  # noqa: E402
from app.models.orm import Session as SessionORM  # noqa: E402
from app.providers.base import ChatChunk, ChatCompletion, ChatMessage, ChatProvider, Usage  # noqa: E402

get_settings.cache_clear()


def _hr(title: str) -> None:
    print(f"\n{'─' * 6} {title} {'─' * max(0, 60 - len(title))}")


def _short(text: str, n: int = 240) -> str:
    text = text.replace("\n", " ⏎ ")
    return text if len(text) <= n else text[:n] + " …"


class LoggingChatProvider(ChatProvider):
    """Decorator that prints every LLM request/response in detail."""

    def __init__(self, inner: ChatProvider) -> None:
        self._inner = inner
        self._call = 0

    @property
    def name(self) -> str:
        return self._inner.name

    @property
    def model(self) -> str:
        return self._inner.model

    @property
    def context_window(self) -> int:
        return self._inner.context_window

    async def complete(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        stop: list[str] | None = None,
    ) -> ChatCompletion:
        self._call += 1
        n = self._call
        _hr(f"LLM CALL #{n} — request")
        for m in messages:
            print(f"  [{m.role}] {_short(m.content, 500)}")
        print(f"  params: temperature={temperature} max_tokens={max_tokens}")
        result = await self._inner.complete(
            messages, temperature=temperature, max_tokens=max_tokens, stop=stop
        )
        _hr(f"LLM CALL #{n} — response")
        print(f"  model:  {result.model}")
        print(f"  tokens: prompt={result.usage.prompt_tokens} completion={result.usage.completion_tokens}")
        print(f"  content: {_short(result.content, 600)}")
        return result

    async def stream(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        stop: list[str] | None = None,
    ) -> AsyncIterator[ChatChunk]:
        # Not exercised by generate_answer (which uses complete); keep delegating
        async for chunk in self._inner.stream(
            messages, temperature=temperature, max_tokens=max_tokens, stop=stop
        ):
            yield chunk

    async def test_connection(self):  # type: ignore[override]
        return await self._inner.test_connection()


async def _seed() -> tuple[uuid.UUID, uuid.UUID]:
    sm = _get_sessionmaker()
    async with sm() as db:
        chat = ChatORM(id=uuid.uuid4(), name="trace-demo")
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


async def _build_initial_state(
    db, *, chat_id: uuid.UUID, session_id: uuid.UUID, question: str
) -> AgentState:
    """Mirror QAService._build_initial_state but inline so we can inspect."""
    return AgentState(chat_id=chat_id, session_id=session_id, question=question)


async def main() -> int:
    s = get_settings()
    _hr("CONFIG")
    print(f"  LLM_PROVIDER  = {s.llm_provider}")
    print(f"  LLM_API_URL   = {s.llm_api_url}")
    print(f"  LLM_MODEL     = {s.llm_model}")
    print(f"  VESPA_ENABLED = {s.vespa_enabled}")

    chat_id, session_id = await _seed()
    _hr("SEED")
    print(f"  chat_id    = {chat_id}")
    print(f"  session_id = {session_id}")

    try:
        sm = _get_sessionmaker()
        async with sm() as db:
            raw_chat_provider, retrieval_service = await _build_providers(db, chat_id)
        chat_provider = LoggingChatProvider(raw_chat_provider)
        _hr("PROVIDERS")
        print(f"  chat_provider     = {type(raw_chat_provider).__name__} (model={chat_provider.model})")
        print(f"  retrieval_service = {type(retrieval_service).__name__}")

        question = "Summarise the documents in this chat in one sentence."
        _hr("QUESTION")
        print(f"  {question}")

        @asynccontextmanager
        async def _session_ctx():
            async with sm() as s_:
                yield s_

        deps = ToolDeps(
            retrieval_service=retrieval_service,
            chat_provider=chat_provider,
            session_factory=_session_ctx,
        )
        bm = ContextBudgetManager()
        store = InMemoryMessageStore()
        graph = build_graph(
            deps=deps, chat_provider=chat_provider, budget_manager=bm, message_store=store
        )

        async with sm() as db:
            initial = await _build_initial_state(
                db, chat_id=chat_id, session_id=session_id, question=question
            )

        _hr("RUN GRAPH")
        result = await graph.ainvoke({"state": initial.model_dump()})
        final = AgentState.model_validate(result["state"])

        # --- 1) Event log ---
        _hr("EVENT LOG (debug_trace)")
        for i, evt in enumerate(final.debug_trace.events):
            ts = evt.ts.strftime("%H:%M:%S.%f")[:-3]
            payload_str = json.dumps(evt.payload, default=str, ensure_ascii=False)
            print(f"  [{i:>2}] {ts}  {evt.kind:<18} {evt.name}")
            if evt.payload:
                print(f"        {_short(payload_str, 200)}")

        # --- 2) Tool invocation fingerprints ---
        _hr("TOOL INVOCATIONS")
        if not final.tool_invocations_fingerprints:
            print("  (none)")
        for fp in sorted(final.tool_invocations_fingerprints):
            print(f"  - {fp}")

        # --- 3) Evidence collected ---
        _hr("EVIDENCE ITEMS")
        if not final.evidence_items:
            print("  (none)")
        for ev in final.evidence_items:
            print(f"  • id={ev.evidence_id}  source={ev.source_type}  doc={ev.document_id}")
            print(f"      title:   {ev.section_title or '-'}")
            print(f"      pages:   {ev.page_start}-{ev.page_end}")
            print(f"      excerpt: {_short(ev.content, 200)}")

        # --- 4) Citations ---
        _hr("CITATIONS")
        if not final.citations:
            print("  (none)")
        for c in final.citations:
            print(f"  • {c.citation_id}  doc={c.document_id}  pages={c.page_start}-{c.page_end}")
            print(f"      section: {c.section_title or '-'}")
            print(f"      excerpt: {_short(c.excerpt, 200)}")

        # --- 5) Errors / uncertainty ---
        _hr("ERRORS / POLICY VIOLATIONS")
        if not final.errors:
            print("  (none)")
        for err in final.errors:
            print(f"  • {err.code}: {err.detail} (tool={err.tool_name})")

        # --- 6) Final answer ---
        _hr("FINAL ANSWER")
        print(textwrap.fill(final.answer or "(empty)", width=88, initial_indent="  ", subsequent_indent="  "))

        return 0
    finally:
        await _cleanup(chat_id)
        _hr("CLEANUP")
        print(f"  removed chat {chat_id}")


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
