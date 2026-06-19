# 07 · QA Pipeline — End-to-End

This is the operational walkthrough of what happens between *"user clicks
Send"* and *"the answer finishes streaming"*. Every step lists the
component that does the work and the contract it has to obey.

> Prereqs you should have skimmed first:
> [`03-postgresql-schema.md`](./03-postgresql-schema.md),
> [`04-vespa-schema.md`](./04-vespa-schema.md),
> [`05-agent-workflow.md`](./05-agent-workflow.md),
> [`06-architecture.md`](./06-architecture.md).

---

## 1. Pre-conditions (ingestion already happened)

Before any QA call works, the chat must contain *indexed* documents.
Ingestion is its own pipeline; the short version is:

```
PDF upload
  → document row inserted (status='uploaded')
  → ingestion_job(pending) created
  → MinerU hybrid client called; output cached under data/parsed/<doc>/
  → middle.json mapped → document_nodes
  → enrichment runs summaries / facts / manifest
  → embedding provider embeds chunks
  → VespaFeedClient.feed_chunks(...) PUTs them to Vespa
  → document.status = 'indexed', ingestion_job.state = 'succeeded'
```

Idempotency comes from `VespaFeedClient.delete_by_document(chat_id,
document_id)` being called *before* each feed, plus the
`documents.checksum_sha256` uniqueness check (see
`docs/03-postgresql-schema.md` §5).

---

## 2. Step-by-step — one question, one answer

The numbered steps below correspond to actual code paths; the names in
**bold** are functions or modules you can grep for.

### 2.1 Browser → backend

The Chat UI posts the user's message to:

```
POST /chats/{chat_id}/sessions/{session_id}/messages
Accept: text/event-stream
Content-Type: application/json

{
  "content": "Compare LightRAG and GraphRAG on accuracy.",
  "max_answer_tokens": null,
  "temperature": null,
  "context_window": null
}
```

The request body is `MessageRequest` (`src/app/api/messages.py:57+`). Per-
request generation overrides are optional; when absent the chat's default
provider profile (or the env-level `LLM_*` fallback) decides.

### 2.2 API layer

Router: `src/app/api/messages.py:239-298`. Responsibilities, in order:

1. **`session_service.get_session_by_id(chat_id, session_id)`** — verifies
   the session belongs to the chat. Cross-chat URLs (`{chat_id_X}/sessions/
   {session_id_Y}`) raise `SessionNotFound` → HTTP 404. Layer 3 of the
   isolation contract.
2. **Resolve the chat provider:**
   - if `Chat.default_chat_profile_id` is set, decrypt the API key with
     **`security.decrypt`** and instantiate the right adapter (`OpenAIChatProvider`
     / `GeminiNativeChatProvider` / `OpenAICompatChatProvider`);
   - otherwise build an `OpenAICompatChatProvider` from `LLM_*` env vars (dev/demo);
   - otherwise fall back to `ExtractiveEvidenceChatProvider`, which only
     summarizes retrieved evidence and emits normal citation markers.
3. **Build `GenerationConfig`** from the request body, clamped against the
   provider's `context_window`.
4. **Call `QAService.stream(...)`** and return the async generator as a
   FastAPI `StreamingResponse(media_type="text/event-stream")`.

### 2.3 QAService — boundary between FastAPI and LangGraph

`src/app/services/qa_service.py:stream`. What it does *before* the agent:

1. Load the last N session messages (filtered by `session_id` + `chat_id`)
   into `conversation_history`.
2. Construct `AgentState` (see `docs/05-agent-workflow.md` §3): chat_id,
   session_id, question, history, generation_config, empty plan / evidence /
   tool_calls / debug_trace.
3. Construct `RetrievalService` with Vespa native embeddings/rerank. The
   production path sends `input.query(qvec)=embed(e5, @user_query)` to Vespa;
   no per-user embedding or reranker profile is required.
4. Compile the LangGraph via **`build_graph(chat_provider, retrieval_service,
   ...)`** and call `graph.ainvoke({"state": state.model_dump()})`.

What it does *after* the agent: see §2.7.

### 2.4 The graph executes (15 nodes)

The full node list and conditional edges live in
[`docs/05-agent-workflow.md` §1](./05-agent-workflow.md#1-stategraph-nodes).
Here is what each node does *in terms of which subsystems it touches*:

| # | Node | Touches |
|---|---|---|
| 1 | `load_chat_and_session` | Postgres (chats, sessions, messages) |
| 2 | `inspect_scope` | Postgres (chat manifest snapshot) via `inspect_chat` tool |
| 3 | `plan_information_needs` | **LLM** (chat provider) — first call |
| 4 | `enforce_scope_and_policies` | pure Python (P1–P4) |
| 5 | `execute_retrieval_tools` | Postgres + Vespa via tools |
| 6 | `merge_evidence_workspace` | pure Python (dedup, score normalisation) |
| 7 | `check_context_budget` | tiktoken / heuristic (`ContextBudgetManager`) |
| 8 | `aggregate_sources_node` | **LLM** — only on overflow (P6) |
| 9 | `check_coverage` | pure Python; may loop |
| 10 | `plan_gap_retrieval` | **LLM** — only when coverage incomplete |
| 11 | `verify_critical_claims` | Postgres (`structured_facts`) + numeric cross-check (P8) |
| 12 | `generate_answer` | **LLM** — final answer |
| 13 | `validate_citations` | pure Python (P12 / P13) |
| 14 | `validate_scope_isolation` | pure Python; last guard |
| 15 | `persist_messages` | Postgres (insert user + assistant `messages`) |

Three LLM calls is the common case. Adding `aggregate_sources` and/or a gap
pass pushes it to four or five. Policy P9 caps the tool-call iteration count
at 2 — there is no unbounded ReAct loop.

### 2.5 What gets called by tools

| Tool | Concrete call | Returns |
|---|---|---|
| `inspect_chat` | `chat_service.get_manifest(chat_id)` | doc count, titles, source-type histogram |
| `inspect_document` | `document_service.get_manifest(chat_id, document_id)` | per-doc page count, sections, has-table/figure |
| `fetch_structural_nodes` | `document_service.fetch_nodes(...)` with the restricted filter | `list[NodeRecord]` in reading order |
| `search_hybrid` | **`RetrievalService.search(RetrievalRequest(...))`** | `list[SearchHit]` with full score breakdown |
| `query_structured_facts` | `facts_service.query(FactsFilter(...))` | `list[StructuredFact]` |
| `aggregate_sources` | chat provider chat call with built-in template | `list[EvidenceItem]` (compacted) |
| `expand_evidence` | `document_service.fetch_siblings(...)` | additional `EvidenceItem`s |

Inside `search_hybrid`, the call chain is:

```
search_hybrid (agent tool)
  └─ RetrievalService.search(req)
       ├─ _yql_where(req)            ──► "chat_id contains '<cid>' and …"
       ├─ asyncio.gather(
       │     vespa.query(bm25 query),
       │     vespa.query(nearestNeighbor(embedding, vector)),
       │  )
       ├─ reciprocal_rank_fusion(bm25_hits, ann_hits)
       ├─ optional second-pass rerank (native or cross-encoder)
       └─ return list[SearchHit]
```

The agent itself never builds a YQL string and never imports `pyvespa`.

### 2.6 Citations are assembled, not invented

`generate_answer` produces an answer body plus a list of `CitationDraft`s
that point at evidence items already in `AgentState`. Two filters then run:

- **P12 — `citation_chat_isolation`**: drop any draft where `chat_id !=
  state.chat_id`. (Cannot normally happen — evidence comes from `RetrievalService`
  which already enforced this — but the policy is the last line of defence.)
- **P13 — `citation_doc_association`**: drop any draft whose `document_id`
  is not in this chat's `chat_documents` association.

What survives is upgraded to a `Citation` (with `citation_id`,
`document_name`, `excerpt`, `page_start/end`, `section_title`). Anything
left in the answer text that *would* have referenced a dropped citation is
detached, so the client never sees a citation it cannot resolve.

### 2.7 Streaming back to the browser

`QAService.stream` re-hydrates the final `AgentState` from
`graph.ainvoke`'s output and yields three event types over SSE:

```
event: token
data: {"delta": "LightRAG "}

event: token
data: {"delta": "reports "}

…

event: citation
data: {"citation_id": "…", "chat_id": "…", "document_id": "…",
       "document_name": "lightrag.pdf", "page_start": 4, "page_end": 4,
       "section_title": "Experiments", "excerpt": "…"}

event: error      # only on failure
data: {"code": "AGENT_ERROR", "detail": "policy P7 …"}
```

- Tokens are emitted *post-hoc* from `state.answer` (split on whitespace)
  rather than streamed token-by-token from the underlying LLM — this lets
  policy P8 (`numeric_cross_check`) rewrite the answer before the client
  sees a single word.
- Citations are emitted after the last token, one event per citation.
- Client disconnect sets a `stop_event`; the generator exits and the
  partial state is still persisted by node 15 if `generate_answer` had
  already run.

### 2.8 Persistence

Node 15 (`persist_messages`) inserts two rows into `messages`:

| role | content | citations | tool_trace |
|---|---|---|---|
| `user` | the question | — | — |
| `assistant` | the final answer (with `[unverified]` tags if P8 fired) | serialized `list[Citation]` | serialized `ToolTrace` (every tool call, params, status, token estimate) |

The next session GET will replay this history; future agent turns will read
it via P2-scoped queries.

---

## 3. What's invoked at each stage — quick index

```
HTTP
  ├─ FastAPI router            src/app/api/messages.py
  ├─ session_service           src/app/services/session_service.py
  ├─ chat_service              src/app/services/chat_service.py
  └─ qa_service                src/app/services/qa_service.py

Agent
  ├─ build_graph               src/app/agent/graph.py
  ├─ nodes                     src/app/agent/nodes/
  ├─ tools                     src/app/agent/tools/
  ├─ policies                  src/app/agent/policies.py
  └─ budget                    src/app/agent/budget.py

Data layer
  ├─ Postgres (SQLAlchemy)     src/app/models/orm.py · services/* · facts_service · document_service
  ├─ Vespa (only via)          src/app/retrieval/service.py
  └─ Vespa client / feed       src/app/vespa/

External
  ├─ ChatProvider              src/app/providers/chat/*  (OpenAI / Gemini / vLLM)
  ├─ EmbeddingProvider         src/app/providers/embedding/*
  ├─ RerankerProvider          src/app/providers/reranker/*
  └─ MinerU + vLLM @ :8001     deploy/mineru/
```

---

## 4. Failure modes and what surfaces to the user

| Failure | Where it's caught | What the user sees |
|---|---|---|
| Cross-chat URL | route layer (`session_service`) | HTTP 404 `session not found` |
| Wrong embedding DIM | `RetrievalService.__init__` | SSE `error: INTERNAL` "embedding dimension mismatch" |
| LLM provider down | provider adapter raises | SSE `error: AGENT_ERROR` with sanitised detail (policy P14) — **never** silent model swap |
| No relevant evidence | policies P7 / P11 | answer body becomes the exact string *"There is not enough information in the current chat's documents."* |
| Citation outside chat scope | P12 / P13 / `validate_scope_isolation` | citation silently dropped; the answer remains coherent |
| Tool overflow | P6 | `aggregate_sources_node` compacts evidence and the graph continues |
| Iteration cap reached | P9 → P7 fallback | "not enough information" answer |

All failures are recorded in `AgentDebugTrace` and persisted alongside the
assistant message as `tool_trace`, so a follow-up question can inspect the
trace and the operator can replay the run from `messages.tool_trace` in
Postgres.

---

## 5. Trying it locally

End-to-end smoke against a real LLM (no UI, no DB writes):

```bash
LLM_PROVIDER=openai_compatible \
  LLM_API_URL=http://localhost:8001/v1 \
  LLM_MODEL=gemma-3-27b-it \
  uv run python scripts/smoke_agent_e2e.py
```

End-to-end via the UI (the recommended flow):

```bash
docker compose -f deploy/docker-compose.yml up -d
uv run python scripts/deploy_vespa.py    # one-shot
open http://localhost:3000
```

Then: **New chat → Upload PDF → wait for `indexed` → ask a question.** The
full ingestion + QA chain described in this document fires on every send.
