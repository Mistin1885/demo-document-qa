# 06 · Architecture & Component Interaction

A bird's-eye view of how the pieces fit together. This document is a
cross-reference; the schemas, contracts, and policies it points at live in
`docs/03–05`, `CLAUDE.md`, and the source itself.

---

## 1. System diagram

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                                BROWSER                                       │
│  Next.js 15 App Router · React 19 · TanStack Query · SSE EventSource         │
│  Chat │ Documents │ Sessions │ Settings                                      │
└─────────────┬────────────────────────────────────────────────────────────────┘
              │  HTTP/JSON   (NEXT_PUBLIC_API_BASE_URL)
              │  multipart   POST /chats/{cid}/documents
              │  SSE         POST /chats/{cid}/sessions/{sid}/messages
              ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                          FASTAPI BACKEND  (uvicorn :8000)                    │
│                                                                              │
│  api/         chats · sessions · documents · facts · manifest · messages     │
│  services/    chat_service · session_service · document_service              │
│               ingestion_service · qa_service · facts_service · provider_*    │
│  agent/       graph.py (LangGraph) → nodes/ · tools/ · policies.py · budget  │
│  parsing/     mineru_client · mapping · hierarchy · models                   │
│  enrichment/  summaries · keywords · entities · claims · facts · manifest    │
│  retrieval/   RetrievalService  ◄────── single Vespa entry point             │
│  providers/   ChatProvider · EmbeddingProvider · RerankerProvider (+mocks)   │
│  storage/     local PDF blob storage  (data/storage/)                        │
│  security.py  Fernet encrypt/decrypt + mask_secret                           │
│                                                                              │
└──┬──────────────┬──────────────┬──────────────────────┬──────────────────────┘
   │              │              │                      │
   │ asyncpg /    │ HTTP         │ HTTP                 │ HTTP (OpenAI-compat)
   │ psycopg      │ pyvespa      │ MinerU hybrid client │
   ▼              ▼              ▼                      ▼
┌──────────┐  ┌──────────┐  ┌─────────────────┐   ┌──────────────────────────┐
│ Postgres │  │  Vespa   │  │  MinerU         │   │ Chat / Embedding /       │
│  :5432   │  │  :8080   │  │  hybrid client  │   │ Reranker providers       │
│          │  │  :19071  │  │  + local vLLM   │   │ (OpenAI / Gemini /       │
│ chats    │  │ schema:  │  │  @ :8001        │   │  Gemini-compat /         │
│ sessions │  │ document │  │  serving        │   │  OpenAI-compat / vLLM)   │
│ messages │  │ _chunk.sd│  │  MinerU2.5      │   │                          │
│ docs     │  │ (BM25 +  │  │                 │   │ keys: Fernet-encrypted   │
│ nodes    │  │  HNSW    │  │ output cache:   │   │       in Postgres,       │
│ summaries│  │  ANN)    │  │ data/parsed/    │   │       masked in logs     │
│ facts    │  │          │  │                 │   │                          │
│ providers│  │          │  │                 │   │                          │
│ jobs     │  │          │  │                 │   │                          │
└──────────┘  └──────────┘  └─────────────────┘   └──────────────────────────┘
```

Everything between the dashed boundaries inside FastAPI runs in one Python
process; the four outbound arrows are the only out-of-process dependencies.

---

## 2. Boundaries & responsibilities

| Component | Responsibility | Out of scope |
|---|---|---|
| **Frontend (Next.js)** | render Chat / Documents / Settings, upload PDFs, render SSE answer + citations | no secrets — all keys live server-side; client only sees masked values |
| **API layer (`src/app/api/`)** | URL parsing, auth-scope verification, request validation, SSE stream framing | no domain logic; no SQL; no agent state |
| **Service layer (`src/app/services/`)** | orchestrate transactions, inject `chat_id` into every read, call agent / ingestion | no FastAPI imports; no raw YQL |
| **Agent (`src/app/agent/`)** | LangGraph nodes + tools + policies + budget | no SQL inside nodes; tools speak only to repository functions |
| **Retrieval (`src/app/retrieval/`)** | the *only* Vespa query surface; builds YQL, fuses BM25/ANN, reranks | no Postgres reads |
| **Vespa client (`src/app/vespa/`)** | feed / delete / app-package generation / encoders / mock | no business logic |
| **Parsing (`src/app/parsing/`)** | call MinerU, map `middle.json` → `ParsedBlock`, derive hierarchy | no enrichment / summarisation |
| **Enrichment (`src/app/enrichment/`)** | summaries, keywords, entities, claims, facts, manifest building | no parsing |
| **Providers (`src/app/providers/`)** | OpenAI / Gemini / vLLM adapters + deterministic mocks for tests | no caching |

---

## 3. The agent's eye view of the surrounding system

```
                    ┌──────────────────────────────────┐
                    │           AgentState             │
                    │  chat_id  session_id  question   │
                    │  plan  evidence  budget  errors  │
                    └─────────────────┬────────────────┘
                                      │ (service-injected; LLM-unreachable)
                                      │
              ┌───────────────────────┼───────────────────────┐
              │                       │                       │
              ▼                       ▼                       ▼
   ┌──────────────────┐    ┌─────────────────────┐   ┌──────────────────┐
   │   PostgreSQL     │    │   Vespa hybrid      │   │  Chat provider   │
   │  via service /   │    │   via RetrievalService    │  (LLM)           │
   │  repository fns  │    │   (chat_id always   │   │                  │
   │                  │    │    in YQL WHERE)    │   │                  │
   └────────┬─────────┘    └──────────┬──────────┘   └────────┬─────────┘
            │                         │                       │
   ┌────────┴────────┐         ┌──────┴──────┐         ┌──────┴──────┐
   │ inspect_chat    │         │ search_hybrid│        │ plan_information│
   │ inspect_document│         │              │        │ generate_answer │
   │ fetch_structural│         │              │        │ aggregate_sources│
   │ query_facts     │         │              │        │ verify_claims    │
   │ grep_chunks     │         │              │        │ llm_replan       │
   │ expand_evidence │         │              │        │                  │
   └─────────────────┘         └──────────────┘        └──────────────────┘
```

Three rules colour everything in this diagram:

1. **`chat_id` flows down, never up.** Routers read it from the URL → put
   it on `AgentState` → service layer attaches it to every tool / retrieval
   request. Tools accept it through their *runtime* call, not their schema;
   the LLM cannot author it.
2. **Postgres for deterministic / structural fetches; Vespa for similarity.**
   Whole-chat overviews, whole-section walks, and structured facts go via
   Postgres. Conceptual / cross-paragraph queries go via Vespa. Policy P3
   prevents top-k from impersonating a summary.
3. **`RetrievalService.search` is the only public surface to Vespa.** Both
   `search_hybrid` (agent tool) and any future call paths must go through it
   so the chat-isolation guard cannot be skipped.

The `llm_replan` node sits inside the right-hand "Chat provider" lane:
it is the only place where the LLM authors structured retrieval intent.
Its output is a Pydantic-validated `ReplanDecision`, gated by policy P15
to a whitelist of retrieval tools, and dispatched through the same
`execute_retrieval_tools → PolicyEngine → RetrievalService` path —
never directly. See [`08-deep-qa.md`](./08-deep-qa.md) for how this
interacts with Deep QA mode (which only widens budgets / retrieval, never
weakens isolation).

---

## 4. Cross-component flow examples

### 4.1 Upload a PDF

```
Browser ── multipart ──► api/documents.upload
                            ▼
                  document_service.create_document
                  └─ inserts documents + chat_documents (Postgres)
                            ▼
                  ingestion_service.run_ingestion
                  ├─ parsing/mineru_client  ──HTTP──► MinerU hybrid (vLLM @ :8001)
                  ├─ parsing/mapping        →  ParsedBlock list
                  ├─ parsing/hierarchy      →  document_nodes rows
                  ├─ enrichment/*           →  summaries + facts rows
                  ├─ providers/embedding.embed_chunks
                  └─ vespa/feed.feed_chunks ──HTTP──► Vespa
                            ▼
                  ingestion_jobs.state = 'succeeded'
                  documents.status     = 'indexed'
```

### 4.2 Ask a question

```
Browser ── SSE POST ──► api/messages
                            ▼
                  session_service._require_session  (404 on cross-chat)
                            ▼
                  qa_service.stream
                  └─ build AgentState{chat_id, session_id, question, …}
                            ▼
                  agent/graph.build_graph().ainvoke
                  └─ 15 nodes → 0–N tool calls → answer + citations
                            ▼
                  qa_service emits SSE:
                  ├─ event: token     {delta}
                  ├─ event: citation  {chat_id, document_id, page, excerpt, …}
                  └─ event: error?    {code, detail}
                            ▼
                  api/messages StreamingResponse → Browser
```

Detailed step-by-step lives in `docs/07-qa-pipeline.md`.

---

## 5. Deployment topology (docker-compose)

```
deploy/docker-compose.yml

 ┌──────────────────────────────────────────┐
 │  network: default (compose)              │
 │                                          │
 │  ┌──────────┐  ┌─────────┐               │
 │  │ postgres │  │  vespa  │               │
 │  └────┬─────┘  └────┬────┘               │
 │       │             │                    │
 │       │ DATABASE_URL│ VESPA_ENDPOINT     │
 │       ▼             ▼                    │
 │  ┌──────────────────────────┐            │
 │  │        backend           │            │
 │  │  (alembic + uvicorn)     │            │
 │  └─────────────┬────────────┘            │
 │                │ NEXT_PUBLIC_API_BASE_URL │
 │                ▼ (build-time, inlined)   │
 │  ┌──────────────────────────┐            │
 │  │        frontend          │            │
 │  │  (next start standalone) │            │
 │  └──────────────────────────┘            │
 │                                          │
 │  extra_hosts:                            │
 │   host.docker.internal:host-gateway      │
 │   → reach host's vLLM @ :8001            │
 └──────────────────────────────────────────┘
```

Browser-facing ports: `frontend :3000`, `backend :8000`, `vespa :8080` and
`:19071`, `postgres :5432`. Persistent volumes: `postgres_data`,
`vespa_data`, `backend_data` (uploaded PDFs + MinerU cache).

---

## 6. Where each contract is enforced (one-page cheat sheet)

| Contract | Layer 1 | Layer 2 | Layer 3 | Layer 4 |
|---|---|---|---|---|
| **chat isolation** | Postgres WHERE in services | Vespa YQL `chat_id contains` | Route layer 404 | Policies P1 / P12 / P13 |
| **session isolation** | history filtered by `(session_id, chat_id)` | — | route 404 on cross-chat session | policy P2 |
| **API key safety** | Fernet at rest | masked in logs | never returned to FE | provider tests sanitise errors |
| **No model-knowledge answers** | empty evidence/facts | policy P11 fallback string | — | — |
| **Citations belong to current chat** | citation_draft assembly uses state.chat_id | policy P12 strips violators | policy P13 enforces `chat_documents` membership | final `validate_scope_isolation` |
