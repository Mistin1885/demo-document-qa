# CLAUDE.md — Paper Notebook Agent Architecture Contract

> This file is the **single source of truth that every development phase (including sub-agents) MUST obey.**
> If any phase needs to deviate, update this file first and record the decision in `PROGRESS.md`; otherwise it is a violation.
> The full product spec lives in `GUIDE.md`; this file is its executable, distilled, and non-negotiable form.

---

## 0. Non-negotiable Rules

1. **All package management and execution go through `uv`. No exceptions.**
   - Install: `uv add <pkg>` (dev deps: `uv add --dev <pkg>`).
   - Run anything: `uv run <cmd>` (e.g. `uv run pytest`, `uv run ruff check .`, `uv run uvicorn ...`, `uv run alembic ...`, `uv run mineru ...`).
   - **Forbidden:** bare `pip install`, `python xxx.py`, `pytest`, manually activating a venv and running tools directly.
   - Commit `uv.lock`; commit it again whenever dependencies change.
2. **MinerU parsing is validated FIRST.** No downstream phase (Foundation onward) starts until the **MinerU Parsing PoC gate (Phase 1)** proves MinerU produces reliable markdown for arXiv PDFs. See §6 and `DEVELOPMENT_PLAN.md` Phase 1.
3. **Chat isolation is the highest-priority invariant (mandatory gate).** Every retrieval / SQL / API / citation path MUST be bound to `current_chat_id`, and `chat_id` **must never be passed in by the LLM or the agent** — it is injected only by the service layer from `AgentState`. Code that violates this is a defect.
4. **Never hard-code answers to pass tests, never delete important tests, never lower isolation/citation standards.**
5. **Never answer from the model's own knowledge.** If the info is not in the chat's documents, respond "there is not enough information in the current chat's documents."
6. **After each phase, immediately run the Self-Repair Validation Loop (§10) and update `PROGRESS.md`.**

---

## 1. Product in One Sentence

A NotebookLM-like multi-document Agentic QA system. Phase one targets arXiv paper PDFs: parse → index → summarize → cited Q&A. Search and vector DB use **Vespa** (BM25 + vector + metadata filtering + multi-stage rerank). The agent uses a **LangGraph StateGraph**. Document parsing uses **MinerU (hybrid backend)**.

---

## 2. Core Isolation Model (memorize this)

```
Chat    = smallest document-isolation boundary (owns documents, Vespa docs, Sessions, default model profile)
Session = one conversation thread under a Chat (shares documents, NOT history, may use a different model)
Document= always belongs to a Chat (Document.chat_id); cross-chat reuse requires an explicit ChatDocument association
```

Isolation is guaranteed by **four layers together** — none may be skipped:
- Relational DB query (`WHERE chat_id = :current_chat_id`)
- Vespa filter (every query forces `chat_id contains current_chat_id`)
- API authorization scope (route layer verifies chat ownership)
- Repository / Service layer (the single trusted point where the filter is injected)

> Session isolation: a Session must never read another Session's message history.

---

## 3. Tech Stack (locked — do not swap casually)

| Layer | Technology |
|---|---|
| Language / Runtime | Python 3.12+, `uv` end-to-end |
| Backend | FastAPI, Uvicorn, Pydantic v2, SQLAlchemy 2.x, Alembic |
| Agent | LangGraph, LangChain Core |
| DB | PostgreSQL |
| **Parsing** | **MinerU 3.3+ (hybrid backend, `hybrid-http-client`)** as the primary parser; PyMuPDF (fitz) / pdfplumber optional for bbox cross-check & fallback; tiktoken for token estimates |
| Providers | OpenAI SDK, Google GenAI / Gemini OpenAI-compatible, Generic OpenAI-compatible, self-hosted vLLM |
| Search | **Vespa + pyvespa** (BM25 / HNSW ANN / hybrid / rank profiles / multi-stage rerank) |
| HTTP | HTTPX (async) |
| Test / Quality | pytest, pytest-asyncio, ruff, mypy |
| Frontend | Next.js, React, TypeScript, Tailwind CSS, TanStack Query |
| Infra | Docker Compose (vespa / postgres / backend / frontend) |

**Forbidden:** FAISS, Chroma, using `rank-bm25` as the production search engine. BM25 / vector / filter / ranking all go through Vespa. (Unit tests may use a mock search adapter; integration tests must hit a real Vespa.)

---

## 4. Repository Structure (target layout)

Top-level layout deliberately separates **source (`src/`)**, **data (`data/`)**, and **deployment (`deploy/`)**. The backend Python import root is **`app`** (under `src/`).

```
demo-document-qa/                       # repo root (this directory)
├── CLAUDE.md                           # this file (architecture contract)
├── DEVELOPMENT_PLAN.md                 # phased plan + sub-agent dispatch (zh-Hant)
├── PROGRESS.md                         # progress tracker, updated every phase (zh-Hant)
├── GUIDE.md                            # original product spec
├── README.md                           # run instructions (kept current from Phase 2)
├── Makefile                            # thin wrappers over `uv run` commands
├── pyproject.toml / uv.lock            # uv-managed
├── .env.example                        # NO real keys
│
├── src/
│   ├── app/                            # backend application (import root: `app`)
│   │   ├── __init__.py
│   │   ├── main.py                     # FastAPI entrypoint
│   │   ├── config.py                   # pydantic-settings; env overrides
│   │   ├── api/                        # FastAPI routers (thin; no domain logic)
│   │   │   ├── chats.py  sessions.py  documents.py  messages.py  provider_profiles.py
│   │   ├── agent/                      # LangGraph: state, nodes/, tools/, policies, budget, graph
│   │   ├── models/                     # Pydantic v2 domain models + SQLAlchemy ORM
│   │   ├── parsing/                    # MinerU client + output mapping → ParsedBlock/hierarchy
│   │   │   ├── mineru_client.py        # invokes MinerU hybrid-http-client / parses outputs
│   │   │   ├── mapping.py              # content_list.json + middle.json → ParsedBlock
│   │   │   ├── hierarchy.py            # sections / abstract / references / appendix
│   │   │   └── models.py               # ParsedBlock, etc.
│   │   ├── enrichment/                 # summaries, keywords, entities, claims, facts, manifest
│   │   ├── retrieval/                  # RetrievalService (the ONLY Vespa query entry point)
│   │   ├── vespa/                      # application-package helpers, feed, query wrappers
│   │   ├── storage/                    # original-PDF file storage
│   │   ├── providers/                  # chat / embedding / reranker adapters (+ deterministic mocks)
│   │   ├── services/                   # chat / session / document / ingestion / qa / provider / facts
│   │   └── evaluation/                 # retrieval / parser / goal-score evaluators
│   └── frontend/                       # Next.js (app/ components/ features/ hooks/ lib/ types/)
│
├── data/                               # all data artifacts (mostly gitignored)
│   ├── sample/arxiv/                   # downloaded sample PDFs (gitignored; via script)
│   ├── fixtures/                       # SMALL committed test fixtures (tiny PDFs, golden JSON)
│   ├── storage/                        # local object storage for original PDFs (gitignored)
│   └── parsed/                         # MinerU output cache: md / content_list.json / middle.json (gitignored)
│
├── deploy/                             # everything deployment-related
│   ├── docker-compose.yml              # postgres / vespa / backend / frontend
│   ├── postgres/                       # init scripts
│   ├── vespa/application/              # services.xml, hosts.xml, validation-overrides.xml,
│   │   └── schemas/document_chunk.sd   #   schema (DIM driven by config)
│   └── mineru/                         # README + config for the vLLM + MinerU OpenAI-compatible server
│
├── migrations/                         # Alembic env + versions
├── scripts/                            # mineru_poc / deploy_vespa / ingest_sample_arxiv / run_evaluation / seed_demo
├── tests/                              # unit/ integration/ e2e/ evaluation/ fixtures/
└── artifacts/
    ├── repair-loop/                    # iteration-{n}.md
    └── evaluation/                     # retrieval-report.{json,md}, parser-report.*, goal-score.*
```

> `pyproject.toml` must set the package path to `src/app` (e.g. hatchling `packages = ["src/app"]` / `tool.uv` src layout). All backend commands run from repo root via `uv run`.

---

## 5. Data Layer

### 5.1 PostgreSQL tables (ORM)
`chats, sessions, messages, documents, document_nodes, summaries, structured_facts, provider_profiles, ingestion_jobs`, plus `chat_documents` (association).
- Every document-scoped table carries `chat_id` and is indexed on it.
- `ingestion_jobs` must be **idempotent and retryable**.

### 5.2 Vespa schema (`deploy/vespa/application/schemas/document_chunk.sd`)
Fields at least: `vespa_document_id, chat_id, document_id, source_node_id, parent_node_id, source_type, title, heading_path, content, keywords, technical_keywords, entities, page_start, page_end, order_index, token_count, embedding(tensor<float>(x[DIM])), created_at`.
- `embedding`: `indexing: attribute | index`, `distance-metric: angular`.
- **DIM is driven by deployment config.** MVP: one deployment = one embedding dimension; switching must not silently reuse the old schema, and the UI must say so.
- `source_type` values: `raw_block, chunk, section_summary, compact_section_summary, chapter_summary, compact_chapter_summary, document_overview, technology_card, claim, definition, performance_fact, table_record, figure_caption`.
- **Every query forcibly injects** `chat_id contains "<current>"`; if documents are specified, also `document_id in current_chat_document_ids`.

### 5.3 File storage
Original PDFs live in `data/storage/` (local) or object storage. Deleting a document MUST synchronously remove the PostgreSQL records, Vespa documents, the file, and the ChatDocument association.

---

## 6. Document Parsing — MinerU Hybrid (primary path)

**MinerU runs in hybrid mode and is validated before anything else (Phase 1 gate).** Rationale: MinerU's VLM already handles layout analysis, multi-column reading order, tables, formulas, and figures, producing reliable markdown + structured JSON — so we map its output into our model instead of hand-writing column detection.

### 6.1 Hybrid backend topology
- A remote **OpenAI-compatible VLM server** runs the MinerU model. **The user already serves it via vLLM at `http://localhost:8001`** — prefer this endpoint. (MinerU can also start its own wrapper: `mineru-openai-server --port <p>`.)
- The backend acts as a **lightweight hybrid client** that does PDF pre/post-processing locally and offloads VLM inference to the server.
- `hybrid-http-client` requires local `mineru[pipeline]` + torch pipeline deps; `vlm-http-client` is the no-torch lightweight variant. **We use `hybrid-http-client`.**

### 6.2 Install & invoke (always via `uv`)
```bash
uv add "mineru[pipeline]"                  # hybrid client deps (local pipeline + torch)
# parse one PDF against the remote vLLM server:
uv run mineru -p <input_pdf> -o <output_dir> -b hybrid-http-client -u http://localhost:8001
```
- The server URL is **configurable** via `app.config` (`MINERU_SERVER_URL`, default `http://localhost:8001`) and surfaced as a provider-style setting. Never hard-code it in business logic.
- **Open verification (resolve in Phase 1):** confirm the raw vLLM endpoint at `:8001` is compatible with MinerU's `hybrid-http-client` `-u` expectation; if it expects a `mineru-openai-server` wrapper, document the exact server launch in `deploy/mineru/` and adjust the URL/port. Record findings in `PROGRESS.md`.

### 6.3 MinerU outputs we consume (post-processed)

`scripts/mineru_poc.py` runs MinerU then immediately post-processes the per-doc
output directory so that **only three things survive** (everything else is
deleted to keep `data/parsed/` lean and the schema predictable):

- `<name>.md` — markdown, **rebuilt from `content_list.json`** with two changes:
  - every PDF page wrapped in `<Page N>...</Page N>` markers (literal text;
    not valid HTML — markdown renderers pass it through as plain text, so it
    does not collide with `#`/`##` heading parsing).
  - every `images/...` link rewritten to the renamed basename.
- `<name>_middle.json` — **primary structural source** (per-page `pdf_info`
  with `preproc_blocks` / `para_blocks` / `discarded_blocks`, bbox in PDF-point
  space, `page_size`, `_backend`, `_version_name`). All `image_path` values
  are rewritten to renamed basenames so the references match the files on
  disk.
- `images/<doc>_p<page>_<short_hash>.<ext>` — figure / table / formula crops,
  renamed deterministically (`<short_hash>` = first 8 chars of the original
  sha256-derived stem, `<page>` is 1-indexed).

Intermediate files (`<name>_content_list.json`, `<name>_content_list_v2.json`,
`<name>_model.json`, `<name>_layout.pdf`, `<name>_origin.pdf`) are deleted by
the post-processor. The PoC and Phase 4 mapping must read from
`<name>_middle.json` (structure) + `<name>.md` (rendered text) only.

### 6.4 Mapping into our model (`src/app/parsing/`)
- `mapping.py`: walk `middle.json:pdf_info[].preproc_blocks` in order →
  `ParsedBlock` (GUIDE §7.3 fields: `block_id, chat_id, document_id,
  page_number, block_type, text, bbox, reading_order, font_size?, font_name?,
  column_index?, confidence`). `text` for `text`/`title`/`ref_text` blocks
  comes from joining each block's `lines[].spans[].content`;
  `interline_equation` blocks carry the LaTeX in their span; `image`/`table`
  blocks reference the renamed `images/...` basename and the caption sits in
  a sibling `image_caption`/`table_caption` nested block.
- `hierarchy.py`: derive title/authors/abstract/headings/subsections/
  references-boundary/appendix + tables/figure captions from MinerU title
  `level` (`1` = doc title, `2` = section/subsection heading) plus
  proximity-to-image/table heuristics. **Preserve original layout metadata;
  do not let one piece of content land in multiple sections; keep heuristic
  vs LLM refinements traceable** (GUIDE §24).
- `<Page N>...</Page N>` markers in the markdown give a deterministic anchor
  for page-bounded retrieval excerpts (citations carry `page_start`/`page_end`).
- PyMuPDF/pdfplumber are **optional** cross-checks (e.g., verify page count
  / bbox) and a degraded fallback if the MinerU server is unavailable.

### 6.5 Retrieval routing (do NOT make everything top-k)
- **Structural / deterministic fetch (PostgreSQL):** whole-doc / whole-chat / specific-section summaries, anything requiring every document be covered → fetch-all; **never replace with top-k**.
- **Hybrid retrieval (Vespa):** technical details, concepts, cross-paragraph, definitions, claims, method limitations, partial experiments.
- **Structured fact query (PostgreSQL `structured_facts`):** numbers, benchmarks, performance, datasets, metrics first; then Vespa for context.

---

## 7. Vespa Hybrid Retrieval + Rerank

**Hybrid flow:** BM25 candidates + nearestNeighbor candidates → fusion → first-phase ranking → second/global-phase rerank → top-k.
- Default (all configurable): BM25=60, ANN=60, fusion=80, rerank=30, final top-k=8–12.
- **First-version fusion uses RRF (Reciprocal Rank Fusion);** never directly add un-normalized BM25 + vector scores.
- Implement at least two rerankers: (A) Vespa native rank profile; (B) cross-encoder / OpenAI-compatible scoring; keep per-stage scores for debug.
- `SearchHit` must carry `bm25_score / vector_score / fusion_score / rerank_score / final_rank` plus chat/document/node IDs and page range.

> `RetrievalService` is the **only** outward Vespa query entry point; the chat filter is injected here.

---

## 8. LangGraph Agent

Explicit **StateGraph**, not an unbounded ReAct loop. Workflow:
```
load_chat_and_session → inspect_scope → plan_information_needs → enforce_scope_and_policies
→ execute_retrieval_tools → merge_evidence_workspace → check_context_budget
   (overflow → aggregate_sources)
→ check_coverage (incomplete → plan_gap_retrieval → execute_retrieval_tools)
→ verify_critical_claims → generate_answer → validate_citations
→ validate_scope_isolation → persist_messages → END
```

**Tools (map to data capabilities, not to fixed user questions):** `inspect_chat, inspect_document, fetch_structural_nodes, search_hybrid, query_structured_facts, aggregate_sources, expand_evidence`.
- `chat_id` for all tools is injected from `AgentState`; the LLM cannot pass it.
- `query_structured_facts` may only emit a restricted filter schema; **never raw SQL.**
- Every tool result carries `status` (incl. `overflow`), token estimate, sources; on overflow do not truncate — aggregate.

**Code-enforced policies (NOT just prompt) — GUIDE §13.2 (14 rules).** Key: restrict to chat_id; no cross-Session reads; summaries use fetch-all; numbers prefer facts; results pass rerank; do not answer on incomplete coverage; citations all belong to current chat; provider failure returns an explicit error, no silent model switching.

**ContextBudget:** default context 10,000 tokens (actual from provider profile `context_window`); allocation per GUIDE §14. When a provider tokenizer is unavailable, approximate with tiktoken and mark `token_count_is_estimate=true` in the debug trace. Fixed max tool rounds; never repeat the same tool with the same params.

---

## 9. Provider Settings & Security

- Provider types: OpenAI / Gemini Native / Gemini OpenAI-compatible / Generic OpenAI-compatible / self-hosted vLLM.
- Chat / Embedding / Reranker are **independent** profiles (do not assume chat and embedding share a provider). Fields per GUIDE §6.
- Connection tests: Test Chat / Embedding / Reranker → return success/failure, model, latency, sanitized error.
- **API key security (hard rules):** never logged, never returned to the frontend, never stored in plaintext (encrypt with at least an application encryption key), UI shows only masked values, `.env.example` contains no real keys.
- All LLM calls support async; provider adapters are replaceable.

---

## 10. Self-Repair Validation Loop (run after every phase)

```
Implement → Static Validation → Unit → Integration → Goal Tests
→ Analyze Failures → Root Cause → Repair Plan → Minimal Repair
→ Re-run Relevant → Full Regression → Goal Coverage → Continue/Stop
```
- Each iteration writes `artifacts/repair-loop/iteration-{n}.md` (fields per GUIDE §20.2) and records `failed/passed count, goal_score, new_regressions`.
- Failure taxonomy per GUIDE §20.3. `MAX_REPAIR_ITERATIONS = 8` (extendable to 12).
- Two consecutive same-class failures → re-examine architectural assumptions; three consecutive no-progress → emit blocker + alternative, but keep fixing other issues.
- **Stop condition is meeting the Definition of Done (GUIDE §28), not "no compiler errors."**

### 10.1 Standard validation commands (always `uv run` / `npm`)
```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src/app
uv run pytest tests/unit
uv run pytest tests/integration
uv run pytest tests/evaluation
# frontend
npm --prefix src/frontend run lint
npm --prefix src/frontend run typecheck
npm --prefix src/frontend run test
npm --prefix src/frontend run build
# infra
docker compose -f deploy/docker-compose.yml config
docker compose -f deploy/docker-compose.yml up -d postgres vespa
```

---

## 11. Goal Coverage Scoring (out of 100, min 90)

Backend 10 / Parsing 15 / Vespa Retrieval 20 / Agent QA 20 / Isolation 15 / Provider 10 / Frontend 10 (details in GUIDE §21).
**Mandatory gates (any failure ⇒ cannot claim done):** Chat isolation, Session isolation, Vespa hybrid retrieval, Citations, arXiv parsing, LangGraph QA, Provider settings.

---

## 12. Code Quality Requirements (excerpt of GUIDE §26 — violation = defect)

- All public functions have type hints; domain logic does not depend on FastAPI directly; LangGraph nodes do not write SQL directly; Vespa queries all go through `RetrievalService`.
- Core data models do not use unbounded `dict[str, Any]` (use Pydantic v2).
- Never log API keys; never return raw exceptions to the frontend; strip sensitive data from debug traces.
- Tests never depend on real paid APIs; provide deterministic mock LLM / embedding / reranker.
- Vespa feed is repeatable; deleting a document clears Vespa; ingestion is idempotent.

---

## 13. Answer Contract

`QAResponse{answer, citations[], documents_used[], coverage, uncertainty[], session_id, message_id, debug_trace?}`;
`Citation{citation_id, chat_id, document_id, document_name, page_start, page_end, section_title?, source_node_id, excerpt}`.
**The final response must not contain any citation that does not belong to the current `chat_id`.**

---

## 14. Relationship to PROGRESS.md / DEVELOPMENT_PLAN.md

- `DEVELOPMENT_PLAN.md`: phase breakdown, sub-agent task cards, dependencies, acceptance. Execute top-down from it.
- `PROGRESS.md`: live status per phase / sub-agent task, decisions, blockers, goal score. **Update immediately after each task or repair-loop iteration.**
- On conflict among the three, **CLAUDE.md wins**; when a spec conflict appears, update CLAUDE.md first, then continue.
- Language convention: **CLAUDE.md is English; DEVELOPMENT_PLAN.md and PROGRESS.md are Traditional Chinese.**
