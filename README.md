# Paper Notebook Agent

A NotebookLM-like multi-document Agentic QA system targeting arXiv research papers. Upload PDFs to a Chat, the system parses + indexes them, and a LangGraph agent answers questions over **that chat's documents only**, with citations back to the original pages.

> Architecture contract: [`CLAUDE.md`](CLAUDE.md) (English, normative).
> Phased development plan: [`DEVELOPMENT_PLAN.md`](DEVELOPMENT_PLAN.md) (zh-Hant).
> Live progress + decisions: [`PROGRESS.md`](PROGRESS.md) (zh-Hant).
> Original spec: [`GUIDE.md`](GUIDE.md).
> Final delivery report: [`artifacts/final-report.md`](artifacts/final-report.md).

---

## 1. Highlights

- **Multi-Chat isolation**: every document, retrieval, citation and SQL query is bound to `chat_id`. Cross-chat reads are rejected at four independent layers (DB, Vespa filter, API scope, service layer).
- **MinerU hybrid parsing**: arXiv PDFs flow through MinerU's VLM (running on a local vLLM at `http://localhost:8001`) producing reliable markdown + a structured `middle.json`; the post-processor renames image crops and adds page anchors.
- **Vespa hybrid retrieval**: BM25 + HNSW ANN candidates → RRF fusion → optional Vespa native or cross-encoder reranker. Every query forcibly injects `chat_id`.
- **LangGraph StateGraph**: explicit 15-node graph, 7 tools, 14 code-enforced policies; tool `chat_id` is **always** injected from `AgentState`, never from the LLM.
- **Goal coverage score: 100/100** with every mandatory gate passing — see `artifacts/evaluation/goal-score.md`.

---

## 2. Tech stack

| Layer | Tech |
|---|---|
| Language / runtime | Python 3.12+ (entire pipeline via `uv`) |
| Backend | FastAPI, Uvicorn, Pydantic v2, SQLAlchemy 2.x, Alembic |
| Agent | LangGraph + LangChain Core |
| Storage | PostgreSQL (Alembic-managed schema) |
| Parsing | MinerU 3.3+ hybrid backend (vLLM-backed OpenAI-compatible VLM) |
| Search | Vespa 8 + pyvespa (BM25 / HNSW ANN / hybrid / multi-stage rerank) |
| Providers | OpenAI, Gemini Native + OpenAI-compatible, self-hosted vLLM |
| Frontend | Next.js 15 (App Router), React 19, TypeScript, Tailwind 3, TanStack Query v5 |
| Tests | pytest + pytest-asyncio (backend), vitest (frontend) |
| Quality | ruff, mypy, vitest, tsc, Next.js lint |

---

## 3. Repository layout

```
.
├── CLAUDE.md                      # architecture contract (English, normative)
├── DEVELOPMENT_PLAN.md / PROGRESS.md / GUIDE.md
├── src/
│   ├── app/                       # backend (import root = `app`)
│   │   ├── api/                   # FastAPI routers (thin)
│   │   ├── agent/                 # LangGraph state, nodes/, tools/, policies, budget, graph
│   │   ├── parsing/               # MinerU client + mapping → ParsedBlock / hierarchy
│   │   ├── enrichment/            # summaries, keywords, claims, facts, manifest
│   │   ├── retrieval/             # RetrievalService (single Vespa entry-point)
│   │   ├── vespa/                 # app package, feed, encoders, mock
│   │   ├── providers/             # chat/embedding/reranker adapters (+ deterministic mocks)
│   │   ├── services/              # chat / session / document / ingestion / qa / facts
│   │   ├── models/                # ORM + Pydantic v2 domain models
│   │   └── evaluation/            # parser, retrieval, QA, goal-score harnesses
│   └── frontend/                  # Next.js 15 App Router (Chat / Documents / Sessions / Settings)
├── data/                          # fixtures (committed) + parsed cache + storage (gitignored)
├── deploy/                        # docker-compose, vespa application package, mineru wrapper docs
├── migrations/                    # Alembic env + versions
├── scripts/                       # MinerU PoC, deploy_vespa, ingest_sample_arxiv, eval/score runners
├── tests/                         # unit/ integration/ e2e/ evaluation/ fixtures/
└── artifacts/                     # repair-loop iterations + evaluation reports
```

Backend import root: `app` (under `src/app/`). All Python commands run from repo root via `uv`.

---

## 4. Quick start

### 4.1 Prerequisites

- `uv` ≥ 0.4 — installer: <https://docs.astral.sh/uv/>
- Docker Engine + Compose v2 (for Vespa + Postgres dev stack)
- A local MinerU server (uses your own vLLM @ `http://localhost:8001`; see `deploy/mineru/README.md`)
- Node.js 20+ for the frontend (Next.js 15)

### 4.2 First-time setup

```bash
uv sync                                          # install Python deps from uv.lock
docker compose -f deploy/docker-compose.yml up -d postgres vespa
uv run alembic upgrade head                      # create DB schema
cp .env.example .env                             # then edit if needed
uv run uvicorn app.main:app --reload             # http://localhost:8000
npm --prefix src/frontend install
npm --prefix src/frontend run dev                # http://localhost:3000
```

### 4.3 Parse a sample PDF (Phase 1 PoC sanity check)

```bash
uv run python scripts/ingest_sample_arxiv.py     # downloads a small arXiv paper
uv run python scripts/mineru_poc.py              # hybrid-http-client → data/parsed/
```

---

## 5. Running tests

```bash
uv run ruff check .                              # lint
uv run mypy src/app                              # type-check (95 source files clean)
uv run pytest -q                                 # full suite (~387 tests, ~2 s)
uv run pytest tests/unit -q
uv run pytest tests/integration -q
uv run pytest tests/evaluation -q                # parser + retrieval + QA + goal score
uv run pytest tests/e2e -q                       # cross-stack chat / session isolation

# Frontend
npm --prefix src/frontend run lint
npm --prefix src/frontend run typecheck
npm --prefix src/frontend run test
npm --prefix src/frontend run build
```

CLAUDE.md §12.1 caps each test file at ≤ 10 collect-only items; the suite enforces this via:

```bash
uv run pytest --collect-only -q | awk -F'::' '/::/{print $1}' | sort | uniq -c | sort -rn | head
```

---

## 6. Evaluation harnesses & reports

All harnesses are deterministic and run without paid APIs.

```bash
# Parser evaluation (LightRAG golden corpus)
uv run python scripts/run_parser_eval.py
# → artifacts/evaluation/parser-report.{json,md}

# Retrieval evaluation (5 modes, leakage check)
uv run python scripts/run_retrieval_eval.py
# → artifacts/evaluation/retrieval-report.{json,md}

# Golden QA evaluation (GUIDE §19, 7 case kinds)
uv run python scripts/run_qa_eval.py
# → artifacts/evaluation/qa-report.{json,md}

# Goal coverage score (GUIDE §21) — depends on the three above
uv run python scripts/run_goal_score.py
# → artifacts/evaluation/goal-score.{json,md}
```

Current results (this commit):

| Report | Result |
|---|---|
| Parser eval | LightRAG `2410.05779v3`: gate PASS (heading-F1=0.33, math-recall=1.0, refs=21, figs=6, tables=6) |
| Retrieval eval | 5 modes; all `recall@10=1.0`; **leakage=0** in every mode; rerank does not regress nDCG |
| Golden QA | **7/7 cases pass**; cross-chat refusal verified |
| Goal coverage | **100/100**, mandatory gates all PASS |

---

## 7. Isolation contract (CLAUDE.md §2)

Four layers — none may be skipped:

1. **DB query** — every document-scoped table carries `chat_id`; service-layer queries always include `WHERE chat_id = :current_chat_id`.
2. **Vespa filter** — `RetrievalService._yql_where` injects `chat_id contains "<current>"` before any user-supplied filter; the YQL builder is the only public Vespa entry point.
3. **API scope** — route layer verifies session/document ownership via `session_service` / `document_service`; cross-chat URLs return 404.
4. **Agent layer** — `chat_id` lives in `AgentState` and is propagated to tools by the service layer; tool parameter schemas use `extra="forbid"` and do **not** include `chat_id`, so the LLM cannot inject it.

Citations are checked twice: `PolicyEngine.enforce_citations` drops drafts whose `chat_id != state.chat_id` (policy 12) and whose `document_id` is not in the `ChatDocument` association (policy 13).

---

## 8. Provider settings

Chat, embedding and reranker profiles are independent. Supported provider types:

- OpenAI (`gpt-*` etc.)
- Gemini Native (Google GenAI SDK)
- Gemini OpenAI-compatible
- Generic OpenAI-compatible (vLLM, OpenRouter, etc.)

API keys are encrypted at rest via Fernet (`APP_ENCRYPTION_KEY`) — never logged, never returned to the frontend (masked everywhere). Connection-test endpoints validate model+latency and return sanitised error strings.

For demo without a DB row, env-level fallback variables (`LLM_PROVIDER` / `LLM_API_URL` / `LLM_MODEL` / `LLM_API_KEY`) build a `OpenAICompatChatProvider` on the fly — see `scripts/smoke_agent_e2e.py` for an end-to-end demo.

---

## 9. Known limitations & next steps

### Not validated in this environment
- **Vespa cluster (live)** — the application package deploys via `scripts/deploy_vespa.py --dry-run` and the schema validates locally; spinning up `docker compose ... vespa` and feeding live data was previously executed but is not part of the CI regression. Retrieval evaluation uses a fake transport that replays the real RetrievalService logic against a fixture corpus.
- **MinerU multi-paper sweep** — Phase 1 PoC was run on one paper (LightRAG `2410.05779v3`) and passed all reliability checks; the harness supports more papers, but the included golden corpus has one entry.
- **Real LLM E2E** — `scripts/smoke_agent_e2e.py` was run during Phase 7 against Gemma-4-31B-it via vLLM; the regression suite uses deterministic mocks only.

### Three most important next improvements
1. **Live Vespa CI lane** — extend `docker compose up vespa` + retrieval feed/query smoke tests into the regression so the YQL builder is exercised against the real engine, not only the fake transport.
2. **Multi-paper parser sweep** — expand `data/fixtures/golden/` with 2–3 additional arXiv papers covering different layouts and re-tighten parser-eval thresholds.
3. **Real-LLM CI nightly** — keep mocks for fast tests but add a nightly nightly that runs `smoke_agent_e2e.py` against the configured vLLM endpoint so prompt regressions are caught.

---

## 10. Phase summary

| Phase | Status | Goal block | Highlight |
|---|---|---|---|
| 0 — Bootstrap | ✅ | — | `src/app` layout + uv + Makefile |
| 1 — MinerU PoC | ✅ | — | hybrid client direct to vLLM @ 8001 (no wrapper) |
| 2 — Foundation | ✅ | Backend 10/10 | Alembic, ORM, providers, encryption, compose |
| 3 — Chat/Session/Doc | ✅ | Isolation core | 4-layer isolation tests |
| 4 — Parsing pipeline | ✅ | Parsing 15/15 | middle.json → ParsedBlock + hierarchy |
| 5 — Enrichment | ✅ | — | section/doc summaries, facts, manifest |
| 6 — Vespa retrieval | ✅ | Retrieval 20/20 | RRF + native/cross-encoder rerank, leakage=0 |
| 7 — LangGraph agent | ✅ | Agent QA 20/20 | 15 nodes, 7 tools, 14 policies, SSE |
| 8 — Frontend | ✅ | Frontend 10/10 | Next.js 15 + 3-region UI + Settings |
| 9 — Evaluation & repair | ✅ | Goal 100/100 | Golden QA + E2E + goal-score scorer |

See `artifacts/final-report.md` for the full 20-item delivery summary (GUIDE §29).
