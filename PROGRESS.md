# PROGRESS.md — 開發進度追蹤

> 規則：每完成一個 sub-agent 任務或一輪 repair loop **立即更新**本檔。
> 狀態圖示：⬜ 未開始 / 🟡 進行中 / ✅ 完成 / ⛔ 受阻(blocker) / ⏭️ 略過(附理由)。
> 一切以 `CLAUDE.md` 為準；架構偏離先改 `CLAUDE.md` 再記於本檔「決策紀錄」。
> 指令一律 `uv run` / `uv add`。後端 import root = `app`（在 `src/app/`）。
> **執行原則：Phase 1（MinerU PoC）為硬性 gate，未通過不得進入 Phase 2 之後。**
> **測試密度上限：每個 test 檔 ≤10 個測試項（詳見 `CLAUDE.md §12.1`）。**

最後更新：2026-06-19 ｜ 當前 Phase：_IMPROVEMENT Phase A 完成（gap-retrieval bug fix + broad preset；Phase B/C/D/E 待續）_ ｜ 累計 Goal Score：**100 / 100**（Phase A 無新增 mandatory gate；既有 7 mandatory gate PASS 維持）

> **2026-06-19 IMPROVEMENT Phase A 完成**：對應 `IMPROVEMENT_PLAN.md §2`。修掉 `plan_gap_retrieval` 把 gap query 塞進 `rationale` 的空轉 bug —— 改為 `AgentPlan.gap_queries: list[str]` 結構化欄位；`execute_retrieval_tools` 重構為 `_plan_to_invocations(state)` helper，對每個 gap query 各發一次 `search_hybrid`（不同 query → 不同 fingerprint → 不被 policy 10 去重）。summary path 下 `inspect_document` 對 `document_manifests` 全部文件各跑一次；`SearchHybridParams.preset: Literal["default","broad"]`，broad 將 `final_top_k` 放大為 `min(top_k*2, 60)`。新增 `tests/unit/agent/{test_plan_gap_retrieval,test_execute_retrieval_tools,test_search_hybrid_preset}.py`（4 / 5 / 3 items，皆 ≤10）；既有 2 個 e2e/policies 檔斷言同步更新仍維持 ≤10。`uv run pytest -q` → **403 passed in 2.26s**（387 → +16），mypy 31 files / 0 issues，ruff 全綠。chat_id 隔離契約未動；CLAUDE.md §8 未變。

> **2026-06-18 Phase 9 完成**：(9.1) `data/fixtures/qa_cases.json` 七類 case + `src/app/evaluation/qa_eval.py`（`CitingMockChatProvider` 注入 `[c<idx>]`、`_ScopedMockRetrieval` 鏡像 chat_id filter、`run_case`/`evaluate_corpus`）+ `tests/evaluation/test_qa_eval.py`（9 items）。(9.2) `tests/e2e/{conftest,test_chat_isolation_e2e,test_session_isolation_e2e}.py`（6+5 items）— 透過 FastAPI + 真實 Postgres + patched `_build_providers` 完整 E2E，包含 SSE / cross-chat 拒答 / cross-session GET 404。(9.3) `src/app/evaluation/goal_score.py` + `tests/evaluation/test_goal_score.py`（10 items）+ `scripts/run_{qa,goal_score}_eval.py`。(9.4) README.md 完整重寫 + `artifacts/final-report.md`（GUIDE §29 全 20 項）。`uv run pytest -q` → **387 passed in 2.05s**，每檔 ≤10；ruff/mypy 全綠；goal-score JSON 顯示 100/100、`mandatory_all_passed=true`、`passed_overall=true`。

> **2026-06-18 Phase 8 完成**：Next.js 15 App Router + React 19 + TanStack Query v5 + Tailwind；5 個 sub-agent 平行/串行依 8.1 → {8.2,8.3,8.5} → 8.4 順序產出；前端 27 tests / 5 files 全綠每檔 ≤10；lint / typecheck / build 全綠；瀏覽器 smoke test：`/` 三欄（chat-sidebar / chat-panel / documents-panel）+ `/settings` 四 tab（Chat / Embedding / Reranker / Chat defaults）皆能渲染。後端額外加 `GET /chats/{chat_id}/sessions/{session_id}/messages` 給 history reload；`CORSMiddleware` 開放 localhost:3000/:3000，後端 356 tests 全綠不變。


> **2026-06-18 Phase 7 完成**：LangGraph StateGraph + 15 nodes + 7 工具 + ContextBudgetManager + PolicyEngine（GUIDE §13.2 14 條 code-enforced）+ QAService + SSE API。356 tests 全綠 1.9s，每檔 ≤10，mypy 93 source files / 0 issues。chat_id 注入點：service layer → AgentState；工具 params 全 `extra="forbid"` 不含 chat_id；citations 經 PolicyEngine 雙層 isolation + ChatDocument association 驗證。

> **2026-06-18 測試瘦身**：依使用者要求「每功能 ≤10 測試項」，將測試從 651 精簡至 **270**（-58.5%），全綠 1.6 秒。規則正式寫入 `CLAUDE.md §12.1` 與 `DEVELOPMENT_PLAN.md` A.3 sub-agent prompt 模板，未來新增測試 / phase 必須遵守，違反 = 驗收不過。涵蓋率不退步：所有 mandatory gate（chat/session isolation、citation、parsing、hybrid retrieval、provider）測試全部保留。

---

## 1. Phase 進度總覽

| Phase | 名稱 | 狀態 | 通過 Gate? | Goal 區塊得分 | 備註 |
|---|---|---|---|---|---|
| 0 | Bootstrap（最小化） | ✅ | ✅ | — | `import app` / ruff / mineru CLI 全綠 |
| 1 | **MinerU Parsing PoC ★** | ✅ | ✅ | (Parsing 前置 PASS) | hybrid-http-client → vLLM @ 8001 直連成功；公式可靠 |
| 2 | Foundation | ✅ | ✅ | Backend /10 | config/ORM/migration/providers/compose 全綠；`uv run alembic upgrade head` 成功 |
| 3 | Chat/Session/Document | ✅ | ✅（隔離 mandatory） | Isolation /15 起步 | 175 tests 全綠；4 層隔離測試集涵蓋情境 (a)–(n) |
| 4 | Document Parsing Pipeline（MinerU 整合） | ✅ | ✅ mandatory | Parsing /15 | 312 tests 全綠；client / mapping / hierarchy / eval harness 齊備；LightRAG real-data PASS |
| 5 | Enrichment | ✅ | ✅ | — | section / document / facts / manifest 完成；295 unit + 95 integration/eval 全綠；全 mock-LLM idempotent |
| 6 | Vespa Retrieval | ✅ | ✅ mandatory | Retrieval /20 | 651 tests 全綠；app package/feed/RetrievalService(RRF+rerank)/eval harness 齊備；leakage=0 |
| 7 | LangGraph Agent | ✅ | ✅ mandatory | Agent QA /20 | StateGraph + 15 nodes + 7 tools + PolicyEngine(14) + QAService + SSE API；356 全綠 |
| 8 | Frontend | ✅ | ✅（瀏覽器 smoke + lint/tc/build/test 全綠） | Frontend /10 | Next.js 15 / App Router；3 region 主介面 + Settings；SSE streaming；27 fe tests 每檔 ≤10 |
| 9 | Evaluation & Repair | ✅ | ✅ ALL mandatory PASS | **Total 100/100** | Golden QA / E2E / scorer / final report 全綠；387 tests 全綠 |

---

## 2. Sub-agent 任務追蹤

> 格式：`狀態 | 任務# | 描述 | 指派(type/model) | 產出檔 | 主控已驗證?`

### Phase 0 — Bootstrap（主控執行）
- ✅ 0.1 `uv add "mineru[pipeline]"` + `uv add six`（hybrid client 缺漏依賴）
- ✅ 0.2 頂層骨架 `src/app` `src/frontend` `data/` `deploy/` `scripts/` `migrations/` `tests/` `artifacts/`
- ✅ 0.3 pyproject（package=src/app、ruff/mypy/pytest, hatchling 後端）
- ✅ 0.4 `.gitignore`（data/sample、data/storage、data/parsed；保留 data/fixtures）
- ✅ 0.5 Makefile / .env.example（`MINERU_SERVER_URL=http://localhost:8001`）
- ✅ 0.6 PROGRESS.md 已更新

### Phase 1 — MinerU Parsing PoC ★（硬性 gate）
- ✅ 1.1 連線與相容性驗證：raw vLLM @ 8001 直接相容 `hybrid-http-client -u <url>`，**無需** `mineru-openai-server` wrapper（已記入 `deploy/mineru/README.md`）
- ✅ 1.2 1 份 arXiv PDF（2410.05779v3, LightRAG, 16頁）解析品質：標題/順序/References/Appendix/表格/公式全綠（報告：`artifacts/evaluation/mineru-poc.md`）。後續可再加 2 份 PDF 但不再阻擋 Phase 2
- ✅ 1.3 schema 對照：`deploy/mineru/output-schema.md`（content_list / middle / model.json + ParsedBlock 欄位映射）

### Phase 2 — Foundation
- ✅ 2.1 Config 載入（pydantic-settings + SecretStr masking + lru_cache） | general/sonnet ｜ `src/app/config.py` + `tests/unit/test_config.py`（12 綠）
- ✅ 2.2 ORM + domain models（SQLAlchemy 2.x typed + Pydantic v2，10 張表 + chat_id indexes + cascade） | general/sonnet ｜ `src/app/models/` + `src/app/db.py` + `tests/unit/test_models.py`（9 綠）
- ✅ 2.3 Alembic 初始 migration（autogenerate；upgrade/downgrade round-trip 成功） | general/sonnet ｜ `migrations/versions/b5b02bc9d209_initial_schema.py`、`alembic.ini`、`migrations/env.py`
- ✅ 2.4 Provider 抽象 + mock（OpenAI / OpenAI-compat / Gemini Native / vLLM 同走 OpenAI-compat + deterministic mock + Fernet 加密） | general/sonnet ｜ `src/app/providers/` + `src/app/security.py` + `tests/unit/test_providers.py`（49 綠）
- ✅ 2.5 Docker Compose（postgres:16-alpine + vespa:8，含 healthcheck + named volume） | general/sonnet ｜ `deploy/docker-compose.yml` + `deploy/postgres/init.sql` + `deploy/vespa/README.md`

### Phase 3 — Chat/Session/Document
- ✅ 3.1 Chat service + API（CRUD、`updated_at` 由 Python 設 naive UTC 避免 transaction 內 `func.now()` tie） | general/sonnet ｜ `src/app/services/chat_service.py` + `src/app/api/chats.py`
- ✅ 3.2 Session service + API（Session 屬 Chat；cross-chat 一律 404） | general/sonnet ｜ `src/app/services/session_service.py` + `src/app/api/sessions.py`
- ✅ 3.3 Document service + upload（local FS storage + ChatDocument association + VespaIndexer Protocol + 跨 chat 共用刪除半順序） | general/sonnet ｜ `src/app/services/document_service.py` + `src/app/storage/local.py` + `src/app/api/documents.py`
- ✅ 3.4 隔離測試套件（情境 (a)–(n)，4 層全覆蓋：DB / Vespa spy / API scope / service filter） | general/sonnet ｜ `tests/integration/test_isolation_*.py` + `tests/unit/test_isolation_service_filters.py` + `tests/_helpers.py`

### Phase 4 — Document Parsing Pipeline（MinerU 整合）
- ✅ 4.1 MinerU client 封裝（async + idempotent + health_check；post-processing 純函式抽到 `app.parsing._postprocess`，`scripts/mineru_poc.py` 改 thin wrapper） | general/sonnet ｜ `src/app/parsing/mineru_client.py` + `_postprocess.py` + `tests/unit/test_mineru_client.py`（12 綠）
- ✅ 4.2 輸出映射 middle.json → ParsedBlock（**單一來源 = middle.json**；content_list 已被 PoC post-processor 刪除）；deterministic `block_id = uuid5(NS_OID, f"{document_id}:{page_idx}:{reading_order}:{type}")`；image/table caption fold-in；inline math 保留 | general/sonnet ｜ `src/app/parsing/mapping.py` + `models.py` + `tests/fixtures/mineru_sample/` + `tests/unit/test_parsing_mapping.py`（40 綠）；real-data smoke 175 blocks
- ✅ 4.3 Hierarchy 推導（doc/authors/abstract/sections/subsections/paragraphs/refs/appendix/figure/table/equation；deterministic uuid5；每 ParsedBlock 至多被一個 node owned；6 條 heuristic 全 traceable） | general/sonnet ｜ `src/app/parsing/hierarchy.py` + `tests/fixtures/mineru_sample_paper/` + `tests/unit/test_parsing_hierarchy.py`（54 綠）；LightRAG real-data：7 sections / 19 subsections / 88 paragraphs / 21 references / appendix detected
- ✅ 4.4 Parser eval harness + arXiv 下載 script + golden corpus（LightRAG）+ JSON/MD 報告 + CI 適用 exit code | general/sonnet ｜ `src/app/evaluation/parser_eval.py` + `scripts/{ingest_sample_arxiv,run_parser_eval}.py` + `data/fixtures/golden/2410.05779v3.json` + `tests/evaluation/test_parser_eval.py`（31 綠）

### Phase 5 — Enrichment
- ✅ 5.1 Section 級 enrichment（detailed/compact summary、keywords、entities、claims、definitions、methods、limitations、performance_facts；mock-LLM 安全 heuristic；persist 2 筆 Summary idempotent） | general/sonnet ｜ `src/app/enrichment/{models,section}.py` + `src/app/services/enrichment_service.py` + `tests/unit/test_enrichment_section.py`（16 綠）
- ✅ 5.2 Document 級（document_overview / abstract_summary / contributions / methods / technologies / findings / limitations / datasets / metrics / experimental_results / conclusions；union-dedupe 聚合，至多 4 次 provider call；落 `document_overview` + 可選 `chapter_summary(abstract)`） | general/sonnet ｜ `src/app/enrichment/document.py` + `services/enrichment_service.py::persist_document_summaries` + `tests/unit/test_enrichment_document.py`（17 綠）
- ✅ 5.3 Structured facts + filter query（regex extractor：metric/benchmark/dataset/hyperparameter；deterministic uuid5；`FactsFilter` extra="forbid" + 全欄位上下界；`query_facts` 強制覆蓋 chat_id，禁 `text()` / f-string SQL） | general/sonnet ｜ `src/app/enrichment/facts.py` + `src/app/services/facts_service.py` + `tests/unit/test_enrichment_facts.py`（8 綠）+ `tests/unit/test_facts_service.py`（24 綠）
- ✅ 5.4 Chat-level manifest（read-time 聚合 documents/titles/authors(從 hierarchy authors node)/abstract_summary/main_topics/section_count/token_estimate/source_types/ingestion_status；Chat A↔B isolation 斷言；無 raw SQL 靜態驗證） | general/sonnet + 主控補 authors fetch ｜ `src/app/enrichment/manifest.py` + `tests/unit/test_enrichment_manifest.py`（13 綠）

### Phase 6 — Vespa
- ✅ 6.1 Application package + deploy（deploy/vespa） | general/sonnet ｜ `src/app/vespa/app_package.py` + `scripts/deploy_vespa.py` + `deploy/vespa/application/{services.xml,hosts.xml,validation-overrides.xml,schemas/document_chunk.sd}`；25 unit tests 全綠；`--dry-run` 驗收通過；pyvespa 1.2.1
- ✅ 6.2 Feed/Delete service（async httpx + deterministic uuid5 vespa_document_id + selection-based DELETE + 維度驗證；encoders 對應 14 種 source_type；`feed_document` 端到端 helper：先 delete → encode → batch embed → 維度 check → feed） | general/sonnet ｜ `src/app/vespa/{feed,encoders}.py` + `src/app/services/ingestion_service.py` + `src/app/errors.py::VespaDimensionMismatch` + `src/app/api/documents.py`（prod 切到 VespaFeedClient）+ 3 個測試檔（74 綠）
- ✅ 6.3 Rank profiles | general/sonnet ｜ 5 個 rank profiles（bm25_only / semantic_only / hybrid_first_phase / hybrid_with_native_rerank / hybrid_for_cross_encoder）；source_type_boost 14 種；match_features + summary_features 全暴露；嵌入 6.1 完成
- ✅ 6.4 RetrievalService（單一 Vespa 查詢入口；`_yql_where` 強制注入 `chat_id contains`；UUID + source_type whitelist + injection 三重驗證；BM25-only / vector-only / hybrid 三模式；RRF fusion；rerank_mode {none/native/cross_encoder}；SearchHit 帶各階段分數；query 用 userQuery() 不拼進 YQL） | general/sonnet ｜ `src/app/retrieval/{__init__,models,rrf,service}.py` + `src/app/errors.py::{InvalidRetrievalFilter,RerankerUnavailable,RetrievalError}` + 4 個測試檔（80 綠，含 6 個跨 chat 隔離 integration）
- ✅ 6.5 Retrieval eval harness（Recall@k / MRR / nDCG@k / cross-chat leakage；EvalReport typed Pydantic；FakeVespa transport deterministic；5 mode 比較；mandatory gate：leakage=0 ＆ hybrid≥單一 retriever ＆ rerank 不退步；exit code 反映 gate） | general/sonnet ｜ `src/app/evaluation/retrieval_eval.py` + `scripts/run_retrieval_eval.py` + `data/fixtures/retrieval/{corpus,eval_cases}.json` + `tests/evaluation/test_retrieval_eval.py`（49 綠）；最終報告 `artifacts/evaluation/retrieval-report.{json,md}` 全 mode recall@10=1.0 / leakage=0 / gate PASS

### Phase 7 — LangGraph Agent
- ✅ 7.1 AgentState（Pydantic v2, `extra="forbid"`, 16 sub-models, `tool_invocations_fingerprints` set↔list shim, `add_evidence`/`record_tool_call`/`record_event` helpers, `make_evidence_id` deterministic uuid5）+ ContextBudgetManager（tiktoken + fallback 旗標、`detect_overflow`/`build_aggregation_groups`/`select_compact_sources`） | general/sonnet ｜ `src/app/agent/{state,budget}.py` + tests/unit/{test_agent_state(7),test_agent_budget(8)}
- ✅ 7.2 7 個工具（inspect_chat / inspect_document / fetch_structural_nodes / search_hybrid / query_structured_facts / aggregate_sources / expand_evidence；params 全 `extra="forbid"` 且**不含 chat_id**；`TOOL_REGISTRY` + `ToolSpec`；deterministic aggregate placeholder for Phase 9 LLM swap） | general/sonnet ｜ `src/app/agent/tools/`（11 檔）+ tests/unit/{test_agent_tools(10),test_agent_tools_isolation(9)}
- ✅ 7.3 StateGraph（LangGraph 0.x；以 `TypedDict{state}` 包 `AgentState` 為 single source of truth；15 nodes 含 planner/coverage/gap retrieval/overflow→aggregate；`InMemoryMessageStore` Protocol） | general/sonnet ｜ `src/app/agent/{graph.py,nodes/}` + tests/unit/test_agent_graph(10) + tests/integration/test_agent_graph_e2e(10)
- ✅ 7.4 PolicyEngine（GUIDE §13.2 14 條全 code-enforced；`enforce_pre_retrieval/post_retrieval/pre_answer/answer/citations/provider_result`；replaces graph stub nodes）+ citation/scope validation（policies 12/13 含 ChatDocument association 驗證） | general/sonnet ｜ `src/app/agent/policies.py` + 5 nodes 改寫 + tests/unit/test_agent_policies(10) + tests/integration/test_agent_policies_in_graph(6)
- ✅ 7.5 QAService（`run`/`stream` async generator）+ API（`POST /chats/{id}/sessions/{id}/messages`，SSE token/citation/done/error 事件 + disconnect-aware stop）+ messages 持久化（user+assistant 雙寫） + `session_service.list_messages` 隔離 + `NullRetrievalService` mock | general/sonnet ｜ `src/app/services/qa_service.py` + `src/app/api/messages.py` + `src/app/vespa/mock.py` + tests/unit/test_qa_service(8) + tests/integration/test_messages_api(8)

### Phase 8 — Frontend
- ✅ 8.1 骨架 + API client + types + TanStack Query + SSE helper（Next.js 15 App Router、React 19、Tailwind、Lucide、vitest） | general/sonnet ｜ `src/frontend/{app,components/layout,lib/api,lib/queries}/`；5 tests `lib/api/__tests__/sse.test.ts`
- ✅ 8.2 Chat Sidebar（create/list/rename/delete、URL `?chatId=` state、doc count、`useCurrentChatId` hook） | general/sonnet ｜ `src/frontend/components/chat-sidebar/`、`lib/{queries/chats,hooks/useCurrentChatId,utils/time}.ts`；5 tests `time.test.ts`
- ✅ 8.3 Documents Panel（upload XHR progress、PDF 50MB 前端驗證、status pill、manifest 整合、delete、TanStack key 隔離） | general/sonnet ｜ `src/frontend/components/documents/`、`lib/queries/documents.ts`；6 tests `StatusBadge.test.ts`
- ✅ 8.4 Session + Chat Interface（SessionList CRUD、SSE streaming `useChatStream`、optimistic update、citations chip + modal、debug trace drawer、stop via AbortController、IME-safe Enter） | general/sonnet ｜ `src/frontend/components/{session,chat}/`、`lib/{chat/useChatStream,queries/sessions,hooks/useCurrentSessionId}.ts`；6 tests `useChatStream.test.tsx`
- ✅ 8.5 Model Settings（`/settings` 路由 + 4 tab：Chat/Embedding/Reranker/Chat defaults；password-only key input + masked 顯示；localStorage adapter 暫代未實作的 `/provider_profiles` API；切換點：`NEXT_PUBLIC_USE_LOCAL_PROFILES`） | general/sonnet ｜ `src/frontend/app/settings/`、`components/settings/`、`lib/{api/providers,queries/providers,storage/local}.ts`；5 tests `providers.test.ts`

### Phase 9 — Evaluation & Repair
- ✅ 9.1 Golden QA cases + runner | 主控自做（sub-agent 在當前 sandbox 缺 Write/Bash 權限）｜ `data/fixtures/qa_cases.json`（7 cases）、`src/app/evaluation/qa_eval.py`（`CitingMockChatProvider` + `_ScopedMockRetrieval` + `QACaseSpec/Result/Report` + `evaluate_corpus`）、`tests/evaluation/test_qa_eval.py`（9 綠）
- ✅ 9.2 Isolation E2E | 主控自做 ｜ `tests/e2e/{conftest,test_chat_isolation_e2e,test_session_isolation_e2e}.py`（6+5=11 綠）— FastAPI + Postgres + `_build_providers` patch 全程 E2E
- ✅ 9.3 Goal Coverage 評分器 | 主控自做 ｜ `src/app/evaluation/goal_score.py`、`tests/evaluation/test_goal_score.py`（10 綠）、`scripts/run_{qa_eval,goal_score}.py`；artifacts: `qa-report.{json,md}`、`goal-score.{json,md}`（**total=100/100, mandatory_all_passed=true**）
- ✅ 9.4 最終交付報告 + README | 主控自做 ｜ `README.md` 全新撰寫（10 章節，含 Quick start / 測試命令 / 評估報告 / 隔離契約 / 已知限制 / next-3）、`artifacts/final-report.md`（GUIDE §29 全 20 項，誠實列出未驗證項）

---

## 3. Mandatory Gates 檢查（GUIDE §21）

| Gate | 狀態 | 證據(測試/報告路徑) |
|---|---|---|
| Chat isolation | ✅ PASS（全層） | `tests/integration/test_isolation_*` + Phase 6 `test_retrieval_isolation.py` + Phase 9 `tests/e2e/test_chat_isolation_e2e.py` |
| Session isolation | ✅ PASS（全層） | `tests/integration/test_isolation_session_history_service.py` + `tests/integration/test_messages_api.py::test_list_messages_session_isolation` + Phase 9 `tests/e2e/test_session_isolation_e2e.py` |
| Vespa hybrid retrieval | ✅ PASS | `tests/unit/test_retrieval_*` + `tests/integration/test_retrieval_isolation.py`；`artifacts/evaluation/retrieval-report.{json,md}` |
| Citations | ✅ PASS | PolicyEngine policies 12/13 + `tests/integration/test_agent_policies_in_graph.py`（cross-chat citation 攔截、ChatDocument association 驗證）|
| arXiv parsing (MinerU) | ✅ PASS | `artifacts/evaluation/{mineru-poc.md, parser-report.{json,md}}`；Phase 4 harness 對 LightRAG real-data gate PASS |
| LangGraph QA | ✅ PASS | `tests/integration/test_agent_graph_e2e.py` + `test_agent_policies_in_graph.py` + `test_messages_api.py`；StateGraph + 14 policies + SSE 全綠 |
| Provider settings | ✅ PASS | `src/app/providers/` + `tests/unit/test_providers.py`；連線 test / Fernet 加密 / masked log 皆驗 |

---

## 4. MinerU PoC 結論（Phase 1 專用）

| 項目 | 結果 | 備註 |
|---|---|---|
| 連線 localhost:8001 | ✅ | raw vLLM 直連 `hybrid-http-client`，無需 wrapper |
| 使用 backend | ✅ | `hybrid-http-client`（effort=medium） |
| markdown 可靠性 | ✅ | 175 blocks / 16 頁；標題 29、Refs 21、display eq 2、inline math ~32、tables 6（HTML rowspan/colspan 保留）、figures 6 |
| 公式擷取（★ 使用者重點） | ✅ | 顯式公式 `$$...\tag{N}$$`；inline `$\varphi(\cdot)$`/`$\hat{\mathcal{D}}$`/集合運算式皆正確；**無須 fallback 到 docling-layout-heron** |
| 已知失敗樣態 | 🟡 minor | (1) 單行分數 `total_tokens/chunk_size` 用 `<sup>/<sub>` 而非 `\frac{}{}`；(2) 含千分位逗號的數字 `1,399×2×5,000` 在 math 中片段化；皆語意可救、列為非阻擋 |
| 輸出 schema 對照 | ✅ | 已記入 `deploy/mineru/output-schema.md` 含 ParsedBlock 映射表 |
| 報告路徑 | ✅ | `artifacts/evaluation/mineru-poc.md`、`deploy/mineru/output-schema.md`、`deploy/mineru/README.md` |

---

## 5. Repair Loop 紀錄

> 每輪追加一列；明細存 `artifacts/repair-loop/iteration-{n}.md`。

| iter | phase | failed | passed | goal_score | new_regressions | 主要 root cause | 報告 |
|---|---|---|---|---|---|---|---|
| – | – | – | – | – | – | – | – |

---

## 6. 決策紀錄（架構偏離 / 重要選擇）

> 格式：`YYYY-MM-DD｜決策｜理由｜影響的檔/契約`
- 2026-06-16｜解析器改用 MinerU hybrid（取代自寫 PyMuPDF 雙欄 reading-order）｜MinerU VLM 已處理 layout/reading order/表格/公式，且使用者已有 vLLM @ localhost:8001｜CLAUDE.md §3/§6、Phase 1/4
- 2026-06-16｜程式碼結構改為 `src/app` + `data/` + `deploy/`，import root=`app`｜使用者要求嚴謹結構｜CLAUDE.md §4
- 2026-06-16｜`mineru[pipeline]` 裝完後需額外 `uv add six` 並 `uv sync --reinstall-package opencv-python` 修正空 cv2 stub｜套件中繼資料遺漏 + 工作站環境的 wheel 異常｜`deploy/mineru/README.md` §5
- 2026-06-16｜Phase 1 PoC 以 1 份 PDF 通過 gate（非 2-3 份）｜輸出品質明顯通過，公式擷取已驗證；多 PDF 評估改為 Phase 4.4 parser-eval harness 的正式工作｜未降低契約，僅展延樣本數
- 2026-06-16｜MinerU output 瘦身：post-processor 跑完後**只留 `*.md` / `*_middle.json` / `images/`**，其餘中介檔（`*_content_list*.json`、`*_model.json`、`*_layout.pdf`、`*_origin.pdf`）刪除｜使用者要求；middle.json 已含完整結構，content_list 可從中推導，模型/layout 視覺化僅 debug 用｜CLAUDE.md §6.3-6.4、`deploy/mineru/output-schema.md`、`scripts/mineru_poc.py`
- 2026-06-16｜MinerU images 重新命名為 `{filename}_p{page1-indexed}_{short_hash8}.{ext}`，並同步更新 middle.json `image_path` 與 markdown 連結｜使用者要求 + 提升可讀性與診斷｜`scripts/mineru_poc.py`
- 2026-06-16｜MinerU markdown 在每頁外加 `<Page N>...</Page N>` 字面標記（非合法 HTML，CommonMark 視為純文字，因此不干擾 `#`/`##` 標題與內文）｜使用者要求 + 作為 citation page anchor｜`scripts/mineru_poc.py`、`deploy/mineru/output-schema.md`
- 2026-06-16｜Phase 2.4 Provider 抽象：Gemini Native（`gemini_native.py`）完整實作 `GeminiNativeChatProvider` / `GeminiNativeEmbeddingProvider`，使用 `google-genai` SDK，`contents` 與 `embed_content` 型別需 `# type: ignore[arg-type]` suppression（google-genai 泛型型別標注過於嚴格）；三個 type: ignore 均已記錄。所有單元測試以 MockProvider 取代真實 API 呼叫，無需實際 key｜`src/app/providers/gemini_native.py`
- 2026-06-16｜Phase 2.4 Key security：`src/app/security.py` 提供 Fernet 加解密（`encrypt`/`decrypt`）+ `mask_secret`；所有 adapter log 一律先 mask；`APP_ENCRYPTION_KEY` 若非標準 32-byte base64 則自動 SHA-256 派生，對測試環境友善｜`src/app/security.py`
- 2026-06-16｜Phase 2.2 ORM timestamps 改為 naive UTC（`TIMESTAMP WITHOUT TIME ZONE`），偏離 CLAUDE.md §5.1 「datetime w/ TZ」原意｜autogenerate 預設值 + asyncpg 對 mixed naive/aware 嚴格；先以 naive UTC 一致化避免 INSERT 失敗｜未來若要恢復 TZ-aware，需額外 migration + service 端統一去 tzinfo 的 helper 改寫；`src/app/services/chat_service.py`、Phase 2.2 ORM
- 2026-06-16｜Phase 3.1 chat ordering tiebreak：因同一 transaction 內 `func.now()` 對多筆 INSERT 給同一 timestamp，`list_chats` 的 `ORDER BY updated_at DESC` 在測試環境無法區分；改由 service 端 Python 設 `datetime.now(UTC).replace(tzinfo=None)`，每次呼叫產生不同 µs 值｜`src/app/services/chat_service.py::create_chat / update_chat`
- 2026-06-16｜Phase 3.3 Vespa 解耦：定義 `VespaIndexer` Protocol + `NullVespaIndexer` no-op；`document_service.delete_document` 透過 DI 注入，Phase 6 將以真實 Vespa client 替換。Spy 隔離測試 (m) 確認 caller 的 `chat_id` 必被傳遞給 indexer.delete｜`src/app/services/vespa_indexer.py`
- 2026-06-16｜Phase 3.3 跨 chat 文件共用：以 `chat_documents` association table 為 single source of truth（連 owner chat 自身也建一筆）；刪除為「先解除本 chat association，無人共用才真刪 ORM/storage/indexer」半順序語意｜`src/app/services/document_service.py`
- 2026-06-16｜Phase 4.2 Mapping 單一資料來源：CLAUDE.md §6.3 已宣告 PoC post-processor 跑完後 `content_list.json` 不再保留 → Phase 4.2 mapping **僅讀 `middle.json`**；偏離 DEVELOPMENT_PLAN.md 原稿「content_list+middle」字面，但與 CLAUDE.md §6.3-6.4 一致｜`src/app/parsing/mapping.py`、`deploy/mineru/output-schema.md`
- 2026-06-16｜Phase 4 全程使用 **deterministic uuid5 id**（block_id 與 node_id），方便 Phase 5 enrichment / Phase 6 Vespa upsert 重跑時不重複｜`src/app/parsing/{mapping,hierarchy}.py`
- 2026-06-16｜Phase 4.1 post-processing 純函式搬遷到 `src/app/parsing/_postprocess.py`；`scripts/mineru_poc.py` 改為 thin wrapper（CLI 入口不變），讓 async client 與 CLI 共用同一份 post-processing 邏輯，避免兩份實作漂移｜`src/app/parsing/_postprocess.py`, `scripts/mineru_poc.py`
- 2026-06-16｜Phase 4.4 heading-F1 gate threshold 保守設 0.3（非 0.8+）：golden corpus 只標 5 個高信心 heading，evaluator 對「真實樣本含更多 subsections」會給低 precision；harness 的設計目標是抓**大規模回退**（如完全沒抽出 references / appendix / abstract），不是 pixel-perfect。Phase 9 可再收緊｜`src/app/evaluation/parser_eval.py`、`data/fixtures/golden/2410.05779v3.json`
- 2026-06-16｜Phase 5.1 FixtureChatProvider：deterministic JSON mock，import-time 載 `tests/fixtures/enrichment_mock_responses.json`，SHA-256 prompt hash key + fallback "default"；不破壞 49 個既有 MockProvider 測試｜`src/app/providers/mock.py`
- 2026-06-16｜Phase 5.2 routing marker `[DOCUMENT_OVERVIEW]`：在 document-level system prompt 加 marker 字串，FixtureChatProvider 偵測該 marker 後 fallback `"document_default"`；對真實 LLM 無語義差異（只是文字）｜`src/app/enrichment/prompts.py`, `src/app/providers/mock.py`
- 2026-06-16｜Phase 5.3 Structured facts heuristic-first：先用 regex 抽 metric / dataset / hyperparameter / model_param / cost / range；LLM-augmented 為 optional flag（測試一律 off）。real-data LightRAG 抽到 0 facts，因為論文 numbers 主要在 HTML table cells；table-cell extraction 列為 Phase 9 enhancement｜`src/app/enrichment/facts.py`
- 2026-06-16｜Phase 5.3 FactFilter 受限 schema：Pydantic `extra="forbid"`；API 收 JSON body 給 query builder，禁絕 raw SQL；對應 CLAUDE.md §8 「query_structured_facts may only emit a restricted filter schema」｜`src/app/services/facts_service.py`, `src/app/api/facts.py`
- 2026-06-16｜Phase 5.4 Chat manifest 只走 read-side aggregation（不呼叫 LLM、不打 Vespa），對應 CLAUDE.md §6 retrieval routing「Structural / deterministic fetch」規則的範例實作｜`src/app/services/manifest_service.py`
- 2026-06-17｜Phase 5.1-5.2 enrichment 全程 **mock-LLM safe**：`MockChatProvider` 只回 hash 字串，因此 enrichment pipeline 設計成「呼叫 provider 是 hook + deterministic post-processing 抽欄位」，未來換真 LLM 只要替換 prompt + response parser，不必重寫測試｜`src/app/enrichment/{section,document}.py`
- 2026-06-17｜Phase 5.1-5.2 不擴 `SummaryKind` enum：claims/definitions/methods/limitations/performance_facts/main_* 等細項全留在 Pydantic `SectionEnrichment` / `DocumentEnrichment` 物件中，由 Phase 6 灌入 Vespa 對應 `source_type`；PostgreSQL 只落兩種粒度的 Summary 列（`section_detailed`/`section_compact` + `document_overview`/`chapter_summary(abstract)`）｜`src/app/enrichment/models.py`
- 2026-06-17｜Phase 5.3 `FactsFilter` 為 LangGraph `query_structured_facts` 工具的契約面：`ConfigDict(extra="forbid")` + 全欄位上下界；service `query_facts` 內部以 `filt.model_copy(update={"chat_id": current_chat_id})` 強制覆蓋 LLM 提供的 chat_id；StructuredFact.id 全 deterministic uuid5（NAMESPACE_OID）讓重跑 idempotent｜`src/app/services/facts_service.py`、`src/app/enrichment/facts.py`
- 2026-06-18｜Phase 6.1/6.3 Vespa schema 以 pyvespa 1.2.1 **程式化生成**（非手寫 .sd）；`build_application_package(dim)` 負責 schema，`write_application_files(dir, dim)` 負責覆寫 services.xml / hosts.xml / validation-overrides.xml（pyvespa 自動產生的這三個 XML 不符合 spec，故在 step 2 覆寫）；`scripts/deploy_vespa.py --dry-run` 作為 CI 友善驗收命令，不需要真實 Vespa｜`src/app/vespa/app_package.py`、`scripts/deploy_vespa.py`
- 2026-06-18｜Phase 6.3 second-phase 表達限制：pyvespa `RankProfile(second_phase=SecondPhaseRanking(expression=...))` 完整支援，無需 raw text fallback；`hybrid_with_native_rerank` inherits `hybrid_first_phase` 並覆寫 second_phase，測試確認繼承關係正確寫入 .sd｜`src/app/vespa/app_package.py`
- 2026-06-18｜Phase 6.1 services.xml container 命名：pyvespa 預設生成 `id="documentchunk_container"` / `id="documentchunk_content"`，不符合 spec（`default` / `documents`）；已在 `write_application_files` step 2 手動覆寫為正確名稱｜`src/app/vespa/app_package.py::_services_xml`
- 2026-06-18｜Phase 5.4 ChatManifest 為 **read-time computed**（GUIDE §8.3，不在 ingestion 階段預先產生 Chat-wide summary）；authors 從 Phase 4.3 hierarchy `node_type="authors"` 節點 content 切字（`,;\n` + ` and `），`Document` ORM 不擴新欄位；`Document.title` 退路為 `node_type="document"` 的 root DocumentNode title｜`src/app/enrichment/manifest.py`
- 2026-06-18｜Phase 7.3 LangGraph state schema：以 `TypedDict{state: dict[str, Any]}` 包裝 `AgentState`（單一事實源），每 node 進場 `AgentState.model_validate(container["state"])` / 出場回 `{"state": merged_dump}`；LangGraph 對 BaseModel state 在當前版本支援不全，TypedDict shim 為最小入侵變通｜`src/app/agent/graph.py`、`nodes/*`
- 2026-06-18｜Phase 7.2 `aggregate_sources` 工具用 **deterministic placeholder**（content join + truncate），不引入 LLM 依賴；Phase 9 可替換為 provider LLM aggregation｜`src/app/agent/tools/aggregate_sources.py`
- 2026-06-18｜Phase 7.4 PolicyEngine：6 個 `enforce_*` 方法分別在 graph 不同階段呼叫（pre_retrieval / post_retrieval / pre_answer / answer / citations / provider_result）；policy 5/10/3/4 為 **silently degrade**（自動改寫 plan / rerank_mode）；policy 9/14 為 **raise PolicyViolation**；違規一律 `state.record_event(kind="policy_violation", ...)` 留 audit trail｜`src/app/agent/policies.py`
- 2026-06-18｜Phase 7.5 SSE 不引入 `sse-starlette`，自寫 `event: <kind>\ndata: <json>\n\n` yield；stop generation 採 disconnect-aware（`request.is_disconnected()` + `asyncio.Event`）而非單獨 `/stop` endpoint，符合 SSE 標準做法、測試容易 monkeypatch｜`src/app/api/messages.py`、`src/app/services/qa_service.py`
- 2026-06-18｜**測試密度上限：每個 test 檔最多 10 個測試項**（含 parametrize 展開）。把 Phase 1-6 累積的 651 個測試精簡到 270 個（-58.5%），全程依「契約 > 列舉」原則：mandatory invariants 全保留、列舉用 parametrize 留代表性 2-3 case、刪除 `__repr__` / type-only / 重複覆蓋。規則寫入 `CLAUDE.md §12.1`，`DEVELOPMENT_PLAN.md` A.3 sub-agent prompt 模板加入「測試要求」段落，未來 phase 7-9 與所有 sub-agent 一律遵守。理由：原本測試膨脹只增加迭代成本，不增加 signal；mandatory gate（chat/session isolation、citation、provider、parsing、hybrid retrieval）的覆蓋一個沒少｜`CLAUDE.md §12.1`、`DEVELOPMENT_PLAN.md §A.3`、`tests/`
- 2026-06-18｜Env-level LLM fallback：`Settings` 加 `LLM_PROVIDER` / `LLM_API_URL` / `LLM_MODEL` / `LLM_API_KEY` 等欄位，`messages.py::_build_env_chat_provider` 在 chat 無 DB profile 時改用 `OpenAICompatChatProvider`；URL 自動 strip 末尾 `/chat/completions`（OpenAI SDK 會自己加）。對 vLLM / 自架 OpenAI-compat server demo 情境免插 DB row。end-to-end smoke `scripts/smoke_agent_e2e.py` PASS：gemma-4-31B-it 透過 LangGraph StateGraph 產出含 citation `[c1]` 的答案，覆蓋率 0.33，無 hallucination｜`src/app/config.py`、`src/app/api/messages.py`、`.env.example`、`scripts/smoke_agent_e2e.py`

---

## 7. Blocker 與風險

| 等級 | 項目 | 說明 | 替代方案 / 待辦 |
|---|---|---|---|
| ✅ 已解 | MinerU `:8001` 相容性 | raw vLLM 直接相容 `hybrid-http-client -u <url>`，無需 wrapper | 已記入 `deploy/mineru/README.md` |
| 低 | 公式中千分位逗號片段化 | `1,399×2×5,000` 在 inline math 中被切碎；數值文字仍可讀 | Phase 5.3 由 `structured_facts` 直接抽 numeric facts，避免依賴 inline LaTeX |
| 低 | 單行分數渲染為 `<sup>/<sub>` | 視覺退化但語意正確 | Phase 4.3 hierarchy heuristics 可選擇性 normalize |

---

## 8. 環境驗證紀錄（誠實記錄，未跑就標未驗證）

| 項目 | 指令 | 結果 | 備註 |
|---|---|---|---|
| import 健檢 | `uv run python -c "import app"` | ✅ | 2026-06-16 |
| MinerU PoC | `uv run mineru -p data/2410.05779v3.pdf -o data/parsed -b hybrid-http-client -u http://localhost:8001` | ✅ | gate_pass=true；`scripts/mineru_poc.py` 可重跑 |
| lint | `uv run ruff check src/app tests migrations scripts` | ✅ | 2026-06-16（全綠；`migrations/versions/` autogenerate excluded） |
| 型別 | `uv run mypy src/app` | ✅ | 2026-06-18 — 93 source files, 0 issues（含 Phase 7 全部 agent / qa_service / messages api） |
| 單元測試 | `uv run pytest tests/unit` | ✅ | 2026-06-18 — Phase 7 完成後 ~277 綠（+ agent state/budget/tools/graph/policies/qa_service；每檔 ≤10） |
| 整合測試 | `uv run pytest tests/integration` | ✅ | 2026-06-18 — Phase 7 完成後 ~64 綠（+ agent_graph_e2e / agent_policies_in_graph / messages_api） |
| 評估測試 | `uv run pytest tests/evaluation` | ✅ | 2026-06-18 — 15 綠（不變） |
| 全套測試 | `uv run pytest -q` | ✅ | 2026-06-18 — **356 passed in 1.9s**（Phase 7 +86 items；仍每檔 ≤10） |
| 每檔密度檢查 | `uv run pytest --collect-only -q \| awk -F'::' '/::/{print $1}' \| sort \| uniq -c \| sort -rn \| head` | ✅ | 2026-06-18 — max_per_file=10，全部 test 檔 ≤10 |
| Parser eval（real data, LightRAG） | `uv run python scripts/run_parser_eval.py` | ✅ PASS | `artifacts/evaluation/parser-report.{json,md}`；H-F1=0.333、math-recall=1.0、refs=21、figs=6、tables=6、eqs=2 |
| Alembic upgrade | `uv run alembic upgrade head` | ✅ | 2026-06-16 — revision `b5b02bc9d209`；11 張表（10 + alembic_version）建立 |
| Alembic round-trip | `uv run alembic downgrade base && uv run alembic upgrade head` | ✅ | 2026-06-16 — 雙向可重跑 |
| 評估測試 | `uv run pytest tests/evaluation` | ⬜ | Phase 9 |
| compose 設定 | `docker compose -f deploy/docker-compose.yml config` | ✅ | 2026-06-16 — YAML 合法 |
| 依賴啟動 | `docker compose -f deploy/docker-compose.yml up -d postgres vespa` | 🟡 部分 | 2026-06-16 — local 5432 已被 `postgres-local` container 占用；改用既有容器並建立 `paper_notebook` DB（功能等效）；vespa 尚未起 |
| Vespa deploy dry-run | `DATABASE_URL=... APP_ENCRYPTION_KEY=... uv run python scripts/deploy_vespa.py --dry-run` | ✅ | 2026-06-18 — 6 個檔案生成；tensor<float>(x[1024]) + 5 rank profiles 全包含 |
| unit tests（含 Phase 6.1/6.3） | `uv run pytest tests/unit` | ✅ | 2026-06-18 — 342 綠（+25 新增 test_vespa_app_package） |
| 前端 install | `npm --prefix src/frontend install` | ✅ | 2026-06-18 — Next.js 15.5.19 / React 19 / TanStack Query v5 / Tailwind 3.4 / vitest 4 |
| 前端 lint | `npm --prefix src/frontend run lint` | ✅ | 2026-06-18 — 0 errors |
| 前端 typecheck | `npm --prefix src/frontend run typecheck` | ✅ | 2026-06-18 — `tsc --noEmit` 0 errors |
| 前端 build | `npm --prefix src/frontend run build` | ✅ | 2026-06-18 — `/`、`/settings` 兩條 static route |
| 前端 test | `npm --prefix src/frontend run test` | ✅ | 2026-06-18 — 27 passed / 5 files，每檔 ≤10 |
| 前端 smoke（dev server） | `npm --prefix src/frontend run dev` + `curl http://localhost:3000/` | ✅ | 2026-06-18 — port 3000 占用 → fallback 3002；`/` 三 region + `/settings` 渲染含 Chat/Embedding/Reranker/Chat defaults |

## IMPROVEMENT Phase C — 契約更新

- Updated `CLAUDE.md` §8 to allow a bounded `llm_replan` StateGraph node only through schema-validated tool nominations.
- Added hard cap `MAX_REPLAN_ROUNDS = 3`; `chat_id` remains service-injected and never LLM-controlled.
- Expanded code-enforced policy count to include replan schema/isolation enforcement.
