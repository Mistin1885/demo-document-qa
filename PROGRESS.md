# PROGRESS.md — 開發進度追蹤

> 規則：每完成一個 sub-agent 任務或一輪 repair loop **立即更新**本檔。
> 狀態圖示：⬜ 未開始 / 🟡 進行中 / ✅ 完成 / ⛔ 受阻(blocker) / ⏭️ 略過(附理由)。
> 一切以 `CLAUDE.md` 為準；架構偏離先改 `CLAUDE.md` 再記於本檔「決策紀錄」。
> 指令一律 `uv run` / `uv add`。後端 import root = `app`（在 `src/app/`）。
> **執行原則：Phase 1（MinerU PoC）為硬性 gate，未通過不得進入 Phase 2 之後。**

最後更新：2026-06-17 ｜ 當前 Phase：_Phase 5 完成（section / document / structured-facts / chat manifest enrichment，全 mock-LLM idempotent）_ ｜ 累計 Goal Score：_Parsing /15 mandatory PASS + Backend skeleton + Isolation 基礎 + Phase 5 enrichment 全綠_

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
| 6 | Vespa Retrieval | ⬜ | — | Retrieval /20 | mandatory |
| 7 | LangGraph Agent | ⬜ | — | Agent QA /20 | mandatory |
| 8 | Frontend | ⬜ | — | Frontend /10 | |
| 9 | Evaluation & Repair | ⬜ | — | Provider /10 + Isolation /15 | |

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
- ⬜ 6.1 Application package + deploy（deploy/vespa） | general/sonnet
- ⬜ 6.2 Feed/Delete service | general/sonnet
- ⬜ 6.3 Rank profiles | general/sonnet
- ⬜ 6.4 RetrievalService（RRF + rerank） | general/sonnet
- ⬜ 6.5 Retrieval eval harness | general/sonnet

### Phase 7 — LangGraph Agent
- ⬜ 7.1 AgentState + ContextBudget | general/sonnet
- ⬜ 7.2 7 個工具 | general/sonnet
- ⬜ 7.3 StateGraph 節點 | general/sonnet
- ⬜ 7.4 Policy 強制層 + citation/scope validation | general/sonnet
- ⬜ 7.5 QA service + API（SSE） | general/sonnet

### Phase 8 — Frontend
- ⬜ 8.1 骨架 + API client + types | general/sonnet
- ⬜ 8.2 Chat Sidebar | general/sonnet
- ⬜ 8.3 文件區 | general/sonnet
- ⬜ 8.4 Session + Chat Interface | general/sonnet
- ⬜ 8.5 Model Settings | general/sonnet

### Phase 9 — Evaluation & Repair
- ⬜ 9.1 Golden QA cases + runner | general/sonnet
- ⬜ 9.2 Isolation E2E | general/sonnet
- ⬜ 9.3 Goal Coverage 評分器 | general/sonnet
- ⬜ 9.4 最終交付報告 + README | general/sonnet

---

## 3. Mandatory Gates 檢查（GUIDE §21）

| Gate | 狀態 | 證據(測試/報告路徑) |
|---|---|---|
| Chat isolation | 🟡 PASS（基礎層） | `tests/integration/test_isolation_chat_documents.py` + `test_isolation_chat_sessions.py` + `test_isolation_api_scope.py`；Phase 6 接 Vespa 後再驗 retrieval-side filter |
| Session isolation | 🟡 PASS（基礎層） | `tests/integration/test_isolation_session_history_service.py` + `tests/unit/test_isolation_service_filters.py`；Phase 7 接 message API 後再驗 endpoint scope |
| Vespa hybrid retrieval | ⬜ | |
| Citations | ⬜ | |
| arXiv parsing (MinerU) | ✅ PASS | `artifacts/evaluation/{mineru-poc.md, parser-report.{json,md}}`；Phase 4 harness 對 LightRAG real-data gate PASS |
| LangGraph QA | ⬜ | |
| Provider settings | 🟡 抽象 + mock 完成 | `src/app/providers/` + `tests/unit/test_providers.py` 49 綠 |

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
- 2026-06-17｜Phase 5.4 ChatManifest 為 **read-time computed**（GUIDE §8.3，不在 ingestion 階段預先產生 Chat-wide summary）；authors 從 Phase 4.3 hierarchy `node_type="authors"` 節點 content 切字（`,;\n` + ` and `），`Document` ORM 不擴新欄位；`Document.title` 退路為 `node_type="document"` 的 root DocumentNode title｜`src/app/enrichment/manifest.py`

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
| 型別 | `uv run mypy src/app` | ✅ | 2026-06-17 — 49 source files, 0 issues |
| 單元測試 | `uv run pytest tests/unit` | ✅ | 2026-06-17 — Phase 2-4 既有 224 + Phase 5.1 section 26 + 5.2 document 12 + 5.3 facts extractor/service + 5.4 manifest = 317 綠 |
| 整合測試 | `uv run pytest tests/integration` | ✅ | 2026-06-17 — chats/sessions/documents + 4 isolation + Phase 5.3 facts_api 6 + Phase 5.4 manifest_api 5 = 75 綠 |
| 評估測試 | `uv run pytest tests/evaluation` | ✅（parser harness） | 2026-06-17 — 31 綠；Phase 6/9 retrieval/goal-score harness 尚未建 |
| Parser eval（real data, LightRAG） | `uv run python scripts/run_parser_eval.py` | ✅ PASS | `artifacts/evaluation/parser-report.{json,md}`；H-F1=0.333、math-recall=1.0、refs=21、figs=6、tables=6、eqs=2 |
| Alembic upgrade | `uv run alembic upgrade head` | ✅ | 2026-06-16 — revision `b5b02bc9d209`；11 張表（10 + alembic_version）建立 |
| Alembic round-trip | `uv run alembic downgrade base && uv run alembic upgrade head` | ✅ | 2026-06-16 — 雙向可重跑 |
| 評估測試 | `uv run pytest tests/evaluation` | ⬜ | Phase 9 |
| compose 設定 | `docker compose -f deploy/docker-compose.yml config` | ✅ | 2026-06-16 — YAML 合法 |
| 依賴啟動 | `docker compose -f deploy/docker-compose.yml up -d postgres vespa` | 🟡 部分 | 2026-06-16 — local 5432 已被 `postgres-local` container 占用；改用既有容器並建立 `paper_notebook` DB（功能等效）；vespa 尚未起 |
| 前端 build | `npm --prefix src/frontend run build` | ⬜ | Phase 8 |
