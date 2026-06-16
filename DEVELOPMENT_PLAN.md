# DEVELOPMENT_PLAN.md — Paper Notebook Agent 開發計畫

> 本檔給 **orchestrator agent（主控）** 依序往下執行。主控負責切分任務、派發 sub-agent、驗收、跑 repair loop、更新 `PROGRESS.md`。
> 先讀 `CLAUDE.md`（架構契約，全英文）與 `GUIDE.md`（完整規格）。一切以 `CLAUDE.md` 為準。
> **所有指令一律 `uv`（`uv add` 裝套件、`uv run` 跑任務），無例外。**
> **執行原則：MinerU 解析必須最先驗證（Phase 1 為硬性 gate），確認能產生可靠 markdown 後才往下開發。**

---

## A. 執行模型（Orchestrator ⇄ Sub-agents）

### A.1 角色
- **Orchestrator（主控，本對話）**：閱讀計畫、派發 sub-agent、整合與驗收、跑 §10 repair loop、維護 `PROGRESS.md`、決定 phase 是否通過 gate。
- **Sub-agent（執行者）**：實際寫程式碼/測試的工人。**一律使用 `model: "sonnet"`**（Sonnet 4.6）。預設 `subagent_type: "general-purpose"`；純探索用 `Explore`，架構規劃用 `Plan`。

### A.2 派發規則
1. **同一 phase 內無相依的任務 → 一則訊息內多個 `Agent` 呼叫並行派發。** 有相依者依序派發。
2. 每個 sub-agent 的 prompt 必須 **自包含**（見 A.3），因為 sub-agent 看不到本對話歷史。
3. sub-agent 回報後，**主控必須親自驗證實際檔案與測試結果**（trust but verify），不可僅憑回報標記完成。
4. sub-agent 完成 → 主控更新 `PROGRESS.md`。
5. 跨 phase 嚴格依賴：未通過前一 phase 的 gate，不得啟動下一 phase 的實作任務（探索類可提前）。**Phase 1（MinerU）未通過前，不得進入 Phase 2 之後的實作。**

### A.3 Sub-agent Prompt 模板（每次派發都要帶齊）
```
你是 Paper Notebook Agent 專案的開發者。先讀 repo 根目錄的 CLAUDE.md（架構契約，全英文，必須遵守）
與 GUIDE.md 的第 <X> 節。本專案全程使用 uv：裝套件用 `uv add`，跑任何指令用 `uv run`，禁止 pip/裸跑。
程式碼結構：後端在 src/app/（import root = app）、部署在 deploy/、資料在 data/。

【任務】<一句話目標>
【背景】<為何需要、上游已完成什麼、下游會用到什麼>
【需建立/修改的檔案】<明確路徑清單（用 src/app/、deploy/、data/、tests/）>
【實作要求】<對應 CLAUDE.md / GUIDE 的硬性約束，逐條列出>
【測試要求】<要寫哪些測試、放哪、用 deterministic mock，不得碰真實付費 API>
【驗收標準】<可量測的完成定義；列出主控會跑的 `uv run` 驗證指令>
【限制】不要超出任務範圍重構；不要 hard-code 測試答案；不要降低 chat/session 隔離與 citation 標準。
【回報】完成後用 <200 字總結：改了哪些檔、跑了哪些 `uv run` 指令、結果、未解問題。
```

### A.4 每個 Phase 的標準收尾流程
1. 整合所有 sub-agent 產出。
2. 跑 `CLAUDE.md §10.1` 對應的驗證指令（該 phase 適用部分）。
3. 跑 §10 Self-Repair Loop，產 `artifacts/repair-loop/iteration-{n}.md`。
4. 計算該 phase 對應 goal score 區塊。
5. 更新 `PROGRESS.md`（狀態、決策、blocker、score、commit message 建議）。
6. 依 `CLAUDE.md` 慣例向使用者提供 commit message（不自動 commit）。

---

## B. 依賴總覽（已調整：MinerU 最先）

```
Phase 0 Bootstrap（最小化，足以跑 MinerU）
   └─> Phase 1 MinerU Parsing PoC & Validation（硬性 GATE）★ 通過才往下
          └─> Phase 2 Foundation（config / DB / models / providers / compose）
                 └─> Phase 3 Chat / Session / Document（+ 隔離測試）
                        ├─> Phase 4 Document Parsing Pipeline（MinerU 整合 → ParsedBlock/hierarchy）─┐
                        └─> Phase 6 Vespa（schema 可提前並行設計）                                     │
                               (Phase 4 與 Phase 6 可並行)                                              │
                                      ├─> Phase 5 Enrichment（依 Phase 4）                              │
                                      └────────────────────────────────> Phase 7 LangGraph Agent ◀─────┘
                                                                              └─> Phase 8 Frontend
                                                                                     └─> Phase 9 Evaluation & Repair
```

---

## Phase 0 — Bootstrap（主控親自執行，最小化）

**目標**：把空 uv 專案整理成可承載 monorepo 的骨架，且 **足以執行 MinerU PoC**。

任務：
1. `uv` 加入 MinerU hybrid client 依賴：`uv add "mineru[pipeline]"`（hybrid 需本地 pipeline + torch）。
2. 建立頂層骨架：`src/app/`、`src/frontend/`（先空）、`data/{sample/arxiv,fixtures,storage,parsed}`、`deploy/{vespa/application/schemas,postgres,mineru}`、`scripts/`、`migrations/`、`tests/{unit,integration,e2e,evaluation,fixtures}`、`artifacts/{repair-loop,evaluation}`。建立 `src/app/__init__.py`。
3. `pyproject.toml`：package 指向 `src/app`；加入 `[tool.ruff]`、`[tool.mypy]`、`[tool.pytest.ini_options]`（`asyncio_mode=auto`）。
4. `.gitignore` 補上 `data/sample/`、`data/storage/`、`data/parsed/`（不提交大型 PDF 與解析快取）。
5. `Makefile`（封裝 `uv run` 指令）、`.env.example`（含 `MINERU_SERVER_URL=http://localhost:8001`，無真實 key）。
6. 初始化 `PROGRESS.md`。

**Gate**：`uv run python -c "import app"` 成功；`uv run ruff check .` 通過。

---

## Phase 1 — MinerU Parsing PoC & Validation ★（硬性 GATE，最先做）

**目標**：在串接後端架構前，先**獨立驗證** MinerU hybrid 模式能對真實 arXiv PDF 產生**可靠 markdown 與結構化 JSON**。未達標不得進入 Phase 2。

**先決條件**：使用者已用 vLLM 將 MinerU 模型架於 `http://localhost:8001`。

| # | Sub-agent 任務 | type/model | 並行 | 交付 | 驗收 |
|---|---|---|---|---|---|
|1.1|連線與相容性驗證：用 `uv run mineru -p <pdf> -o data/parsed/<doc> -b hybrid-http-client -u http://localhost:8001` 對 1 份 arXiv PDF 跑通；確認 `:8001` 是否需 `mineru-openai-server` wrapper，將正確啟動方式寫入 `deploy/mineru/README.md`|general/sonnet|⛔(先做)|`deploy/mineru/README.md`、`scripts/mineru_poc.py`|能產出 `*.md` + `content_list.json` + `middle.json`，無連線錯誤|
|1.2|輸出可靠性評估：對 2–3 份不同 arXiv PDF（雙欄、含表格/figure、含 appendix）跑解析，人工＋腳本檢查 markdown 品質：標題層級、段落順序、表格、公式、references/appendix 邊界、有無亂序/重複/header-footer 洩漏|general/sonnet|⛔(依 1.1)|`scripts/mineru_poc.py` 評估報告、`artifacts/evaluation/mineru-poc.md`|產出可靠性結論與已知失敗樣態|
|1.3|輸出 schema 摸清：記錄 `content_list.json`/`middle.json`/`model.json` 的實際欄位（block type、bbox、page_idx、title level、discarded），作為 Phase 4 mapping 的依據|general/sonnet|⛔(依 1.1)|`deploy/mineru/output-schema.md`|欄位對照表完整，標出可映射到 `ParsedBlock` 的來源|

**Gate（硬性，必過才繼續）**：
- MinerU hybrid client 能連到 `localhost:8001` 並完成解析；
- arXiv PDF 的 markdown **可靠**（標題/段落順序合理、表格/figure 不遺漏、references/appendix 可辨識、無嚴重重複或亂序）；
- 已記錄輸出 schema 與已知限制於 `PROGRESS.md` 與 `artifacts/evaluation/mineru-poc.md`。
- 若不可靠：先在本 phase 嘗試調整 backend 參數 / server 設定 / 改用 `vlm` 或 `pipeline` backend 比較；仍不可靠則輸出 blocker 與替代方案，**暫停往後開發並回報使用者**。

---

## Phase 2 — Foundation（依 Phase 1 通過）

**目標**：configuration、PostgreSQL、migrations、domain models、provider 抽象、Docker Compose。

| # | Sub-agent 任務 | type/model | 並行 | 交付 | 驗收 |
|---|---|---|---|---|---|
|2.1|Config 載入（pydantic-settings，含 `MINERU_SERVER_URL`、DB、encryption key，env 覆寫）|general/sonnet|✅|`src/app/config.py`、`configs/`（或 `deploy/`）|載入設定單測|
|2.2|SQLAlchemy ORM + Pydantic domain models（§5.1 全表 + `chat_documents`）|general/sonnet|✅|`src/app/models/`|model import 單測；mypy 乾淨|
|2.3|Alembic 初始 migration（索引、`chat_id` FK）|general/sonnet|⛔(依 2.2)|`migrations/`|`uv run alembic upgrade head` 對真 PG 成功|
|2.4|Provider 抽象（chat/embedding/reranker adapter + OpenAI/Gemini/OpenAI-compatible/vLLM + deterministic mock）|general/sonnet|✅|`src/app/providers/`|mock adapter 單測；key 不入 log 測試|
|2.5|Docker Compose（postgres + vespa，volume、healthcheck）|general/sonnet|✅|`deploy/docker-compose.yml`|`docker compose -f deploy/docker-compose.yml up -d postgres vespa` 起得來|

**Gate（Backend 10 分相關）**：API 骨架可啟動、migration 正常、provider mock 可用、`uv run mypy src/app` 乾淨。

---

## Phase 3 — Chat / Session / Document（依 Phase 2）

**目標**：CRUD + 上傳 + 儲存 + **隔離測試**。

| # | Sub-agent 任務 | type/model | 並行 | 交付 | 驗收 |
|---|---|---|---|---|---|
|3.1|Chat service + API（`POST/GET/PATCH/DELETE /chats`）|general/sonnet|✅|`src/app/services/chat_service.py`、`src/app/api/chats.py`|CRUD 單測|
|3.2|Session service + API（`/chats/{id}/sessions...`）|general/sonnet|✅|`src/app/services/session_service.py`、`src/app/api/sessions.py`|CRUD + 歸屬校驗單測|
|3.3|Document service + 上傳 + 檔案儲存（`data/storage/`）+ ChatDocument association + 刪除連動|general/sonnet|⛔(依 3.1)|`src/app/services/document_service.py`、`src/app/storage/`、`src/app/api/documents.py`|上傳/刪除單測；刪除清連動測試（Vespa 先 mock）|
|3.4|**隔離測試套件**（Chat A 不可見 Chat B 文件；Session history 不互通；API scope 校驗）|general/sonnet|⛔(依 3.1-3.3)|`tests/unit/test_isolation*.py`、`tests/integration/...`|隔離測試全綠|

**Gate**：Chat/Session 文件與 history 隔離單測通過；錯誤 chat scope 回 403/404。

---

## Phase 4 — Document Parsing Pipeline（MinerU 整合，依 Phase 1 + Phase 3，可與 Phase 6 並行）

**目標**：把 Phase 1 驗證過的 MinerU 輸出，整合進 ingestion pipeline，映射為 `ParsedBlock` 與 hierarchy。**不重寫雙欄 reading-order**（MinerU 已處理），改為映射 + hierarchy 推導。

| # | Sub-agent 任務 | type/model | 並行 | 交付 | 驗收 |
|---|---|---|---|---|---|
|4.1|MinerU client 封裝：以 `MINERU_SERVER_URL` 呼叫 hybrid client、async、產出落 `data/parsed/`、可重試/idempotent|general/sonnet|✅|`src/app/parsing/mineru_client.py`|mock/實機解析測試（CI 用 fixture 輸出）|
|4.2|輸出映射：`content_list.json` + `middle.json` → `ParsedBlock`（§GUIDE 7.3，含 bbox/page/reading_order/block_type/confidence）|general/sonnet|⛔(依 4.1, Phase 1.3 schema)|`src/app/parsing/mapping.py`、`src/app/parsing/models.py`|以 Phase 1 真實輸出當 fixture 的映射測試|
|4.3|Hierarchy 推導：title/authors/abstract/headings/subsections/references boundary/appendix + tables/figure captions；保留原始 metadata、可追蹤 heuristic/LLM 修正、無內容重複|general/sonnet|⛔(依 4.2)|`src/app/parsing/hierarchy.py`|heading precision/recall fixture 測試|
|4.4|Parser evaluation harness（§GUIDE 24）+ 小型 golden annotations + arXiv 下載 script|general/sonnet|⛔(依 4.3)|`src/app/evaluation/parser_eval.py`、`scripts/ingest_sample_arxiv.py`、`data/fixtures/`|`uv run` 產評估報告，無段落重複/header 洩漏|

**Gate（Parsing 15 分，mandatory）**：MinerU→ParsedBlock 映射正確、hierarchy 合理、頁碼/bbox 保留、abstract/references/appendix 可辨識。

---

## Phase 5 — Enrichment（依 Phase 4 + Phase 2.4）

**目標**：section/document 層摘要、keywords、entities、claims、structured facts、document overview、chat manifest。

| # | Sub-agent 任務 | type/model | 並行 | 交付 | 驗收 |
|---|---|---|---|---|---|
|5.1|Section 級 enrichment（detailed/compact summary、keywords、entities、definitions、claims、methods、limitations、performance facts、來源 block IDs）用 provider 抽象 + mock LLM|general/sonnet|✅|`src/app/enrichment/section.py`|mock LLM 下結構正確|
|5.2|Document 級（overview、contributions/methods/findings/limitations/datasets/metrics/results/conclusions）|general/sonnet|⛔(依 5.1)|`src/app/enrichment/document.py`|結構測試|
|5.3|Structured facts 抽取入 `structured_facts` + 受限 filter query API（無任意 SQL）|general/sonnet|✅|`src/app/enrichment/facts.py`、`src/app/services/facts_service.py`|facts query 單測|
|5.4|Chat-level manifest（document list/titles/authors/abstract summaries/topics/token est/source types/status）|general/sonnet|⛔(依 5.2)|`src/app/enrichment/manifest.py`|manifest 測試|

**Gate**：mock LLM 下 enrichment pipeline 可跑、idempotent、來源可追溯。

---

## Phase 6 — Vespa（可與 Phase 4/5 並行設計，feed 需 Phase 5 產出）

**目標**：application package、schema、feed/delete、BM25、ANN、hybrid、phased ranking、rerank、isolation filter。

| # | Sub-agent 任務 | type/model | 並行 | 交付 | 驗收 |
|---|---|---|---|---|---|
|6.1|Vespa application package（`services.xml/hosts.xml/document_chunk.sd/validation-overrides.xml`，DIM 由設定）+ `scripts/deploy_vespa.py`|general/sonnet|✅|`deploy/vespa/application/`、deploy script|`uv run` 部署本地 Vespa、schema 驗證通過|
|6.2|Feed/Delete service（pyvespa，可重複執行、刪文件清 Vespa）|general/sonnet|⛔(依 6.1)|`src/app/vespa/feed.py`|feed fixture→可查；重複 feed 不重複|
|6.3|Rank profiles：`hybrid_first_phase` + second/global-phase rerank（BM25/closeness/exact/heading/entity/source-type boost）|general/sonnet|⛔(依 6.1)|`document_chunk.sd` rank profiles|BM25/ANN query 各自可跑|
|6.4|**RetrievalService**：強制注入 chat_id filter、BM25+ANN 雙查、**RRF fusion**、rerank（Vespa native + cross-encoder/OpenAI-compatible 可切）、`SearchHit` 全分數|general/sonnet|⛔(依 6.2,6.3)|`src/app/retrieval/service.py`|hybrid query 整合測試；分數齊全|
|6.5|Retrieval evaluation harness（§GUIDE 23）：Recall@k/MRR/nDCG/leakage；比較 BM25/Vector/Hybrid±rerank → `artifacts/evaluation/retrieval-report.{json,md}`|general/sonnet|⛔(依 6.4)|`src/app/evaluation/retrieval_eval.py`|leakage=0；hybrid≥單一 retriever；rerank 不退步|

**Gate（Vespa Retrieval 20 分，mandatory）**：BM25/ANN/hybrid 可跑、chat_id filter 正確（leakage 0）、fusion 正確、rerank 正常、debug 分數可查。

---

## Phase 7 — LangGraph Agent（依 Phase 3/5/6）

**目標**：StateGraph、tools、planner、budget、coverage、answer、citations、session persistence、policies。

| # | Sub-agent 任務 | type/model | 並行 | 交付 | 驗收 |
|---|---|---|---|---|---|
|7.1|`AgentState`（§GUIDE 13.1 全欄位）+ ContextBudgetManager（§14，tiktoken 近似標記）|general/sonnet|✅|`src/app/agent/state.py`、`src/app/agent/budget.py`|budget 計算單測|
|7.2|7 個工具（`inspect_chat/inspect_document/fetch_structural_nodes/search_hybrid/query_structured_facts/aggregate_sources/expand_evidence`）；chat_id 由 state 注入，工具回 status/token/sources|general/sonnet|⛔(依 7.1)|`src/app/agent/tools/`|工具單測；LLM 無法注入 chat_id 測試|
|7.3|StateGraph 節點與連線（§8 workflow）+ planner + coverage + gap retrieval + overflow→aggregate|general/sonnet|⛔(依 7.2)|`src/app/agent/graph.py`、`src/app/agent/nodes/`|graph 走通 mock 情境|
|7.4|Policy 強制層（§GUIDE 13.2 共14條，程式碼非 prompt）+ citation/scope validation|general/sonnet|⛔(依 7.3)|`src/app/agent/policies.py`|每條 policy 對應測試|
|7.5|QA service + API（`POST .../messages`，SSE streaming、stop generation）+ 訊息持久化 + `QAResponse/Citation`|general/sonnet|⛔(依 7.3)|`src/app/services/qa_service.py`、`src/app/api/messages.py`|streaming e2e（mock LLM）；citation 全屬 current chat|

**Gate（Agent QA 20 + Isolation/Citation mandatory）**：structural/hybrid/facts/aggregation 工具可選、coverage 驗證、citation 正確且不跨 chat、session 持久化。

---

## Phase 8 — Frontend（依 Phase 7 API 穩定）

**目標**：Next.js + TS + Tailwind + TanStack Query，完成主要流程。

| # | Sub-agent 任務 | type/model | 並行 | 交付 | 驗收 |
|---|---|---|---|---|---|
|8.1|專案骨架 + API client + types（對齊後端 schema）+ TanStack Query setup|general/sonnet|✅|`src/frontend/`|`npm --prefix src/frontend run build` 過|
|8.2|Chat Sidebar（建立/列表/rename/delete/文件數/最後更新）|general/sonnet|⛔(依 8.1)|`src/frontend/features/chat/`|UI 可呼叫 API|
|8.3|文件區（上傳/進度/parsing/enrichment/Vespa indexing status/頁數/section/retry/remove）|general/sonnet|⛔(依 8.1)|`src/frontend/features/documents/`|上傳→狀態輪詢|
|8.4|Session 區 + Chat Interface（message list、streaming、stop、citations、source chips、展開 excerpt、PDF 對應頁、debug trace）|general/sonnet|⛔(依 8.2,8.3)|`src/frontend/features/session/`、`.../chat-ui/`|QA streaming + citation 顯示|
|8.5|Model Settings（chat/embedding/reranker profile、Test Connection、Set global/chat default、masked key）|general/sonnet|✅|`src/frontend/features/settings/`|連線測試 UI 串通|

**Gate（Frontend 10 分）**：Chat 管理、文件上傳、Session 切換、QA streaming、citations、settings 皆可操作（瀏覽器實測；無法實測須在 PROGRESS 註明）。

---

## Phase 9 — Evaluation & Repair

**目標**：golden QA、retrieval eval、parser eval、isolation E2E、repair loop、final goal score、交付報告。

| # | Sub-agent 任務 | type/model | 並行 | 交付 | 驗收 |
|---|---|---|---|---|---|
|9.1|Golden evaluation cases（§GUIDE 19：global summary/method/comparison/performance/multi-doc/chat-isolation/session-isolation）+ runner|general/sonnet|✅|`tests/evaluation/`|`uv run pytest tests/evaluation` 可跑|
|9.2|Isolation E2E（§GUIDE 22：Chat A/B 上傳、跨 chat 問題 → 無洩漏）|general/sonnet|✅|`tests/e2e/`|no cross-chat result|
|9.3|Goal Coverage 自動評分器（§GUIDE 21，100 分制 + mandatory gate 判定）|general/sonnet|⛔(依 9.1,9.2)|`src/app/evaluation/goal_score.py`|產 score 報告|
|9.4|最終交付報告（§GUIDE 29 全 20 項）+ README 完成|general/sonnet|⛔(依 9.3)|`README.md`、`artifacts/`|報告誠實列出未完成/風險|

**最終 Gate（Definition of Done，GUIDE §28 全 22 條）**：goal score ≥ 90 且所有 mandatory gate 通過，否則不得宣稱完成。

---

## C. 跨 Phase 持續事項

- **每完成一個 sub-agent 任務或一輪 repair loop → 立即更新 `PROGRESS.md`。**
- 任何架構偏離 → 先改 `CLAUDE.md`，記入 `PROGRESS.md` 的決策紀錄。
- 所有指令 `uv run` / `uv add`；frontend 用 `npm --prefix src/frontend`。
- 危險操作（刪 DB、force push、改 CI）先問使用者。
- 每個 phase 收尾依 `CLAUDE.md` 慣例提供 conventional commit message（不自動 commit）。
