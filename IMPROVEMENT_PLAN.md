# IMPROVEMENT_PLAN.md — Paper Notebook Agent 深度檢索強化計畫

> 本檔接續 `DEVELOPMENT_PLAN.md`，**只負責 Phase 9 完成之後的後續強化**。一切以 `CLAUDE.md` 為準；若需要修改架構契約（例如 §8 的 StateGraph 行為），**先改 `CLAUDE.md`** 再回到本檔記錄決策（Phase C.0 的職責）。
> 所有指令一律 `uv run` / `uv add`，無例外（CLAUDE.md §0 規則 1）。
> 測試密度上限維持：每個 test 檔 ≤10 個測試項（CLAUDE.md §12.1）。
> 沿用 `DEVELOPMENT_PLAN.md` A.1–A.4 的 orchestrator / sub-agent 工作模式（sub-agent 一律 `model: "sonnet"`）。

---

## 0. 動機與目標題型

現況：QA endpoint 在 LLM provider 接好後雖然會走完 LangGraph，但對下面三題仍會「拉到 ~3–8 個 chunk 就交差」，深度不足：

- **Q1 — Summary this document.**
  走 summary 路徑（`plan_information_needs.py:99`），但 `execute_retrieval_tools._build_default_params` 只對**第一個**未訪問文件呼叫 `inspect_document`，多文件 chat 會漏；且 `fetch_structural_nodes` 預設只抓 `document_overview / chapter_summary / compact_chapter_summary` 三類來源。
- **Q2 — Compare lightRAG with GraphRAG.**
  「compare」**不在** summary / numeric keyword set，落到 default → 單一 `search_hybrid(query=原句, top_k=8)`，無法分別抓 LightRAG 細節與 GraphRAG 細節。
- **Q3 — Performance of ablated versions of LightRAG.**
  走 numeric 路徑，但 `query_structured_facts` 用**空 filter** 呼叫，無 kind / metric_name；search 也只用原句。

額外發現的既有 bug：`plan_gap_retrieval.py:58` 把 gap query 塞進 `plan.rationale` 字串，但 `execute_retrieval_tools._build_default_params()` 不讀此欄位 → 第二輪 gap retrieval 等於空轉、被 fingerprint 去重。**這是「現況只拉幾個 chunks」的直接成因。**

**目標**：讓 agent 在 budget 內主動補齊資訊（多 query / 多輪 / LLM 驅動的下一步決策），使 Q1–Q3 達到 NotebookLM/ChatGPT 等級的答案深度，同時維持：

1. CLAUDE.md §0 規則 3 — chat_id 永遠由 service layer 注入，不經 LLM。
2. CLAUDE.md §0 規則 5 — 只能用 chat 內文件回答，不得用模型自身知識。
3. 工具 schema 仍封閉（`extra="forbid"`），LLM 不能任意傳參數。
4. 有硬性 round / budget 上限，永不變成無限 ReAct。

---

## 1. 階層化計畫（A → B → C → D → E）

| Layer | Phase | 名稱 | 預估規模 | 是否動 CLAUDE.md |
|---|---|---|---|---|
| 1 | **A** | Gap-retrieval bug fix + 廣度調整 | 小（1–2 人日） | 否 |
| 2 | **B** | 啟發式 query decomposition（comparison / ablation） | 中（2–3 人日） | 否 |
| 3 | **C** | LLM 驅動 replan node（bounded） | 中大（3–5 人日） | **是（§8）** |
| 4 | **D** | Embedding-based coverage check | 中（2 人日） | 否 |
| 5 | **E** | 三題基準納入 Golden QA + 再次跑 Goal Score | 小（1 人日） | 否 |

> 依賴：A 是所有後續的前提（修掉 gap-retrieval 才能談「補資料」）。B 與 D 之間沒有強相依，但 C 建議在 B 之後做（B 提供 deterministic baseline，C 才能 fallback）。E 為各 phase 的共同驗收門檻。

---

## 2. Phase A — Gap-retrieval Bug Fix + 廣度調整

**目標**：把 gap-retrieval 從目前的空轉狀態修正成「實際對 unsatisfied requirements 各發一次 `search_hybrid`」；同步把 `inspect_document` / `top_k` 在 summary 場景下擴大。

**改動範圍**（必須對齊 §12 程式品質）：

- `src/app/agent/state.py`
  - 在 `AgentPlan` 新增 `gap_queries: list[str] = Field(default_factory=list)`（Pydantic v2，`extra="forbid"`）。
- `src/app/agent/nodes/plan_gap_retrieval.py`
  - 停止把 gap queries 塞進 `rationale`；改寫入新欄位 `plan.gap_queries`。
- `src/app/agent/nodes/execute_retrieval_tools.py`
  - `_build_default_params` 對 `search_hybrid` 改為：先消化 `state.plan.gap_queries`（pop 一個當 `query`），全部消化完才退回 `state.question`。
  - `inspect_document`：在 summary path 下對 `state.document_manifests` **全數**迭代（不再只取第一個未訪問）。
- `src/app/agent/tools/_models.py`
  - `SearchHybridParams.top_k` 預設維持 8，但新增 `SearchHybridParams.preset: Literal["default","broad"] = "default"`；`broad` 對應 `bm25_top_k / ann_top_k / fusion_top_k / rerank_top_k / final_top_k` 各自放大（例如 ×2，仍受既有 `le=` 上限）。
- `src/app/agent/nodes/plan_information_needs.py`
  - summary path 直接帶 `preset="broad"`；其他維持 default。

**測試**（每檔 ≤10）：

- `tests/unit/agent/test_plan_gap_retrieval.py`（已存在則合併）— 1 個 happy path + 1 個確認 `gap_queries` 真的進 plan + 1 個 unsatisfied=0 時 no-op。
- `tests/unit/agent/test_execute_retrieval_tools.py` — 新增 1 個 case 驗 second iteration 對不同 query 各呼叫一次 `search_hybrid`（fingerprint 不會誤殺）。
- `tests/unit/agent/test_plan_information_needs.py` — summary 題目應帶 `preset="broad"`、`inspect_document` 應出現在 chosen_tools（多文件情境）。

**驗收（主控驗證）**：

```bash
uv run pytest tests/unit/agent -q
uv run pytest tests/integration -q -k "agent or qa"
uv run mypy src/app/agent
uv run ruff check src/app/agent
```

成功條件：
- 每個動到的 test 檔 `pytest --collect-only` 計數 ≤10。
- 既有 mandatory gate 測試（isolation / citations / hybrid retrieval）全綠不退步。
- 跑一個 Q1 樣本（fixture）的 integration test，evidence_items ≥ Layer 1 改造前 × 2。

**Sub-agent 派發**（並行 1 人）：

```
你是 Paper Notebook Agent 專案的開發者。先讀 CLAUDE.md（§0、§8、§12.1）、IMPROVEMENT_PLAN.md Phase A、
GUIDE.md §8/§13。後端 import root = `app`，全程用 `uv` 跑指令。

【任務】修 gap-retrieval 永遠空轉的 bug，並把 summary 場景的廣度拉開。
【需建立/修改的檔案】
 - src/app/agent/state.py（AgentPlan 新增 gap_queries 欄位）
 - src/app/agent/nodes/plan_gap_retrieval.py（改寫入結構化欄位）
 - src/app/agent/nodes/execute_retrieval_tools.py（消化 gap_queries；summary path 走全部文件）
 - src/app/agent/tools/_models.py（SearchHybridParams 加 preset 欄位）
 - src/app/agent/nodes/plan_information_needs.py（summary path 帶 preset="broad"）
 - 對應 unit tests（每檔 ≤10）
【實作要求】
 - 不得在 LLM 端傳 chat_id；plan.gap_queries 由 graph 內部產生。
 - SearchHybridParams 維持 extra="forbid"。
 - 不可降低既有 mandatory gate 任何測試。
【測試要求】依 CLAUDE.md §12.1，每檔 ≤10；改完跑：
   uv run pytest tests/unit/agent --collect-only -q | tail
【驗收】見 IMPROVEMENT_PLAN.md Phase A「驗收（主控驗證）」段。
【回報】<200 字：改了哪些檔、跑的指令與結果、每檔最終測試數、未解問題。
```

---

## 3. Phase B — 啟發式 Query Decomposition（Comparison / Ablation）

**前置**：Phase A 完成。

**目標**：在 deterministic planner 層加上「比較題」「消融題」的拆解路徑，產出多個 sub-query 餵給 `search_hybrid`；ablation 題自動改寫 `query_structured_facts` 的 filter。

**改動範圍**：

- `src/app/agent/nodes/plan_information_needs.py`
  - 新增 `_COMPARISON_PATTERNS`：`compare(d)?` / `\bvs\.?\b` / `\bversus\b` / `between .* and .*` / `對比` / `差異`。
  - 新增 `_ABLATION_PATTERNS`：`\babla(te|ted|tion)s?\b` / `without .* component` / `\bwith and without\b` / `消融`。
  - 當命中 comparison：抽取兩個比較對象 A、B（用 regex `between (\w+) and (\w+)` 或 `(\w+)\s+(vs|versus)\s+(\w+)`），產出 sub-queries `[A, B, "{A} {B} differences", "{A} architecture", "{B} architecture"]` 並寫入 `plan.gap_queries`（一輪即跑掉）。
  - 當命中 ablation：在 `chosen_tools` 加上 `query_structured_facts`，且把 hint 寫入新的 `plan.fact_filter_hints: dict[str, Any]`（例如 `{"kind": "ablation"}`，schema 需檢核 ORM 是否支援）。
- `src/app/agent/tools/_models.py`
  - 確認 `QueryStructuredFactsParams` 已有 `kind` / `metric_name` 欄位；若無，補上（仍 `extra="forbid"`）。
- `src/app/agent/nodes/execute_retrieval_tools.py`
  - `_build_default_params("query_structured_facts", state)` 若 `plan.fact_filter_hints` 有值則套用。
- `src/app/models/orm.py` / `structured_facts` schema
  - 確認 `kind` 欄位真的存在；不存在則本 phase 加 ORM 欄位 + Alembic migration（向後相容）。

**測試**（每檔 ≤10）：

- `tests/unit/agent/test_plan_information_needs.py`（合併到既有檔，總數仍 ≤10）— comparison / ablation 各 1 case。
- `tests/integration/agent/test_decomposition_flow.py` — 用 mock retrieval 驗 comparison 題會跑出 ≥3 個 sub-query，evidence 來自不同子主題。

**驗收**：
- comparison 樣題（Q2）的 evidence 至少涵蓋 A 與 B 兩個主題（用 mock manifest 驗證）。
- ablation 樣題（Q3）的 `tool_calls` 中 `query_structured_facts` 至少帶一個非空 filter。

**Sub-agent 派發**（依任務拆兩個並行，若 ORM 改動需要先做 migration 則串行）：

- B.1 — Plan/Execute 層的拆解邏輯（無 ORM 變更）
- B.2 — `structured_facts` ORM/Alembic + `QueryStructuredFactsParams` 對齊（若需要）

每個 sub-agent 用 §A.3 的模板填入對應【任務】【實作要求】【測試要求】。

---

## 4. Phase C — LLM 驅動 Replan Node（Bounded）

**前置**：Phase A、B 完成；Phase C.0 完成 CLAUDE.md §8 的契約更新。

**目標**：在 `check_coverage` 之後、`generate_answer` 之前加入 `llm_replan` node，讓 LLM 在受限 schema 內提名下一步工具與參數，達到「ChatGPT 風格」的主動補資料行為。整個過程**仍在 bounded StateGraph 中**，有硬性 round/budget 上限。

### 4.1 Phase C.0 — CLAUDE.md §8 契約更新（**必做且最先**）

**規則變更**：
- 將「Explicit StateGraph, not an unbounded ReAct loop.」改寫為：
  > Explicit StateGraph; LLM-driven replanning is permitted **only** through a dedicated `llm_replan` node that emits **schema-validated** next-tool requests (`extra="forbid"` Pydantic model). The graph never grants the LLM free-form tool execution; every nomination is dispatched by `execute_retrieval_tools` under the existing PolicyEngine. Round cap: `MAX_REPLAN_ROUNDS = 3` (configurable in `agent.budget`). `chat_id` is still injected by the service layer and **never** flows through the LLM.
- 在 §8 workflow 圖加入 `llm_replan`：
  ```
  ... → check_coverage → (incomplete → plan_gap_retrieval → execute_retrieval_tools)
        → (coverage_state == "incomplete" AND replan_rounds < MAX_REPLAN_ROUNDS → llm_replan
            → execute_retrieval_tools → merge_evidence_workspace → check_context_budget → check_coverage)
        → verify_critical_claims → generate_answer → ...
  ```
- 在 §8 Tools 區塊不變（沒有新增工具，只是新增 node）；明確指出 `llm_replan` 不能新發明工具名稱、不能傳 `chat_id`、不能繞過 `query_structured_facts` 的 filter schema。
- 在 §10.1 標準驗證指令底下不需要新增；但 §11 mandatory gates **新增第 8 條**：「Replan LLM 不得讓 citations 跨 chat / 不得繞過 isolation」。

**做法**：本 phase 第一個 sub-agent 任務直接送一個小的 patch 給 CLAUDE.md（diff 不超過 ~40 行），並在 PROGRESS.md 開新區塊「IMPROVEMENT Phase C — 契約更新」。

### 4.2 Phase C.1 — 新增 `llm_replan` node 與結構化輸出

**改動範圍**：

- `src/app/agent/nodes/llm_replan.py`（新檔）
  - 輸入：question / chat_manifest titles（不含 chat_id）/ 既有 evidence 的 ≤1k token 摘要 / `replan_rounds` / 剩餘 budget。
  - 輸出 Pydantic 型別：
    ```python
    class ReplanDecision(BaseModel):
        model_config = ConfigDict(extra="forbid")
        action: Literal["search_more", "answer_now", "no_info"]
        tool_calls: list[ReplanToolCall] = []
        reasoning: str = Field(max_length=400)

    class ReplanToolCall(BaseModel):
        model_config = ConfigDict(extra="forbid")
        tool: Literal["search_hybrid", "fetch_structural_nodes",
                      "query_structured_facts", "inspect_document"]
        query: str | None = None       # for search_hybrid
        source_types: list[Literal[...]] | None = None  # for fetch_structural_nodes
        fact_filter: dict[str, str] | None = None       # 只接受白名單 keys
        document_id: UUID | None = None
    ```
  - prompt 強約束：JSON-only、不可改動 `chat_id`、不可發明工具、`max_tokens` 上限低（~300）。
  - JSON parse 失敗 / schema 不合 → fallback：`action="answer_now"`，記錄 `state.errors` + 不抛例外。
- `src/app/agent/budget.py`
  - 新增 `MAX_REPLAN_ROUNDS = 3`（可被 `generation_config` 覆寫但有上限）。
  - `state.replan_rounds: int = 0` 由 `llm_replan` 累加。
- `src/app/agent/graph.py`
  - 在 `check_coverage` 之後加 conditional edge：`coverage_state == "incomplete" AND iteration_count >= 2 AND replan_rounds < MAX_REPLAN_ROUNDS → llm_replan`，否則維持原路徑。
  - `llm_replan → execute_retrieval_tools` 後再跑一次後續流程。
- `src/app/agent/policies.py`
  - 新增 policy 15：`enforce_replan_decision(state, decision)` 檢查 `decision.tool_calls` 全部符合工具白名單與參數 schema；違反則丟 `PolicyViolation` 並 fallback 到 `answer_now`。

**測試**（每檔 ≤10）：

- `tests/unit/agent/test_llm_replan.py`
  - happy path（用 mock ChatProvider 回固定 JSON）
  - schema 違規 → 退到 `answer_now`
  - `chat_id` 嘗試入侵 → 被 policy 15 攔下並記錄
  - round cap：第 3 次 replan 後不再進入此 node
- `tests/integration/agent/test_replan_flow.py`（與 deterministic flow 共用 ≤10）
  - Q2 樣題：replan 後成功補到 GraphRAG 細節

**驗收**：
- 跑 Q1/Q2/Q3 fixture，replan 後 final answer 必含跨主題引用（Q2 要同時引用 LightRAG 與 GraphRAG 段落）。
- mandatory gate 全綠；citations 全屬於同 chat。

### 4.3 Sub-agent 派發（C 階段）

依序：
- **C.0**：派一名 `Plan` agent 起草 CLAUDE.md §8 patch（diff < 40 行），主控覆核後手動寫入 `CLAUDE.md`。
- **C.1**：派 `general-purpose` agent 1 名執行 `llm_replan` node + budget + policy 15 + tests。
- **C.2**：派 `general-purpose` agent 1 名負責 graph 重接 + integration test。

C.1 / C.2 可並行（共用 state 欄位定義，需先約定 `AgentState.replan_rounds` 命名）。

---

## 5. Phase D — Embedding-based Coverage Check

**目標**：把 `check_coverage._token_overlap` 換成語意相似度，解決「ablated」對「without component」這類同義字漏接。

**做法選擇**（擇一，主控決定）：
1. **Vespa native**：用既有 E5 embedder 在 retrieval 階段直接帶回 similarity 分數，coverage 用 evidence 的 `vector_score` 閾值判斷。
2. **本地 embed**：用 deps 中的 `embedding_provider`（若已設定 chat profile）embed `requirement.description`，與 evidence 做 cosine。無 embedding provider → 退回 token overlap。

**改動範圍**：
- `src/app/agent/nodes/check_coverage.py`：抽出 `_score_requirement(req, evidence_items, deps)`，內部依設定走 native / local / fallback。
- `src/app/agent/budget.py`：新增 `COVERAGE_SIMILARITY_THRESHOLD = 0.55`（可調）。

**測試**（每檔 ≤10）：
- `tests/unit/agent/test_check_coverage.py`：保留現有 ≤10；加 1 個 similarity case（mock embedding provider）。

**驗收**：Q3 樣題的 coverage_state 應在第 1 輪即標 satisfied=True（用同義字 evidence）。

**Sub-agent 派發**：1 名 `general-purpose`，串行於 Phase C 之後（避免 graph 邊改）。

---

## 6. Phase E — Golden QA 三題納入 + Goal Score 重算

**目標**：把 Q1/Q2/Q3 的 expected answer / 必出現引用片段 加入 `data/fixtures/qa_cases.json`，並重跑 `scripts/run_qa_eval.py` 與 `scripts/run_goal_score_eval.py`。

**改動範圍**：
- `data/fixtures/qa_cases.json`：新增 3 個 case（用 LightRAG paper PDF 作為固定文件）。
- `src/app/evaluation/qa_eval.py`：若需要新評分維度（例如 cross-topic coverage），補上。
- `tests/evaluation/test_qa_eval.py`：在 ≤10 內加 1 個 case 確保新題型評分流程通過。

**驗收**：
- `uv run pytest tests/evaluation -q` 全綠。
- `uv run python scripts/run_goal_score_eval.py` 產出新版 `artifacts/evaluation/goal-score.{json,md}`，`mandatory_all_passed=true` 維持，total ≥ 90（理想為 95+）。

**Sub-agent 派發**：1 名 `general-purpose`，於各 phase 完成後分別跑一次（A 後跑、B 後跑、C 後跑），紀錄三條改善曲線到 PROGRESS.md。

---

## 7. 各 Phase 驗證指令彙整

```bash
# 通用
uv run ruff check .
uv run ruff format --check .
uv run mypy src/app

# Phase A
uv run pytest tests/unit/agent -q
uv run pytest tests/integration -q -k "agent or qa"

# Phase B
uv run pytest tests/unit/agent tests/integration/agent -q

# Phase C
uv run pytest tests/unit/agent/test_llm_replan.py tests/integration/agent/test_replan_flow.py -q
uv run pytest -q  # 全綠 regression

# Phase D
uv run pytest tests/unit/agent/test_check_coverage.py -q

# Phase E
uv run pytest tests/evaluation -q
uv run python scripts/run_qa_eval.py
uv run python scripts/run_goal_score_eval.py
```

每個 phase 收尾仍跑 CLAUDE.md §10 self-repair loop，產 `artifacts/repair-loop/iteration-{n}.md`。

---

## 8. 風險與決策紀錄欄

| 風險 | 影響 | 緩解 |
|---|---|---|
| Phase C 動到 §8 契約 | 後續所有 isolation 測試需重跑 | Phase C.0 先送 CLAUDE.md patch + 補新 policy 15 測試 |
| LLM replan 多打 1–3 次 LLM，延遲↑、成本↑ | 影響 UX | round cap + budget 檢查；提供 chat-level `enable_llm_replan: bool` |
| Comparison 對象抽錯 | Phase B 拆解失準 | regex 抽不到時退回原句 + 加 search_hybrid `broad` preset |
| `structured_facts.kind` 欄位若未存在 | Phase B 卡關 | B.2 任務獨立完成 ORM/Alembic 後再做 B.1 |
| 既有測試密度上限 | 改動可能讓檔超 10 | 每次新增 test 前先檢查並合併 / 刪除低價值 case（CLAUDE.md §12.1） |

主控（你）若決定**只做 A+B+E**、暫不做 C，IMPROVEMENT_PLAN.md 仍保留 Phase C 章節作為日後 backlog；此情況下 CLAUDE.md §8 不需更動。
