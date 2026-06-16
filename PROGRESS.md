# PROGRESS.md — 開發進度追蹤

> 規則：每完成一個 sub-agent 任務或一輪 repair loop **立即更新**本檔。
> 狀態圖示：⬜ 未開始 / 🟡 進行中 / ✅ 完成 / ⛔ 受阻(blocker) / ⏭️ 略過(附理由)。
> 一切以 `CLAUDE.md` 為準；架構偏離先改 `CLAUDE.md` 再記於本檔「決策紀錄」。
> 指令一律 `uv run` / `uv add`。後端 import root = `app`（在 `src/app/`）。
> **執行原則：Phase 1（MinerU PoC）為硬性 gate，未通過不得進入 Phase 2 之後。**

最後更新：_(YYYY-MM-DD HH:MM)_ ｜ 當前 Phase：_Phase 0_ ｜ 累計 Goal Score：_0 / 100_

---

## 1. Phase 進度總覽

| Phase | 名稱 | 狀態 | 通過 Gate? | Goal 區塊得分 | 備註 |
|---|---|---|---|---|---|
| 0 | Bootstrap（最小化） | ⬜ | — | — | 足以跑 MinerU |
| 1 | **MinerU Parsing PoC ★** | ⬜ | — | (Parsing 前置) | **硬性 gate**；vLLM @ localhost:8001 |
| 2 | Foundation | ⬜ | — | Backend /10 | |
| 3 | Chat/Session/Document | ⬜ | — | (Isolation 起步) | |
| 4 | Document Parsing Pipeline（MinerU 整合） | ⬜ | — | Parsing /15 | mandatory |
| 5 | Enrichment | ⬜ | — | — | |
| 6 | Vespa Retrieval | ⬜ | — | Retrieval /20 | mandatory |
| 7 | LangGraph Agent | ⬜ | — | Agent QA /20 | mandatory |
| 8 | Frontend | ⬜ | — | Frontend /10 | |
| 9 | Evaluation & Repair | ⬜ | — | Provider /10 + Isolation /15 | |

---

## 2. Sub-agent 任務追蹤

> 格式：`狀態 | 任務# | 描述 | 指派(type/model) | 產出檔 | 主控已驗證?`

### Phase 0 — Bootstrap（主控執行）
- ⬜ 0.1 `uv add "mineru[pipeline]"`（hybrid client 依賴）
- ⬜ 0.2 頂層骨架 `src/app` `src/frontend` `data/` `deploy/` `scripts/` `migrations/` `tests/` `artifacts/`
- ⬜ 0.3 pyproject（package=src/app、ruff/mypy/pytest）
- ⬜ 0.4 `.gitignore`（data/sample、data/storage、data/parsed）
- ⬜ 0.5 Makefile / .env.example（`MINERU_SERVER_URL=http://localhost:8001`）
- ⬜ 0.6 PROGRESS.md 初始化

### Phase 1 — MinerU Parsing PoC ★（硬性 gate）
- ⬜ 1.1 連線與相容性驗證（hybrid-http-client → localhost:8001；確認是否需 mineru-openai-server wrapper） | general/sonnet
- ⬜ 1.2 多份 arXiv PDF 輸出可靠性評估（雙欄/表格/figure/appendix） | general/sonnet
- ⬜ 1.3 輸出 schema 摸清（content_list/middle/model.json 欄位對照） | general/sonnet

### Phase 2 — Foundation
- ⬜ 2.1 Config 載入（含 MINERU_SERVER_URL） | general/sonnet
- ⬜ 2.2 ORM + domain models | general/sonnet
- ⬜ 2.3 Alembic 初始 migration | general/sonnet
- ⬜ 2.4 Provider 抽象 + mock | general/sonnet
- ⬜ 2.5 Docker Compose（deploy/） | general/sonnet

### Phase 3 — Chat/Session/Document
- ⬜ 3.1 Chat service + API | general/sonnet
- ⬜ 3.2 Session service + API | general/sonnet
- ⬜ 3.3 Document service + 上傳 + 儲存（data/storage） | general/sonnet
- ⬜ 3.4 隔離測試套件 | general/sonnet

### Phase 4 — Document Parsing Pipeline（MinerU 整合）
- ⬜ 4.1 MinerU client 封裝（async/idempotent，落 data/parsed） | general/sonnet
- ⬜ 4.2 輸出映射 content_list+middle → ParsedBlock | general/sonnet
- ⬜ 4.3 Hierarchy 推導（含 references/appendix） | general/sonnet
- ⬜ 4.4 Parser eval + 下載 script | general/sonnet

### Phase 5 — Enrichment
- ⬜ 5.1 Section 級 enrichment | general/sonnet
- ⬜ 5.2 Document 級 | general/sonnet
- ⬜ 5.3 Structured facts + filter query | general/sonnet
- ⬜ 5.4 Chat-level manifest | general/sonnet

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
| Chat isolation | ⬜ | |
| Session isolation | ⬜ | |
| Vespa hybrid retrieval | ⬜ | |
| Citations | ⬜ | |
| arXiv parsing (MinerU) | ⬜ | |
| LangGraph QA | ⬜ | |
| Provider settings | ⬜ | |

---

## 4. MinerU PoC 結論（Phase 1 專用）

| 項目 | 結果 | 備註 |
|---|---|---|
| 連線 localhost:8001 | ⬜ | 是否需 mineru-openai-server wrapper？ |
| 使用 backend | ⬜ | hybrid-http-client（預設）/ 比較 vlm / pipeline |
| markdown 可靠性 | ⬜ | 標題/順序/表格/figure/references/appendix |
| 已知失敗樣態 | ⬜ | |
| 輸出 schema 對照 | ⬜ | content_list / middle / model.json |
| 報告路徑 | ⬜ | `artifacts/evaluation/mineru-poc.md`、`deploy/mineru/output-schema.md` |

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

---

## 7. Blocker 與風險

| 等級 | 項目 | 說明 | 替代方案 / 待辦 |
|---|---|---|---|
| 待驗證 | MinerU `:8001` 相容性 | 不確定 raw vLLM 是否直接相容 hybrid-http-client，或需 mineru-openai-server wrapper | Phase 1.1 驗證；必要時於 deploy/mineru 記錄正確啟動方式 |

---

## 8. 環境驗證紀錄（誠實記錄，未跑就標未驗證）

| 項目 | 指令 | 結果 | 備註 |
|---|---|---|---|
| import 健檢 | `uv run python -c "import app"` | ⬜ | |
| MinerU PoC | `uv run mineru -p <pdf> -o data/parsed/<doc> -b hybrid-http-client -u http://localhost:8001` | ⬜ | Phase 1 |
| lint | `uv run ruff check .` | ⬜ | |
| 型別 | `uv run mypy src/app` | ⬜ | |
| 單元測試 | `uv run pytest tests/unit` | ⬜ | |
| 整合測試 | `uv run pytest tests/integration` | ⬜ | |
| 評估測試 | `uv run pytest tests/evaluation` | ⬜ | |
| compose 設定 | `docker compose -f deploy/docker-compose.yml config` | ⬜ | |
| 依賴啟動 | `docker compose -f deploy/docker-compose.yml up -d postgres vespa` | ⬜ | |
| 前端 build | `npm --prefix src/frontend run build` | ⬜ | |
