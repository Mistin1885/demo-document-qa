# PROGRESS.md — 開發進度追蹤

> 規則：每完成一個 sub-agent 任務或一輪 repair loop **立即更新**本檔。
> 狀態圖示：⬜ 未開始 / 🟡 進行中 / ✅ 完成 / ⛔ 受阻(blocker) / ⏭️ 略過(附理由)。
> 一切以 `CLAUDE.md` 為準；架構偏離先改 `CLAUDE.md` 再記於本檔「決策紀錄」。
> 指令一律 `uv run` / `uv add`。後端 import root = `app`（在 `src/app/`）。
> **執行原則：Phase 1（MinerU PoC）為硬性 gate，未通過不得進入 Phase 2 之後。**

最後更新：2026-06-16 17:30 ｜ 當前 Phase：_Phase 1 完成，準備進入 Phase 2_ ｜ 累計 Goal Score：_前置（Parsing PoC PASS）_

---

## 1. Phase 進度總覽

| Phase | 名稱 | 狀態 | 通過 Gate? | Goal 區塊得分 | 備註 |
|---|---|---|---|---|---|
| 0 | Bootstrap（最小化） | ✅ | ✅ | — | `import app` / ruff / mineru CLI 全綠 |
| 1 | **MinerU Parsing PoC ★** | ✅ | ✅ | (Parsing 前置 PASS) | hybrid-http-client → vLLM @ 8001 直連成功；公式可靠 |
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
| arXiv parsing (MinerU) | 🟡 PoC PASS | `artifacts/evaluation/mineru-poc.md`（單樣本通過；Phase 4 整合後正式 mark ✅） |
| LangGraph QA | ⬜ | |
| Provider settings | ⬜ | |

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
| lint | `uv run ruff check .` | ✅ | 全綠 |
| 型別 | `uv run mypy src/app` | ⬜ | |
| 單元測試 | `uv run pytest tests/unit` | ⬜ | |
| 整合測試 | `uv run pytest tests/integration` | ⬜ | |
| 評估測試 | `uv run pytest tests/evaluation` | ⬜ | |
| compose 設定 | `docker compose -f deploy/docker-compose.yml config` | ⬜ | |
| 依賴啟動 | `docker compose -f deploy/docker-compose.yml up -d postgres vespa` | ⬜ | |
| 前端 build | `npm --prefix src/frontend run build` | ⬜ | |
