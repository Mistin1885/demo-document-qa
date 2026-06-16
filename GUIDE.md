你是一位資深 AI 系統架構師、搜尋工程師、LangGraph Agent 工程師與全端開發者。

請建立一個完整、可執行、可測試的開源 repository，實作一套類似 NotebookLM 的多文件 Agentic QA 系統。

第一階段主要針對從 arxiv.org 下載的研究論文 PDF 進行解析、索引、摘要與問答。

不得只產生概念性程式碼、展示性 notebook 或大量 TODO。請實際建立 repository、啟動必要服務、執行測試，並透過循環驗證與自我修復，使結果盡可能符合以下 Goal 與驗收標準。

---

# 1. Product Goal

建立一個類似 NotebookLM 的 Web 應用，具備以下能力：

1. 使用者可以建立多個獨立 Chat。
2. 每個 Chat 可以上傳一份或多份 PDF 文件。
3. 系統解析文件後，可以針對該 Chat 所屬文件進行問答。
4. 不同 Chat 的文件、索引、對話與回答必須完全隔離。
5. Chat A 不得檢索或回答 Chat B 上傳的文件內容。
6. 支援 Chat 與 Session 切分。
7. 同一個 Chat 可以有多個 conversation session。
8. Session 保存自己的 conversation history，但共享該 Chat 的文件集合。
9. 使用者可以建立新 Chat，並上傳完全不同的文件。
10. 回答必須提供文件名稱、頁碼、章節與引用內容。
11. 支援整份文件摘要、跨文件摘要、技術比較、實驗結果比較、數值查詢與一般 QA。
12. 使用者可在獨立設定頁面設定 LLM 與 embedding provider。
13. 支援 OpenAI、Gemini，以及 self-hosted vLLM 的 OpenAI-compatible endpoint。
14. UI 中可以設定 URL、model、API key、max tokens、temperature、timeout 等參數。
15. 主要搜尋引擎與向量資料庫使用 Vespa。
16. Vespa 必須同時支援 BM25、向量搜尋、metadata filtering 與多階段 reranking。
17. AI Agent 使用 LangGraph。
18. 系統應優先針對 arXiv 論文的典型格式進行解析與驗證。

最終產品的主要使用流程：

```text
建立 Chat
→ 上傳一份或多份 arXiv PDF
→ 系統解析文件
→ 顯示解析與索引狀態
→ 使用者建立或進入 Session
→ 提問
→ LangGraph Agent 選擇工具
→ Vespa / 結構化資料庫取得資訊
→ 必要時進行 aggregation
→ 產生附引用的答案
```

---

# 2. 核心隔離模型

請明確區分以下概念。

## 2.1 Chat

Chat 是文件隔離的最小邊界。

每個 Chat 擁有：

* 自己的文件集合
* 自己的 Vespa searchable documents
* 自己的預先產生資訊
* 自己的 Sessions
* 自己的預設模型設定，可選擇繼承全域設定
* 自己的建立時間與名稱

所有 retrieval query 都必須包含：

```text
chat_id == current_chat_id
```

不得只依賴 prompt 告訴模型不要跨 Chat。

隔離必須由：

* relational database query
* Vespa filter
* API authorization scope
* repository/service layer

共同保證。

## 2.2 Session

Session 是同一 Chat 底下的一條 conversation thread。

每個 Session 擁有：

* chat_id
* session_id
* conversation history
* session title
* message records
* selected model profile
* creation/update time

同一 Chat 下的不同 Session：

* 共享該 Chat 文件
* 不共享 conversation history
* 可使用不同模型設定

## 2.3 Document

文件一定屬於一個 Chat：

```text
Document.chat_id
```

一份文件不可在未建立 association 的情況下被另一個 Chat 搜尋。

若未來支援文件複用，必須透過明確的 ChatDocument association，而不能只共享 Vespa document ID。

---

# 3. 主要技術棧

請使用：

## Backend

```text
Python 3.11+
FastAPI
Uvicorn
LangGraph
LangChain Core
Pydantic v2
SQLAlchemy 2.x
PostgreSQL
Alembic
PyMuPDF
pdfplumber
tiktoken
HTTPX
OpenAI Python SDK
Google GenAI SDK 或 Gemini OpenAI-compatible adapter
pytest
pytest-asyncio
ruff
mypy
```

## Search and Retrieval

```text
Vespa
pyvespa
BM25
HNSW approximate nearest-neighbor search
Hybrid retrieval
Vespa rank profiles
First-phase ranking
Second-phase 或 global-phase reranking
Optional cross-encoder reranker
```

不要再使用：

```text
FAISS
Chroma
rank-bm25 作為正式搜尋引擎
```

BM25、向量索引、metadata filter 與 ranking 統一由 Vespa 負責。

本地單元測試可以使用 mock search adapter，但 integration test 必須實際連接 Vespa。

## Frontend

請建立可操作的 Web UI。

推薦：

```text
Next.js
React
TypeScript
Tailwind CSS
TanStack Query
```

如有更合理選擇可以調整，但必須實際完成基本介面。

## Infrastructure

```text
Docker Compose
Vespa container
PostgreSQL container
Backend container
Frontend container
```

開發模式應可以使用：

```bash
docker compose up
```

啟動主要依賴。

---

# 4. 高階系統架構

```text
Web UI
  ├── Chat Sidebar
  ├── File Upload Panel
  ├── Session Panel
  ├── Chat Interface
  ├── Source Viewer
  └── Model Settings

FastAPI Backend
  ├── Chat Service
  ├── Session Service
  ├── Document Service
  ├── Ingestion Service
  ├── Provider Settings Service
  ├── QA Service
  └── LangGraph Agent

Data Layer
  ├── PostgreSQL
  │    ├── Chats
  │    ├── Sessions
  │    ├── Messages
  │    ├── Documents
  │    ├── Document Nodes
  │    ├── Summaries
  │    ├── Structured Facts
  │    ├── Provider Profiles
  │    └── Ingestion Jobs
  │
  ├── Vespa
  │    ├── Raw Blocks
  │    ├── Section Summaries
  │    ├── Chapter Summaries
  │    ├── Technology Cards
  │    ├── Claims
  │    ├── Definitions
  │    ├── Table Records
  │    └── Embedding Vectors
  │
  └── Local / Object File Storage
       └── Original PDFs
```

---

# 5. Repository 結構

請建立清楚的 monorepo：

```text
paper-notebook-agent/
├── README.md
├── Makefile
├── docker-compose.yml
├── .env.example
├── pyproject.toml
├── package.json
├── configs/
│   ├── backend.yaml
│   ├── prompts.yaml
│   └── vespa/
│       └── application/
│           ├── services.xml
│           ├── hosts.xml
│           ├── schemas/
│           │   └── document_chunk.sd
│           └── validation-overrides.xml
├── apps/
│   ├── backend/
│   │   └── src/
│   │       └── paperqa/
│   │           ├── api/
│   │           ├── agent/
│   │           ├── models/
│   │           ├── parsing/
│   │           ├── enrichment/
│   │           ├── retrieval/
│   │           ├── vespa/
│   │           ├── storage/
│   │           ├── providers/
│   │           ├── services/
│   │           └── evaluation/
│   └── frontend/
│       ├── app/
│       ├── components/
│       ├── features/
│       ├── hooks/
│       ├── lib/
│       └── types/
├── scripts/
│   ├── deploy_vespa.py
│   ├── ingest_sample_arxiv.py
│   ├── run_evaluation.py
│   └── seed_demo.py
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── e2e/
│   ├── evaluation/
│   └── fixtures/
└── sample_data/
    └── arxiv/
```

---

# 6. Provider Settings

建立獨立的「Model Settings」區塊。

使用者可建立多個 Provider Profile。

## 6.1 Provider 類型

至少支援：

```text
OpenAI
Gemini Native API
Gemini OpenAI-compatible API
Generic OpenAI-compatible API
Self-hosted vLLM
```

## 6.2 Chat Model 設定

每個 profile 至少包含：

```python
provider_type
profile_name
base_url
api_key
model_name
max_output_tokens
context_window
temperature
top_p
timeout_seconds
max_retries
supports_tool_calling
supports_structured_output
supports_vision
enabled
```

## 6.3 Embedding Model 設定

請不要假設 chat model 與 embedding model 使用相同 provider。

embedding profile 至少包含：

```python
provider_type
profile_name
base_url
api_key
model_name
embedding_dimension
batch_size
timeout_seconds
normalize_embeddings
enabled
```

## 6.4 Reranker 設定

至少支援：

```text
Vespa native rank expression
Cross-encoder HTTP endpoint
OpenAI-compatible scoring model fallback
Disable reranking
```

設定欄位：

```python
reranker_type
model_name
base_url
api_key
candidate_count
rerank_count
timeout_seconds
```

## 6.5 API Key 安全

API key：

* 不得寫入 log
* 不得回傳到前端
* 資料庫中不得明文保存
* 至少使用 application encryption key 加密
* UI 只能顯示 masked value
* 更新時可重新輸入
* `.env.example` 不可包含真實 key

建立 provider connection test：

```text
Test Chat Connection
Test Embedding Connection
Test Reranker Connection
```

測試結果需顯示：

* success / failure
* model
* latency
* sanitized error message

---

# 7. arXiv 論文 Parsing 目標

第一階段主要針對 arxiv.org 上的論文 PDF。

典型 arXiv 論文結構包括：

```text
Title
Authors
Affiliations
Abstract
1 Introduction
2 Related Work
3 Method / Architecture
4 Experiments / Evaluation
5 Conclusion
References
Appendix
Figures
Tables
Equations
Captions
```

## 7.1 Parser 必須盡可能識別

* paper title
* authors
* abstract
* section heading
* subsection heading
* appendix
* references boundary
* paragraphs
* figure captions
* table captions
* page number
* bounding boxes
* reading order
* multi-column layout
* equations附近的文字描述
* table blocks
* figure blocks

## 7.2 Layout 與閱讀順序

arXiv 論文常見雙欄格式。

不能只使用：

```python
page.get_text()
```

並假設文字順序正確。

請建立 block-based reading-order algorithm，至少考慮：

* x/y positions
* column detection
* full-width title/abstract blocks
* left/right column ordering
* figure/table placement
* footer/header removal
* repeated page headers
* references section

## 7.3 Parser 輸出

```python
class ParsedBlock(BaseModel):
    block_id: str
    chat_id: str
    document_id: str
    page_number: int
    block_type: Literal[
        "title",
        "author",
        "abstract",
        "heading",
        "paragraph",
        "table",
        "figure",
        "caption",
        "equation",
        "reference",
        "header",
        "footer",
    ]
    text: str
    bbox: list[float]
    reading_order: int
    font_size: float | None
    font_name: str | None
    column_index: int | None
    confidence: float
```

---

# 8. 文件階層與預先產生資訊

建立：

```text
Document
├── Abstract
├── Section
│   ├── Subsection
│   ├── Paragraphs
│   ├── Tables
│   └── Figures
├── References
└── Appendix
```

## 8.1 每個 Section 產生

```text
detailed summary
compact summary
keywords
technical keywords
entities
definitions
claims
methods
limitations
performance facts
related table IDs
related figure IDs
source block IDs
```

## 8.2 每份 Document 產生

```text
document overview
abstract summary
section summaries
main contributions
main methods
main technologies
main findings
main limitations
main datasets
main metrics
main experimental results
main conclusions
```

## 8.3 跨文件層級

一個 Chat 可有多份文件。

另外產生 Chat-level manifest，但不要在 ingestion 階段強制生成單一 Chat summary，因為文件可能持續新增。

Chat-level manifest 至少包含：

```text
document list
document titles
authors
abstract summaries
main topics per document
token estimates
available source types
ingestion status
```

---

# 9. Vespa Schema 設計

Vespa document schema 必須支援多 Chat、多文件、多 node type。

至少包含：

```text
vespa_document_id
chat_id
document_id
source_node_id
parent_node_id
source_type
title
heading_path
content
keywords
technical_keywords
entities
page_start
page_end
order_index
token_count
embedding
created_at
```

建議 source_type：

```text
raw_block
chunk
section_summary
compact_section_summary
chapter_summary
compact_chapter_summary
document_overview
technology_card
claim
definition
performance_fact
table_record
figure_caption
```

## 9.1 強制隔離條件

每次 Vespa query 都必須有：

```text
chat_id contains current_chat_id
```

如果 query 指定 document IDs，還需有：

```text
document_id in current_chat_document_ids
```

請在 RetrievalService 層強制注入 filter。

不要允許 Agent 自己傳入任意 chat_id。

工具只能從 AgentState 取得 current chat_id。

## 9.2 Embedding 欄位

使用 Vespa tensor field，例如：

```text
field embedding type tensor<float>(x[EMBEDDING_DIM]) {
    indexing: attribute | index
    attribute {
        distance-metric: angular
    }
}
```

dimension 必須由部署設定決定。

當使用者切換 embedding dimension 時，不可默默沿用舊 schema。

MVP 可限制一個 deployment 使用一種 embedding dimension，並在 UI 明確提示。

---

# 10. Vespa Hybrid Retrieval

需要向量搜尋時，不可只做 ANN。

請實作 hybrid retrieval：

```text
BM25 candidate retrieval
+
nearestNeighbor candidate retrieval
+
candidate fusion
+
reranking
```

## 10.1 Candidate Retrieval

候選至少來自：

### Lexical branch

```text
userQuery()
```

使用 BM25，搜尋：

* title
* heading_path
* content
* keywords
* technical_keywords
* entities

不同欄位需要可設定權重。

### Vector branch

```text
nearestNeighbor(embedding, query_embedding)
```

使用 HNSW ANN。

### Hybrid YQL

概念上使用：

```text
chat filter AND
(
    userQuery()
    OR
    nearestNeighbor(embedding, query_embedding)
)
```

實作時請依 Vespa 正式 YQL 語法處理 annotation、target hits 與 query tensor。

## 10.2 First-phase Ranking

建立 rank profile，例如：

```text
hybrid_first_phase
```

綜合：

```text
BM25 score
vector closeness
exact title match
heading match
entity match
source type boost
page / structural features
```

不要直接把未正規化的 BM25 與 vector score 任意相加。

請選擇並清楚實作以下一種：

1. Reciprocal Rank Fusion
2. Query normalization 後的 weighted fusion
3. Vespa ranking expression 中可解釋的 score transformation

第一版優先使用穩定且容易測試的 RRF。

若 RRF 必須在 application layer 合併兩次 Vespa query，可以接受，但最終 rerank 必須由 Vespa rank profile 或明確 reranker 完成。

## 10.3 Reranking

使用多階段 ranking funnel：

```text
BM25 / ANN 取得較大的候選集合
→ first-phase cheap ranking
→ second-phase 或 global-phase reranking
→ 回傳 top-k evidence
```

建議預設：

```text
BM25 candidates: 60
ANN candidates: 60
fusion candidates: 80
rerank candidates: 30
final top-k: 8–12
```

所有參數必須可設定。

## 10.4 Reranker 選項

至少實作兩種：

### Option A：Vespa native rerank

使用 Vespa rank profile，利用：

* BM25
* closeness
* exact matches
* field match
* structural metadata
* source type boost

完成 second-phase/global-phase rerank。

### Option B：Cross-encoder reranker

對融合後候選使用：

```text
query + candidate content
→ relevance score
```

然後將重排結果回傳。

Cross-encoder 可以是：

* 本地 sentence-transformers cross-encoder
* self-hosted HTTP endpoint
* 可設定的 reranker provider

若使用 application-side cross-encoder，仍須保留 Vespa first-phase ranking 結果與各分數，供 debug UI 顯示。

## 10.5 Search Result

每筆結果需包含：

```python
class SearchHit(BaseModel):
    vespa_document_id: str
    chat_id: str
    document_id: str
    source_node_id: str
    source_type: str
    content: str
    title: str | None
    heading_path: list[str]
    page_start: int
    page_end: int
    bm25_score: float | None
    vector_score: float | None
    fusion_score: float | None
    rerank_score: float | None
    final_rank: int
```

---

# 11. 不應對所有問題使用 Vespa Retrieval

即使 Vespa 提供 hybrid retrieval，仍不可把所有問題都變成 top-k search。

## 11.1 Structural Retrieval

以下情況使用 PostgreSQL deterministic fetch：

* 摘要一整份文件
* 摘要 Chat 中所有文件
* 摘要特定 section
* 取得全部 document overview
* 取得全部 section summary
* 需要保證每份文件都被涵蓋

例如：

```text
請總結這個 Chat 中的三份論文
```

必須：

```text
fetch three document overviews
或 fetch all required section summaries
```

不得只以 Vespa top-k 搜尋。

## 11.2 Vespa Hybrid Retrieval

適合：

* 技術細節
* 特定概念
* 問法與原文不同
* 跨段落搜尋
* 定義
* claim
* 方法限制
* 部分實驗內容
* 相關 evidence

## 11.3 Structured Fact Query

數值、benchmark、performance、dataset、metric 優先查 PostgreSQL structured facts，再利用 Vespa 補上下文。

---

# 12. Agent 工具

工具應對應資料取得能力，而不是對應固定使用者問題。

至少提供以下工具。

## 12.1 inspect_chat

回傳：

* Chat 文件數
* 文件列表
* ingestion status
* 文件標題
* section count
* source type statistics
* token estimates
* 可用 structured facts
* Vespa indexing status

## 12.2 inspect_document

回傳單份文件：

* hierarchy
* sections
* summaries
* token estimates
* figures/tables
* available source types

## 12.3 fetch_structural_nodes

deterministic fetch：

* 所有 document overviews
* 某份文件所有 section summaries
* 指定 section
* 指定 node IDs
* 指定 parent 的所有 children

若超過 budget：

```text
status = overflow
```

不得截斷。

## 12.4 search_hybrid

透過 Vespa：

```text
BM25 + ANN + fusion + rerank
```

工具輸入：

```python
query
source_types
document_ids optional
top_k
token_budget
```

chat_id 不允許由 LLM 傳入，必須由 AgentState 注入。

## 12.5 query_structured_facts

用於：

* metrics
* performance
* latency
* throughput
* memory
* token cost
* dataset
* model comparison
* table values

LLM 只能生成受限 filter schema，不可直接生成任意 SQL。

## 12.6 aggregate_sources

當資訊超過 context 時：

```text
query-focused map-reduce
```

必須支援：

```python
required_source_ids
required_document_ids
all_documents_in_chat
required_entities
required_dimensions
preserve_numeric_facts
preserve_citations
output_token_budget
```

## 12.7 expand_evidence

從 summary、fact 或 Vespa hit 回讀：

* 原始 block
* 前後段落
* table context
* figure caption
* page
* bbox

---

# 13. LangGraph Agent

請使用明確 StateGraph，不要只建立無限制 ReAct loop。

建議 workflow：

```text
START
  ↓
load_chat_and_session
  ↓
inspect_scope
  ↓
plan_information_needs
  ↓
enforce_scope_and_policies
  ↓
execute_retrieval_tools
  ↓
merge_evidence_workspace
  ↓
check_context_budget
  ├── overflow → aggregate_sources
  └── within budget
  ↓
check_coverage
  ├── incomplete → plan_gap_retrieval
  │                    ↓
  │              execute_retrieval_tools
  └── complete
  ↓
verify_critical_claims
  ↓
generate_answer
  ↓
validate_citations
  ↓
validate_scope_isolation
  ↓
persist_messages
  ↓
END
```

## 13.1 AgentState

至少包含：

```python
chat_id
session_id
question
conversation_history
chat_manifest
document_manifests
plan
tool_calls
evidence_items
structured_facts
coverage_requirements
coverage_state
context_token_estimate
remaining_budget
iteration_count
answer
citations
errors
debug_trace
```

## 13.2 Agent Policies

以下政策必須由程式碼執行，不得只寫在 prompt：

1. 所有 retrieval 限定 current chat_id。
2. Session 不得讀取其他 Session history。
3. 全文件或全 Chat 摘要不得以 top-k search 代替 fetch-all。
4. 數值問題優先查 structured facts。
5. Vespa 結果必須經過 hybrid retrieval 與 rerank。
6. overflow 時不得截斷，必須 aggregate。
7. incomplete coverage 時不得直接回答。
8. 重大數值需回查原始 evidence。
9. 最多執行固定工具輪數。
10. 不得重複相同工具與相同參數。
11. 文件沒有資訊時，不可使用模型自身知識補答。
12. final citations 必須全部屬於 current chat。
13. 每個 citation 的 document_id 必須存在於 current ChatDocument association。
14. provider failure 時需回傳明確錯誤，不得靜默切換模型，除非使用者設定 fallback。

---

# 14. Context Budget

預設模型 context：

```text
10,000 tokens
```

但必須從 provider profile 取得實際 context_window。

預設分配：

```text
system and tool instructions: 1,200
conversation history: 1,000
question and plan: 500
document evidence: 5,000
answer reserve: 2,000
miscellaneous: 300
```

建立 ContextBudgetManager：

```python
count_tokens
estimate_tool_result
estimate_conversation
calculate_available_evidence_budget
detect_overflow
select_compact_sources
build_aggregation_groups
```

如果使用 provider tokenizer 不可取得，可使用 tiktoken 作近似，但需在 debug trace 標示：

```text
token_count_is_estimate = true
```

---

# 15. Web UI

建立類似 NotebookLM 的基本介面。

## 15.1 左側 Chat Sidebar

提供：

* 建立 Chat
* Chat 列表
* rename
* delete
* 顯示文件數
* 顯示最後更新時間

## 15.2 Chat 文件區

顯示：

* 上傳按鈕
* 文件名稱
* 上傳進度
* parsing status
* enrichment status
* Vespa indexing status
* 頁數
* section 數
* retry ingestion
* remove document

刪除文件時需同步：

* PostgreSQL records
* Vespa records
* file storage
* ChatDocument association

## 15.3 Session 區

同一 Chat 可：

* New Session
* 切換 Session
* rename Session
* delete Session

每個 Session 顯示獨立 message history。

## 15.4 Chat Interface

包含：

* message list
* streaming answer
* stop generation
* citations
* source chips
* 展開 source excerpt
* 點擊後顯示 PDF 對應頁面
* debug mode 顯示 tool trace

## 15.5 Model Settings

獨立頁面或 drawer：

### Chat Provider

```text
Provider Type
Base URL
Model
API Key
Context Window
Max Output Tokens
Temperature
Top P
Timeout
Max Retries
```

### Embedding Provider

```text
Provider Type
Base URL
Model
API Key
Dimension
Batch Size
Normalize
```

### Reranker

```text
Reranker Type
Base URL
Model
API Key
Candidate Count
Rerank Count
```

提供：

```text
Save Profile
Test Connection
Set as Global Default
Set as Chat Default
```

---

# 16. API

至少建立：

## Chats

```http
POST   /chats
GET    /chats
GET    /chats/{chat_id}
PATCH  /chats/{chat_id}
DELETE /chats/{chat_id}
```

## Documents

```http
POST   /chats/{chat_id}/documents
GET    /chats/{chat_id}/documents
GET    /chats/{chat_id}/documents/{document_id}
DELETE /chats/{chat_id}/documents/{document_id}
POST   /chats/{chat_id}/documents/{document_id}/retry
```

## Sessions

```http
POST   /chats/{chat_id}/sessions
GET    /chats/{chat_id}/sessions
GET    /chats/{chat_id}/sessions/{session_id}
PATCH  /chats/{chat_id}/sessions/{session_id}
DELETE /chats/{chat_id}/sessions/{session_id}
```

## QA

```http
POST /chats/{chat_id}/sessions/{session_id}/messages
```

支援 SSE 或 WebSocket streaming。

## Provider Profiles

```http
POST   /provider-profiles
GET    /provider-profiles
PATCH  /provider-profiles/{profile_id}
DELETE /provider-profiles/{profile_id}
POST   /provider-profiles/{profile_id}/test
```

---

# 17. 回答格式

API response：

```python
class QAResponse(BaseModel):
    answer: str
    citations: list[Citation]
    documents_used: list[str]
    coverage: CoverageResult
    uncertainty: list[str]
    session_id: str
    message_id: str
    debug_trace: DebugTrace | None
```

Citation：

```python
class Citation(BaseModel):
    citation_id: str
    chat_id: str
    document_id: str
    document_name: str
    page_start: int
    page_end: int
    section_title: str | None
    source_node_id: str
    excerpt: str
```

final response 不得包含不屬於 current chat_id 的 citation。

---

# 18. 驗證資料集

第一階段以 arXiv 論文作驗證。

請在 sample/evaluation 設計支援：

* 單篇論文
* 同一 Chat 內多篇論文
* 不同 Chat 使用不同論文
* 雙欄格式
* 包含表格與 figure
* 包含 Appendix
* 包含 References

不要將大型 PDF commit 進 repository。

可以：

* 提供下載 script
* 使用小型 fixture
* README 說明如何放入 arXiv PDF
* integration test 使用自行生成的小 PDF fixture

---

# 19. Golden Evaluation Cases

建立 evaluation cases YAML/JSON。

至少包含以下類別。

## 19.1 Global Summary

```text
請摘要整篇論文。
```

驗證：

* 所有主要 sections 被涵蓋
* 不得只使用 top-k search
* 包含研究目的、方法、實驗、結果、限制

## 19.2 Method Explanation

```text
詳細說明論文提出的方法與資料流。
```

驗證：

* 方法 section evidence
* architecture / algorithm evidence
* 不得只使用 abstract

## 19.3 Technical Comparison

```text
比較論文中的方法 A 與 baseline B。
```

驗證：

* A、B 都有 evidence
* comparison dimensions 完整
* performance、架構、限制分開呈現

## 19.4 Performance

```text
方法 A 是否真的優於方法 B？
```

驗證：

* dataset
* metric
* experimental setting
* numeric result
* limitation
* 不可只引用作者結論

## 19.5 Multi-document

```text
比較這個 Chat 中三篇論文採用的方法。
```

驗證：

* 三篇都被涵蓋
* 每個 claim 有對應文件 citation

## 19.6 Chat Isolation

Chat A 上傳論文 A。

Chat B 上傳論文 B。

在 Chat A 問論文 B 的獨特內容。

預期：

```text
目前 Chat 的文件中沒有足夠資訊。
```

不得取得 Chat B citation。

## 19.7 Session Isolation

同一 Chat 的 Session 1 與 Session 2 具有不同 conversation history。

驗證：

* Session 2 不得讀取 Session 1 的 user messages
* 兩者仍可查詢相同文件

---

# 20. Self-Repair Validation Loop

請建立一套 coding agent 在開發過程中必須執行的循環驗證機制。

不得在第一次執行失敗後停止，也不得只列出未解決錯誤。

每個開發 Phase 完成後，執行以下 loop。

## 20.1 Loop 流程

```text
Implement
→ Static Validation
→ Unit Tests
→ Integration Tests
→ Goal Tests
→ Analyze Failures
→ Identify Root Cause
→ Create Repair Plan
→ Apply Minimal Repair
→ Re-run Relevant Tests
→ Re-run Full Regression
→ Evaluate Goal Coverage
→ Continue or Stop
```

## 20.2 每輪輸出 Repair Report

建立：

```text
artifacts/repair-loop/iteration-{n}.md
```

內容包括：

```text
Iteration number
Current implementation scope
Commands executed
Passed tests
Failed tests
Observed symptoms
Root cause hypotheses
Confirmed root cause
Files to modify
Repair plan
Changes made
Regression result
Remaining risks
Goal coverage score
```

## 20.3 Failure 分類

將失敗分類：

```text
BUILD_FAILURE
TYPE_FAILURE
LINT_FAILURE
UNIT_TEST_FAILURE
INTEGRATION_FAILURE
VESPA_SCHEMA_FAILURE
VESPA_QUERY_FAILURE
PARSER_FAILURE
RETRIEVAL_QUALITY_FAILURE
RERANK_FAILURE
CHAT_ISOLATION_FAILURE
SESSION_ISOLATION_FAILURE
CITATION_FAILURE
CONTEXT_OVERFLOW_FAILURE
PROVIDER_FAILURE
FRONTEND_FAILURE
E2E_FAILURE
```

## 20.4 修復政策

每輪必須：

1. 先重現問題。
2. 找出最小失敗案例。
3. 不可為通過測試而 hard-code 特定答案。
4. 優先修正 root cause。
5. 修正後執行受影響測試。
6. 再執行完整 regression。
7. 若同類問題連續失敗兩次，重新檢查架構假設。
8. 若連續三次沒有進展，輸出 blocker 與替代方案，但繼續處理其他可修復問題。
9. 不得刪除重要測試來讓 pipeline 通過。
10. 不得降低 Chat isolation 或 citation correctness 標準。

## 20.5 最大循環次數

預設：

```text
MAX_REPAIR_ITERATIONS = 8
```

但若仍有明確可修復問題，可繼續至 12 輪。

每輪需記錄：

```text
failed_test_count
passed_test_count
goal_score
new_regressions
```

停止條件不是「沒有 compiler error」，而是符合 Definition of Done。

---

# 21. Goal Coverage Scoring

建立自動化 goal score，總分 100。

## Backend 基礎：10 分

* API 可啟動
* PostgreSQL migration 正常
* 錯誤處理正常

## Parsing：15 分

* arXiv 雙欄閱讀順序合理
* section hierarchy 正確
* 頁碼與 bbox 保留
* abstract / references / appendix 可辨識

## Vespa Retrieval：20 分

* BM25 可搜尋
* ANN 可搜尋
* chat_id filter 正確
* hybrid fusion 正確
* rerank 正常
* debug score 可查看

## Agent QA：20 分

* LangGraph workflow 正常
* structural fetch
* hybrid search
* structured facts
* aggregation
* coverage validation
* citations

## Isolation：15 分

* Chat document isolation
* Vespa filter isolation
* Session history isolation
* citation scope validation

## Provider Settings：10 分

* OpenAI
* Gemini
* vLLM/OpenAI-compatible
* connection test
* encrypted API key

## Frontend：10 分

* Chat 管理
* 文件上傳
* Session 切換
* QA streaming
* citations
* settings

最低接受分數：

```text
90 / 100
```

且以下項目為 mandatory gates：

```text
Chat isolation
Session isolation
Vespa hybrid retrieval
Citations
arXiv parsing
LangGraph QA
Provider settings
```

任何 mandatory gate 失敗，不能宣稱完成。

---

# 22. 必須執行的驗證命令

至少執行：

```bash
ruff check .
ruff format --check .
mypy apps/backend/src
pytest tests/unit
pytest tests/integration
pytest tests/evaluation
```

Frontend：

```bash
npm run lint
npm run typecheck
npm run test
npm run build
```

Infrastructure：

```bash
docker compose config
docker compose up -d postgres vespa
```

Vespa：

```text
deploy application package
validate schema
feed fixture documents
run BM25 query
run ANN query
run hybrid query
run rerank query
run cross-chat isolation query
```

E2E：

```text
create Chat A
upload paper A
wait for ingestion
create Session A1
ask question
verify citation

create Chat B
upload paper B
ask paper A-only question
verify no cross-chat result
```

---

# 23. Vespa Retrieval Evaluation

建立一小組人工標記的 retrieval cases。

每個 case 包含：

```python
chat_id
query
expected_document_ids
expected_source_node_ids
forbidden_document_ids
```

評估：

```text
Recall@k
MRR
nDCG@k
Chat leakage count
Rerank improvement
```

至少比較：

```text
BM25 only
Vector only
Hybrid before rerank
Hybrid after rerank
```

輸出報告：

```text
artifacts/evaluation/retrieval-report.json
artifacts/evaluation/retrieval-report.md
```

成功標準：

* Hybrid Recall@k 不低於較佳單一 retriever。
* Rerank 後 MRR 或 nDCG 不低於 rerank 前。
* Chat leakage count 必須為 0。
* forbidden document 不得出現在結果。

---

# 24. Parsing Evaluation

針對 arXiv fixture 建立 expected structure：

```text
title
abstract
section count
section titles
page ranges
references start page
appendix presence
```

評估：

```text
heading precision
heading recall
reading order correctness
paragraph duplication
header/footer leakage
```

可使用人工 fixture 或小型 golden annotations。

若 parser 失敗，不要直接用 LLM 掩蓋所有 parsing 問題。

LLM 可以協助 hierarchy refinement，但：

* 原始 layout metadata 必須保存
* 必須可追蹤 heuristic 與 LLM 修正
* 不可讓同一內容重複存入多個 section

---

# 25. 實作 Phase

## Phase 1：Foundation

* monorepo
* configuration
* PostgreSQL
* migrations
* domain models
* Docker Compose
* provider abstraction

## Phase 2：Chat / Session / Document

* Chat CRUD
* Session CRUD
* document upload
* storage
* isolation tests

## Phase 3：arXiv Parser

* PyMuPDF blocks
* column detection
* hierarchy
* tables/captions
* parser evaluation

## Phase 4：Enrichment

* summaries
* keywords
* entities
* claims
* facts
* document overview

## Phase 5：Vespa

* application package
* schema
* feeding
* delete
* BM25
* ANN
* hybrid retrieval
* phased ranking
* reranking
* isolation filter

## Phase 6：LangGraph

* state
* tools
* planner
* budget
* coverage
* answer
* citations
* session persistence

## Phase 7：Frontend

* Chat sidebar
* uploads
* ingestion status
* Session UI
* chat streaming
* citations
* settings

## Phase 8：Evaluation and Repair

* golden QA
* retrieval evaluation
* parser evaluation
* isolation E2E
* repair loop
* final score

每完成一個 Phase，立即執行 validation loop。

---

# 26. 程式品質要求

* 所有 public functions 有 type hints。
* Domain logic 不直接依賴 FastAPI。
* LangGraph nodes 不直接寫 SQL。
* Vespa query 統一透過 RetrievalService。
* Chat filter 由 service 強制注入。
* Agent 工具不能自行指定其他 chat_id。
* 不使用無限制 `dict[str, Any]` 作為核心資料模型。
* provider adapter 可替換。
* 所有 LLM 呼叫支援 async。
* ingestion job 可重試且 idempotent。
* Vespa feed 需可重複執行。
* 刪除文件需清除 Vespa documents。
* tool result 必須包含 status、token estimate、sources。
* 不記錄 API key。
* 不允許 raw exception 直接回傳前端。
* debug trace 需移除敏感資訊。
* 測試不能依賴真實付費 API。
* 提供 deterministic mock LLM、embedding 與 reranker。

---

# 27. MVP 非目標

第一版暫時不要求：

* OCR scanned PDF
* Word、PowerPoint、Excel
* 音訊與影片
* 多使用者登入與權限系統
* collaborative editing
* production-grade distributed queue
* GraphRAG community detection
* 完整數學公式理解
* 完整 figure VLM 解讀
* mobile application

但架構應保留 parser adapter 與 provider adapter。

---

# 28. Definition of Done

只有符合以下所有條件，才能宣稱 MVP 完成：

1. 可以建立多個 Chat。
2. 每個 Chat 可上傳多份 PDF。
3. arXiv PDF 能解析出合理 section hierarchy。
4. 文件可成功 feed 到 Vespa。
5. Vespa BM25 搜尋可運作。
6. Vespa ANN 搜尋可運作。
7. Hybrid retrieval 可運作。
8. Rerank 可運作。
9. LangGraph Agent 可選擇 structural、hybrid、facts 與 aggregation 工具。
10. 可以針對整份論文進行摘要。
11. 可以回答方法與 performance 問題。
12. 回答包含正確 citation。
13. Chat A 不會檢索 Chat B 文件。
14. Session history 相互隔離。
15. OpenAI profile 可設定與測試。
16. Gemini profile 可設定與測試。
17. vLLM OpenAI-compatible profile 可設定與測試。
18. Frontend 可以完成主要流程。
19. 所有 mandatory tests 通過。
20. Goal score 至少 90。
21. Repair loop 報告存在。
22. README 清楚說明執行方式與限制。

---

# 29. 最終交付報告

完成後輸出：

1. Repository tree
2. 架構說明
3. Chat / Session isolation 設計
4. arXiv parsing 流程
5. Vespa schema
6. BM25 + ANN hybrid retrieval 設計
7. rerank 設計
8. LangGraph workflow
9. Provider settings 設計
10. Frontend 頁面
11. 執行方式
12. 測試命令
13. 測試結果
14. Retrieval evaluation 結果
15. Parser evaluation 結果
16. Goal score
17. Repair loop iteration 摘要
18. 未完成與簡化項目
19. 已知風險
20. 下一步最重要的三項改善

不要宣稱未實作的項目已完成。

若環境限制導致 Vespa、Docker、模型端點或 frontend 無法執行，必須清楚說明：

* 執行過什麼
* 失敗原因
* 錯誤訊息摘要
* 哪些部分仍未驗證
* 使用者應如何重現