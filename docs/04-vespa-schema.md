# 04 · Vespa Schema & Usage

Vespa is the **only** engine that answers "find the most relevant chunks of
this chat's documents." BM25, vector ANN, hybrid fusion, native rerank — all
of them run inside Vespa; the application talks to it through one entry
point (`RetrievalService.search`).

> CLAUDE.md §3 forbids FAISS / Chroma / `rank-bm25` as the production search
> engine. Unit tests may use the `app.vespa.mock` transport; integration and
> evaluation tests must hit a real Vespa.

---

## 1. Cluster topology

`deploy/vespa/application/` is a self-contained Vespa **application package**:

```
deploy/vespa/application/
├── services.xml                    # container + content cluster definitions
├── hosts.xml                       # single-node dev topology
├── validation-overrides.xml        # allow embed-dimension changes during dev
└── schemas/
    └── document_chunk.sd           # the only document type we feed
```

`services.xml` declares:

| Cluster | id | role |
|---|---|---|
| `<container>` | `default` | search front-end, document-API, document-processing; search timeout 5.0 s |
| `<content>` | `documents` | one document type (`document_chunk`); redundancy 1 |

`hosts.xml` is a single host (`localhost`, alias `node1`) — fine for the
docker-compose dev stack. Production deployments scale the content cluster
horizontally; the application code does not care because everything goes
through `RetrievalService`.

---

## 2. `document_chunk` schema

One field per concept, no nested structs. Generated programmatically by
`src/app/vespa/app_package.py:build_application_package(embedding_dim)` so
the embedding dimension stays config-driven.

### 2.1 Identity & isolation

| Field | Type | Indexing | Notes |
|---|---|---|---|
| `vespa_document_id` | string | `attribute | summary`, fast-search | UUID-5 from `(chat_id, document_id, node_id, source_type)` — see `src/app/vespa/encoders.py` |
| `chat_id` | string | `attribute | summary`, fast-search | **MUST appear in every WHERE** |
| `document_id` | string | `attribute | summary`, fast-search | document scope |
| `source_node_id` | string | `attribute | summary` | links a chunk back to `document_nodes.id` |
| `parent_node_id` | string | `attribute | summary` | parent section / paragraph |
| `source_type` | string | `attribute | summary`, fast-search | 14 enum values (raw_block, chunk, section_summary, compact_section_summary, chapter_summary, compact_chapter_summary, document_overview, technology_card, claim, definition, performance_fact, table_record, figure_caption, …) |

### 2.2 Searchable text fields (BM25-weighted)

| Field | Type | BM25 weight | Notes |
|---|---|---|---|
| `title` | string | 2.0 | per-chunk title / heading |
| `heading_path` | string | 1.5 | breadcrumb up to the document root |
| `content` | string | 1.0 | the chunk body |
| `keywords` | array<string> | 1.2 | enrichment-extracted keywords |
| `technical_keywords` | array<string> | 1.2 | technical terms |
| `entities` | array<string> | 1.0 | named entities |

All BM25 fields use `indexing: index | summary` (and `attribute | index | summary`
for arrays) with `enable-bm25`.

### 2.3 Numeric / location

`page_start`, `page_end`, `order_index`, `token_count` → `int`, `attribute | summary`.
`created_at` → `long` epoch-ms.

### 2.4 Vector

```text
field embedding type tensor<float>(x[<DIM>]) {
    indexing: attribute | index
    attribute { distance-metric: angular }
    index { hnsw { max-links-per-node: 16, neighbors-to-explore-at-insert: 200 } }
}
```

- Distance metric is **angular** (cosine on unit-normalised vectors).
- `<DIM>` is substituted at app-package build time from
  `Settings.embedding_dim` (default 384, matching the bundled E5-small Vespa
  embedder).

---

## 3. Rank profiles

All defined in `src/app/vespa/app_package.py`:

| Profile | Phase 1 (first-phase) | Phase 2 (rerank) | When the agent picks it |
|---|---|---|---|
| `bm25_only` | weighted BM25 only | — | diagnostics / BM25-only retrieval |
| `semantic_only` | `closeness(embedding)` | — | pure-vector retrieval (rare) |
| `hybrid_first_phase` | weighted BM25 + 100·closeness | firstPhase + 0.5·heading_match_boost + firstPhase·(source_type_boost − 1.0); rerank-count 200 | default hybrid |
| `hybrid_with_native_rerank` | inherits hybrid_first_phase | firstPhase · source_type_boost + 0.5 · heading_match_boost; rerank-count 200 | when `rerank_mode="native"` |
| `hybrid_for_cross_encoder` | first-phase only | — (rerank done in Python) | when `rerank_mode="cross_encoder"` |

The cross-encoder profile deliberately *omits* a second phase — pulling raw
candidates back to the application lets us call an external reranker
(OpenAI-compatible scorer) and recombine, while still keeping every per-stage
score in `SearchHit`.

---

## 4. RetrievalService — the one public Vespa entry point

`src/app/retrieval/service.py` exposes a single async method:

```python
async def search(req: RetrievalRequest) -> RetrievalResponse: ...
```

Everything else in the file is private. `RetrievalRequest` carries the
caller's `chat_id`, the query string, optional `document_ids`,
`source_types`, `top_k`, `rerank_mode`, and a `max_tokens` budget; the
response carries `SearchHit`s with full score breakdown.

### 4.1 Chat-id injection is non-optional

`_yql_where()` (`src/app/retrieval/service.py:96-152`) builds the YQL
`WHERE` clause and **always** starts it with

```text
chat_id contains "<validated-uuid>"
```

Inputs flow through `_validate_uuid_str` (regex `^[0-9a-fA-F-]{36}$`) and
`_safe_str` (rejects `"`, `\`, newline). Any attempt to bypass — empty
chat_id, raw user input, malformed UUID — raises
`InvalidRetrievalFilter` before the query leaves the application.

### 4.2 Hybrid flow

1. **Two parallel queries** (`asyncio.gather`):
   - BM25-only `weakAnd` over `title / heading_path / content / keywords /
     technical_keywords / entities`.
   - `nearestNeighbor(embedding, query_vector)` ANN.
2. **RRF fusion** — `reciprocal_rank_fusion(bm25_hits, ann_hits, k=60)` in
   `src/app/retrieval/fusion.py`. Crucially, the first version explicitly
   **does not** sum un-normalised BM25 + vector scores (CLAUDE.md §7).
3. **Rerank** based on `req.rerank_mode`:
   - `none` → take fused order as-is;
   - `native` → re-query Vespa with `hybrid_with_native_rerank` over the
     fused candidate IDs;
   - `cross_encoder` → call the configured `RerankerProvider` (cross-encoder
     or OpenAI scorer) on the candidate texts.
4. **Truncate** to `req.top_k` (default 8–12) and emit `RetrievalResponse`.

Every `SearchHit` carries:

```
vespa_document_id, chat_id, document_id, source_node_id, parent_node_id,
source_type, title, heading_path, content, page_start, page_end, order_index,
bm25_score, vector_score, fusion_score,
native_rerank_score?, cross_encoder_score?, final_score, final_rank
```

— so the debug trace can show *which stage* moved a hit up or down.

### 4.3 Dimension safety

`RetrievalService.__init__` checks `embedding_provider.dimension == embedding_dim`;
mismatch raises `VespaDimensionMismatch` immediately, before any query goes
out. The same check fires inside `VespaFeedClient` so a stale schema cannot
silently accept wrong-width vectors.

---

## 5. Feeding & deleting documents

`src/app/vespa/feed.py` exposes the only Vespa write surface:

| Method | What it does |
|---|---|
| `VespaFeedClient.feed_chunks(chunks)` | POST `/document/v1/{namespace}/document_chunk/docid/{vespa_document_id}` — one HTTP call per chunk. Production feed omits `embedding`; Vespa computes it from `content` via `input content \| embed e5 \| attribute \| index`. Tests may still inject precomputed `{"embedding": {"values": [...]}}`. |
| `VespaFeedClient.delete_by_document(chat_id, document_id)` | selection-based delete: `document_chunk.chat_id == "<cid>" and document_chunk.document_id == "<did>"` |

`delete_by_document` is what makes ingestion idempotent — see
`docs/03-postgresql-schema.md` §5.

---

## 6. Deployment & operations

### 6.1 Bring the cluster up

```bash
docker compose -f deploy/docker-compose.yml up -d vespa
```

Container exposes:

- `:19071` — config / health (`/state/v1/health`)
- `:8080` — search + document API

### 6.2 Deploy the application package

```bash
uv run python scripts/deploy_vespa.py            # default endpoint http://localhost:8080
uv run python scripts/deploy_vespa.py --dry-run  # render only; no HTTP
```

The script:

1. Reads `Settings.embedding_dim` and calls `build_application_package(dim)`.
2. Writes `services.xml / hosts.xml / schemas/document_chunk.sd /
   validation-overrides.xml` into `deploy/vespa/application/`.
3. Uses `pyvespa` to POST the package to `:19071/application/v2/tenant/default/prepareandactivate`.

### 6.3 Changing the embedding dimension

The schema is generated, but Vespa still treats DIM as a breaking change.
The recipe:

1. Update `EMBEDDING_DIM` / `Settings.embedding_dim` only if you also change
   the Vespa embedder model and schema dimension.
2. Re-run `scripts/deploy_vespa.py` — `validation-overrides.xml` permits
   `embed-dimension-change` for the next ~30 days during dev.
3. Re-ingest every document so existing chunks pick up the new width
   (`VespaFeedClient` will reject mixed widths). The MVP rule (CLAUDE.md §5.2)
   is "one deployment = one embedding dimension."

---

## 7. Querying outside the agent (debugging only)

Vespa accepts raw YQL on `:8080/search/`. **Never** call this from
application code — only from a terminal while diagnosing a query plan:

```bash
curl -s http://localhost:8080/search/ \
  --data-urlencode 'yql=select * from document_chunk where chat_id contains "<cid>" and userQuery()' \
  --data-urlencode 'query=hybrid retrieval rerank' \
  --data-urlencode 'ranking=hybrid_with_native_rerank' \
  --data-urlencode 'input.query(query_vector)=[...]' \
  --data-urlencode 'hits=10'
```

— and even there, the `chat_id contains "<cid>"` predicate is mandatory.
Any production code path that wants to query Vespa goes through
`RetrievalService.search` (or it is a bug).
