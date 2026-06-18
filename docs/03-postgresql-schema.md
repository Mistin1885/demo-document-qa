# 03 · PostgreSQL Schema & Usage

PostgreSQL is the **system of record** for everything that is not full-text
or vector search. Vespa holds chunk embeddings and BM25 indexes; Postgres
holds the canonical chat/session/document/node/summary/fact rows, every
isolation key (`chat_id`, `document_id`), and the ingestion state machine.

> The 4-layer isolation contract (CLAUDE.md §2) starts here: every
> document-scoped row carries `chat_id`, every service-layer query injects
> `WHERE chat_id = :current_chat_id`, and the FK graph cascades cleanly on
> chat deletion.

---

## 1. Connection & session factory

| Concern | Where | Notes |
|---|---|---|
| URL source | `src/app/config.py` — `Settings.database_url` | required env `DATABASE_URL`; no default |
| Runtime driver | `postgresql+asyncpg://…` | SQLAlchemy 2.x async engine |
| Alembic driver | rewritten to `postgresql+psycopg://…` | see `migrations/env.py:_get_sync_url` |
| Engine | `src/app/db.py:_get_engine` | `pool_pre_ping=True`, lazy creation |
| Session factory | `src/app/db.py:_get_sessionmaker` | `async_sessionmaker(expire_on_commit=False)` |
| FastAPI dependency | `src/app/db.py:get_session` | yields `AsyncSession`; auto-commit on success, rollback on raise |

`alembic.ini` deliberately does **not** hard-code `sqlalchemy.url` — the
migration env imports the same `Settings`, so dev / CI / docker-compose all
share one truth.

---

## 2. Tables

All ORM models live in `src/app/models/orm.py`. Every PK is `UUID` and every
timestamp is `DateTime(timezone=False)` with `server_default=now()`. FK
ondelete behaviour is annotated below.

### 2.1 `chats` — Chat (the isolation root)

| Column | Type | Notes |
|---|---|---|
| `id` | UUID | PK |
| `name` | String(255) | not null |
| `description` | Text | null |
| `default_chat_profile_id` | UUID | → `provider_profiles.id`, indexed |
| `default_embedding_profile_id` | UUID | → `provider_profiles.id`, indexed |
| `default_reranker_profile_id` | UUID | → `provider_profiles.id`, indexed |
| `created_at` / `updated_at` | DateTime | server-managed |

Chats own everything below; deleting a chat cascades through
`sessions`, `documents`, `summaries`, `structured_facts`, `ingestion_jobs`,
and `chat_documents`.

### 2.2 `sessions` — one conversation thread under a chat

| Column | Type | Notes |
|---|---|---|
| `id` | UUID | PK |
| `chat_id` | UUID | → `chats.id` **(ondelete CASCADE, indexed)** — mandatory isolation key |
| `name` | String(255) | nullable |
| `chat_profile_id` | UUID | → `provider_profiles.id`, indexed; overrides chat default |
| `created_at` / `updated_at` | DateTime | server-managed |

**Session isolation rule (CLAUDE.md §2):** a session must never read another
session's message history. Enforced in `session_service._require_session`
which filters on `(Session.id, Session.chat_id)` together.

### 2.3 `messages` — one chat turn

| Column | Type | Notes |
|---|---|---|
| `id` | UUID | PK |
| `session_id` | UUID | → `sessions.id` (ondelete CASCADE, indexed) |
| `role` | String(20) | `'user' | 'assistant' | 'system' | 'tool'` |
| `content` | Text | not null |
| `citations` | JSONB | serialized `list[Citation]` (see `app.models.domain`) |
| `tool_trace` | JSONB | serialized `ToolTrace` |
| `token_count` | Integer | nullable |
| `created_at` | DateTime | server-managed |

`chat_id` is reached transitively through `session_id`; session-level cascade
covers cleanup.

### 2.4 `documents` — uploaded / ingested document

| Column | Type | Notes |
|---|---|---|
| `id` | UUID | PK |
| `chat_id` | UUID | → `chats.id` (ondelete CASCADE, indexed) — isolation key |
| `source_type` | String(20) | `'upload' | 'arxiv' | 'url'` |
| `original_filename` | String(512) | not null |
| `storage_path` | String(1024) | not null; under `data/storage/` |
| `mime_type` | String(128) | not null |
| `page_count` | Integer | nullable |
| `status` | String(30) | `'uploaded' | 'parsing' | 'parsed' | 'enriching' | 'indexed' | 'failed'` |
| `checksum_sha256` | String(64) | dedup / idempotency |
| `created_at` / `updated_at` | DateTime | server-managed |

Composite index `ix_documents_chat_status(chat_id, status)` supports the
"list documents in chat where status = indexed" query the UI hits often.

### 2.5 `document_nodes` — parsed structural tree

| Column | Type | Notes |
|---|---|---|
| `id` | UUID | PK |
| `document_id` | UUID | → `documents.id` (ondelete CASCADE, indexed) |
| `chat_id` | UUID | → `chats.id` — **denormalized** for fast scope filtering |
| `parent_id` | UUID | → `document_nodes.id` (self-ref, indexed) |
| `node_type` | String(30) | `document / section / subsection / paragraph / figure / table / equation / reference` |
| `title` | Text | nullable |
| `content` | Text | full text or caption |
| `page_start` / `page_end` | Integer | 1-indexed |
| `order_index` | Integer | reading order |
| `level` | Integer | depth (1 = document, 2 = section, …) |
| `bbox` | JSONB | MinerU bbox in PDF-point space |
| `metadata` | JSONB | default `{}` |

Index `ix_document_nodes_doc_order(document_id, order_index)` powers the
deterministic structural fetch path used by the agent's
`fetch_structural_nodes` tool.

### 2.6 `summaries`

`(chat_id, document_id, source_node_id?, kind, content, keywords, entities, token_count)`.
`source_node_id` is `NULL` for document-level summaries (overview / compact
overview); per-section summaries point to the originating
`document_nodes.id` and `ondelete SET NULL` keeps the row when only the
node is removed.

### 2.7 `structured_facts`

`(chat_id, document_id, source_node_id?, kind, key, value JSONB, unit?, context_excerpt?, page?)`.
This is the table the agent's `query_structured_facts` tool reads — numbers,
benchmark scores, dataset names, hyperparameters. Filters never accept raw
SQL; only a restricted `FactsFilter` Pydantic schema is honoured.

### 2.8 `provider_profiles`

`(id, kind, provider_type, name, base_url?, model, api_key_encrypted LargeBinary?, config JSONB, context_window?, is_default, …)`.
- `kind` ∈ `{chat, embedding, reranker}` — the three independent profile
  families a chat can wire up.
- `provider_type` ∈ `{openai, gemini_native, gemini_compat, openai_compat, vllm}`.
- `api_key_encrypted` — Fernet ciphertext; never returned to the frontend,
  never logged. See §4.
- Composite index `ix_provider_profiles_kind_is_default(kind, is_default)`
  serves the "find the default chat provider" lookup.

### 2.9 `ingestion_jobs`

`(id, chat_id, document_id, state, attempt, last_error?, started_at?, finished_at?, created_at)`.
- `state` ∈ `{pending, running, succeeded, failed}`.
- One job row per `(chat_id, document_id)` — see §5.

### 2.10 `chat_documents` — association for cross-chat reuse

Composite primary key `(chat_id, document_id)`. Both columns reference their
parent table with `ondelete CASCADE`. A document originally created in chat A
becomes visible in chat B by inserting `(B, doc_id)` — Vespa filters and
service-layer queries then naturally let chat B "see" it without copying any
chunks. Deleting chat A only removes its associations; the document survives
as long as any other association still points at it.

---

## 3. Isolation enforcement pattern

Every read path through Postgres ends in a `WHERE chat_id = :chat_id`. Two
representative call-sites:

- `src/app/services/document_service.py:94-102` —
  ```python
  stmt = (
      select(Document)
      .join(ChatDocument, ChatDocument.document_id == Document.id)
      .where(ChatDocument.chat_id == chat_id, Document.id == document_id)
  )
  ```
- `src/app/services/session_service.py:53-58` —
  ```python
  await db.execute(
      select(SessionORM).where(
          SessionORM.id == session_id,
          SessionORM.chat_id == chat_id,
      )
  )
  ```

`chat_id` is **never** taken from a request body; routers read it from the
URL path (`/chats/{chat_id}/...`) and pass it to the service. The agent's
`AgentState.chat_id` is service-injected and unreachable from the LLM (see
`docs/05-agent-workflow.md`).

---

## 4. Provider-credential encryption

`src/app/security.py` exposes three primitives:

| Function | Purpose |
|---|---|
| `encrypt(plaintext) → bytes` | Fernet (AES-128-CBC + HMAC-SHA256) ciphertext |
| `decrypt(token) → str` | inverse; raises `InvalidToken` on tamper |
| `mask_secret(value) → str` | log-safe redaction (`sk-...7890`) |

The Fernet key is derived from `APP_ENCRYPTION_KEY` (`Settings.app_encryption_key`):

- if it is already a 32-byte URL-safe base64 string, used directly;
- otherwise the SHA-256 of the bytes is base64-url-encoded → Fernet key.

Round-trip is deterministic, so the same env value always decrypts existing
rows. **Never log a raw API key**; always wrap it in `mask_secret()`.

---

## 5. Ingestion idempotency

`IngestionJob` is the state machine, but the *effective* idempotency comes
from `src/app/services/ingestion_service.py` running

```python
await vespa_client.delete_by_document(chat_id, document_id)
```

before feeding new chunks (≈ line 90). Re-running ingestion against the same
document is therefore always safe — old chunks are dropped first, the
checksum prevents duplicate `documents` rows, and the job row is upserted on
`(chat_id, document_id)` so the `attempt` / `last_error` columns reflect the
*latest* attempt rather than appending a new row per retry.

---

## 6. Running migrations

```bash
# Local dev
uv run alembic upgrade head
uv run alembic revision --autogenerate -m "describe change"

# Inside docker compose (backend container)
docker compose -f deploy/docker-compose.yml exec backend alembic upgrade head
```

The backend container's `CMD` already runs `alembic upgrade head` before
starting Uvicorn, so a fresh `docker compose up` brings the schema up to
date automatically. Only one revision exists today
(`migrations/versions/b5b02bc9d209_initial_schema.py`) — every later schema
change must be a new revision; never edit the initial migration in-place.

---

## 7. Quick troubleshooting

| Symptom | Where to look |
|---|---|
| 404 on `GET /chats/{cid}/documents/{did}` | the document exists but no row in `chat_documents(chat_id=cid)` |
| Document stuck in `parsing` | `SELECT * FROM ingestion_jobs WHERE document_id=…` — check `state`, `last_error`, `attempt` |
| `InvalidToken` on provider use | `APP_ENCRYPTION_KEY` differs from the value used to encrypt the row; rotate by re-saving the profile |
| Retrieval returns 0 hits across all modes | check the `chat_id` denormalization on `document_nodes` matches the chat actually queried; verify a corresponding Vespa feed has happened |
