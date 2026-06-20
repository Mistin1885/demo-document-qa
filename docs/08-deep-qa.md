# 08 · Deep QA Mode

Deep QA mode is an opt-in **per-message** generation profile that trades
latency and tokens for broader evidence, larger answers, and
session-aware follow-up reasoning. Everything else (chat isolation,
citation contract, policy enforcement) is unchanged — Deep QA only
relaxes the **soft budget** and widens the **retrieval surface** that the
agent already operates inside.

> Prereqs: skim [`05-agent-workflow.md`](./05-agent-workflow.md) and
> [`07-qa-pipeline.md`](./07-qa-pipeline.md) first — the nodes, tools, and
> policies named below are defined there.

---

## 1. What it actually changes

Deep QA mode is a single boolean `deep_qa_mode` on the request body. When
true, the backend changes behaviour in five concrete places — and only
those five places.

| Where | Default mode | Deep QA mode |
|---|---|---|
| `MessageRequest.max_answer_tokens` (default) | `LLM_MAX_TOKENS` env | **`32_768`** if user did not override |
| `MessageRequest.context_window` (default) | `LLM_CONTEXT_WINDOW` env (≈ 10k) | **`200_000`** if user did not override |
| `ContextBudgetManager.ignore_budget` | `False` (overflow → `aggregate_sources_node`) | **`True`** — `detect_overflow()` always returns `False`, so evidence is never compacted by P6 |
| `_search_params` (`execute_retrieval_tools`) | `preset="balanced"`, `top_k=10`, `max_tokens≈6k` | **`preset="broad"`**, **`top_k=20`**, **`max_tokens=12_000`** |
| `_grep_params` (`grep_document_chunks`) | `context_chars=1_200`, `scan_limit=500`, `limit=8`, `max_tokens=6_000` | **`context_chars=2_000`**, **`scan_limit=1_000`**, **`limit=12`**, **`max_tokens=10_000`** |
| `generate_answer` user message | evidence + question only | evidence + question **+ last 8 same-session turns** as a labelled memory block |
| `plan_information_needs` query rewrite | only triggers on ambiguous follow-up markers ("which is better", "那", …) | **always** prepends recent history to the retrieval query |

The chat-isolation contract (CLAUDE.md §2) is **not** touched: `chat_id`
is still service-injected, all four isolation layers still run, and
citations still pass policies P12/P13/P14.

Source: `src/app/api/messages.py:81-95`, `src/app/agent/state.py:289-313`,
`src/app/agent/nodes/execute_retrieval_tools.py:56-116`,
`src/app/agent/nodes/generate_answer.py:126-133`,
`src/app/agent/nodes/plan_information_needs.py:178-201`,
`src/app/agent/budget.py:214-232`,
`src/app/services/qa_service.py:417-429,544-554`.

---

## 2. Enabling it

### 2.1 From the UI

1. Open a chat → focus the chat input → click **⚙ Advanced**.
2. Tick **"Deep QA mode"** at the top of the panel.
   The badge in the header turns to `custom`.
3. Send the question. The flag is sent on **that message only** but the
   panel persists the choice per-chat (`useGenerationPrefs`).

Source: `src/frontend/components/chat/GenerationPrefsPanel.tsx:60-76`,
`src/frontend/lib/chat/useGenerationPrefs.ts`.

![Deep QA in the chat panel](./images/deep-q1.png)

### 2.2 From the API

```bash
curl -N -X POST \
  "http://localhost:8000/chats/${CID}/sessions/${SID}/messages" \
  -H "Accept: text/event-stream" \
  -H "Content-Type: application/json" \
  -d '{
        "question": "Compare graph generation modules of LightRAG and GraphRAG.",
        "deep_qa_mode": true
      }'
```

Optional siblings on the body (all `extra="forbid"`):

| Field | Effect | Range |
|---|---|---|
| `max_answer_tokens` | overrides the Deep-QA default of 32 768 | `[1, 32_768]` |
| `temperature` | unchanged by Deep QA | `[0.0, 2.0]` |
| `context_window` | overrides the Deep-QA default of 200 000 | `[1_000, 200_000]` |
| `selected_document_ids` | first-message session document scope; backend then locks it via `document_scope_locked` | `list[UUID]` |

If `deep_qa_mode` is absent or `false`, the env-level fallbacks
(`LLM_MAX_TOKENS`, `LLM_CONTEXT_WINDOW`, `LLM_TEMPERATURE`) apply
exactly as before.

---

## 3. What stays the same (important)

Deep QA mode is **not** a bypass for any contract — only for the soft
budget. The following are still enforced byte-for-byte:

- **Chat isolation.** P1 / P12 / P13 / `validate_scope_isolation` still
  run. Vespa YQL still has `chat_id contains "<current>"` injected by
  `RetrievalService._yql_where`.
- **Session isolation (P2).** Memory included in the answer prompt is
  filtered to `(session_id, chat_id)` — never another session's history.
- **Numeric facts first (P4) / structural fetch-all (P3).** Plan shape
  is unchanged; only retrieval widths are widened.
- **Citation scope (P12/P13).** Citations whose `document_id` is not in
  this chat's `chat_documents` are still dropped after generation.
- **Provider failure (P14).** No silent model swap, even when Deep QA
  bumps `max_tokens` past a provider ceiling.
- **No model knowledge (P11).** When no evidence/facts survive, the
  answer is still the exact fallback string.

The `llm_replan` bounded loop (CLAUDE.md §8 amendment) runs in **both**
modes — it is governed by `MAX_REPLAN_ROUNDS = 3` and policy P15, not by
`deep_qa_mode`.

---

## 4. Why "ignore soft budget" is safe

`ContextBudgetManager.detect_overflow()` early-returns `False` when
`ignore_budget=True` (`src/app/agent/budget.py:221-222`). That short-
circuits the P6 (`overflow_must_aggregate`) routing to
`aggregate_sources_node` — meaning evidence is **not** LLM-summarised
before reaching `generate_answer`.

Two hard ceilings still protect the run:

1. The **provider's** real context window. Once the request is over the
   wire, the LLM hosts its own truncation policy. We do not pretend it
   away — Deep QA only stops *our own* pre-emptive compaction.
2. **Policy P9 — `iteration_cap`.** Tool-call iteration is still capped
   at 2 by `_route_coverage`; gap retrieval cannot fire indefinitely.
3. **`MAX_REPLAN_ROUNDS = 3`** on the bounded LLM-driven replan node.

If the provider rejects the request as too large, P14 fires (`AgentError`
recorded, no silent model switching) and the SSE stream emits
`event: error` `{"code": "AGENT_ERROR", …}` — the user sees a clear
failure, not a degraded answer.

---

## 5. End-to-end trace of a Deep QA message

```
POST …/messages  { "question": "…", "deep_qa_mode": true }
  │
  ▼
api/messages._generation_config_from_request
  → GenerationConfig(max_answer_tokens=32_768,
                     context_window=200_000,
                     deep_qa_mode=True)
  │
  ▼
qa_service.stream
  → ContextBudgetManager(default_context_window=200_000, ignore_budget=True)
  → build_graph(...)
  │
  ▼ load_chat_and_session  · inspect_scope
  ▼ plan_information_needs
      ↳ _question_with_session_context() prepends recent history
        because state.generation_config.deep_qa_mode is True
  ▼ enforce_scope_and_policies                (P1–P4 — unchanged)
  ▼ execute_retrieval_tools
      ↳ _search_params() widens search_hybrid → preset=broad, top_k=20, max_tokens=12k
      ↳ _grep_params()  widens grep_document_chunks → scan_limit=1000, ctx=2000, limit=12
  ▼ merge_evidence_workspace
  ▼ check_context_budget
      ↳ detect_overflow() → False (ignore_budget=True), never routes to aggregate
  ▼ check_coverage      (may still loop via plan_gap_retrieval / llm_replan)
  ▼ verify_critical_claims                    (P8 — unchanged)
  ▼ generate_answer
      ↳ _build_session_memory_block() prepends last-8 turns of (session_id, chat_id)
      ↳ chat_provider.complete(messages, max_tokens=32_768, temperature=…)
  ▼ validate_citations · validate_scope_isolation · persist_messages
```

`event: token` / `event: citation` / `event: error` framing is identical
to a normal QA stream — Deep QA does not change the SSE protocol.

---

## 6. When to use it

Use Deep QA mode when:

- The chat contains **several long papers** and the question asks for a
  cross-document synthesis (overview, comparison, full-chapter summary).
- The user is asking a **session follow-up** that depends on a previous
  answer (pronouns, "which", "那個", "compare them"). Deep QA's history
  prefix to the retrieval query usually fixes the resolution.
- You expect the answer to need **more than ~2k output tokens**
  (default `max_answer_tokens` would truncate).
- You want to bypass the conservative aggregation step so the LLM sees
  raw evidence chunks instead of a summary of them.

Avoid it for:

- **Numeric / single-fact** questions. Normal mode already uses
  `query_structured_facts` first, and the wider retrieval just costs
  tokens.
- **Single-document, single-section** questions. Normal mode is usually
  faster and the answers are equivalent.
- Providers with **small context windows**. Deep QA defaults to a
  200 000-token `context_window` and a 32 768-token max output, which
  many smaller-context providers will reject (you'll see P14 fire).

---

## 7. Knobs you can tune

| Knob | Where | Default | Notes |
|---|---|---|---|
| `deep_qa_mode` | `MessageRequest.deep_qa_mode` | `False` | The on/off switch |
| `max_answer_tokens` | request body | `32_768` in Deep QA | overrides Deep-QA default |
| `context_window` | request body | `200_000` in Deep QA | overrides Deep-QA default |
| `max_replan_rounds` | `GenerationConfig.max_replan_rounds` | `None` → `MAX_REPLAN_ROUNDS=3` | hard cap is 3 |
| `LLM_MAX_TOKENS` | env (server) | `2048` | applies in non-Deep mode |
| `LLM_CONTEXT_WINDOW` | env (server) | `10_000` | applies in non-Deep mode |
| `MAX_REPLAN_ROUNDS` | `src/app/agent/budget.py:21` | `3` | hard ceiling, not request-overridable |

---

## 8. Testing it locally

```bash
# UI path — the easiest reproduction:
docker compose -f deploy/docker-compose.yml up -d
uv run python scripts/deploy_vespa.py
open http://localhost:3000           # Upload PDFs, tick Deep QA, ask

# API smoke — no UI:
LLM_PROVIDER=openai_compatible \
  LLM_API_URL=http://localhost:8001/v1 \
  LLM_MODEL=gemma-3-27b-it \
  uv run python scripts/smoke_agent_e2e.py

# Inspect the SSE trace:
curl -N -X POST "$BASE/chats/$CID/sessions/$SID/messages" \
  -H "Accept: text/event-stream" \
  -H "Content-Type: application/json" \
  -d '{"question":"Summarise both papers.","deep_qa_mode":true}'
```

Deep-QA paths are covered by the same Golden QA / Goal-Score harnesses
described in CLAUDE.md §11 — there is no separate "deep" eval suite,
because Deep QA is a configuration of the same graph.
