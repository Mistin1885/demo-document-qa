# 05 · Agent Workflow & Usage

The QA agent is a **LangGraph `StateGraph`** — not a free-running ReAct
loop. The graph has 16 nodes, 8 tools, and 15 code-enforced policies;
node order, tool inputs, citation scope, and context-budget overflow are
all checked deterministically before the answer reaches the client.
Bounded LLM-driven replanning is restricted to a single dedicated node
(`llm_replan`) whose output is Pydantic-validated and policy-fenced —
the LLM never executes tools directly.

Source map:

```
src/app/agent/
├── graph.py            # build_graph() — 16-node StateGraph
├── state.py            # AgentState, EvidenceItem, ToolCallRecord, …
├── budget.py           # ContextBudgetManager + ContextAllocation + MAX_REPLAN_ROUNDS
├── policies.py         # PolicyEngine — 15 numbered policies
├── nodes/              # one module per StateGraph node (incl. llm_replan)
└── tools/              # 8 capability-shaped tools
```

---

## 1. StateGraph nodes

Order is fixed at compile time in `src/app/agent/graph.py:build_graph` and
the diagram below mirrors `graph.py:220-272`. Conditional edges are marked
with `?`.

```
load_chat_and_session
  → inspect_scope
  → plan_information_needs
  → enforce_scope_and_policies
  → execute_retrieval_tools
  → merge_evidence_workspace
  → check_context_budget
      ?─► aggregate_sources_node     # only if budget overflow (and not deep_qa_mode)
  → check_coverage
      ?─► plan_gap_retrieval         # coverage incomplete & iteration < 2
      │       └─► execute_retrieval_tools           (loops back once)
      ?─► llm_replan                 # bounded LLM-driven replan; round < MAX_REPLAN_ROUNDS (=3)
              └─► execute_retrieval_tools           (loops back once per round)
  → verify_critical_claims
  → generate_answer
  → validate_citations
  → validate_scope_isolation
  → persist_messages
  → END
```

One-line job for each:

1. **`load_chat_and_session`** — fetch chat + session row, conversation history (only this session's messages), provider profile resolution.
2. **`inspect_scope`** — call `inspect_chat` so the planner has a manifest of the chat's documents.
3. **`plan_information_needs`** — LLM produces an `AgentPlan` (goal, information needs, candidate tools).
4. **`enforce_scope_and_policies`** — apply policies P1–P4 to the plan (chat scope, session history, summary-fetch-all, numeric-facts-first).
5. **`execute_retrieval_tools`** — fan out the planned tool calls; record fingerprints to prevent dedup (policy P10).
6. **`merge_evidence_workspace`** — dedupe by deterministic `evidence_id`, normalise scores, attach origin tool.
7. **`check_context_budget`** — conditional: if `detect_overflow → True`, route to aggregate; else continue.
8. **`aggregate_sources_node`** — summarise large evidence groups so the answer prompt fits.
9. **`check_coverage`** — conditional router: if coverage incomplete and `iteration_count < 2` → `plan_gap_retrieval`; else if any requirement is unsatisfied and `replan_rounds < MAX_REPLAN_ROUNDS` (=3) → `llm_replan`; else → `verify_critical_claims`.
10. **`plan_gap_retrieval`** — deterministic targeted second pass; loops back to `execute_retrieval_tools` once.
11. **`llm_replan`** — bounded LLM-driven planner. Returns a JSON `ReplanDecision` (`extra="forbid"`) listing 0–4 whitelisted retrieval tool calls; validated by policy P15 before dispatch; cannot author `chat_id`, raw SQL, or non-whitelisted tools.
12. **`verify_critical_claims`** — re-check numeric / definitional claims against `structured_facts` (policy P8).
13. **`generate_answer`** — final LLM call; produces answer text + `CitationDraft`s. In Deep QA mode, prepends the last 8 same-session turns to the prompt.
14. **`validate_citations`** — policies P12 / P13: drop any citation whose `chat_id ≠ state.chat_id` or whose `document_id` isn't in the chat's `chat_documents`.
15. **`validate_scope_isolation`** — last belt-and-braces check: no evidence, citation, or tool call leaked across `chat_id`.
16. **`persist_messages`** — write the user + assistant messages (plus `tool_trace`) into Postgres.

---

## 2. The 8 tools

Defined under `src/app/agent/tools/`. Every input schema uses Pydantic v2
with `extra="forbid"` and **does not contain `chat_id`** — the LLM cannot
inject it; the service layer always supplies it from `AgentState.chat_id`.

| Tool | Data layer | Purpose |
|---|---|---|
| `inspect_chat` | Postgres | manifest snapshot of the chat (doc count, titles, topics, source-type histogram) |
| `inspect_document` | Postgres | per-document manifest: sections, page count, has-table / has-figure flags |
| `fetch_structural_nodes` | Postgres (`document_nodes`) | deterministic fetch — *do not* replace with top-k for whole-doc / whole-section queries (policy P3) |
| `search_hybrid` | Vespa (`RetrievalService.search`) | BM25 + ANN + RRF + rerank, scoped to chat |
| `grep_document_chunks` | Postgres (`document_nodes`) | deterministic lexical scan with neighbour-context expansion and formula-aware scoring; preferred for exact `Figure N` / `Table N` labels, HTML tables, and literal formulas where hybrid search returns a nearby-but-wrong modality |
| `query_structured_facts` | Postgres (`structured_facts`) | restricted filter schema (`kinds`, `keys`, `page_range`, `numeric_min/max`, `unit_in`) — never raw SQL |
| `aggregate_sources` | LLM only | groups by `(document_id, section_title)` and summarises; used when budget overflows (skipped in Deep QA mode) |
| `expand_evidence` | Postgres | fetch sibling nodes around an existing evidence item (`neighborhood ∈ {section, page, paragraphs}`) |

Every tool call returns a `ToolCallRecord{status, token_estimate, sources, error?}`
attached to `AgentState.tool_calls`. `status` includes the special value
`overflow` — when it fires, the next node MUST be `aggregate_sources_node`
or policy P6 will refuse to continue.

---

## 3. AgentState — the single source of truth

`src/app/agent/state.py` defines a Pydantic model that is the shared
working memory for every node. Notable fields:

| Field | Type | Notes |
|---|---|---|
| `chat_id` | `UUID` | service-injected; LLM-unreachable |
| `session_id` | `UUID` | service-injected |
| `question` | `str` | the user's latest message |
| `conversation_history` | `list[ConversationTurn]` | scoped to `(session_id, chat_id)` |
| `chat_manifest`, `document_manifests` | snapshots | feed the planner |
| `plan` | `AgentPlan` | goal, information_needs, chosen_tools, rationale |
| `tool_calls` | `list[ToolCallRecord]` | full audit trail |
| `evidence_items` | `list[EvidenceItem]` | deterministic `evidence_id` (uuid5 of source_node_id + content hash) |
| `context_token_estimate`, `remaining_budget` | `int` | budget gauges |
| `iteration_count` | `int` | tool-call rounds executed |
| `replan_rounds` | `int` | bounded `llm_replan` rounds executed (hard cap `MAX_REPLAN_ROUNDS = 3`) |
| `replan_tool_calls` | `list[ReplanToolRequest]` | one-shot queue dispatched by `execute_retrieval_tools` on the next pass |
| `scoped_document_ids` | `list[UUID]` | first-message session document scope, locked by `session_service.lock_document_scope_for_qa` |
| `tool_invocations_fingerprints` | `set[str]` | SHA-1(tool_name + params) for dedup |
| `answer`, `citations` | str / list | final outputs |
| `errors` | `list[AgentError]` | structured failures |
| `debug_trace` | `AgentDebugTrace` | append-only event log |
| `generation_config` | `GenerationConfig` | per-request overrides (`max_answer_tokens`, `temperature`, `context_window`, `max_replan_rounds`, `deep_qa_mode`) |

---

## 4. ContextBudgetManager

`src/app/agent/budget.py`. Defaults match CLAUDE.md §8:

| Bucket | Tokens |
|---|---|
| `system_and_tools` | 1,200 |
| `conversation` | 1,000 |
| `question_and_plan` | 500 |
| `document_evidence` | 5,000 |
| `answer_reserve` | 2,000 |
| `miscellaneous` | 300 |
| **total** | **10,000** |

If the chat's `default_chat_profile.context_window` is set, the manager
rescales every bucket proportionally; per-request `generation_config.context_window`
overrides again. Counting uses `tiktoken cl100k_base` when present and falls
back to `len(text) // 4`; the heuristic path sets `last_was_estimate=True`
in the debug trace so the UI can flag it.

When `generation_config.deep_qa_mode` is `True`, the manager is
constructed with `ignore_budget=True` — `detect_overflow()` then always
returns `False`, so policy P6 never routes to `aggregate_sources_node`
and raw evidence reaches `generate_answer` unsummarised. See
[`08-deep-qa.md`](./08-deep-qa.md).

Key operations:

- `count_tokens(text) → int`
- `calculate_available_evidence_budget(state) → int`
- `detect_overflow(state) → bool`
- `select_compact_sources(state, target_budget) → list[EvidenceItem]` — greedy by score desc / token_count asc; guarantees ≥ 1 item per document.
- `build_aggregation_groups(state)` — `(document_id, section_title)` partitions for `aggregate_sources_node`.

---

## 5. The 15 code-enforced policies

`src/app/agent/policies.py`. **None of these are prompt-only.** They run as
code; violation either raises `PolicyViolation` or rewrites state (drop a
citation, append `[unverified]`).

| # | Name | Effect |
|---|---|---|
| P1 | `chat_id_present` | reject nil/empty chat_id |
| P2 | `session_history_isolation` | history loaded only for `(session_id, chat_id)` |
| P3 | `no_topk_for_summary` | summary-style questions must include `fetch_structural_nodes` / `inspect_chat` |
| P4 | `numeric_facts_first` | numeric questions must include `query_structured_facts` |
| P5 | `rerank_required` | `search_hybrid` must use `rerank_mode ≠ "none"` (non-fatal flag) |
| P6 | `overflow_must_aggregate` | overflow detected → `aggregate_sources` must run (skipped when `deep_qa_mode=True`) |
| P7 | `no_answer_on_incomplete_coverage` | bail out with the "not enough information" fallback |
| P8 | `numeric_cross_check` | numbers in the answer must appear in evidence or facts; otherwise tagged `[unverified: N]` |
| P9 | `iteration_cap` | `iteration_count ≤ max_iterations` (default 2) |
| P10 | `dedup_check` | identical tool-call fingerprints are silently skipped |
| P11 | `no_knowledge_fallback` | empty evidence + facts → exact answer "*there is not enough information in the current chat's documents*" |
| P12 | `citation_chat_isolation` | drop citations whose `chat_id ≠ state.chat_id` |
| P13 | `citation_doc_association` | drop citations whose `document_id ∉ chat_documents` |
| P14 | `provider_failure` | record `AgentError` + raise; **never** silently switch models |
| P15 | `replan_decision_schema` | `llm_replan` may only nominate whitelisted retrieval tools (`search_hybrid`, `grep_document_chunks`, `fetch_structural_nodes`, `query_structured_facts`, `inspect_document`); LLM-supplied `chat_id`, raw SQL, or unknown tool names raise `PolicyViolation` |

---

## 6. Entering and leaving the agent

### 6.1 HTTP entry

```
POST /chats/{chat_id}/sessions/{session_id}/messages
```

Router: `src/app/api/messages.py:239-298`. The router:

1. Resolves the session via `session_service.get_session_by_id(chat_id, session_id)`
   — raises 404 on cross-chat lookups.
2. Loads the chat provider profile (or env-level fallback).
3. Calls `QAService.stream(chat_id, session_id, question, chat_provider, stop_event, generation_config)`.

### 6.2 SSE event stream

`messages.py` returns a `StreamingResponse` with three event types:

| Event | Payload |
|---|---|
| `token` | `{"delta": "word "}` — answer streamed word-by-word |
| `citation` | full `Citation` JSON (citation_id, chat_id, document_id, page_start/end, excerpt, …) |
| `error` | `{"code": "NOT_FOUND" | "INTERNAL" | "AGENT_ERROR" | "STOPPED", "detail": …}` |

Disconnect from the client trips `stop_event`, which the generator checks
between tokens — the agent's persistence node still records partial state
to `messages` if it has progressed past `generate_answer`.

### 6.3 What the agent never receives

- The chat_id from a request body (path-only).
- Free-form SQL or YQL.
- Plaintext API keys — `chat_provider` is constructed in the service layer
  from a decrypted `ProviderProfile` row, never from the request body.

---

## 7. Running the agent offline (dev / debugging)

```bash
# Smoke against a real LLM (vLLM, OpenAI-compatible, …)
LLM_PROVIDER=openai_compatible \
  LLM_API_URL=http://localhost:8001/v1 \
  LLM_MODEL=gemma-3-27b-it \
  uv run python scripts/smoke_agent_e2e.py

# Golden QA — deterministic mocks; the regression in the suite
uv run pytest tests/evaluation -q

# Goal-coverage scorer (depends on parser + retrieval + QA reports)
uv run python scripts/run_goal_score.py
```

Each iteration of the self-repair loop (CLAUDE.md §10) re-runs the agent
against the golden corpus and reports per-policy violation counts, so
regressions in policy enforcement are visible immediately.
