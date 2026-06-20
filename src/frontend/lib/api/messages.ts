/**
 * Messages API helpers (non-streaming path only).
 *
 * For streaming (SSE), use lib/api/sse.ts + streamMessage().
 *
 * Routes:
 *   POST /chats/{chat_id}/sessions/{session_id}/messages  (stream=false → QAResponse)
 *   GET  /chats/{chat_id}/sessions/{session_id}/messages?limit&offset
 */

import { apiGet, apiPost } from "./client";
import type { MessageRead, QAResponse } from "./types";

/**
 * Optional per-request generation overrides forwarded to the LLM.
 * All fields fall back to env-level `LLM_*` settings on the backend when
 * left undefined.
 */
export interface GenerationOverrides {
  /** Cap on output tokens (helpful for long summaries). */
  max_answer_tokens?: number;
  /** Sampling temperature in [0, 2]. */
  temperature?: number;
  /** Total input budget — resizes the evidence allocation proportionally. */
  context_window?: number;
  /** Deep QA mode: ignore soft budget and include same-session memory. */
  deep_qa_mode?: boolean;
}

export interface AskOptions extends GenerationOverrides {
  question: string;
  stream?: false;
}

export async function askQuestion(
  chatId: string,
  sessionId: string,
  question: string,
  overrides: GenerationOverrides = {}
): Promise<QAResponse> {
  return apiPost<QAResponse>(
    `/chats/${chatId}/sessions/${sessionId}/messages`,
    { question, stream: false, ...overrides }
  );
}

export async function listMessages(
  chatId: string,
  sessionId: string,
  limit = 200,
  offset = 0
): Promise<MessageRead[]> {
  return apiGet<MessageRead[]>(
    `/chats/${chatId}/sessions/${sessionId}/messages?limit=${limit}&offset=${offset}`
  );
}
