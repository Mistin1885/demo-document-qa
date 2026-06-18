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

export interface AskOptions {
  question: string;
  stream?: false;
}

export async function askQuestion(
  chatId: string,
  sessionId: string,
  question: string
): Promise<QAResponse> {
  return apiPost<QAResponse>(
    `/chats/${chatId}/sessions/${sessionId}/messages`,
    { question, stream: false }
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
