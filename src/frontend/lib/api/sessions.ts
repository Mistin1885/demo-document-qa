/**
 * Session API helpers.
 *
 * Routes:
 *   POST   /chats/{chat_id}/sessions
 *   GET    /chats/{chat_id}/sessions
 *   GET    /chats/{chat_id}/sessions/{session_id}
 *   PATCH  /chats/{chat_id}/sessions/{session_id}
 *   DELETE /chats/{chat_id}/sessions/{session_id}
 */

import { apiDelete, apiGet, apiPatch, apiPost } from "./client";
import type { SessionCreate, SessionRead, SessionUpdate } from "./types";

export async function createSession(
  chatId: string,
  body: Omit<SessionCreate, "chat_id">
): Promise<SessionRead> {
  return apiPost<SessionRead>(`/chats/${chatId}/sessions`, {
    ...body,
    chat_id: chatId,
  });
}

export async function listSessions(chatId: string): Promise<SessionRead[]> {
  return apiGet<SessionRead[]>(`/chats/${chatId}/sessions`);
}

export async function getSession(
  chatId: string,
  sessionId: string
): Promise<SessionRead> {
  return apiGet<SessionRead>(`/chats/${chatId}/sessions/${sessionId}`);
}

export async function updateSession(
  chatId: string,
  sessionId: string,
  body: SessionUpdate
): Promise<SessionRead> {
  return apiPatch<SessionRead>(`/chats/${chatId}/sessions/${sessionId}`, body);
}

export async function deleteSession(chatId: string, sessionId: string): Promise<void> {
  return apiDelete(`/chats/${chatId}/sessions/${sessionId}`);
}
