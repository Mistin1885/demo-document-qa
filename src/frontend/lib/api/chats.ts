/**
 * Chat API helpers — thin wrappers over the core client.
 *
 * Routes:
 *   POST   /chats
 *   GET    /chats
 *   GET    /chats/{chat_id}
 *   PATCH  /chats/{chat_id}
 *   DELETE /chats/{chat_id}
 */

import { apiDelete, apiGet, apiPatch, apiPost } from "./client";
import type { ChatCreate, ChatRead, ChatUpdate } from "./types";

export async function createChat(body: ChatCreate): Promise<ChatRead> {
  return apiPost<ChatRead>("/chats", body);
}

export async function listChats(): Promise<ChatRead[]> {
  return apiGet<ChatRead[]>("/chats");
}

export async function getChat(chatId: string): Promise<ChatRead> {
  return apiGet<ChatRead>(`/chats/${chatId}`);
}

export async function updateChat(chatId: string, body: ChatUpdate): Promise<ChatRead> {
  return apiPatch<ChatRead>(`/chats/${chatId}`, body);
}

export async function deleteChat(chatId: string): Promise<void> {
  return apiDelete(`/chats/${chatId}`);
}
