/**
 * Chat manifest API helper.
 *
 * Routes:
 *   GET /chats/{chat_id}/manifest
 */

import { apiGet } from "./client";
import type { ChatManifest } from "./types";

export async function getChatManifest(chatId: string): Promise<ChatManifest> {
  return apiGet<ChatManifest>(`/chats/${chatId}/manifest`);
}
