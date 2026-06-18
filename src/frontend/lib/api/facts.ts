/**
 * Structured facts API helpers.
 *
 * Routes:
 *   POST /chats/{chat_id}/facts/search   (body: FactFilter)
 *   GET  /chats/{chat_id}/facts/{fact_id}
 */

import { apiGet, apiPost } from "./client";
import type { FactFilter, StructuredFactRead } from "./types";

export async function searchFacts(
  chatId: string,
  filter: FactFilter
): Promise<StructuredFactRead[]> {
  return apiPost<StructuredFactRead[]>(`/chats/${chatId}/facts/search`, filter);
}

export async function getFact(
  chatId: string,
  factId: string
): Promise<StructuredFactRead> {
  return apiGet<StructuredFactRead>(`/chats/${chatId}/facts/${factId}`);
}
