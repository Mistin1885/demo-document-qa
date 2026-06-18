"use client";

/**
 * TanStack Query v5 hooks for Chat resources.
 *
 * All hooks use the queryKeys factory from lib/queries/keys.ts.
 * Mutations invalidate the chats list on success.
 */

import {
  useMutation,
  useQuery,
  useQueryClient,
} from "@tanstack/react-query";
import {
  createChat,
  deleteChat,
  listChats,
  updateChat,
} from "@/lib/api/chats";
import { listDocuments } from "@/lib/api/documents";
import { queryKeys } from "@/lib/queries/keys";
import type { ChatCreate, ChatUpdate } from "@/lib/api/types";

// ---------------------------------------------------------------------------
// Queries
// ---------------------------------------------------------------------------

export function useChats() {
  return useQuery({
    queryKey: queryKeys.chats(),
    queryFn: listChats,
  });
}

/**
 * Returns the document count for a single chat.
 * Only fires when chatId is provided; on error returns undefined (caller shows "—").
 */
export function useChatDocumentCount(chatId: string | null) {
  return useQuery({
    queryKey: queryKeys.documents(chatId ?? ""),
    queryFn: () => listDocuments(chatId!),
    enabled: !!chatId,
    select: (docs) => docs.length,
  });
}

// ---------------------------------------------------------------------------
// Mutations
// ---------------------------------------------------------------------------

export function useCreateChat() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (body: ChatCreate) => createChat(body),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: queryKeys.chats() });
    },
  });
}

export function useUpdateChat() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ chatId, body }: { chatId: string; body: ChatUpdate }) =>
      updateChat(chatId, body),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: queryKeys.chats() });
    },
  });
}

export function useDeleteChat() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (chatId: string) => deleteChat(chatId),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: queryKeys.chats() });
    },
  });
}
