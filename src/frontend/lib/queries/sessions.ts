"use client";

/**
 * TanStack Query v5 hooks for Session resources.
 *
 * All hooks scope queries to a chatId to enforce chat isolation (CLAUDE.md §2).
 * Mutations invalidate the session list for the current chat.
 */

import {
  useMutation,
  useQuery,
  useQueryClient,
} from "@tanstack/react-query";
import {
  createSession,
  deleteSession,
  listSessions,
  updateSession,
} from "@/lib/api/sessions";
import { listMessages } from "@/lib/api/messages";
import { queryKeys } from "@/lib/queries/keys";
import type { SessionCreate, SessionUpdate } from "@/lib/api/types";

// ---------------------------------------------------------------------------
// Queries
// ---------------------------------------------------------------------------

/** List all sessions for a chat (sorted by updated_at descending in display). */
export function useSessions(chatId: string | null) {
  return useQuery({
    queryKey: queryKeys.sessions(chatId ?? ""),
    queryFn: () => listSessions(chatId!),
    enabled: !!chatId,
  });
}

/** List messages for a specific session — initial history load. */
export function useSessionMessages(
  chatId: string | null,
  sessionId: string | null
) {
  return useQuery({
    queryKey: queryKeys.messages(chatId ?? "", sessionId ?? ""),
    queryFn: () => listMessages(chatId!, sessionId!),
    enabled: !!chatId && !!sessionId,
  });
}

// ---------------------------------------------------------------------------
// Mutations
// ---------------------------------------------------------------------------

export function useCreateSession() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({
      chatId,
      body,
    }: {
      chatId: string;
      body: Omit<SessionCreate, "chat_id">;
    }) => createSession(chatId, body),
    onSuccess: (_data, { chatId }) => {
      void queryClient.invalidateQueries({
        queryKey: queryKeys.sessions(chatId),
      });
    },
  });
}

export function useUpdateSession() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({
      chatId,
      sessionId,
      body,
    }: {
      chatId: string;
      sessionId: string;
      body: SessionUpdate;
    }) => updateSession(chatId, sessionId, body),
    onSuccess: (_data, { chatId }) => {
      void queryClient.invalidateQueries({
        queryKey: queryKeys.sessions(chatId),
      });
    },
  });
}

export function useDeleteSession() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({
      chatId,
      sessionId,
    }: {
      chatId: string;
      sessionId: string;
    }) => deleteSession(chatId, sessionId),
    onSuccess: (_data, { chatId }) => {
      void queryClient.invalidateQueries({
        queryKey: queryKeys.sessions(chatId),
      });
    },
  });
}
