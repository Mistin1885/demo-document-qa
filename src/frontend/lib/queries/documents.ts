/**
 * TanStack Query v5 hooks for documents and chat manifest.
 *
 * Chat isolation: every hook requires a non-null chatId; TanStack Query key
 * isolation ensures old chat data is never shown when chatId changes.
 */

"use client";

import {
  useQuery,
  useMutation,
  useQueryClient,
  type UseQueryResult,
} from "@tanstack/react-query";
import {
  listDocuments,
  deleteDocument,
  uploadDocumentWithProgress,
  type UploadProgressOptions,
} from "@/lib/api/documents";
import { getChatManifest } from "@/lib/api/manifest";
import { queryKeys } from "./keys";
import type { DocumentRead, ChatManifest } from "@/lib/api/types";

// ---------------------------------------------------------------------------
// useDocuments — list all documents for a chat
// ---------------------------------------------------------------------------

export function useDocuments(
  chatId: string | null
): UseQueryResult<DocumentRead[], Error> {
  return useQuery({
    queryKey: chatId ? queryKeys.documents(chatId) : ["__no_chat__", "documents"],
    queryFn: () => listDocuments(chatId!),
    enabled: !!chatId,
    // Uploaded PDFs are parsed by a backend background task. Poll so the badge
    // moves from Uploaded/Parsing to Parsed/Failed without a manual refresh.
    refetchInterval: 3000,
  });
}

// ---------------------------------------------------------------------------
// useChatManifest — fetch ChatManifest (section_count, token_estimate, etc.)
// 404 is treated as "not yet generated" and resolves to null instead of error.
// ---------------------------------------------------------------------------

export function useChatManifest(
  chatId: string | null
): UseQueryResult<ChatManifest | null, Error> {
  return useQuery({
    queryKey: chatId ? queryKeys.manifest(chatId) : ["__no_chat__", "manifest"],
    queryFn: async () => {
      try {
        return await getChatManifest(chatId!);
      } catch (err: unknown) {
        // 404 = manifest not generated yet → treat as null, don't error
        if (
          err != null &&
          typeof err === "object" &&
          "status" in err &&
          (err as { status: number }).status === 404
        ) {
          return null;
        }
        throw err;
      }
    },
    enabled: !!chatId,
    retry: false,
  });
}

// ---------------------------------------------------------------------------
// useUploadDocument — mutation that supports progress tracking via XHR
// ---------------------------------------------------------------------------

export interface UploadDocumentVars {
  file: File;
  opts?: UploadProgressOptions;
}

export function useUploadDocument(chatId: string | null) {
  const queryClient = useQueryClient();

  return useMutation<DocumentRead, Error, UploadDocumentVars>({
    mutationFn: ({ file, opts }) => {
      if (!chatId) throw new Error("No chat selected");
      return uploadDocumentWithProgress(chatId, file, opts);
    },
    onSuccess: () => {
      if (!chatId) return;
      queryClient.invalidateQueries({ queryKey: queryKeys.documents(chatId) });
      queryClient.invalidateQueries({ queryKey: queryKeys.manifest(chatId) });
      queryClient.invalidateQueries({ queryKey: queryKeys.chats() });
    },
  });
}

// ---------------------------------------------------------------------------
// useDeleteDocument — mutation to remove a document from the chat
// ---------------------------------------------------------------------------

export function useDeleteDocument(chatId: string | null) {
  const queryClient = useQueryClient();

  return useMutation<void, Error, string>({
    mutationFn: (docId: string) => {
      if (!chatId) throw new Error("No chat selected");
      return deleteDocument(chatId, docId);
    },
    onSuccess: () => {
      if (!chatId) return;
      queryClient.invalidateQueries({ queryKey: queryKeys.documents(chatId) });
      queryClient.invalidateQueries({ queryKey: queryKeys.manifest(chatId) });
      queryClient.invalidateQueries({ queryKey: queryKeys.chats() });
    },
  });
}
