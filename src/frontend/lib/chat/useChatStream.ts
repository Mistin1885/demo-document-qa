"use client";

/**
 * useChatStream — core streaming hook for the chat panel.
 *
 * Responsibilities:
 *  - Load message history from useSessionMessages (server source of truth).
 *  - Manage optimistic user + in-flight assistant messages during streaming.
 *  - Drive SSE via streamMessage(), accumulate token deltas and citations.
 *  - Support stop() via AbortController.
 *  - On done/error: invalidate the messages query so server state takes over.
 *
 * Isolation contract (CLAUDE.md §2, §13):
 *  - chatId and sessionId always come from the caller (URL state).
 *  - No cross-session message leakage: each hook instance is scoped to one
 *    (chatId, sessionId) pair; switching either triggers a fresh query load.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { streamMessage } from "@/lib/api/sse";
import { useSessionMessages } from "@/lib/queries/sessions";
import { queryKeys } from "@/lib/queries/keys";
import type { Citation, MessageRead, QAResponse } from "@/lib/api/types";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/** An in-flight assistant message produced during SSE streaming. */
export interface StreamingMessage {
  /** Temporary client-side id (not a server UUID). */
  id: string;
  role: "assistant";
  /** Accumulated text so far from token events. */
  partial: string;
  citations: Citation[];
  status: "streaming" | "done" | "error" | "stopped";
  /** Final QAResponse — set on done event. */
  finalResponse: QAResponse | null;
  errorDetail: string | null;
}

/** Optimistic user message added immediately on send. */
export interface OptimisticUserMessage {
  id: string;
  role: "user";
  content: string;
  /** ISO timestamp generated client-side. */
  created_at: string;
}

export type ChatMessage =
  | MessageRead
  | OptimisticUserMessage
  | StreamingMessage;

export interface UseChatStreamReturn {
  /** All messages: server history + optimistic user msg + in-flight assistant. */
  messages: ChatMessage[];
  sendMessage: (question: string) => void;
  stop: () => void;
  isStreaming: boolean;
  error: string | null;
}

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------

function tempId(): string {
  return `tmp-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;
}

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------

export function useChatStream(
  chatId: string | null,
  sessionId: string | null
): UseChatStreamReturn {
  const queryClient = useQueryClient();

  // Server-loaded history
  const { data: serverMessages } = useSessionMessages(chatId, sessionId);

  // In-flight optimistic messages (cleared after server invalidation)
  const [optimistic, setOptimistic] = useState<
    (OptimisticUserMessage | StreamingMessage)[]
  >([]);

  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // AbortController for stop()
  const abortRef = useRef<AbortController | null>(null);
  // Streaming message id being updated
  const streamingIdRef = useRef<string | null>(null);

  // Reset optimistic state when chat/session changes (isolation guard)
  useEffect(() => {
    setOptimistic([]);
    setIsStreaming(false);
    setError(null);
    // Abort any in-flight request from previous session
    abortRef.current?.abort();
    abortRef.current = null;
    streamingIdRef.current = null;
  }, [chatId, sessionId]);

  const stop = useCallback(() => {
    abortRef.current?.abort();
    // Mark the streaming message as stopped
    if (streamingIdRef.current) {
      const sid = streamingIdRef.current;
      setOptimistic((prev) =>
        prev.map((m) =>
          m.id === sid && m.role === "assistant"
            ? { ...(m as StreamingMessage), status: "stopped" }
            : m
        )
      );
    }
    setIsStreaming(false);
  }, []);

  const sendMessage = useCallback(
    (question: string) => {
      if (!chatId || !sessionId) return;
      if (isStreaming) return; // prevent double-send

      setError(null);

      // 1. Optimistic user message
      const userMsgId = tempId();
      const userMsg: OptimisticUserMessage = {
        id: userMsgId,
        role: "user",
        content: question,
        created_at: new Date().toISOString(),
      };

      // 2. In-flight assistant placeholder
      const assistantMsgId = tempId();
      streamingIdRef.current = assistantMsgId;
      const assistantMsg: StreamingMessage = {
        id: assistantMsgId,
        role: "assistant",
        partial: "",
        citations: [],
        status: "streaming",
        finalResponse: null,
        errorDetail: null,
      };

      setOptimistic((prev) => [...prev, userMsg, assistantMsg]);
      setIsStreaming(true);

      // 3. Start SSE
      const abort = new AbortController();
      abortRef.current = abort;

      void (async () => {
        try {
          for await (const evt of streamMessage(
            chatId,
            sessionId,
            question,
            abort.signal
          )) {
            if (abort.signal.aborted) break;

            if (evt.kind === "token") {
              const delta = evt.data.delta;
              setOptimistic((prev) =>
                prev.map((m) =>
                  m.id === assistantMsgId && m.role === "assistant"
                    ? {
                        ...(m as StreamingMessage),
                        partial: (m as StreamingMessage).partial + delta,
                      }
                    : m
                )
              );
            } else if (evt.kind === "citation") {
              const citation = evt.data;
              setOptimistic((prev) =>
                prev.map((m) =>
                  m.id === assistantMsgId && m.role === "assistant"
                    ? {
                        ...(m as StreamingMessage),
                        citations: [
                          ...(m as StreamingMessage).citations,
                          citation,
                        ],
                      }
                    : m
                )
              );
            } else if (evt.kind === "done") {
              const response = evt.data as QAResponse;
              setOptimistic((prev) =>
                prev.map((m) =>
                  m.id === assistantMsgId && m.role === "assistant"
                    ? {
                        ...(m as StreamingMessage),
                        status: "done",
                        finalResponse: response,
                      }
                    : m
                )
              );
              setIsStreaming(false);
              // Invalidate server messages — backend has persisted the exchange
              void queryClient.invalidateQueries({
                queryKey: queryKeys.messages(chatId, sessionId),
              });
              // Clear optimistic messages so server history takes over
              setOptimistic([]);
            } else if (evt.kind === "error") {
              const detail = evt.data.detail;
              setOptimistic((prev) =>
                prev.map((m) =>
                  m.id === assistantMsgId && m.role === "assistant"
                    ? {
                        ...(m as StreamingMessage),
                        status: "error",
                        errorDetail: detail,
                      }
                    : m
                )
              );
              setError(detail);
              setIsStreaming(false);
            }
          }
        } catch (err: unknown) {
          if (abort.signal.aborted) {
            // User-triggered stop — already handled above
            return;
          }
          const msg =
            err instanceof Error ? err.message : "Streaming failed";
          setOptimistic((prev) =>
            prev.map((m) =>
              m.id === assistantMsgId && m.role === "assistant"
                ? { ...(m as StreamingMessage), status: "error", errorDetail: msg }
                : m
            )
          );
          setError(msg);
          setIsStreaming(false);
        } finally {
          streamingIdRef.current = null;
        }
      })();
    },
    [chatId, sessionId, isStreaming, queryClient]
  );

  // Merge server history + optimistic messages
  const messages: ChatMessage[] = [
    ...(serverMessages ?? []),
    ...optimistic,
  ];

  return { messages, sendMessage, stop, isStreaming, error };
}
