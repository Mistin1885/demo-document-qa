"use client";

/**
 * ChatPanel — center column: session header + message thread + input.
 *
 * Structure:
 *   <SessionList />          ← session tabs row
 *   <MessageList />          ← scrollable message history
 *   <QuestionInput />        ← textarea + send/stop
 *
 * Isolation: all data is scoped to (chatId, sessionId) from URL params.
 * Switching either param causes useChatStream to reset (isolation guard).
 */

import { Suspense, useMemo } from "react";

import { SessionList } from "@/components/session/SessionList";
import { useChatStream } from "@/lib/chat/useChatStream";
import { useGenerationPrefs } from "@/lib/chat/useGenerationPrefs";
import { useCurrentChatId } from "@/lib/hooks/useCurrentChatId";
import { useCurrentSessionId } from "@/lib/hooks/useCurrentSessionId";
import { useDocuments } from "@/lib/queries/documents";
import { useSession } from "@/lib/queries/sessions";
import { useDocumentScopeSelection } from "@/lib/session/useDocumentScopeSelection";

import { GenerationPrefsPanel } from "./GenerationPrefsPanel";
import { MessageList } from "./MessageList";
import { QuestionInput } from "./QuestionInput";

// ---------------------------------------------------------------------------
// Inner panel (requires hooks that need Suspense boundary above)
// ---------------------------------------------------------------------------

function ChatPanelInner() {
  const { chatId } = useCurrentChatId();
  const { sessionId } = useCurrentSessionId();

  const { prefs, setPrefs, reset: resetPrefs } = useGenerationPrefs(chatId);
  const { data: documents } = useDocuments(chatId);
  const { data: session, isLoading: sessionLoading, isError: sessionError } = useSession(
    chatId,
    sessionId
  );
  const documentIds = useMemo(
    () => (documents ?? []).map((doc) => doc.id),
    [documents]
  );
  const { selectedDocumentIds } = useDocumentScopeSelection(
    chatId,
    sessionId,
    documentIds,
    session?.selected_document_ids,
    session?.document_scope_locked ?? false
  );
  const ready = !!chatId && !!sessionId && session?.chat_id === chatId;
  const statusHint = !chatId
    ? "Select a chat from the sidebar to get started."
    : !sessionId
    ? "Select or create a session above to start a conversation."
    : sessionLoading
    ? "Loading session…"
    : sessionError
    ? "This session does not belong to the selected chat. Please choose another session."
    : "Select or create a session above to start a conversation.";
  const { messages, sendMessage, stop, isStreaming, error } = useChatStream(
    chatId,
    ready ? sessionId : null,
    prefs,
    selectedDocumentIds
  );

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* Session list row */}
      <Suspense
        fallback={
          <div className="px-4 py-2 border-b border-[var(--border)]">
            <span className="text-xs text-[var(--muted)]">Loading sessions…</span>
          </div>
        }
      >
        <SessionList />
      </Suspense>

      {/* Hint when no chat/session selected */}
      {!ready && (
        <div className="flex-1 flex items-center justify-center p-8">
          <p className="text-sm text-[var(--muted)] text-center">
            {statusHint}
          </p>
        </div>
      )}

      {/* Active conversation */}
      {ready && (
        <>
          {/* Error banner */}
          {error && (
            <div className="px-4 py-2 bg-rose-900/40 border-b border-rose-700/40 text-xs text-rose-300 shrink-0">
              Error: {error}
            </div>
          )}

          {/* Message thread */}
          <MessageList messages={messages} />

          {/* Advanced generation overrides */}
          <GenerationPrefsPanel
            prefs={prefs}
            onChange={setPrefs}
            onReset={resetPrefs}
            disabled={isStreaming}
          />

          {/* Input */}
          <QuestionInput
            onSend={sendMessage}
            onStop={stop}
            isStreaming={isStreaming}
          />
        </>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Exported component with Suspense wrapper for useSearchParams
// ---------------------------------------------------------------------------

export function ChatPanel() {
  return (
    <Suspense
      fallback={
        <div className="flex-1 flex items-center justify-center">
          <span className="text-xs text-[var(--muted)]">Loading…</span>
        </div>
      }
    >
      <ChatPanelInner />
    </Suspense>
  );
}
