"use client";

/**
 * SessionList — lists sessions for the current chat.
 *
 * Displayed inside ChatPanel's header area.
 * Supports: select (switch), create, rename (inline), delete.
 * Isolation: scoped strictly to the active chatId.
 */

import { useState, useRef, useEffect } from "react";
import { useCurrentChatId } from "@/lib/hooks/useCurrentChatId";
import { useCurrentSessionId } from "@/lib/hooks/useCurrentSessionId";
import {
  useCreateSession,
  useDeleteSession,
  useSessions,
  useUpdateSession,
} from "@/lib/queries/sessions";
import type { SessionRead } from "@/lib/api/types";

// ---------------------------------------------------------------------------
// Inline rename sub-component
// ---------------------------------------------------------------------------

interface RenameInputProps {
  initialName: string;
  onCommit: (name: string) => void;
  onCancel: () => void;
}

function RenameInput({ initialName, onCommit, onCancel }: RenameInputProps) {
  const [value, setValue] = useState(initialName);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    inputRef.current?.focus();
    inputRef.current?.select();
  }, []);

  function handleKeyDown(e: React.KeyboardEvent<HTMLInputElement>) {
    if (e.key === "Enter" && !e.nativeEvent.isComposing) {
      if (value.trim()) onCommit(value.trim());
    } else if (e.key === "Escape") {
      onCancel();
    }
  }

  return (
    <input
      ref={inputRef}
      value={value}
      onChange={(e) => setValue(e.target.value)}
      onKeyDown={handleKeyDown}
      onBlur={() => {
        if (value.trim() && value.trim() !== initialName) {
          onCommit(value.trim());
        } else {
          onCancel();
        }
      }}
      className="flex-1 min-w-0 bg-[var(--surface-raised)] text-[var(--foreground)] text-xs px-1.5 py-0.5 rounded border border-[var(--accent)] outline-none"
    />
  );
}

// ---------------------------------------------------------------------------
// SessionItem
// ---------------------------------------------------------------------------

interface SessionItemProps {
  session: SessionRead;
  isActive: boolean;
  chatId: string;
  onSelect: () => void;
}

function SessionItem({ session, isActive, chatId, onSelect }: SessionItemProps) {
  const [renaming, setRenaming] = useState(false);
  const updateMutation = useUpdateSession();
  const deleteMutation = useDeleteSession();

  const displayName = session.name ?? `Session ${session.id.slice(0, 6)}`;

  function handleRename(name: string) {
    updateMutation.mutate(
      { chatId, sessionId: session.id, body: { name } },
      { onSuccess: () => setRenaming(false) }
    );
  }

  function handleDelete(e: React.MouseEvent) {
    e.stopPropagation();
    if (!window.confirm(`Delete session "${displayName}"?`)) return;
    deleteMutation.mutate({ chatId, sessionId: session.id });
  }

  return (
    <div
      className={`group flex items-center gap-1.5 px-2 py-1.5 rounded-md cursor-pointer transition-colors ${
        isActive
          ? "bg-[var(--accent)] text-white"
          : "text-[var(--foreground)] hover:bg-[var(--surface-raised)]"
      }`}
      onClick={onSelect}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") onSelect();
      }}
    >
      {renaming ? (
        <RenameInput
          initialName={session.name ?? ""}
          onCommit={handleRename}
          onCancel={() => setRenaming(false)}
        />
      ) : (
        <span className="flex-1 min-w-0 truncate text-xs">{displayName}</span>
      )}

      {!renaming && (
        <div className="flex items-center gap-0.5 opacity-0 group-hover:opacity-100 transition-opacity">
          {/* Rename button */}
          <button
            type="button"
            onClick={(e) => {
              e.stopPropagation();
              setRenaming(true);
            }}
            className={`w-5 h-5 flex items-center justify-center rounded hover:bg-black/20 ${
              isActive ? "text-white/80" : "text-[var(--muted)]"
            }`}
            title="Rename session"
            aria-label="Rename session"
          >
            <svg className="w-3 h-3" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" aria-hidden="true">
              <path d="M11 2l3 3L5 14H2v-3L11 2z" />
            </svg>
          </button>
          {/* Delete button */}
          <button
            type="button"
            onClick={handleDelete}
            disabled={deleteMutation.isPending}
            className={`w-5 h-5 flex items-center justify-center rounded hover:bg-black/20 ${
              isActive ? "text-white/80" : "text-[var(--muted)]"
            }`}
            title="Delete session"
            aria-label="Delete session"
          >
            <svg className="w-3 h-3" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" aria-hidden="true">
              <line x1="3" y1="3" x2="13" y2="13" />
              <line x1="13" y1="3" x2="3" y2="13" />
            </svg>
          </button>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// SessionList
// ---------------------------------------------------------------------------

export function SessionList() {
  const { chatId } = useCurrentChatId();
  const { sessionId, setSessionId } = useCurrentSessionId();
  const { data: sessions, isLoading, isError } = useSessions(chatId);
  const createMutation = useCreateSession();

  function handleCreate() {
    if (!chatId) return;
    createMutation.mutate(
      { chatId, body: {} },
      {
        onSuccess: (created) => {
          setSessionId(created.id, chatId);
        },
      }
    );
  }

  // Sort by updated_at descending
  const sorted = [...(sessions ?? [])].sort(
    (a, b) =>
      new Date(b.updated_at).getTime() - new Date(a.updated_at).getTime()
  );

  if (!chatId) {
    return null;
  }

  return (
    <div className="flex items-center gap-1.5 px-3 py-2 border-b border-[var(--border)] overflow-x-auto shrink-0">
      {/* Label */}
      <span className="text-xs text-[var(--muted)] shrink-0 mr-1">Sessions:</span>

      {isLoading && (
        <span className="text-xs text-[var(--muted)]">Loading…</span>
      )}
      {isError && (
        <span className="text-xs text-red-400">Failed to load sessions</span>
      )}

      {!isLoading && !isError && sorted.length === 0 && (
        <span className="text-xs text-[var(--muted)]">No sessions yet</span>
      )}

      {/* Session chips */}
      <div className="flex items-center gap-1 flex-1 min-w-0 overflow-x-auto">
        {sorted.map((session) => (
          <div key={session.id} className="shrink-0">
            <SessionItem
              session={session}
              isActive={session.id === sessionId}
              chatId={chatId}
              onSelect={() => setSessionId(session.id, chatId)}
            />
          </div>
        ))}
      </div>

      {/* New session button */}
      <button
        type="button"
        onClick={handleCreate}
        disabled={createMutation.isPending}
        className="shrink-0 flex items-center justify-center w-6 h-6 rounded-md text-[var(--muted)] hover:text-[var(--foreground)] hover:bg-[var(--surface-raised)] transition-colors disabled:opacity-40"
        aria-label="New session"
        title="New session"
      >
        <svg className="w-4 h-4" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" aria-hidden="true">
          <line x1="8" y1="3" x2="8" y2="13" />
          <line x1="3" y1="8" x2="13" y2="8" />
        </svg>
      </button>
    </div>
  );
}
