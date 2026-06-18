"use client";

/**
 * ChatSidebar — left panel listing all chats with CRUD actions.
 *
 * Features:
 *  - List all chats (newest first from API)
 *  - Active chat highlighted (driven by URL ?chatId=)
 *  - New chat button → NewChatDialog
 *  - Per-row rename → RenameChatDialog
 *  - Per-row delete → window.confirm guard
 */

import { useState } from "react";
import { useCurrentChatId } from "@/lib/hooks/useCurrentChatId";
import {
  useChats,
  useCreateChat,
  useDeleteChat,
  useUpdateChat,
} from "@/lib/queries/chats";
import { ChatListItem } from "./ChatListItem";
import { NewChatDialog } from "./NewChatDialog";
import { RenameChatDialog } from "./RenameChatDialog";
import type { ChatRead } from "@/lib/api/types";

export function ChatSidebar() {
  const { chatId: currentChatId, setChatId } = useCurrentChatId();

  const { data: chats, isLoading, isError } = useChats();
  const createChatMutation = useCreateChat();
  const updateChatMutation = useUpdateChat();
  const deleteChatMutation = useDeleteChat();

  // New-chat dialog
  const [newChatOpen, setNewChatOpen] = useState(false);

  // Rename dialog
  const [renameTarget, setRenameTarget] = useState<ChatRead | null>(null);

  function handleCreateChat(name: string, description: string) {
    createChatMutation.mutate(
      { name, description: description || null },
      {
        onSuccess: (created) => {
          setNewChatOpen(false);
          setChatId(created.id);
        },
      }
    );
  }

  function handleRenameChat(name: string) {
    if (!renameTarget) return;
    updateChatMutation.mutate(
      { chatId: renameTarget.id, body: { name } },
      { onSuccess: () => setRenameTarget(null) }
    );
  }

  function handleDeleteChat(chat: ChatRead) {
    deleteChatMutation.mutate(chat.id, {
      onSuccess: () => {
        // If the deleted chat was active, clear selection
        if (currentChatId === chat.id) {
          setChatId(null);
        }
      },
    });
  }

  return (
    <>
      {/* Sidebar header */}
      <div className="flex items-center justify-between px-3 py-2.5 border-b border-[var(--border)] shrink-0">
        <span className="text-xs font-semibold uppercase tracking-wider text-[var(--muted)]">
          Chats
        </span>
        <button
          type="button"
          onClick={() => setNewChatOpen(true)}
          className="flex items-center justify-center w-6 h-6 rounded-md text-[var(--muted)] hover:text-[var(--foreground)] hover:bg-[var(--surface-raised)] transition-colors"
          aria-label="New chat"
          title="New chat"
        >
          <svg
            className="w-4 h-4"
            viewBox="0 0 16 16"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.5"
            strokeLinecap="round"
            aria-hidden="true"
          >
            <line x1="8" y1="3" x2="8" y2="13" />
            <line x1="3" y1="8" x2="13" y2="8" />
          </svg>
        </button>
      </div>

      {/* Chat list */}
      <div className="flex-1 overflow-y-auto">
        {isLoading && (
          <div className="flex items-center justify-center py-8">
            <span className="text-xs text-[var(--muted)]">Loading…</span>
          </div>
        )}

        {isError && (
          <div className="px-3 py-4">
            <p className="text-xs text-red-400">Failed to load chats.</p>
          </div>
        )}

        {!isLoading && !isError && chats && chats.length === 0 && (
          <div className="flex flex-col items-center justify-center py-8 px-4 text-center">
            <p className="text-xs text-[var(--muted)]">No chats yet.</p>
            <button
              type="button"
              onClick={() => setNewChatOpen(true)}
              className="mt-2 text-xs text-[var(--accent)] hover:underline"
            >
              Create your first chat
            </button>
          </div>
        )}

        {!isLoading &&
          !isError &&
          chats?.map((chat) => (
            <ChatListItem
              key={chat.id}
              chat={chat}
              isActive={chat.id === currentChatId}
              onClick={() => setChatId(chat.id)}
              onRename={() => setRenameTarget(chat)}
              onDelete={() => handleDeleteChat(chat)}
            />
          ))}
      </div>

      {/* Dialogs */}
      <NewChatDialog
        open={newChatOpen}
        onClose={() => setNewChatOpen(false)}
        onSubmit={handleCreateChat}
        isPending={createChatMutation.isPending}
      />

      {renameTarget && (
        <RenameChatDialog
          open={!!renameTarget}
          currentName={renameTarget.name}
          onClose={() => setRenameTarget(null)}
          onSubmit={handleRenameChat}
          isPending={updateChatMutation.isPending}
        />
      )}
    </>
  );
}
