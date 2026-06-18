"use client";

/**
 * ChatListItem — a single row in the chat sidebar list.
 *
 * Shows: chat name, relative updated_at, document count.
 * Active state: highlighted when chatId matches current.
 * Context menu: Rename / Delete (shown via a "…" button).
 */

import { useRef, useState, useEffect } from "react";
import { formatRelativeTime } from "@/lib/utils/time";
import { useChatDocumentCount } from "@/lib/queries/chats";
import type { ChatRead } from "@/lib/api/types";

interface ChatListItemProps {
  chat: ChatRead;
  isActive: boolean;
  onClick: () => void;
  onRename: () => void;
  onDelete: () => void;
}

export function ChatListItem({
  chat,
  isActive,
  onClick,
  onRename,
  onDelete,
}: ChatListItemProps) {
  const [menuOpen, setMenuOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);

  const { data: docCount } = useChatDocumentCount(chat.id);

  // Close menu when clicking outside
  useEffect(() => {
    if (!menuOpen) return;
    function handleClickOutside(e: MouseEvent) {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        setMenuOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, [menuOpen]);

  function handleDelete() {
    setMenuOpen(false);
    if (window.confirm(`Delete "${chat.name}"? This cannot be undone.`)) {
      onDelete();
    }
  }

  function handleRename() {
    setMenuOpen(false);
    onRename();
  }

  return (
    <div
      className={`group relative flex items-start gap-2 px-3 py-2.5 cursor-pointer transition-colors ${
        isActive
          ? "bg-[var(--accent)]/15 border-l-2 border-[var(--accent)]"
          : "hover:bg-[var(--surface-raised)] border-l-2 border-transparent"
      }`}
      onClick={onClick}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => e.key === "Enter" && onClick()}
      aria-current={isActive ? "page" : undefined}
    >
      {/* Main content */}
      <div className="flex-1 min-w-0">
        <p
          className={`text-sm font-medium truncate leading-snug ${
            isActive ? "text-[var(--accent-hover)]" : "text-[var(--foreground)]"
          }`}
        >
          {chat.name}
        </p>
        <div className="flex items-center gap-1.5 mt-0.5">
          <span className="text-xs text-[var(--muted)]">
            {formatRelativeTime(chat.updated_at)}
          </span>
          <span className="text-[var(--border)]">·</span>
          <span className="text-xs text-[var(--muted)]">
            {docCount !== undefined ? `${docCount} doc${docCount === 1 ? "" : "s"}` : "—"}
          </span>
        </div>
      </div>

      {/* Context menu trigger */}
      <div className="relative" ref={menuRef}>
        <button
          type="button"
          onClick={(e) => {
            e.stopPropagation();
            setMenuOpen((v) => !v);
          }}
          className={`p-0.5 rounded transition-opacity text-[var(--muted)] hover:text-[var(--foreground)] ${
            menuOpen ? "opacity-100" : "opacity-0 group-hover:opacity-100 focus:opacity-100"
          }`}
          aria-label="Chat options"
          aria-haspopup="true"
          aria-expanded={menuOpen}
        >
          <svg
            className="w-4 h-4"
            viewBox="0 0 16 16"
            fill="currentColor"
            aria-hidden="true"
          >
            <circle cx="8" cy="3" r="1.2" />
            <circle cx="8" cy="8" r="1.2" />
            <circle cx="8" cy="13" r="1.2" />
          </svg>
        </button>

        {menuOpen && (
          <div
            className="absolute right-0 top-6 z-50 w-36 rounded-md border border-[var(--border)] bg-[var(--surface-raised)] shadow-lg py-1"
            role="menu"
          >
            <button
              type="button"
              role="menuitem"
              onClick={(e) => {
                e.stopPropagation();
                handleRename();
              }}
              className="w-full text-left px-3 py-1.5 text-xs text-[var(--foreground)] hover:bg-[var(--surface)] transition-colors"
            >
              Rename
            </button>
            <button
              type="button"
              role="menuitem"
              onClick={(e) => {
                e.stopPropagation();
                handleDelete();
              }}
              className="w-full text-left px-3 py-1.5 text-xs text-red-400 hover:bg-[var(--surface)] transition-colors"
            >
              Delete
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
