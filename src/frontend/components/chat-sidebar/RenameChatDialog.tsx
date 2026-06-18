"use client";

/**
 * RenameChatDialog — modal for renaming an existing chat.
 *
 * Uses native <dialog> element.
 * Keyboard: Esc closes, auto-focuses name input pre-filled with current name.
 */

import { useEffect, useRef, useState } from "react";
import type { FormEvent } from "react";

interface RenameChatDialogProps {
  open: boolean;
  currentName: string;
  onClose: () => void;
  onSubmit: (name: string) => void;
  isPending: boolean;
}

export function RenameChatDialog({
  open,
  currentName,
  onClose,
  onSubmit,
  isPending,
}: RenameChatDialogProps) {
  const dialogRef = useRef<HTMLDialogElement>(null);
  const nameRef = useRef<HTMLInputElement>(null);
  const [name, setName] = useState(currentName);

  // Keep local name in sync when currentName changes (re-open for a different chat)
  useEffect(() => {
    setName(currentName);
  }, [currentName]);

  // Sync open state with native dialog
  useEffect(() => {
    const dialog = dialogRef.current;
    if (!dialog) return;
    if (open) {
      if (!dialog.open) dialog.showModal();
      setTimeout(() => {
        const input = nameRef.current;
        if (input) {
          input.focus();
          input.select();
        }
      }, 0);
    } else {
      if (dialog.open) dialog.close();
    }
  }, [open]);

  // Handle native Esc key from dialog element
  useEffect(() => {
    const dialog = dialogRef.current;
    if (!dialog) return;
    const handleCancel = (e: Event) => {
      e.preventDefault();
      onClose();
    };
    dialog.addEventListener("cancel", handleCancel);
    return () => dialog.removeEventListener("cancel", handleCancel);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function handleSubmit(e: FormEvent) {
    e.preventDefault();
    const trimmed = name.trim();
    if (!trimmed) return;
    onSubmit(trimmed);
  }

  return (
    <dialog
      ref={dialogRef}
      className="rounded-lg shadow-xl bg-[var(--surface)] border border-[var(--border)] p-0 w-full max-w-sm backdrop:bg-black/50"
      onClose={onClose}
    >
      <form onSubmit={handleSubmit}>
        <div className="px-5 py-4 border-b border-[var(--border)]">
          <h2 className="text-sm font-semibold text-[var(--foreground)]">Rename Chat</h2>
        </div>

        <div className="px-5 py-4">
          <label
            htmlFor="rename-chat-name"
            className="block text-xs font-medium text-[var(--muted)] mb-1"
          >
            New name <span className="text-red-400">*</span>
          </label>
          <input
            id="rename-chat-name"
            ref={nameRef}
            type="text"
            value={name}
            onChange={(e) => setName(e.target.value)}
            required
            className="w-full rounded-md border border-[var(--border)] bg-[var(--background)] px-3 py-1.5 text-sm text-[var(--foreground)] placeholder:text-[var(--muted)] focus:outline-none focus:ring-1 focus:ring-[var(--accent)]"
          />
        </div>

        <div className="px-5 py-3 flex justify-end gap-2 border-t border-[var(--border)]">
          <button
            type="button"
            onClick={onClose}
            disabled={isPending}
            className="px-3 py-1.5 text-xs rounded-md border border-[var(--border)] text-[var(--muted)] hover:text-[var(--foreground)] transition-colors disabled:opacity-50"
          >
            Cancel
          </button>
          <button
            type="submit"
            disabled={isPending || !name.trim()}
            className="px-3 py-1.5 text-xs rounded-md bg-[var(--accent)] text-white hover:opacity-90 transition-opacity disabled:opacity-50"
          >
            {isPending ? "Saving…" : "Save"}
          </button>
        </div>
      </form>
    </dialog>
  );
}
