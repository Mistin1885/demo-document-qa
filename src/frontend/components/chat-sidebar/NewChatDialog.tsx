"use client";

/**
 * NewChatDialog — modal for creating a new chat.
 *
 * Uses native <dialog> element.
 * Keyboard: Esc closes, auto-focuses name input on open.
 */

import { useEffect, useRef, useState } from "react";
import type { FormEvent } from "react";

interface NewChatDialogProps {
  open: boolean;
  onClose: () => void;
  onSubmit: (name: string, description: string) => void;
  isPending: boolean;
}

export function NewChatDialog({
  open,
  onClose,
  onSubmit,
  isPending,
}: NewChatDialogProps) {
  const dialogRef = useRef<HTMLDialogElement>(null);
  const nameRef = useRef<HTMLInputElement>(null);
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");

  // Sync open state with native dialog
  useEffect(() => {
    const dialog = dialogRef.current;
    if (!dialog) return;
    if (open) {
      if (!dialog.open) dialog.showModal();
      // Auto-focus the name input
      setTimeout(() => nameRef.current?.focus(), 0);
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
      handleClose();
    };
    dialog.addEventListener("cancel", handleCancel);
    return () => dialog.removeEventListener("cancel", handleCancel);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function handleClose() {
    setName("");
    setDescription("");
    onClose();
  }

  function handleSubmit(e: FormEvent) {
    e.preventDefault();
    const trimmed = name.trim();
    if (!trimmed) return;
    onSubmit(trimmed, description.trim());
  }

  return (
    <dialog
      ref={dialogRef}
      className="rounded-lg shadow-xl bg-[var(--surface)] border border-[var(--border)] p-0 w-full max-w-sm backdrop:bg-black/50"
      onClose={handleClose}
    >
      <form onSubmit={handleSubmit}>
        <div className="px-5 py-4 border-b border-[var(--border)]">
          <h2 className="text-sm font-semibold text-[var(--foreground)]">New Chat</h2>
        </div>

        <div className="px-5 py-4 space-y-3">
          <div>
            <label
              htmlFor="new-chat-name"
              className="block text-xs font-medium text-[var(--muted)] mb-1"
            >
              Name <span className="text-red-400">*</span>
            </label>
            <input
              id="new-chat-name"
              ref={nameRef}
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="e.g. Attention Is All You Need"
              required
              className="w-full rounded-md border border-[var(--border)] bg-[var(--background)] px-3 py-1.5 text-sm text-[var(--foreground)] placeholder:text-[var(--muted)] focus:outline-none focus:ring-1 focus:ring-[var(--accent)]"
            />
          </div>

          <div>
            <label
              htmlFor="new-chat-desc"
              className="block text-xs font-medium text-[var(--muted)] mb-1"
            >
              Description (optional)
            </label>
            <input
              id="new-chat-desc"
              type="text"
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="Brief description…"
              className="w-full rounded-md border border-[var(--border)] bg-[var(--background)] px-3 py-1.5 text-sm text-[var(--foreground)] placeholder:text-[var(--muted)] focus:outline-none focus:ring-1 focus:ring-[var(--accent)]"
            />
          </div>
        </div>

        <div className="px-5 py-3 flex justify-end gap-2 border-t border-[var(--border)]">
          <button
            type="button"
            onClick={handleClose}
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
            {isPending ? "Creating…" : "Create"}
          </button>
        </div>
      </form>
    </dialog>
  );
}
