"use client";

/**
 * QuestionInput — textarea + send/stop button for the chat panel.
 *
 * Keyboard:
 *  - Enter (not composing) → sendMessage
 *  - Shift+Enter → newline
 *  - Uses both e.isComposing and e.nativeEvent.isComposing for IME safety.
 */

import { useRef, useCallback } from "react";

interface QuestionInputProps {
  onSend: (question: string) => void;
  onStop: () => void;
  isStreaming: boolean;
  disabled?: boolean;
}

export function QuestionInput({
  onSend,
  onStop,
  isStreaming,
  disabled = false,
}: QuestionInputProps) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const handleSend = useCallback(() => {
    const value = textareaRef.current?.value.trim();
    if (!value) return;
    onSend(value);
    if (textareaRef.current) {
      textareaRef.current.value = "";
      textareaRef.current.style.height = "auto";
    }
  }, [onSend]);

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    // Guard IME composition (Chinese/Japanese/Korean input)
    // nativeEvent.isComposing is the reliable cross-browser flag
    if (e.nativeEvent.isComposing) return;

    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (!isStreaming && !disabled) {
        handleSend();
      }
    }
  }

  function handleInput(e: React.FormEvent<HTMLTextAreaElement>) {
    // Auto-resize textarea
    const el = e.currentTarget;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 200)}px`;
  }

  return (
    <div className="shrink-0 border-t border-[var(--border)] bg-[var(--surface)] px-4 py-3">
      <div className="flex items-end gap-2 rounded-xl border border-[var(--border)] bg-[var(--surface-raised)] px-3 py-2 focus-within:border-[var(--accent)] transition-colors">
        <textarea
          ref={textareaRef}
          rows={1}
          onKeyDown={handleKeyDown}
          onInput={handleInput}
          disabled={disabled}
          placeholder="Ask about your documents… (Enter to send, Shift+Enter for newline)"
          className="flex-1 resize-none bg-transparent text-sm text-[var(--foreground)] placeholder:text-[var(--muted)] outline-none leading-relaxed max-h-[200px] overflow-y-auto disabled:opacity-50"
          aria-label="Question input"
        />

        {isStreaming ? (
          /* Stop button */
          <button
            type="button"
            onClick={onStop}
            className="shrink-0 flex items-center justify-center w-8 h-8 rounded-lg bg-rose-500/20 text-rose-400 hover:bg-rose-500/30 transition-colors"
            aria-label="Stop generation"
            title="Stop generation"
          >
            <svg className="w-4 h-4" viewBox="0 0 16 16" fill="currentColor" aria-hidden="true">
              <rect x="4" y="4" width="8" height="8" rx="1" />
            </svg>
          </button>
        ) : (
          /* Send button */
          <button
            type="button"
            onClick={handleSend}
            disabled={disabled}
            className="shrink-0 flex items-center justify-center w-8 h-8 rounded-lg bg-[var(--accent)] text-white hover:opacity-90 transition-opacity disabled:opacity-40"
            aria-label="Send message"
            title="Send (Enter)"
          >
            <svg className="w-4 h-4" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
              <line x1="14" y1="2" x2="2" y2="8" />
              <line x1="14" y1="2" x2="8" y2="14" />
              <line x1="2" y1="8" x2="8" y2="14" />
            </svg>
          </button>
        )}
      </div>
    </div>
  );
}
