"use client";

/**
 * GenerationPrefsPanel — collapsible "⚙️ Advanced" controls for the chat input.
 *
 * Exposes three knobs (all optional — undefined means "use backend env defaults"):
 *  - Max output tokens (slider 256..32768; stepped 256)
 *  - Temperature       (slider 0..1.5; stepped 0.05)
 *  - Context window    (number; 4000..200000)
 *
 * Values are persisted per-chat via useGenerationPrefs.
 */

import { useState } from "react";

import type { GenerationOverrides } from "@/lib/api/messages";

interface Props {
  prefs: GenerationOverrides;
  onChange: (next: GenerationOverrides) => void;
  onReset: () => void;
  disabled?: boolean;
}

function fieldOrEmpty(n: number | undefined): string {
  return typeof n === "number" ? String(n) : "";
}

export function GenerationPrefsPanel({ prefs, onChange, onReset, disabled = false }: Props) {
  const [open, setOpen] = useState(false);

  const isDirty =
    prefs.max_answer_tokens !== undefined ||
    prefs.temperature !== undefined ||
    prefs.context_window !== undefined ||
    prefs.deep_qa_mode === true;

  function patch(p: Partial<GenerationOverrides>) {
    onChange({ ...prefs, ...p });
  }

  return (
    <div className="border-t border-[var(--border)] bg-[var(--surface)]">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex items-center gap-1.5 px-4 py-1.5 text-xs text-[var(--muted)] hover:text-[var(--foreground)] transition-colors"
        aria-expanded={open}
        aria-controls="gen-prefs-body"
      >
        <span className="font-medium">⚙ Advanced</span>
        {isDirty && (
          <span className="rounded-full bg-amber-500/20 px-1.5 py-0.5 text-[10px] text-amber-300">
            custom
          </span>
        )}
        <span className="ml-auto">{open ? "▾" : "▸"}</span>
      </button>

      {open && (
        <div id="gen-prefs-body" className="grid gap-3 px-4 py-3 text-xs">
          <label className="flex items-start gap-3 rounded-lg border border-[var(--border)] bg-[var(--surface-raised)] p-2">
            <input
              type="checkbox"
              checked={prefs.deep_qa_mode === true}
              disabled={disabled}
              onChange={(e) => patch({ deep_qa_mode: e.target.checked || undefined })}
              className="mt-0.5 accent-[var(--accent)]"
            />
            <span>
              <span className="block font-medium text-[var(--foreground)]">Deep QA mode</span>
              <span className="text-[11px] text-[var(--muted)]">
                Ignore soft LLM budget and include same-session memory for follow-up questions.
              </span>
            </span>
          </label>

          {/* max_answer_tokens slider */}
          <label className="flex items-center gap-3">
            <span className="w-32 shrink-0 text-[var(--muted)]">Max output tokens</span>
            <input
              type="range"
              min={256}
              max={32_768}
              step={256}
              value={prefs.max_answer_tokens ?? 2048}
              disabled={disabled}
              onChange={(e) =>
                patch({ max_answer_tokens: Number(e.target.value) })
              }
              className="flex-1 accent-[var(--accent)]"
            />
            <input
              type="number"
              min={1}
              max={32_768}
              value={fieldOrEmpty(prefs.max_answer_tokens)}
              placeholder="auto"
              disabled={disabled}
              onChange={(e) => {
                const v = e.target.value.trim();
                patch({
                  max_answer_tokens: v === "" ? undefined : Number(v),
                });
              }}
              className="w-20 rounded border border-[var(--border)] bg-[var(--surface-raised)] px-2 py-1 text-right tabular-nums"
            />
          </label>

          {/* temperature slider */}
          <label className="flex items-center gap-3">
            <span className="w-32 shrink-0 text-[var(--muted)]">Temperature</span>
            <input
              type="range"
              min={0}
              max={1.5}
              step={0.05}
              value={prefs.temperature ?? 0.0}
              disabled={disabled}
              onChange={(e) => patch({ temperature: Number(e.target.value) })}
              className="flex-1 accent-[var(--accent)]"
            />
            <input
              type="number"
              min={0}
              max={2}
              step={0.05}
              value={fieldOrEmpty(prefs.temperature)}
              placeholder="auto"
              disabled={disabled}
              onChange={(e) => {
                const v = e.target.value.trim();
                patch({ temperature: v === "" ? undefined : Number(v) });
              }}
              className="w-20 rounded border border-[var(--border)] bg-[var(--surface-raised)] px-2 py-1 text-right tabular-nums"
            />
          </label>

          {/* context_window number */}
          <label className="flex items-center gap-3">
            <span className="w-32 shrink-0 text-[var(--muted)]">Context window</span>
            <input
              type="range"
              min={4_000}
              max={64_000}
              step={1_000}
              value={prefs.context_window ?? 10_000}
              disabled={disabled}
              onChange={(e) => patch({ context_window: Number(e.target.value) })}
              className="flex-1 accent-[var(--accent)]"
            />
            <input
              type="number"
              min={1_000}
              max={200_000}
              step={1_000}
              value={fieldOrEmpty(prefs.context_window)}
              placeholder="auto"
              disabled={disabled}
              onChange={(e) => {
                const v = e.target.value.trim();
                patch({ context_window: v === "" ? undefined : Number(v) });
              }}
              className="w-20 rounded border border-[var(--border)] bg-[var(--surface-raised)] px-2 py-1 text-right tabular-nums"
            />
          </label>

          <div className="flex items-center justify-between text-[11px] text-[var(--muted)]">
            <span>
              Blank = use server defaults (LLM_MAX_TOKENS / LLM_TEMPERATURE / LLM_CONTEXT_WINDOW).
            </span>
            <button
              type="button"
              onClick={onReset}
              disabled={!isDirty || disabled}
              className="rounded px-2 py-0.5 text-[var(--accent)] hover:bg-[var(--surface-raised)] disabled:opacity-40"
            >
              Reset
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
