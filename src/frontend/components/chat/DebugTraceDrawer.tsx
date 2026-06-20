"use client";

/**
 * DebugTraceDrawer — collapsible display for LangGraph decision/tool trace.
 *
 * Shows per-step tool name, status, token estimate, and note.
 * Collapsed by default; toggled by clicking the header.
 */

import { useState } from "react";
import type { ToolTrace } from "@/lib/api/types";

interface DebugTraceDrawerProps {
  trace: ToolTrace;
}

const STATUS_COLORS: Record<string, string> = {
  ok: "text-emerald-400",
  info: "text-sky-400",
  overflow: "text-amber-400",
  error: "text-rose-400",
};

export function DebugTraceDrawer({ trace }: DebugTraceDrawerProps) {
  const [open, setOpen] = useState(false);

  return (
    <div className="mt-2 border border-[var(--border)] rounded-lg overflow-hidden text-xs">
      {/* Toggle header */}
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="w-full flex items-center justify-between px-3 py-1.5 bg-[var(--surface-raised)] text-[var(--muted)] hover:text-[var(--foreground)] transition-colors"
        aria-expanded={open}
        aria-label="Toggle reasoning trace"
      >
        <span className="font-mono">
          Reasoning trace · {trace.total_rounds} round{trace.total_rounds !== 1 ? "s" : ""}
          {trace.token_count_is_estimate && " (est.)"}
        </span>
        <svg
          className={`w-3.5 h-3.5 transition-transform ${open ? "rotate-180" : ""}`}
          viewBox="0 0 16 16"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.5"
          strokeLinecap="round"
          aria-hidden="true"
        >
          <polyline points="4 6 8 10 12 6" />
        </svg>
      </button>

      {/* Steps */}
      {open && (
        <div className="divide-y divide-[var(--border)]">
          {trace.steps.map((step, i) => (
            <div key={i} className="px-3 py-1.5 flex items-start gap-3 bg-[var(--surface)]">
              <span className="font-mono text-[var(--muted)] w-5 shrink-0 text-right">
                {i + 1}.
              </span>
              <span className="font-mono text-[var(--foreground)] flex-1 min-w-0 truncate">
                {step.tool_name}
              </span>
              <span
                className={`font-mono shrink-0 ${STATUS_COLORS[step.status] ?? "text-[var(--muted)]"}`}
              >
                {step.status}
              </span>
              {step.token_estimate !== null && (
                <span className="font-mono text-[var(--muted)] shrink-0">
                  ~{step.token_estimate}tok
                </span>
              )}
              {step.note && (
                <span className="text-[var(--muted)] shrink-0 max-w-[120px] truncate" title={step.note}>
                  {step.note}
                </span>
              )}
            </div>
          ))}

          {trace.steps.length === 0 && (
            <div className="px-3 py-2 text-[var(--muted)]">No steps recorded.</div>
          )}
        </div>
      )}
    </div>
  );
}
