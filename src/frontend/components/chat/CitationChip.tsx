"use client";

/**
 * CitationChip — inline chip that shows citation metadata on click.
 *
 * Click opens a modal with excerpt, document name, page range, section title.
 * Isolation: citation must belong to the current chat (caller is responsible
 * for passing only chat-scoped citations from the agent response).
 */

import { useState } from "react";
import type { Citation } from "@/lib/api/types";

interface CitationDetailModalProps {
  citation: Citation;
  onClose: () => void;
}

function CitationDetailModal({ citation, onClose }: CitationDetailModalProps) {
  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60"
      onClick={onClose}
      role="dialog"
      aria-modal="true"
      aria-label="Citation detail"
    >
      <div
        className="relative max-w-lg w-full bg-[var(--surface)] rounded-xl shadow-2xl border border-[var(--border)] p-5"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-start justify-between mb-3">
          <div>
            <p className="text-xs font-semibold text-[var(--foreground)] leading-tight">
              {citation.document_name}
            </p>
            <p className="text-xs text-[var(--muted)] mt-0.5">
              Pages {citation.page_start}–{citation.page_end}
              {citation.section_title && ` · ${citation.section_title}`}
            </p>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="ml-3 shrink-0 text-[var(--muted)] hover:text-[var(--foreground)] transition-colors"
            aria-label="Close"
          >
            <svg className="w-4 h-4" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" aria-hidden="true">
              <line x1="3" y1="3" x2="13" y2="13" />
              <line x1="13" y1="3" x2="3" y2="13" />
            </svg>
          </button>
        </div>

        {/* Excerpt */}
        <blockquote className="text-sm text-[var(--foreground)] leading-relaxed border-l-2 border-[var(--accent)] pl-3 italic whitespace-pre-wrap break-words">
          {citation.excerpt}
        </blockquote>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// CitationChip
// ---------------------------------------------------------------------------

interface CitationChipProps {
  citation: Citation;
  index: number;
}

export function CitationChip({ citation, index }: CitationChipProps) {
  const [open, setOpen] = useState(false);

  return (
    <>
      <button
        type="button"
        onClick={() => setOpen(true)}
        className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-xs bg-[var(--accent)]/15 text-[var(--accent)] hover:bg-[var(--accent)]/25 transition-colors border border-[var(--accent)]/30 leading-none"
        title={`${citation.document_name} p.${citation.page_start}–${citation.page_end}`}
        aria-label={`Citation ${index + 1}: ${citation.document_name}`}
      >
        <span>[{index + 1}]</span>
        <span className="max-w-[80px] truncate opacity-70">
          {citation.document_name}
        </span>
      </button>

      {open && (
        <CitationDetailModal
          citation={citation}
          onClose={() => setOpen(false)}
        />
      )}
    </>
  );
}
