"use client";

import { Trash2, AlertCircle } from "lucide-react";
import type { DocumentRead, DocumentManifestEntry } from "@/lib/api/types";
import { StatusBadge } from "./StatusBadge";

interface DocumentListItemProps {
  doc: DocumentRead;
  manifestEntry?: DocumentManifestEntry | null;
  onDelete: (docId: string, name: string) => void;
  isDeleting: boolean;
  selected: boolean;
  onToggleSelected: (docId: string) => void;
  selectionLocked: boolean;
}

function fmt(n: number | null | undefined): string {
  if (n == null) return "—";
  return n.toLocaleString();
}

export function DocumentListItem({
  doc,
  manifestEntry,
  onDelete,
  isDeleting,
  selected,
  onToggleSelected,
  selectionLocked,
}: DocumentListItemProps) {
  const sectionCount =
    manifestEntry != null ? manifestEntry.section_count : null;
  const tokenEstimate =
    manifestEntry != null ? manifestEntry.token_estimate : null;

  return (
    <li className="group flex flex-col gap-1.5 rounded-lg px-3 py-2.5 hover:bg-[var(--surface-raised)] transition-colors">
      {/* Top row: filename + delete */}
      <div className="flex items-start justify-between gap-2 min-w-0">
        <label className="flex items-start gap-2 min-w-0 flex-1">
          <input
            type="checkbox"
            checked={selected}
            disabled={selectionLocked}
            onChange={() => onToggleSelected(doc.id)}
            className="mt-0.5 h-3.5 w-3.5 rounded border-[var(--border)] accent-[var(--accent)] disabled:opacity-60"
            aria-label={`Use ${doc.original_filename} for this session QA`}
          />
          <span
            className="text-xs font-medium text-[var(--foreground)] truncate"
            title={doc.original_filename}
          >
            {doc.original_filename}
          </span>
        </label>
        <button
          aria-label={`Delete ${doc.original_filename}`}
          disabled={isDeleting}
          onClick={() => onDelete(doc.id, doc.original_filename)}
          className="shrink-0 opacity-0 group-hover:opacity-100 transition-opacity text-[var(--muted)] hover:text-rose-400 disabled:opacity-40 disabled:cursor-not-allowed"
        >
          <Trash2 size={13} />
        </button>
      </div>

      {/* Status + meta row */}
      <div className="flex items-center gap-2 flex-wrap">
        <StatusBadge status={doc.status} />
        <span className="text-[10px] text-[var(--muted)]">
          {fmt(doc.page_count)} pp
        </span>
        <span className="text-[10px] text-[var(--muted)]">
          {fmt(sectionCount)} sec
        </span>
        <span className="text-[10px] text-[var(--muted)]">
          ~{fmt(tokenEstimate)} tok
        </span>
      </div>

      {/* Retry hint when failed */}
      {doc.status === "failed" && (
        <div className="flex items-center gap-1 text-[10px] text-rose-400">
          <AlertCircle size={11} />
          <span>Processing failed — re-upload to retry</span>
        </div>
      )}
    </li>
  );
}
