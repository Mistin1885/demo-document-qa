"use client";

import type { DocumentRead, ChatManifest } from "@/lib/api/types";
import { DocumentListItem } from "./DocumentListItem";

interface DocumentListProps {
  documents: DocumentRead[];
  manifest: ChatManifest | null | undefined;
  deletingIds: Set<string>;
  onDelete: (docId: string, name: string) => void;
  selectedIds: Set<string>;
  onToggleSelected: (docId: string) => void;
  onSelectAll: () => void;
  onClearSelection: () => void;
  selectionLocked: boolean;
  allSelected: boolean;
}

export function DocumentList({
  documents,
  manifest,
  deletingIds,
  onDelete,
  selectedIds,
  onToggleSelected,
  onSelectAll,
  onClearSelection,
  selectionLocked,
  allSelected,
}: DocumentListProps) {
  if (documents.length === 0) {
    return (
      <p className="px-4 py-6 text-xs text-[var(--muted)] text-center">
        No documents yet. Upload a PDF to get started.
      </p>
    );
  }

  // Build a lookup map from manifest entries by document_id
  const manifestMap = new Map(
    manifest?.documents?.map((e) => [e.document_id, e]) ?? []
  );

  return (
    <div className="px-2">
      <div className="flex items-center justify-between gap-2 px-1 pb-2">
        <span className="text-[10px] text-[var(--muted)]">
          QA scope {selectionLocked ? "locked" : "select before first QA"}
        </span>
        {!selectionLocked && (
          <button
            type="button"
            onClick={allSelected ? onClearSelection : onSelectAll}
            className="text-[10px] text-[var(--accent)] hover:underline"
          >
            {allSelected ? "Clear" : "All"}
          </button>
        )}
      </div>
      <ul className="flex flex-col gap-0.5">
        {documents.map((doc) => (
          <DocumentListItem
            key={doc.id}
            doc={doc}
            manifestEntry={manifestMap.get(doc.id) ?? null}
            onDelete={onDelete}
            isDeleting={deletingIds.has(doc.id)}
            selected={selectedIds.has(doc.id)}
            onToggleSelected={onToggleSelected}
            selectionLocked={selectionLocked}
          />
        ))}
      </ul>
    </div>
  );
}
