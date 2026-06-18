"use client";

import type { DocumentRead, ChatManifest } from "@/lib/api/types";
import { DocumentListItem } from "./DocumentListItem";

interface DocumentListProps {
  documents: DocumentRead[];
  manifest: ChatManifest | null | undefined;
  deletingIds: Set<string>;
  onDelete: (docId: string, name: string) => void;
}

export function DocumentList({
  documents,
  manifest,
  deletingIds,
  onDelete,
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
    <ul className="flex flex-col gap-0.5 px-2">
      {documents.map((doc) => (
        <DocumentListItem
          key={doc.id}
          doc={doc}
          manifestEntry={manifestMap.get(doc.id) ?? null}
          onDelete={onDelete}
          isDeleting={deletingIds.has(doc.id)}
        />
      ))}
    </ul>
  );
}
