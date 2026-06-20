"use client";

import { useCallback, useMemo, useState } from "react";
import { useSearchParams } from "next/navigation";
import { RefreshCw } from "lucide-react";
import {
  useDocuments,
  useChatManifest,
  useUploadDocument,
  useDeleteDocument,
} from "@/lib/queries/documents";
import { useSession } from "@/lib/queries/sessions";
import { useDocumentScopeSelection } from "@/lib/session/useDocumentScopeSelection";
import { DocumentList } from "./DocumentList";
import { UploadZone } from "./UploadZone";

/**
 * DocumentsPanel — right-rail panel for the active chat.
 *
 * Chat isolation: chatId is read exclusively from the URL search param
 * `?chatId=`. TanStack Query key scoping ensures data from a previous
 * chatId is never shown for the current chatId.
 *
 * Phase 8.2 note: once useCurrentChatId() is merged, the single
 * `useSearchParams().get("chatId")` call here can be replaced with that
 * hook; the semantic is identical.
 */
export function DocumentsPanel() {
  // Read chatId from URL — same source as Phase 8.2's useCurrentChatId
  const searchParams = useSearchParams();
  const chatId = searchParams.get("chatId");
  const sessionId = searchParams.get("sessionId");

  // Upload progress state lives here so UploadZone stays presentational
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploadError, setUploadError] = useState<string | null>(null);

  // Track in-flight deletes so the list can show disabled state
  const [deletingIds, setDeletingIds] = useState<Set<string>>(new Set());

  // Queries
  const {
    data: documents,
    isLoading: docsLoading,
    error: docsError,
    refetch: refetchDocs,
  } = useDocuments(chatId);

  const { data: manifest } = useChatManifest(chatId);
  const { data: session } = useSession(chatId, sessionId);
  const availableDocumentIds = useMemo(
    () => (documents ?? []).map((doc) => doc.id),
    [documents]
  );
  const docScope = useDocumentScopeSelection(
    chatId,
    sessionId,
    availableDocumentIds,
    session?.selected_document_ids,
    session?.document_scope_locked ?? false
  );

  // Mutations
  const uploadMutation = useUploadDocument(chatId);
  const deleteMutation = useDeleteDocument(chatId);

  // Handlers
  const handleUpload = useCallback(
    (file: File) => {
      setUploadError(null);
      setUploadProgress(0);
      uploadMutation.mutate(
        {
          file,
          opts: {
            onProgress: setUploadProgress,
          },
        },
        {
          onSuccess: () => {
            setUploadProgress(100);
            // Brief delay so user sees 100% before resetting
            setTimeout(() => setUploadProgress(0), 800);
          },
          onError: (err: Error) => {
            setUploadProgress(0);
            setUploadError(err.message ?? "Upload failed");
          },
        }
      );
    },
    [uploadMutation]
  );

  const handleDelete = useCallback(
    (docId: string, name: string) => {
      if (
        !window.confirm(
          `Remove "${name}" from this chat? This cannot be undone.`
        )
      ) {
        return;
      }
      setDeletingIds((prev) => new Set(prev).add(docId));
      deleteMutation.mutate(docId, {
        onSettled: () => {
          setDeletingIds((prev) => {
            const next = new Set(prev);
            next.delete(docId);
            return next;
          });
        },
      });
    },
    [deleteMutation]
  );

  // -------------------------------------------------------------------------
  // Render
  // -------------------------------------------------------------------------

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-[var(--border)] shrink-0">
        <span className="text-sm font-semibold text-[var(--foreground)]">
          Documents
          {documents && documents.length > 0 && (
            <span className="ml-1.5 text-xs font-normal text-[var(--muted)]">
              ({documents.length})
            </span>
          )}
        </span>
        {chatId && (
          <button
            aria-label="Refresh document list"
            onClick={() => refetchDocs()}
            className="text-[var(--muted)] hover:text-[var(--foreground)] transition-colors"
          >
            <RefreshCw size={13} />
          </button>
        )}
      </div>

      {/* Body */}
      <div className="flex-1 overflow-y-auto flex flex-col gap-2 py-2">
        {!chatId ? (
          /* No chat selected */
          <div className="flex-1 flex items-center justify-center px-4">
            <p className="text-xs text-[var(--muted)] text-center leading-relaxed">
              Select a chat first to view and manage its documents.
            </p>
          </div>
        ) : docsLoading ? (
          /* Loading skeleton */
          <div className="flex flex-col gap-2 px-3">
            {[1, 2, 3].map((i) => (
              <div
                key={i}
                className="h-14 rounded-lg bg-[var(--surface-raised)] animate-pulse"
              />
            ))}
          </div>
        ) : docsError ? (
          /* Error state */
          <div className="px-4 py-6 text-center">
            <p className="text-xs text-rose-400 mb-2">
              Failed to load documents.
            </p>
            <button
              onClick={() => refetchDocs()}
              className="text-xs text-[var(--accent)] hover:underline"
            >
              Try again
            </button>
          </div>
        ) : (
          /* Document list */
          <DocumentList
            documents={documents ?? []}
            manifest={manifest}
            deletingIds={deletingIds}
            onDelete={handleDelete}
            selectedIds={new Set(docScope.selectedDocumentIds)}
            onToggleSelected={docScope.toggleDocumentId}
            onSelectAll={docScope.selectAll}
            onClearSelection={docScope.clearSelection}
            selectionLocked={session?.document_scope_locked ?? false}
            allSelected={docScope.allSelected}
          />
        )}
      </div>

      {/* Upload zone — only shown when a chat is selected */}
      {chatId && (
        <div className="shrink-0 border-t border-[var(--border)] pt-2">
          <UploadZone
            onUpload={handleUpload}
            isUploading={uploadMutation.isPending}
            uploadProgress={uploadProgress}
            uploadError={uploadError}
            onDismissError={() => setUploadError(null)}
          />
        </div>
      )}
    </div>
  );
}
