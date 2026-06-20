"use client";

import { useCallback, useEffect, useMemo, useState } from "react";

const EVENT_NAME = "document-scope-selection-changed";

function storageKey(chatId: string | null, sessionId: string | null): string | null {
  if (!chatId || !sessionId) return null;
  return `document-scope:${chatId}:${sessionId}`;
}

function readStored(key: string | null): string[] | null {
  if (!key || typeof window === "undefined") return null;
  const raw = window.localStorage.getItem(key);
  if (!raw) return null;
  try {
    const parsed = JSON.parse(raw) as unknown;
    return Array.isArray(parsed) && parsed.every((v) => typeof v === "string")
      ? parsed
      : null;
  } catch {
    return null;
  }
}

function writeStored(key: string | null, value: string[]): void {
  if (!key || typeof window === "undefined") return;
  window.localStorage.setItem(key, JSON.stringify(value));
  window.dispatchEvent(new CustomEvent(EVENT_NAME, { detail: { key } }));
}

export interface DocumentScopeSelection {
  selectedDocumentIds: string[];
  setSelectedDocumentIds: (ids: string[]) => void;
  toggleDocumentId: (id: string) => void;
  allSelected: boolean;
  selectAll: () => void;
  clearSelection: () => void;
}

export function useDocumentScopeSelection(
  chatId: string | null,
  sessionId: string | null,
  availableDocumentIds: string[],
  lockedDocumentIds: string[] | null | undefined,
  locked: boolean
): DocumentScopeSelection {
  const key = storageKey(chatId, sessionId);
  const fallbackIds = useMemo(
    () => (locked ? lockedDocumentIds ?? [] : availableDocumentIds),
    [availableDocumentIds, locked, lockedDocumentIds]
  );
  const [selected, setSelected] = useState<string[]>(fallbackIds);

  useEffect(() => {
    if (!key) {
      setSelected([]);
      return;
    }
    if (locked) {
      setSelected(lockedDocumentIds ?? []);
      return;
    }
    const stored = readStored(key);
    const available = new Set(availableDocumentIds);
    const next = stored ? stored.filter((id) => available.has(id)) : availableDocumentIds;
    setSelected(next);
  }, [availableDocumentIds, key, locked, lockedDocumentIds]);

  useEffect(() => {
    if (!key || typeof window === "undefined") return;
    const onStorage = () => {
      if (locked) return;
      const stored = readStored(key);
      if (stored) setSelected(stored);
    };
    window.addEventListener(EVENT_NAME, onStorage);
    window.addEventListener("storage", onStorage);
    return () => {
      window.removeEventListener(EVENT_NAME, onStorage);
      window.removeEventListener("storage", onStorage);
    };
  }, [key, locked]);

  const setSelectedDocumentIds = useCallback(
    (ids: string[]) => {
      if (locked) return;
      const available = new Set(availableDocumentIds);
      const next = ids.filter((id) => available.has(id));
      setSelected(next);
      writeStored(key, next);
    },
    [availableDocumentIds, key, locked]
  );

  const toggleDocumentId = useCallback(
    (id: string) => {
      if (locked) return;
      setSelectedDocumentIds(
        selected.includes(id) ? selected.filter((v) => v !== id) : [...selected, id]
      );
    },
    [locked, selected, setSelectedDocumentIds]
  );

  const selectAll = useCallback(
    () => setSelectedDocumentIds(availableDocumentIds),
    [availableDocumentIds, setSelectedDocumentIds]
  );
  const clearSelection = useCallback(() => setSelectedDocumentIds([]), [setSelectedDocumentIds]);

  return {
    selectedDocumentIds: selected,
    setSelectedDocumentIds,
    toggleDocumentId,
    allSelected: availableDocumentIds.length > 0 && selected.length === availableDocumentIds.length,
    selectAll,
    clearSelection,
  };
}
