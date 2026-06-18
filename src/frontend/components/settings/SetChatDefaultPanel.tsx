"use client";

/**
 * SetChatDefaultPanel — assigns per-chat default profiles for chat/embedding/reranker.
 *
 * Reads chats from useChats() and profiles from useProviderProfiles(kind).
 * Writes to localStorage (local-mode) or backend once the API is wired.
 */

import { useState } from "react";
import { useChats } from "@/lib/queries/chats";
import { useProviderProfiles, useSetChatDefault } from "@/lib/queries/providers";
import type { ProviderKind } from "@/lib/api/types";

const KINDS: ProviderKind[] = ["chat", "embedding", "reranker"];

interface KindRowProps {
  chatId: string;
  kind: ProviderKind;
  onSaved: () => void;
}

function KindRow({ chatId, kind, onSaved }: KindRowProps) {
  const { data: profiles = [] } = useProviderProfiles(kind);
  const { mutateAsync, isPending } = useSetChatDefault();
  const [selected, setSelected] = useState<string>("");
  const [saved, setSaved] = useState(false);

  async function handleSet() {
    if (!selected) return;
    await mutateAsync({ chatId, body: { kind, profile_id: selected } });
    setSaved(true);
    setTimeout(() => setSaved(false), 2000);
    onSaved();
  }

  return (
    <div className="flex items-center gap-2">
      <span className="w-24 text-xs text-[var(--muted)] capitalize shrink-0">{kind}</span>
      <select
        value={selected}
        onChange={(e) => { setSelected(e.target.value); setSaved(false); }}
        className="flex-1 min-w-0 px-2 py-1.5 text-xs rounded bg-[var(--surface)] border border-[var(--border)] text-[var(--foreground)] focus:outline-none focus:border-[var(--accent)]"
      >
        <option value="">(pick profile…)</option>
        {profiles.map((p) => (
          <option key={p.id} value={p.id}>
            {p.name} — {p.model}
          </option>
        ))}
      </select>
      <button
        onClick={handleSet}
        disabled={!selected || isPending}
        className="px-3 py-1.5 text-xs rounded bg-[var(--accent)] text-white hover:bg-[var(--accent-hover)] disabled:opacity-40 transition-colors shrink-0"
      >
        {saved ? "Saved" : "Set"}
      </button>
    </div>
  );
}

export function SetChatDefaultPanel() {
  const { data: chats = [], isLoading } = useChats();
  const [chatId, setChatId] = useState<string>("");
  const [savedMsg, setSavedMsg] = useState(false);

  function onSaved() {
    setSavedMsg(true);
    setTimeout(() => setSavedMsg(false), 2500);
  }

  return (
    <div className="space-y-4">
      <h3 className="text-sm font-semibold text-[var(--foreground)]">
        Set chat-level default profiles
      </h3>
      <p className="text-xs text-[var(--muted)]">
        Override global defaults for a specific chat. In local-mode the
        selection is persisted to localStorage under{" "}
        <code className="font-mono text-[var(--accent)]">
          pna_chat_defaults_&#123;chatId&#125;
        </code>.
      </p>

      {/* Chat selector */}
      <div>
        <label className="block text-xs font-medium text-[var(--muted)] mb-1">
          Chat
        </label>
        {isLoading ? (
          <p className="text-xs text-[var(--muted)]">Loading chats…</p>
        ) : (
          <select
            value={chatId}
            onChange={(e) => setChatId(e.target.value)}
            className="w-full px-3 py-2 text-sm rounded bg-[var(--surface)] border border-[var(--border)] text-[var(--foreground)] focus:outline-none focus:border-[var(--accent)]"
          >
            <option value="">(select chat…)</option>
            {chats.map((c) => (
              <option key={c.id} value={c.id}>
                {c.name}
              </option>
            ))}
          </select>
        )}
        {!isLoading && chats.length === 0 && (
          <p className="text-xs text-[var(--muted)] mt-1">
            No chats yet — create one from the main view.
          </p>
        )}
      </div>

      {/* Per-kind rows */}
      {chatId && (
        <div className="space-y-2 border border-[var(--border)] rounded-lg p-4 bg-[var(--surface-raised)]">
          {KINDS.map((k) => (
            <KindRow key={k} chatId={chatId} kind={k} onSaved={onSaved} />
          ))}
          {savedMsg && (
            <p className="text-xs text-green-400 mt-1">Saved to localStorage.</p>
          )}
        </div>
      )}
    </div>
  );
}
