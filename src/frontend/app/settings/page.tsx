"use client";

/**
 * /settings — Model Settings UI
 *
 * Three tabs: Chat / Embedding / Reranker provider profiles, plus
 * a "Set chat default" panel.
 *
 * Security (CLAUDE.md §9):
 *   - All API key inputs use type="password".
 *   - Keys are never rendered in list views.
 *   - LocalStorage banner warns about demo-mode key storage.
 */

import { useState } from "react";
import Link from "next/link";
import { ProviderProfileList } from "@/components/settings/ProviderProfileList";
import { SetChatDefaultPanel } from "@/components/settings/SetChatDefaultPanel";
import type { ProviderKind } from "@/lib/api/types";

type Tab = ProviderKind | "chat_defaults";

const TABS: { id: Tab; label: string }[] = [
  { id: "chat", label: "Chat" },
  { id: "embedding", label: "Embedding" },
  { id: "reranker", label: "Reranker" },
  { id: "chat_defaults", label: "Chat defaults" },
];

export default function SettingsPage() {
  const [tab, setTab] = useState<Tab>("chat");

  return (
    <div className="min-h-screen bg-[var(--background)] text-[var(--foreground)]">
      {/* Page header */}
      <header className="border-b border-[var(--border)] bg-[var(--surface)] px-6 py-4 flex items-center gap-4">
        <Link
          href="/"
          className="text-xs text-[var(--muted)] hover:text-[var(--foreground)] transition-colors"
        >
          ← Back
        </Link>
        <h1 className="text-base font-semibold">Model Settings</h1>
      </header>

      {/* Demo mode banner */}
      <div className="px-6 py-2 bg-yellow-900/20 border-b border-yellow-800/30">
        <p className="text-xs text-yellow-400/80">
          Demo mode: provider profiles and API keys are stored in your browser
          localStorage only. Backend API not yet wired.
        </p>
      </div>

      <div className="max-w-3xl mx-auto px-6 py-8">
        {/* Tab bar */}
        <div className="flex gap-1 mb-8 border-b border-[var(--border)]">
          {TABS.map((t) => (
            <button
              key={t.id}
              onClick={() => setTab(t.id)}
              className={`px-4 py-2 text-sm font-medium transition-colors border-b-2 -mb-px ${
                tab === t.id
                  ? "border-[var(--accent)] text-[var(--accent)]"
                  : "border-transparent text-[var(--muted)] hover:text-[var(--foreground)]"
              }`}
            >
              {t.label}
            </button>
          ))}
        </div>

        {/* Tab content */}
        <div className="min-h-[400px]">
          {tab === "chat" && <ProviderProfileList kind="chat" />}
          {tab === "embedding" && <ProviderProfileList kind="embedding" />}
          {tab === "reranker" && <ProviderProfileList kind="reranker" />}
          {tab === "chat_defaults" && <SetChatDefaultPanel />}
        </div>
      </div>
    </div>
  );
}
