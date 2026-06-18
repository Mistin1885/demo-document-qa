"use client";

/**
 * AppShell — three-column workspace layout.
 *
 * Columns:
 *   1. Left sidebar   (w-64)  — Chat list + navigation (Phase 8.2: ChatSidebar).
 *   2. Center panel   (flex-1) — Active session chat UI.
 *                               Phase 8.3 sub-agent: replace placeholder with ChatPanel.
 *   3. Right rail     (w-72)  — Documents panel for the active chat.
 *                               Phase 8.4 sub-agent: replace placeholder with DocumentsPanel.
 *
 * All three regions are full-height and scroll independently.
 */

import { Suspense } from "react";
import { ChatSidebar } from "@/components/chat-sidebar/ChatSidebar";
import { DocumentsPanel } from "@/components/documents/DocumentsPanel";
import { ChatPanel } from "@/components/chat/ChatPanel";

export function AppShell() {
  return (
    <div className="flex h-full overflow-hidden bg-[var(--background)]">
      {/* ------------------------------------------------------------------ */}
      {/* LEFT SIDEBAR — ChatSidebar (Phase 8.2)                             */}
      {/* ------------------------------------------------------------------ */}
      <aside
        className="w-64 shrink-0 flex flex-col border-r border-[var(--border)] bg-[var(--surface)]"
        data-region="chat-sidebar"
      >
        {/* Suspense boundary required because ChatSidebar reads useSearchParams */}
        <Suspense
          fallback={
            <div className="flex-1 flex items-center justify-center">
              <span className="text-xs text-[var(--muted)]">Loading…</span>
            </div>
          }
        >
          <ChatSidebar />
        </Suspense>
      </aside>

      {/* ------------------------------------------------------------------ */}
      {/* CENTER PANEL — ChatPanel (Phase 8.4)                              */}
      {/* ------------------------------------------------------------------ */}
      <main
        className="flex-1 flex flex-col overflow-hidden"
        data-region="chat-panel"
      >
        <ChatPanel />
      </main>

      {/* ------------------------------------------------------------------ */}
      {/* RIGHT RAIL — DocumentsPanel (Phase 8.3)                           */}
      {/* ------------------------------------------------------------------ */}
      <aside
        className="w-72 shrink-0 flex flex-col border-l border-[var(--border)] bg-[var(--surface)]"
        data-region="documents-panel"
      >
        {/* Suspense boundary required because DocumentsPanel reads useSearchParams */}
        <Suspense
          fallback={
            <div className="flex-1 flex items-center justify-center">
              <span className="text-xs text-[var(--muted)]">Loading…</span>
            </div>
          }
        >
          <DocumentsPanel />
        </Suspense>
      </aside>
    </div>
  );
}
