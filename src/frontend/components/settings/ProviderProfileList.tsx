"use client";

/**
 * ProviderProfileList — renders a list of provider profiles for one kind.
 *
 * Security (CLAUDE.md §9):
 *   - API keys are NEVER shown in the list view.
 *   - Key presence is indicated by "••••••••" or "(none)".
 */

import { useState } from "react";
import type { ProviderKind } from "@/lib/api/types";
import type { ProviderProfileCreate, ProviderProfileRead } from "@/lib/api/providers";
import {
  useProviderProfiles,
  useCreateProfile,
  useUpdateProfile,
  useDeleteProfile,
  useSetGlobalDefault,
} from "@/lib/queries/providers";
import { ProviderProfileForm } from "./ProviderProfileForm";
import { TestConnectionButton } from "./TestConnectionButton";

interface Props {
  kind: ProviderKind;
}

type Mode = { type: "list" } | { type: "create" } | { type: "edit"; profile: ProviderProfileRead };

export function ProviderProfileList({ kind }: Props) {
  const { data: profiles = [], isLoading } = useProviderProfiles(kind);
  const create = useCreateProfile(kind);
  const update = useUpdateProfile(kind);
  const del = useDeleteProfile(kind);
  const setDefault = useSetGlobalDefault(kind);
  const [mode, setMode] = useState<Mode>({ type: "list" });

  async function handleCreate(data: ProviderProfileCreate) {
    await create.mutateAsync(data);
    setMode({ type: "list" });
  }

  async function handleUpdate(id: string, data: ProviderProfileCreate) {
    await update.mutateAsync({ id, body: data });
    setMode({ type: "list" });
  }

  async function handleDelete(id: string) {
    if (!window.confirm("Delete this profile?")) return;
    await del.mutateAsync(id);
  }

  if (mode.type === "create") {
    return (
      <div>
        <h3 className="text-sm font-semibold text-[var(--foreground)] mb-4">
          New {kind} profile
        </h3>
        <ProviderProfileForm
          kind={kind}
          onSubmit={handleCreate}
          onCancel={() => setMode({ type: "list" })}
          loading={create.isPending}
        />
      </div>
    );
  }

  if (mode.type === "edit") {
    const p = mode.profile;
    return (
      <div>
        <h3 className="text-sm font-semibold text-[var(--foreground)] mb-4">
          Edit: {p.name}
        </h3>
        <ProviderProfileForm
          kind={kind}
          existing={p}
          onSubmit={(data) => handleUpdate(p.id, data)}
          onCancel={() => setMode({ type: "list" })}
          loading={update.isPending}
        />
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-[var(--foreground)] capitalize">
          {kind} profiles
        </h3>
        <button
          onClick={() => setMode({ type: "create" })}
          className="px-3 py-1 text-xs rounded bg-[var(--accent)] text-white hover:bg-[var(--accent-hover)] transition-colors"
        >
          + New
        </button>
      </div>

      {isLoading && (
        <p className="text-xs text-[var(--muted)]">Loading…</p>
      )}

      {!isLoading && profiles.length === 0 && (
        <p className="text-xs text-[var(--muted)] py-4 text-center">
          No {kind} profiles yet.
        </p>
      )}

      <ul className="space-y-3">
        {profiles.map((p) => (
          <li
            key={p.id}
            className="rounded-lg border border-[var(--border)] bg-[var(--surface-raised)] p-4 space-y-3"
          >
            {/* Header row */}
            <div className="flex items-start justify-between gap-2">
              <div className="min-w-0">
                <div className="flex items-center gap-2 flex-wrap">
                  <span className="text-sm font-medium text-[var(--foreground)] truncate">
                    {p.name}
                  </span>
                  {p.is_default && (
                    <span className="px-1.5 py-0.5 text-xs rounded bg-[var(--accent)]/20 text-[var(--accent)]">
                      default
                    </span>
                  )}
                </div>
                <p className="text-xs text-[var(--muted)] mt-0.5">
                  {p.provider_type} · {p.model}
                  {p.base_url && ` · ${p.base_url}`}
                  {p.context_window && ` · ${p.context_window.toLocaleString()} ctx`}
                </p>
                {/* Key: masked — never show plaintext */}
                <p className="text-xs text-[var(--muted)] mt-0.5">
                  key: ••••••••
                </p>
              </div>
              <div className="flex gap-1 shrink-0">
                <button
                  onClick={() => setMode({ type: "edit", profile: p })}
                  className="px-2 py-1 text-xs rounded bg-[var(--surface)] border border-[var(--border)] text-[var(--foreground)] hover:bg-[var(--border)] transition-colors"
                >
                  Edit
                </button>
                <button
                  onClick={() => handleDelete(p.id)}
                  disabled={del.isPending}
                  className="px-2 py-1 text-xs rounded bg-[var(--surface)] border border-[var(--border)] text-red-400 hover:bg-red-900/20 disabled:opacity-50 transition-colors"
                >
                  Delete
                </button>
              </div>
            </div>

            {/* Actions row */}
            <div className="flex flex-wrap items-center gap-2">
              <TestConnectionButton profileId={p.id} />
              {!p.is_default && (
                <button
                  onClick={() => setDefault.mutateAsync(p.id)}
                  disabled={setDefault.isPending}
                  className="px-3 py-1 text-xs rounded border border-[var(--border)] text-[var(--muted)] hover:text-[var(--foreground)] hover:bg-[var(--surface-raised)] disabled:opacity-50 transition-colors"
                >
                  Set global default
                </button>
              )}
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
}
