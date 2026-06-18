"use client";

/**
 * ProviderProfileForm — create or edit a provider profile.
 *
 * Fields: name, provider_type, base_url, model, api_key_plaintext (password),
 *         context_window, is_default.
 *
 * Security (CLAUDE.md §9):
 *   - api_key_plaintext uses type="password".
 *   - Existing key is shown as masked placeholder; submitting empty leaves key unchanged.
 *   - Key is NEVER logged or returned from list view.
 */

import { useState } from "react";
import type { ProviderKind, ProviderType } from "@/lib/api/types";
import type { ProviderProfileCreate, ProviderProfileRead } from "@/lib/api/providers";

interface Props {
  kind: ProviderKind;
  existing?: ProviderProfileRead;
  onSubmit: (data: ProviderProfileCreate) => void;
  onCancel: () => void;
  loading?: boolean;
}

const PROVIDER_TYPES: { value: ProviderType; label: string }[] = [
  { value: "openai", label: "OpenAI" },
  { value: "gemini_native", label: "Gemini Native" },
  { value: "gemini_compat", label: "Gemini OpenAI-compat" },
  { value: "openai_compat", label: "Generic OpenAI-compat" },
  { value: "vllm", label: "Self-hosted vLLM" },
];

export function ProviderProfileForm({ kind, existing, onSubmit, onCancel, loading }: Props) {
  const [name, setName] = useState(existing?.name ?? "");
  const [providerType, setProviderType] = useState<ProviderType>(
    existing?.provider_type ?? "openai"
  );
  const [baseUrl, setBaseUrl] = useState(existing?.base_url ?? "");
  const [model, setModel] = useState(existing?.model ?? "");
  const [apiKey, setApiKey] = useState("");
  const [contextWindow, setContextWindow] = useState(
    existing?.context_window != null ? String(existing.context_window) : ""
  );
  const [isDefault, setIsDefault] = useState(existing?.is_default ?? false);
  const [errors, setErrors] = useState<Record<string, string>>({});

  function validate(): boolean {
    const e: Record<string, string> = {};
    if (!name.trim()) e.name = "Name is required";
    if (!model.trim()) e.model = "Model is required";
    setErrors(e);
    return Object.keys(e).length === 0;
  }

  function handleSubmit(ev: React.FormEvent) {
    ev.preventDefault();
    if (!validate()) return;
    const payload: ProviderProfileCreate = {
      kind,
      provider_type: providerType,
      name: name.trim(),
      base_url: baseUrl.trim() || null,
      model: model.trim(),
      api_key_plaintext: apiKey || null,
      context_window: contextWindow ? Number(contextWindow) : null,
      is_default: isDefault,
    };
    onSubmit(payload);
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      {/* Name */}
      <div>
        <label className="block text-xs font-medium text-[var(--muted)] mb-1">
          Name <span className="text-red-400">*</span>
        </label>
        <input
          type="text"
          value={name}
          onChange={(e) => setName(e.target.value)}
          className="w-full px-3 py-2 text-sm rounded bg-[var(--surface)] border border-[var(--border)] text-[var(--foreground)] focus:outline-none focus:border-[var(--accent)]"
          placeholder="e.g. GPT-4o main"
        />
        {errors.name && <p className="text-xs text-red-400 mt-1">{errors.name}</p>}
      </div>

      {/* Provider type */}
      <div>
        <label className="block text-xs font-medium text-[var(--muted)] mb-1">
          Provider type
        </label>
        <select
          value={providerType}
          onChange={(e) => setProviderType(e.target.value as ProviderType)}
          className="w-full px-3 py-2 text-sm rounded bg-[var(--surface)] border border-[var(--border)] text-[var(--foreground)] focus:outline-none focus:border-[var(--accent)]"
        >
          {PROVIDER_TYPES.map((pt) => (
            <option key={pt.value} value={pt.value}>
              {pt.label}
            </option>
          ))}
        </select>
      </div>

      {/* Base URL */}
      <div>
        <label className="block text-xs font-medium text-[var(--muted)] mb-1">
          Base URL <span className="text-[var(--muted)]">(optional)</span>
        </label>
        <input
          type="url"
          value={baseUrl}
          onChange={(e) => setBaseUrl(e.target.value)}
          className="w-full px-3 py-2 text-sm rounded bg-[var(--surface)] border border-[var(--border)] text-[var(--foreground)] focus:outline-none focus:border-[var(--accent)]"
          placeholder="https://api.openai.com/v1"
        />
      </div>

      {/* Model */}
      <div>
        <label className="block text-xs font-medium text-[var(--muted)] mb-1">
          Model <span className="text-red-400">*</span>
        </label>
        <input
          type="text"
          value={model}
          onChange={(e) => setModel(e.target.value)}
          className="w-full px-3 py-2 text-sm rounded bg-[var(--surface)] border border-[var(--border)] text-[var(--foreground)] focus:outline-none focus:border-[var(--accent)]"
          placeholder="e.g. gpt-4o"
        />
        {errors.model && <p className="text-xs text-red-400 mt-1">{errors.model}</p>}
      </div>

      {/* API Key */}
      <div>
        <label className="block text-xs font-medium text-[var(--muted)] mb-1">
          API Key{" "}
          {existing && (
            <span className="text-[var(--muted)]">
              (leave blank to keep existing)
            </span>
          )}
        </label>
        <input
          type="password"
          value={apiKey}
          onChange={(e) => setApiKey(e.target.value)}
          autoComplete="new-password"
          className="w-full px-3 py-2 text-sm rounded bg-[var(--surface)] border border-[var(--border)] text-[var(--foreground)] focus:outline-none focus:border-[var(--accent)]"
          placeholder={existing ? "••••••••" : "sk-…"}
        />
        <p className="text-xs text-yellow-500/70 mt-1">
          Keys are stored in your browser localStorage; this is only for the demo.
        </p>
      </div>

      {/* Context window */}
      <div>
        <label className="block text-xs font-medium text-[var(--muted)] mb-1">
          Context window <span className="text-[var(--muted)]">(tokens, optional)</span>
        </label>
        <input
          type="number"
          value={contextWindow}
          onChange={(e) => setContextWindow(e.target.value)}
          min={0}
          className="w-full px-3 py-2 text-sm rounded bg-[var(--surface)] border border-[var(--border)] text-[var(--foreground)] focus:outline-none focus:border-[var(--accent)]"
          placeholder="128000"
        />
      </div>

      {/* Is default */}
      <label className="flex items-center gap-2 cursor-pointer">
        <input
          type="checkbox"
          checked={isDefault}
          onChange={(e) => setIsDefault(e.target.checked)}
          className="accent-[var(--accent)]"
        />
        <span className="text-sm text-[var(--foreground)]">
          Set as global default for {kind}
        </span>
      </label>

      {/* Actions */}
      <div className="flex gap-2 pt-2">
        <button
          type="submit"
          disabled={loading}
          className="flex-1 py-2 text-sm font-medium rounded bg-[var(--accent)] text-white hover:bg-[var(--accent-hover)] disabled:opacity-50 transition-colors"
        >
          {loading ? "Saving…" : existing ? "Update" : "Create"}
        </button>
        <button
          type="button"
          onClick={onCancel}
          className="flex-1 py-2 text-sm font-medium rounded bg-[var(--surface-raised)] text-[var(--foreground)] hover:bg-[var(--border)] transition-colors"
        >
          Cancel
        </button>
      </div>
    </form>
  );
}
