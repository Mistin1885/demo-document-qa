/**
 * Provider profile API helpers.
 *
 * Modes:
 *   LOCAL (default, NEXT_PUBLIC_USE_LOCAL_PROFILES=true or unset):
 *     All CRUD goes to localStorage. Test connection returns a stub error.
 *   REMOTE (NEXT_PUBLIC_USE_LOCAL_PROFILES=false):
 *     Delegates to the backend REST API once it is implemented.
 *
 * Switch point: set NEXT_PUBLIC_USE_LOCAL_PROFILES=false in .env.local and
 *   add the backend routes POST/GET/PATCH/DELETE /provider_profiles.
 *
 * Security (CLAUDE.md §9):
 *   - api_key_plaintext is never logged.
 *   - ProviderProfileRead (returned from list/get) never includes the key.
 *   - Local storage holds the key only in LocalProviderEntry which is an
 *     internal type — it is never returned to UI components directly.
 */

import { localGet, localSet } from "@/lib/storage/local";
import { apiGet, apiPost, apiPatch, apiDelete } from "./client";
import type { ProviderKind, ProviderType } from "./types";

// ---------------------------------------------------------------------------
// Public shape mirroring backend ProviderProfileRead (no key field)
// ---------------------------------------------------------------------------

export interface ProviderProfileRead {
  id: string;
  kind: ProviderKind;
  provider_type: ProviderType;
  name: string;
  base_url: string | null;
  model: string;
  context_window: number | null;
  is_default: boolean;
  created_at: string;
  updated_at: string;
}

export interface ProviderProfileCreate {
  kind: ProviderKind;
  provider_type: ProviderType;
  name: string;
  base_url?: string | null;
  model: string;
  api_key_plaintext?: string | null;
  context_window?: number | null;
  is_default?: boolean;
}

export interface ProviderProfileUpdate {
  name?: string | null;
  base_url?: string | null;
  model?: string | null;
  api_key_plaintext?: string | null;
  context_window?: number | null;
  is_default?: boolean | null;
}

export interface TestConnectionResult {
  ok: boolean;
  model?: string;
  latency_ms?: number;
  error?: string;
  stub?: boolean;
}

export interface SetChatDefaultBody {
  kind: ProviderKind;
  profile_id: string;
}

// ---------------------------------------------------------------------------
// Internal local-storage entry (includes masked key flag, never returns key)
// ---------------------------------------------------------------------------

interface LocalProviderEntry extends ProviderProfileRead {
  _has_key: boolean; // true when a key was provided at create/update time
}

const LS_KEY = "pna_provider_profiles";

function isLocalMode(): boolean {
  if (typeof process === "undefined") return true;
  const v = process.env.NEXT_PUBLIC_USE_LOCAL_PROFILES;
  return v === undefined || v === "" || v === "true";
}

// ---------------------------------------------------------------------------
// LocalStorage adapter
// ---------------------------------------------------------------------------

function lsAll(): LocalProviderEntry[] {
  return localGet<LocalProviderEntry[]>(LS_KEY) ?? [];
}

function lsSave(entries: LocalProviderEntry[]): void {
  localSet(LS_KEY, entries);
}

function nowIso(): string {
  return new Date().toISOString();
}

function genId(): string {
  // crypto.randomUUID with SSR fallback
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  return `local-${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

// ---------------------------------------------------------------------------
// Public API functions
// ---------------------------------------------------------------------------

export async function listProviderProfiles(
  kind: ProviderKind
): Promise<ProviderProfileRead[]> {
  if (isLocalMode()) {
    return lsAll()
      .filter((e) => e.kind === kind)
      .map(({ _has_key: _k, ...rest }) => rest);
  }
  return apiGet<ProviderProfileRead[]>(`/provider_profiles?kind=${kind}`);
}

export async function createProviderProfile(
  body: ProviderProfileCreate
): Promise<ProviderProfileRead> {
  if (isLocalMode()) {
    const now = nowIso();
    const entry: LocalProviderEntry = {
      id: genId(),
      kind: body.kind,
      provider_type: body.provider_type,
      name: body.name,
      base_url: body.base_url ?? null,
      model: body.model,
      context_window: body.context_window ?? null,
      is_default: body.is_default ?? false,
      created_at: now,
      updated_at: now,
      _has_key: !!(body.api_key_plaintext && body.api_key_plaintext.trim()),
    };
    // If setting as default, unset others in same kind
    let all = lsAll();
    if (entry.is_default) {
      all = all.map((e) =>
        e.kind === entry.kind ? { ...e, is_default: false } : e
      );
    }
    all.push(entry);
    lsSave(all);
    const { _has_key: _k, ...out } = entry;
    return out;
  }
  return apiPost<ProviderProfileRead>("/provider_profiles", body);
}

export async function updateProviderProfile(
  id: string,
  body: ProviderProfileUpdate
): Promise<ProviderProfileRead> {
  if (isLocalMode()) {
    let all = lsAll();
    const idx = all.findIndex((e) => e.id === id);
    if (idx === -1) throw new Error(`Profile ${id} not found`);
    const existing = all[idx];
    const updated: LocalProviderEntry = {
      ...existing,
      name: body.name ?? existing.name,
      base_url: body.base_url !== undefined ? body.base_url : existing.base_url,
      model: body.model ?? existing.model,
      context_window:
        body.context_window !== undefined
          ? body.context_window
          : existing.context_window,
      is_default:
        body.is_default !== undefined && body.is_default !== null
          ? body.is_default
          : existing.is_default,
      _has_key:
        body.api_key_plaintext !== undefined && body.api_key_plaintext !== null
          ? !!(body.api_key_plaintext.trim())
          : existing._has_key,
      updated_at: nowIso(),
    };
    if (updated.is_default && !existing.is_default) {
      all = all.map((e) =>
        e.kind === updated.kind && e.id !== id ? { ...e, is_default: false } : e
      );
    }
    all[all.findIndex((e) => e.id === id)] = updated;
    lsSave(all);
    const { _has_key: _k, ...out } = updated;
    return out;
  }
  return apiPatch<ProviderProfileRead>(`/provider_profiles/${id}`, body);
}

export async function deleteProviderProfile(id: string): Promise<void> {
  if (isLocalMode()) {
    lsSave(lsAll().filter((e) => e.id !== id));
    return;
  }
  return apiDelete(`/provider_profiles/${id}`);
}

export async function testConnection(id: string): Promise<TestConnectionResult> {
  if (isLocalMode()) {
    return {
      ok: false,
      error: "Backend provider profiles API not yet implemented",
      stub: true,
    };
  }
  return apiPost<TestConnectionResult>(`/provider_profiles/${id}/test_connection`, {});
}

export async function setGlobalDefault(id: string): Promise<void> {
  if (isLocalMode()) {
    const all = lsAll();
    const target = all.find((e) => e.id === id);
    if (!target) throw new Error(`Profile ${id} not found`);
    lsSave(
      all.map((e) =>
        e.kind === target.kind ? { ...e, is_default: e.id === id } : e
      )
    );
    return;
  }
  return apiPost<void>(`/provider_profiles/${id}/set_global_default`, {});
}

export async function setChatDefault(
  chatId: string,
  body: SetChatDefaultBody
): Promise<void> {
  if (isLocalMode()) {
    // Store chat-level defaults in localStorage
    const key = `pna_chat_defaults_${chatId}`;
    const current = localGet<Record<string, string>>(key) ?? {};
    current[body.kind] = body.profile_id;
    localSet(key, current);
    return;
  }
  return apiPost<void>(`/chats/${chatId}/default_profiles`, body);
}

/** Returns the masked key display string */
export function maskedKey(hasKey: boolean): string {
  return hasKey ? "••••••••" : "(none)";
}
