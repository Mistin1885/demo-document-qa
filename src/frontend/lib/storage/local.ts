/**
 * Typed localStorage wrapper with SSR safeguard.
 *
 * - All reads return `null` when running server-side.
 * - JSON serialise/parse with type assertion; caller owns the type contract.
 * - Never log or expose raw values containing API keys.
 */

const isBrowser = (): boolean => typeof window !== "undefined";

export function localGet<T>(key: string): T | null {
  if (!isBrowser()) return null;
  try {
    const raw = window.localStorage.getItem(key);
    if (raw === null) return null;
    return JSON.parse(raw) as T;
  } catch {
    return null;
  }
}

export function localSet<T>(key: string, value: T): void {
  if (!isBrowser()) return;
  try {
    window.localStorage.setItem(key, JSON.stringify(value));
  } catch {
    // quota exceeded or private-browsing restriction — silently ignore
  }
}

export function localRemove(key: string): void {
  if (!isBrowser()) return;
  try {
    window.localStorage.removeItem(key);
  } catch {
    // ignore
  }
}

export function localGetOrDefault<T>(key: string, fallback: T): T {
  const v = localGet<T>(key);
  return v === null ? fallback : v;
}
