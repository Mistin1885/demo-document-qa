/**
 * Core HTTP client for the Paper Notebook Agent backend.
 *
 * All API calls go through this module.  Never import fetch wrappers directly
 * in UI components — use the typed helpers in chats.ts / sessions.ts / etc.
 *
 * Security rules (CLAUDE.md §9):
 * - No API keys are ever stored or logged here.
 * - Base URL comes exclusively from NEXT_PUBLIC_API_BASE_URL.
 */

import type { ApiErrorBody } from "./types";

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

// NEXT_PUBLIC_API_BASE_URL is baked in at build time. Leave it unset in
// Docker so the browser uses the relative /api/proxy prefix, which the
// Next.js Route Handler forwards to the backend container server-side
// (so the internal Docker hostname never reaches the browser).
export const API_BASE_URL =
  (typeof process !== "undefined" && process.env.NEXT_PUBLIC_API_BASE_URL) ||
  "/api/proxy";

// ---------------------------------------------------------------------------
// ApiError — thrown by every fetch helper on non-2xx responses
// ---------------------------------------------------------------------------

export class ApiError extends Error {
  constructor(
    public readonly status: number,
    public readonly detail: string,
    public readonly body?: unknown
  ) {
    super(`API ${status}: ${detail}`);
    this.name = "ApiError";
  }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

async function _parseError(res: Response): Promise<ApiError> {
  let body: unknown;
  try {
    body = await res.json();
  } catch {
    body = undefined;
  }

  let detail = res.statusText || "Unknown error";
  if (body && typeof body === "object") {
    const err = body as ApiErrorBody;
    if (typeof err.detail === "string") {
      detail = err.detail;
    } else if (Array.isArray(err.detail) && err.detail.length > 0) {
      detail = err.detail.map((e) => e.msg).join("; ");
    }
  }

  return new ApiError(res.status, detail, body);
}

function _url(path: string): string {
  // path must start with "/"
  return `${API_BASE_URL}${path}`;
}

// ---------------------------------------------------------------------------
// JSON GET / POST / PATCH / DELETE
// ---------------------------------------------------------------------------

export async function apiFetch<T>(
  path: string,
  options?: RequestInit
): Promise<T> {
  const res = await fetch(_url(path), {
    headers: { "Content-Type": "application/json", ...options?.headers },
    ...options,
  });

  if (!res.ok) {
    throw await _parseError(res);
  }

  // 204 No Content
  if (res.status === 204) {
    return undefined as unknown as T;
  }

  return res.json() as Promise<T>;
}

export function apiGet<T>(path: string, init?: RequestInit): Promise<T> {
  return apiFetch<T>(path, { method: "GET", ...init });
}

export function apiPost<T>(path: string, body: unknown, init?: RequestInit): Promise<T> {
  return apiFetch<T>(path, {
    method: "POST",
    body: JSON.stringify(body),
    ...init,
  });
}

export function apiPatch<T>(path: string, body: unknown, init?: RequestInit): Promise<T> {
  return apiFetch<T>(path, {
    method: "PATCH",
    body: JSON.stringify(body),
    ...init,
  });
}

export function apiDelete(path: string, init?: RequestInit): Promise<void> {
  return apiFetch<void>(path, { method: "DELETE", ...init });
}

// ---------------------------------------------------------------------------
// Multipart (file upload)
// ---------------------------------------------------------------------------

export async function apiUpload<T>(
  path: string,
  formData: FormData,
  signal?: AbortSignal
): Promise<T> {
  // Do NOT set Content-Type — let the browser set multipart boundary.
  const res = await fetch(_url(path), {
    method: "POST",
    body: formData,
    signal,
  });

  if (!res.ok) {
    throw await _parseError(res);
  }

  return res.json() as Promise<T>;
}

// ---------------------------------------------------------------------------
// Health check — used by app/page.tsx to display Backend status
// ---------------------------------------------------------------------------

export interface HealthResponse {
  status: string;
}

export async function healthCheck(): Promise<HealthResponse> {
  return apiGet<HealthResponse>("/health");
}
