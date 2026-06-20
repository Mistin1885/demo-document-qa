"use client";

/**
 * useCurrentSessionId — read/write the active session from the URL search param.
 *
 * URL shape: /?chatId=<uuid>&sessionId=<uuid>
 *
 * Public API:
 *   const { sessionId, setSessionId } = useCurrentSessionId();
 *   sessionId          — string | null (null = no session selected)
 *   setSessionId(id)   — push /?chatId=...&sessionId=<id> (preserves chatId)
 *   setSessionId(null) — clears sessionId param (keeps chatId)
 */

import { useRouter, useSearchParams } from "next/navigation";
import { useCallback } from "react";

export interface UseCurrentSessionIdReturn {
  sessionId: string | null;
  setSessionId: (id: string | null, chatIdOverride?: string | null) => void;
}

export function buildSessionHref(chatId: string | null, sessionId: string | null): string {
  const params = new URLSearchParams();
  if (chatId) params.set("chatId", chatId);
  if (sessionId) params.set("sessionId", sessionId);
  const qs = params.toString();
  return qs ? `/?${qs}` : "/";
}

export function useCurrentSessionId(): UseCurrentSessionIdReturn {
  const searchParams = useSearchParams();
  const router = useRouter();

  const sessionId = searchParams.get("sessionId");

  const setSessionId = useCallback(
    (id: string | null, chatIdOverride?: string | null) => {
      const chatId = chatIdOverride !== undefined ? chatIdOverride : searchParams.get("chatId");
      router.push(buildSessionHref(chatId, id));
    },
    [router, searchParams]
  );

  return { sessionId, setSessionId };
}
