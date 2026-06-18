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
  setSessionId: (id: string | null) => void;
}

export function useCurrentSessionId(): UseCurrentSessionIdReturn {
  const searchParams = useSearchParams();
  const router = useRouter();

  const sessionId = searchParams.get("sessionId");

  const setSessionId = useCallback(
    (id: string | null) => {
      const chatId = searchParams.get("chatId");
      const params = new URLSearchParams();
      if (chatId) params.set("chatId", chatId);
      if (id) params.set("sessionId", id);
      const qs = params.toString();
      router.push(qs ? `/?${qs}` : "/");
    },
    [router, searchParams]
  );

  return { sessionId, setSessionId };
}
