"use client";

/**
 * useCurrentChatId — read/write the active chat from the URL search param.
 *
 * URL shape:  /?chatId=<uuid>
 *
 * Public API (used by Phase 8.3 / 8.4 as well):
 *   const { chatId, setChatId } = useCurrentChatId();
 *   chatId   — string | null (null = no chat selected)
 *   setChatId(id)   — navigate to /?chatId=<id>  (shallow)
 *   setChatId(null) — navigate to /              (clears selection)
 */

import { useRouter, useSearchParams } from "next/navigation";
import { useCallback } from "react";

export interface UseCurrentChatIdReturn {
  chatId: string | null;
  setChatId: (id: string | null) => void;
}

export function useCurrentChatId(): UseCurrentChatIdReturn {
  const searchParams = useSearchParams();
  const router = useRouter();

  const chatId = searchParams.get("chatId");

  const setChatId = useCallback(
    (id: string | null) => {
      if (id) {
        router.push(`/?chatId=${encodeURIComponent(id)}`);
      } else {
        router.push("/");
      }
    },
    [router]
  );

  return { chatId, setChatId };
}
