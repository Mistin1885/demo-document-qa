/**
 * Per-chat generation preferences (max output tokens / temperature /
 * context window) persisted in localStorage.
 *
 * Key shape: `pna_gen_prefs_{chatId}` → JSON of GenerationOverrides.
 * A `default` key (`pna_gen_prefs_default`) supplies the fallback when no
 * per-chat record exists.
 */
"use client";

import { useCallback, useEffect, useState } from "react";

import type { GenerationOverrides } from "@/lib/api/messages";
import { localGet, localSet } from "@/lib/storage/local";

const DEFAULT_KEY = "pna_gen_prefs_default";
const chatKey = (chatId: string | null) =>
  chatId ? `pna_gen_prefs_${chatId}` : DEFAULT_KEY;

/** Sensible UI defaults — empty means "let backend env defaults win". */
const EMPTY_PREFS: GenerationOverrides = {};

export interface GenerationPrefsApi {
  prefs: GenerationOverrides;
  setPrefs: (next: GenerationOverrides) => void;
  reset: () => void;
}

export function useGenerationPrefs(chatId: string | null): GenerationPrefsApi {
  const key = chatKey(chatId);
  const [prefs, setPrefsState] = useState<GenerationOverrides>(EMPTY_PREFS);

  // Hydrate from localStorage when chat changes
  useEffect(() => {
    const stored =
      localGet<GenerationOverrides>(key) ??
      localGet<GenerationOverrides>(DEFAULT_KEY) ??
      EMPTY_PREFS;
    setPrefsState(stored);
  }, [key]);

  const setPrefs = useCallback(
    (next: GenerationOverrides) => {
      // Drop undefined / null / empty-string fields so the backend can apply
      // env-level defaults for omitted keys.
      const cleaned: GenerationOverrides = {};
      if (typeof next.max_answer_tokens === "number") {
        cleaned.max_answer_tokens = next.max_answer_tokens;
      }
      if (typeof next.temperature === "number") {
        cleaned.temperature = next.temperature;
      }
      if (typeof next.context_window === "number") {
        cleaned.context_window = next.context_window;
      }
      setPrefsState(cleaned);
      localSet(key, cleaned);
    },
    [key]
  );

  const reset = useCallback(() => {
    setPrefsState(EMPTY_PREFS);
    localSet(key, EMPTY_PREFS);
  }, [key]);

  return { prefs, setPrefs, reset };
}
