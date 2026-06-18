"use client";

/**
 * TestConnectionButton — fires test_connection and renders result inline.
 *
 * In local-mode the backend API is not wired; returns a stub badge
 * indicating "stub: backend not wired".
 */

import { useState } from "react";
import { useTestConnection } from "@/lib/queries/providers";
import type { TestConnectionResult } from "@/lib/api/providers";

interface Props {
  profileId: string;
}

type Status = "idle" | "loading" | "success" | "error";

export function TestConnectionButton({ profileId }: Props) {
  const { mutateAsync } = useTestConnection();
  const [status, setStatus] = useState<Status>("idle");
  const [result, setResult] = useState<TestConnectionResult | null>(null);

  async function handleClick() {
    setStatus("loading");
    setResult(null);
    try {
      const res = await mutateAsync(profileId);
      setResult(res);
      setStatus(res.ok ? "success" : "error");
    } catch (err) {
      setResult({
        ok: false,
        error: err instanceof Error ? err.message : "Unknown error",
      });
      setStatus("error");
    }
  }

  return (
    <div className="flex flex-col gap-1">
      <button
        onClick={handleClick}
        disabled={status === "loading"}
        className="px-3 py-1 text-xs rounded border border-[var(--border)] text-[var(--foreground)] bg-[var(--surface)] hover:bg-[var(--surface-raised)] disabled:opacity-50 transition-colors"
      >
        {status === "loading" ? "Testing…" : "Test connection"}
      </button>

      {result && (
        <div
          className={`flex flex-wrap items-center gap-1.5 text-xs px-2 py-1 rounded ${
            result.ok
              ? "bg-green-900/30 text-green-300"
              : result.stub
              ? "bg-yellow-900/30 text-yellow-300"
              : "bg-red-900/30 text-red-300"
          }`}
        >
          {result.stub ? (
            <span>stub: backend not wired</span>
          ) : result.ok ? (
            <>
              <span className="font-semibold">OK</span>
              {result.model && <span>model: {result.model}</span>}
              {result.latency_ms != null && (
                <span>{result.latency_ms}ms</span>
              )}
            </>
          ) : (
            <>
              <span className="font-semibold">Failed</span>
              {result.error && <span>{result.error}</span>}
            </>
          )}
        </div>
      )}
    </div>
  );
}
