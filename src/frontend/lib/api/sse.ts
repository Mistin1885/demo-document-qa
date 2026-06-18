/**
 * SSE streaming helper for POST .../messages with stream=true.
 *
 * Uses fetch + ReadableStream (NOT EventSource) because:
 * 1. EventSource only supports GET — we need POST with a JSON body.
 * 2. We need AbortController support to stop generation.
 *
 * Wire format (one event block per message):
 *   event: <kind>\n
 *   data: <JSON>\n
 *   \n
 *
 * Supported kinds: token | citation | done | error
 *
 * Usage:
 *   const abort = new AbortController();
 *   for await (const evt of streamMessage(chatId, sessionId, question, abort.signal)) {
 *     if (evt.kind === "token") { ... }
 *     if (evt.kind === "done")  { break; }
 *   }
 *   // To cancel early: abort.abort()
 */

import { API_BASE_URL, ApiError } from "./client";
import type { SSEEvent } from "./types";

// ---------------------------------------------------------------------------
// Internal SSE text-stream parser
// ---------------------------------------------------------------------------

interface RawEvent {
  kind: string;
  rawData: string;
}

function* parseSseChunk(buffer: string): Generator<RawEvent> {
  // An SSE message block ends with \n\n
  const blocks = buffer.split("\n\n");
  for (const block of blocks) {
    if (!block.trim()) continue;

    let kind = "message";
    let data = "";

    for (const line of block.split("\n")) {
      if (line.startsWith("event:")) {
        kind = line.slice("event:".length).trim();
      } else if (line.startsWith("data:")) {
        data = line.slice("data:".length).trim();
      }
    }

    if (data) {
      yield { kind, rawData: data };
    }
  }
}

// ---------------------------------------------------------------------------
// Public async iterator
// ---------------------------------------------------------------------------

export async function* streamMessage(
  chatId: string,
  sessionId: string,
  question: string,
  signal?: AbortSignal
): AsyncIterable<SSEEvent> {
  const url = `${API_BASE_URL}/chats/${chatId}/sessions/${sessionId}/messages`;

  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, stream: true }),
    signal,
  });

  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = (await res.json()) as { detail?: string };
      if (typeof body.detail === "string") detail = body.detail;
    } catch {
      // ignore parse error
    }
    throw new ApiError(res.status, detail);
  }

  if (!res.body) {
    throw new ApiError(0, "Response body is empty");
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";

  try {
    while (true) {
      // Check abort between reads
      if (signal?.aborted) break;

      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      // Process complete event blocks from the buffer
      const lastDoubleNewline = buffer.lastIndexOf("\n\n");
      if (lastDoubleNewline === -1) continue;

      const toProcess = buffer.slice(0, lastDoubleNewline + 2);
      buffer = buffer.slice(lastDoubleNewline + 2);

      for (const raw of parseSseChunk(toProcess)) {
        let parsed: unknown;
        try {
          parsed = JSON.parse(raw.rawData);
        } catch {
          continue; // skip malformed data lines
        }

        // Yield typed SSE events
        switch (raw.kind) {
          case "token":
            yield { kind: "token", data: parsed } as SSEEvent;
            break;
          case "citation":
            yield { kind: "citation", data: parsed } as SSEEvent;
            break;
          case "done":
            yield { kind: "done", data: parsed } as SSEEvent;
            return; // stream complete
          case "error":
            yield { kind: "error", data: parsed } as SSEEvent;
            return;
          default:
            // Unknown event kinds are silently ignored
            break;
        }
      }
    }
  } finally {
    reader.cancel().catch(() => undefined);
  }
}
