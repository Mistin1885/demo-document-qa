/**
 * SSE parser unit tests.
 *
 * Mocks fetch + ReadableStream to simulate backend SSE chunks.
 * CLAUDE.md §12.1: ≤10 test items per file.
 */

import { describe, it, expect, vi, beforeEach } from "vitest";
import { ApiError } from "../client";

// ---------------------------------------------------------------------------
// Helpers to build a mock ReadableStream from SSE text
// ---------------------------------------------------------------------------

function sseBlock(kind: string, data: unknown): string {
  return `event: ${kind}\ndata: ${JSON.stringify(data)}\n\n`;
}

function makeStream(chunks: string[]): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();
  let i = 0;
  return new ReadableStream({
    pull(controller) {
      if (i < chunks.length) {
        controller.enqueue(encoder.encode(chunks[i++]));
      } else {
        controller.close();
      }
    },
  });
}

function mockFetch(status: number, body: ReadableStream<Uint8Array> | null) {
  vi.stubGlobal(
    "fetch",
    vi.fn().mockResolvedValue({
      ok: status >= 200 && status < 300,
      status,
      statusText: status === 200 ? "OK" : "Error",
      body,
      json: async () => ({ detail: "error detail" }),
    })
  );
}

// ---------------------------------------------------------------------------
// Tests (5 parametrized cases — well within the ≤10 cap)
// ---------------------------------------------------------------------------

describe("streamMessage SSE parsing", () => {
  beforeEach(() => {
    vi.unstubAllGlobals();
  });

  it("yields token events from a single chunk", async () => {
    const chunk = sseBlock("token", { delta: "Hello", index: 0 });
    mockFetch(200, makeStream([chunk + sseBlock("done", { answer: "Hello", citations: [], documents_used: [], coverage: 1, uncertainty: [], session_id: "s1", message_id: "m1", debug_trace: null })]));

    const { streamMessage } = await import("../sse");
    const events = [];
    for await (const evt of streamMessage("chat1", "sess1", "q?")) {
      events.push(evt);
    }

    expect(events[0]).toMatchObject({ kind: "token", data: { delta: "Hello", index: 0 } });
    expect(events[events.length - 1].kind).toBe("done");
  });

  it("stops iteration after receiving done event", async () => {
    const chunks = [
      sseBlock("token", { delta: "A", index: 0 }),
      sseBlock("done", { answer: "A", citations: [], documents_used: [], coverage: 1, uncertainty: [], session_id: "s1", message_id: "m1", debug_trace: null }),
      sseBlock("token", { delta: "B", index: 1 }), // should not be yielded
    ];
    mockFetch(200, makeStream(chunks));

    const { streamMessage } = await import("../sse");
    const events = [];
    for await (const evt of streamMessage("chat1", "sess1", "q?")) {
      events.push(evt);
    }

    const kinds = events.map((e) => e.kind);
    expect(kinds).not.toContain("B");
    expect(kinds[kinds.length - 1]).toBe("done");
  });

  it("throws ApiError on non-2xx response", async () => {
    mockFetch(404, null);

    const { streamMessage } = await import("../sse");
    await expect(async () => {
      for await (const _event of streamMessage("chat1", "sess1", "q?")) {
        // no-op
      }
    }).rejects.toBeInstanceOf(ApiError);
  });

  it("yields error event and stops", async () => {
    const chunks = [sseBlock("error", { code: "STOPPED", detail: "client disconnected" })];
    mockFetch(200, makeStream(chunks));

    const { streamMessage } = await import("../sse");
    const events = [];
    for await (const evt of streamMessage("chat1", "sess1", "q?")) {
      events.push(evt);
    }

    expect(events).toHaveLength(1);
    expect(events[0]).toMatchObject({ kind: "error", data: { code: "STOPPED" } });
  });

  it("handles multi-chunk SSE blocks split across reads", async () => {
    // Split the event block across two reads to test buffer reassembly
    const full = sseBlock("token", { delta: "Hi", index: 0 });
    const mid = Math.floor(full.length / 2);
    const part1 = full.slice(0, mid);
    const part2 = full.slice(mid) + sseBlock("done", { answer: "Hi", citations: [], documents_used: [], coverage: 1, uncertainty: [], session_id: "s1", message_id: "m1", debug_trace: null });

    mockFetch(200, makeStream([part1, part2]));

    const { streamMessage } = await import("../sse");
    const events = [];
    for await (const evt of streamMessage("chat1", "sess1", "q?")) {
      events.push(evt);
    }

    expect(events.some((e) => e.kind === "token")).toBe(true);
    expect(events[events.length - 1].kind).toBe("done");
  });
});
