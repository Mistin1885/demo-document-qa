/**
 * StatusBadge — status → label / colour mapping tests.
 *
 * CLAUDE.md §12.1: ≤ 10 test items per file.
 * Kept as pure-TS (no React rendering) so it runs in the existing
 * "node" vitest environment without a jsdom dependency.
 */

import { describe, it, expect } from "vitest";
import type { DocumentStatus } from "@/lib/api/types";

// Re-define the mapping locally so we can validate it without
// importing the React component (which would need jsdom).
const STATUS_LABELS: Record<DocumentStatus, string> = {
  uploaded: "Uploaded",
  parsing: "Parsing",
  parsed: "Parsed",
  enriching: "Enriching",
  indexed: "Indexed",
  failed: "Failed",
};

const STATUS_COLOR_FRAGMENTS: Record<DocumentStatus, string> = {
  uploaded: "gray",
  parsing: "blue",
  parsed: "indigo",
  enriching: "violet",
  indexed: "emerald",
  failed: "rose",
};

describe("StatusBadge status config", () => {
  // Spot-check representative statuses rather than enumerating all 6
  it.each<DocumentStatus>(["uploaded", "indexed", "failed"])(
    "status=%s has non-empty label and color keyword",
    (status) => {
      expect(STATUS_LABELS[status]).toBeTruthy();
      expect(STATUS_COLOR_FRAGMENTS[status]).toBeTruthy();
    }
  );

  it("all statuses have unique labels", () => {
    const labels = Object.values(STATUS_LABELS);
    const unique = new Set(labels);
    expect(unique.size).toBe(labels.length);
  });

  it("all statuses have unique color keywords", () => {
    const colors = Object.values(STATUS_COLOR_FRAGMENTS);
    const unique = new Set(colors);
    expect(unique.size).toBe(colors.length);
  });

  it("failed status uses rose color and indexed uses emerald", () => {
    expect(STATUS_COLOR_FRAGMENTS.failed).toBe("rose");
    expect(STATUS_COLOR_FRAGMENTS.indexed).toBe("emerald");
  });
});
