/**
 * Tests for formatRelativeTime.
 * Uses a fixed "now" so results are deterministic.
 * ≤ 10 items (CLAUDE.md §12.1).
 */

import { describe, it, expect } from "vitest";
import { formatRelativeTime } from "./time";

// Fixed reference point: 2024-03-15T12:00:00Z
const NOW = new Date("2024-03-15T12:00:00Z");

describe("formatRelativeTime", () => {
  it("returns 'just now' for < 60 s ago", () => {
    const ts = new Date(NOW.getTime() - 30_000).toISOString();
    expect(formatRelativeTime(ts, NOW)).toBe("just now");
  });

  it("returns minutes for 2–59 min ago", () => {
    const ts = new Date(NOW.getTime() - 2 * 60_000).toISOString();
    const result = formatRelativeTime(ts, NOW);
    expect(result).toMatch(/2 min/);
  });

  it("returns hours for 3 hours ago", () => {
    const ts = new Date(NOW.getTime() - 3 * 3_600_000).toISOString();
    const result = formatRelativeTime(ts, NOW);
    expect(result).toMatch(/3 hour/);
  });

  it("returns days for 2 days ago", () => {
    const ts = new Date(NOW.getTime() - 2 * 86_400_000).toISOString();
    const result = formatRelativeTime(ts, NOW);
    expect(result).toMatch(/2 day/);
  });

  it("returns short date (same year) for >= 7 days", () => {
    // 10 days ago: 2024-03-05
    const ts = new Date(NOW.getTime() - 10 * 86_400_000).toISOString();
    const result = formatRelativeTime(ts, NOW);
    expect(result).toBe("Mar 5");
  });
});
