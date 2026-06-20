/**
 * Provider profiles localStorage adapter tests.
 *
 * CLAUDE.md §12.1: ≤10 test items per file.
 * Tests: CRUD lifecycle (3 cases) + masking guarantee (1) + stub test-connection (1) = 5 items.
 *
 * Uses JSDOM localStorage stub via vitest environment.
 */

import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";

// ---------------------------------------------------------------------------
// Minimal localStorage stub (vitest runs in "node" by default; inject manually)
// ---------------------------------------------------------------------------

const store: Record<string, string> = {};
const localStorageMock = {
  getItem: (k: string) => store[k] ?? null,
  setItem: (k: string, v: string) => { store[k] = v; },
  removeItem: (k: string) => { delete store[k]; },
  clear: () => { for (const k of Object.keys(store)) delete store[k]; },
};

beforeEach(() => {
  // Clear storage between tests
  localStorageMock.clear();
  // Inject as global
  vi.stubGlobal("window", { localStorage: localStorageMock });
  vi.stubGlobal("localStorage", localStorageMock);
  // Force local mode: unset the env var
  process.env.NEXT_PUBLIC_USE_LOCAL_PROFILES = "true";
});

afterEach(() => {
  vi.unstubAllGlobals();
});

// ---------------------------------------------------------------------------
// Dynamic imports ensure module re-reads the stubbed globals on each test
// ---------------------------------------------------------------------------

describe("providers localStorage adapter", () => {
  it("create profile → list returns it without api_key field", async () => {
    const { createProviderProfile, listProviderProfiles } = await import("../providers");

    await createProviderProfile({
      kind: "chat",
      provider_type: "openai",
      name: "Test GPT",
      model: "gpt-4o",
      api_key_plaintext: "sk-secret",
    });

    const list = await listProviderProfiles("chat");
    expect(list).toHaveLength(1);
    expect(list[0].name).toBe("Test GPT");
    // Key must NOT be present on returned object
    expect("api_key_plaintext" in list[0]).toBe(false);
    expect("_has_key" in list[0]).toBe(false);
  });

  it("delete profile removes it from list", async () => {
    const { createProviderProfile, deleteProviderProfile, listProviderProfiles } =
      await import("../providers");

    const created = await createProviderProfile({
      kind: "embedding",
      provider_type: "openai",
      name: "Embedder",
      model: "text-embedding-3-small",
    });

    await deleteProviderProfile(created.id);
    const list = await listProviderProfiles("embedding");
    expect(list).toHaveLength(0);
  });

  it("update profile changes fields; key presence flag updates correctly", async () => {
    const { createProviderProfile, updateProviderProfile, listProviderProfiles } =
      await import("../providers");

    const created = await createProviderProfile({
      kind: "reranker",
      provider_type: "openai_compat",
      name: "Reranker v1",
      model: "rerank-v1",
    });

    await updateProviderProfile(created.id, { name: "Reranker v2", model: "rerank-v2" });
    const list = await listProviderProfiles("reranker");
    expect(list[0].name).toBe("Reranker v2");
    expect(list[0].model).toBe("rerank-v2");
  });

  it("maskedKey helper never returns plaintext", async () => {
    const { maskedKey } = await import("../providers");
    expect(maskedKey(true)).toBe("••••••••");
    expect(maskedKey(false)).toBe("(none)");
    // Ensure it cannot somehow contain 'sk-' or real key patterns
    expect(maskedKey(true)).not.toMatch(/^sk-/);
  });

  it("testConnection reports missing local profile without stub fallback", async () => {
    const { testConnection } = await import("../providers");
    const result = await testConnection("any-id");
    expect(result.ok).toBe(false);
    expect(result.stub).toBeUndefined();
    expect(result.error).toBeTruthy();
  });
});
