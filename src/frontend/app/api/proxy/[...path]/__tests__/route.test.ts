import { describe, expect, it } from "vitest";
import { buildBackendTarget } from "../route-utils";

describe("buildBackendTarget", () => {
  it("preserves a trailing slash before query string", () => {
    expect(
      buildBackendTarget(
        "http://backend:8000",
        "/api/proxy/chats/",
        "?limit=10"
      )
    ).toBe("http://backend:8000/chats/?limit=10");
  });

  it("does not add a trailing slash when the request does not have one", () => {
    expect(buildBackendTarget("http://backend:8000", "/api/proxy/chats")).toBe(
      "http://backend:8000/chats"
    );
  });

  it("normalizes backend base trailing slash", () => {
    expect(buildBackendTarget("http://backend:8000/", "/api/proxy/health")).toBe(
      "http://backend:8000/health"
    );
  });
});
