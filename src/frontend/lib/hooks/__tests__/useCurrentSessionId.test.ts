import { describe, expect, it } from "vitest";
import { buildSessionHref } from "../useCurrentSessionId";

describe("buildSessionHref", () => {
  it("pins a newly-created session to the explicit chat id", () => {
    expect(buildSessionHref("chat-a", "session-new")).toBe(
      "/?chatId=chat-a&sessionId=session-new"
    );
  });

  it("clears only the session id while preserving chat selection", () => {
    expect(buildSessionHref("chat-a", null)).toBe("/?chatId=chat-a");
  });

  it("returns root when chat and session are both absent", () => {
    expect(buildSessionHref(null, null)).toBe("/");
  });
});
