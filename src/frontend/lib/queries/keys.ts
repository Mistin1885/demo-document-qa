/**
 * TanStack Query v5 key factory.
 *
 * All query keys are derived from this factory to ensure consistency.
 * Never hardcode query key strings in components — always import from here.
 */

export const queryKeys = {
  // Chats
  chats: () => ["chats"] as const,
  chat: (chatId: string) => ["chats", chatId] as const,

  // Sessions (scoped to a chat)
  sessions: (chatId: string) => ["chats", chatId, "sessions"] as const,
  session: (chatId: string, sessionId: string) =>
    ["chats", chatId, "sessions", sessionId] as const,

  // Documents (scoped to a chat)
  documents: (chatId: string) => ["chats", chatId, "documents"] as const,
  document: (chatId: string, docId: string) =>
    ["chats", chatId, "documents", docId] as const,

  // Messages (scoped to a session)
  messages: (chatId: string, sessionId: string) =>
    ["chats", chatId, "sessions", sessionId, "messages"] as const,

  // Manifest (scoped to a chat)
  manifest: (chatId: string) => ["chats", chatId, "manifest"] as const,

  // Facts (scoped to a chat)
  facts: (chatId: string) => ["chats", chatId, "facts"] as const,
  fact: (chatId: string, factId: string) =>
    ["chats", chatId, "facts", factId] as const,

  // Provider profiles (global, scoped by kind)
  providerProfiles: (kind: string) => ["provider_profiles", kind] as const,
} as const;
