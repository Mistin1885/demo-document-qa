"use client";

/**
 * TanStack Query v5 provider.
 *
 * Wrap the entire app in <QueryProvider> (done in app/layout.tsx).
 * QueryClient is created once per browser session with sensible defaults.
 */

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useState } from "react";

function makeQueryClient(): QueryClient {
  return new QueryClient({
    defaultOptions: {
      queries: {
        staleTime: 30_000,        // 30 s before refetch
        retry: 1,                 // one retry on failure
        refetchOnWindowFocus: false,
      },
    },
  });
}

// Keep a module-level reference so server components that access the same
// module don't get a new instance every render (SSR safety).
let browserQueryClient: QueryClient | undefined;

function getQueryClient(): QueryClient {
  if (typeof window === "undefined") {
    // Server: always create a new client (no singleton leak across requests).
    return makeQueryClient();
  }
  if (!browserQueryClient) {
    browserQueryClient = makeQueryClient();
  }
  return browserQueryClient;
}

export function QueryProvider({ children }: { children: React.ReactNode }) {
  // useState ensures the client is not recreated on every re-render.
  const [queryClient] = useState(() => getQueryClient());

  return (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
}
