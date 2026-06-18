"use client";

/**
 * TanStack Query v5 hooks for Provider Profile resources.
 *
 * In local-mode, all mutations write to localStorage via the providers adapter.
 * Once the backend API is wired (NEXT_PUBLIC_USE_LOCAL_PROFILES=false), the
 * same hooks delegate to real HTTP calls — no component changes needed.
 */

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  createProviderProfile,
  deleteProviderProfile,
  listProviderProfiles,
  setGlobalDefault,
  setChatDefault,
  testConnection,
  updateProviderProfile,
} from "@/lib/api/providers";
import { queryKeys } from "@/lib/queries/keys";
import type { ProviderKind } from "@/lib/api/types";
import type {
  ProviderProfileCreate,
  ProviderProfileUpdate,
  SetChatDefaultBody,
} from "@/lib/api/providers";

// ---------------------------------------------------------------------------
// Queries
// ---------------------------------------------------------------------------

export function useProviderProfiles(kind: ProviderKind) {
  return useQuery({
    queryKey: queryKeys.providerProfiles(kind),
    queryFn: () => listProviderProfiles(kind),
    staleTime: 0, // local-mode changes go to localStorage, always fresh
  });
}

// ---------------------------------------------------------------------------
// Mutations
// ---------------------------------------------------------------------------

export function useCreateProfile(kind: ProviderKind) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: ProviderProfileCreate) => createProviderProfile(body),
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: queryKeys.providerProfiles(kind) });
    },
  });
}

export function useUpdateProfile(kind: ProviderKind) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ id, body }: { id: string; body: ProviderProfileUpdate }) =>
      updateProviderProfile(id, body),
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: queryKeys.providerProfiles(kind) });
    },
  });
}

export function useDeleteProfile(kind: ProviderKind) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => deleteProviderProfile(id),
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: queryKeys.providerProfiles(kind) });
    },
  });
}

export function useSetGlobalDefault(kind: ProviderKind) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => setGlobalDefault(id),
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: queryKeys.providerProfiles(kind) });
    },
  });
}

export function useSetChatDefault() {
  return useMutation({
    mutationFn: ({ chatId, body }: { chatId: string; body: SetChatDefaultBody }) =>
      setChatDefault(chatId, body),
  });
}

export function useTestConnection() {
  return useMutation({
    mutationFn: (id: string) => testConnection(id),
  });
}
