/**
 * TypeScript types mirroring the backend Pydantic v2 domain models.
 *
 * Conventions:
 * - All UUID fields are `string` (JSON-serialised uuid.UUID).
 * - All timestamp fields are `string` (ISO 8601 from datetime).
 * - Literal union types mirror Python `Literal[...]` fields.
 * - No `any` — every structured sub-field has its own named type.
 */

// ---------------------------------------------------------------------------
// Enumerations
// ---------------------------------------------------------------------------

export type ProviderKind = "chat" | "embedding" | "reranker";
export type ProviderType =
  | "openai"
  | "gemini_native"
  | "gemini_compat"
  | "openai_compat"
  | "vllm";
export type MessageRole = "user" | "assistant" | "system" | "tool";
export type DocumentSourceType = "upload" | "arxiv" | "url";
export type DocumentStatus =
  | "uploaded"
  | "parsing"
  | "parsed"
  | "enriching"
  | "indexed"
  | "failed";
export type FactKind =
  | "metric"
  | "benchmark"
  | "dataset"
  | "hyperparameter"
  | "ablation"
  | "other";
export type IngestionState = "pending" | "running" | "succeeded" | "failed";

// ---------------------------------------------------------------------------
// Sub-structures
// ---------------------------------------------------------------------------

export interface Citation {
  citation_id: string;
  chat_id: string;
  document_id: string;
  document_name: string;
  page_start: number;
  page_end: number;
  section_title: string | null;
  source_node_id: string | null;
  excerpt: string;
}

export interface ToolTraceStep {
  tool_name: string;
  status: "ok" | "overflow" | "error";
  token_estimate: number | null;
  note: string | null;
}

export interface ToolTrace {
  steps: ToolTraceStep[];
  total_rounds: number;
  token_count_is_estimate: boolean;
}

export interface FactValue {
  raw: string | null;
  numeric: number | null;
  items: string[] | null;
  [key: string]: string | number | boolean | string[] | null | undefined;
}

// ---------------------------------------------------------------------------
// Chat
// ---------------------------------------------------------------------------

export interface ChatCreate {
  name: string;
  description?: string | null;
  default_chat_profile_id?: string | null;
  default_embedding_profile_id?: string | null;
  default_reranker_profile_id?: string | null;
}

export interface ChatUpdate {
  name?: string | null;
  description?: string | null;
  default_chat_profile_id?: string | null;
  default_embedding_profile_id?: string | null;
  default_reranker_profile_id?: string | null;
}

export interface ChatRead {
  id: string;
  name: string;
  description: string | null;
  default_chat_profile_id: string | null;
  default_embedding_profile_id: string | null;
  default_reranker_profile_id: string | null;
  created_at: string;
  updated_at: string;
}

// ---------------------------------------------------------------------------
// Session
// ---------------------------------------------------------------------------

export interface SessionCreate {
  chat_id: string;
  name?: string | null;
  chat_profile_id?: string | null;
}

export interface SessionUpdate {
  name?: string | null;
  chat_profile_id?: string | null;
}

export interface SessionRead {
  id: string;
  chat_id: string;
  name: string | null;
  chat_profile_id: string | null;
  created_at: string;
  updated_at: string;
}

// ---------------------------------------------------------------------------
// Message
// ---------------------------------------------------------------------------

export interface MessageRead {
  id: string;
  session_id: string;
  role: MessageRole;
  content: string;
  citations: Citation[] | null;
  tool_trace: ToolTrace | null;
  token_count: number | null;
  created_at: string;
}

// ---------------------------------------------------------------------------
// Document
// ---------------------------------------------------------------------------

export interface DocumentRead {
  id: string;
  chat_id: string;
  source_type: DocumentSourceType;
  original_filename: string;
  storage_path: string;
  mime_type: string;
  page_count: number | null;
  status: DocumentStatus;
  checksum_sha256: string;
  created_at: string;
  updated_at: string;
}

// ---------------------------------------------------------------------------
// QAResponse (non-streaming)
// ---------------------------------------------------------------------------

export interface QAResponse {
  answer: string;
  citations: Citation[];
  documents_used: string[];
  coverage: number;
  uncertainty: string[];
  session_id: string;
  message_id: string;
  debug_trace: ToolTrace | null;
}

// ---------------------------------------------------------------------------
// Structured facts
// ---------------------------------------------------------------------------

export interface FactFilter {
  kind?: FactKind | null;
  key?: string | null;
  document_ids?: string[] | null;
  page?: number | null;
}

export interface StructuredFactRead {
  id: string;
  chat_id: string;
  document_id: string;
  source_node_id: string | null;
  kind: FactKind;
  key: string;
  value: FactValue;
  unit: string | null;
  context_excerpt: string | null;
  page: number | null;
  created_at: string;
}

// ---------------------------------------------------------------------------
// Chat Manifest (Phase 5.4)
// ---------------------------------------------------------------------------

export interface DocumentManifestEntry {
  document_id: string;
  title: string | null;
  authors: string[];
  page_count: number | null;
  abstract_summary: string | null;
  main_topics: string[];
  section_count: number;
  token_estimate: number;
  available_source_types: string[];
  ingestion_status: IngestionState;
}

export interface ChatManifest {
  chat_id: string;
  generated_at: string;
  document_count: number;
  total_token_estimate: number;
  documents: DocumentManifestEntry[];
  ingestion_summary: Record<IngestionState, number>;
}

// ---------------------------------------------------------------------------
// SSE event union (stream=true path)
//
// Format over the wire:
//   event: token\ndata: {"delta":"...","index":0}\n\n
//   event: citation\ndata: <Citation JSON>\n\n
//   event: done\ndata: <QAResponse JSON>\n\n
//   event: error\ndata: {"code":"...","detail":"..."}\n\n
// ---------------------------------------------------------------------------

export interface SSETokenData {
  delta: string;
  index: number;
}

export interface SSEErrorData {
  code: string;
  detail: string;
}

export type SSEEventKind = "token" | "citation" | "done" | "error";

export type SSEEvent =
  | { kind: "token"; data: SSETokenData }
  | { kind: "citation"; data: Citation }
  | { kind: "done"; data: QAResponse }
  | { kind: "error"; data: SSEErrorData };

// ---------------------------------------------------------------------------
// API error shape
// ---------------------------------------------------------------------------

export interface ApiErrorBody {
  detail: string | { msg: string; type: string }[];
}
