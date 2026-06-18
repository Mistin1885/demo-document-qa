/**
 * Document API helpers.
 *
 * Routes:
 *   POST   /chats/{chat_id}/documents          (multipart: file + optional description)
 *   GET    /chats/{chat_id}/documents
 *   GET    /chats/{chat_id}/documents/{doc_id}
 *   DELETE /chats/{chat_id}/documents/{doc_id}
 *   POST   /chats/{chat_id}/documents/{doc_id}/associate
 */

import { apiDelete, apiGet, apiUpload, apiFetch, API_BASE_URL, ApiError } from "./client";
import type { DocumentRead } from "./types";

export async function uploadDocument(
  chatId: string,
  file: File,
  description?: string,
  signal?: AbortSignal
): Promise<DocumentRead> {
  const form = new FormData();
  form.append("file", file);
  if (description) {
    form.append("description", description);
  }
  return apiUpload<DocumentRead>(`/chats/${chatId}/documents`, form, signal);
}

export async function listDocuments(chatId: string): Promise<DocumentRead[]> {
  return apiGet<DocumentRead[]>(`/chats/${chatId}/documents`);
}

export async function getDocument(
  chatId: string,
  docId: string
): Promise<DocumentRead> {
  return apiGet<DocumentRead>(`/chats/${chatId}/documents/${docId}`);
}

export async function deleteDocument(chatId: string, docId: string): Promise<void> {
  return apiDelete(`/chats/${chatId}/documents/${docId}`);
}

export async function associateDocument(
  chatId: string,
  docId: string
): Promise<void> {
  return apiFetch<void>(`/chats/${chatId}/documents/${docId}/associate`, {
    method: "POST",
  });
}

// ---------------------------------------------------------------------------
// Upload with progress events (XMLHttpRequest — fetch does not expose upload progress)
// ---------------------------------------------------------------------------

export interface UploadProgressOptions {
  onProgress?: (percent: number) => void;
  signal?: AbortSignal;
}

export function uploadDocumentWithProgress(
  chatId: string,
  file: File,
  opts: UploadProgressOptions = {}
): Promise<DocumentRead> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    const form = new FormData();
    form.append("file", file);

    xhr.upload.addEventListener("progress", (e) => {
      if (e.lengthComputable && opts.onProgress) {
        opts.onProgress(Math.round((e.loaded / e.total) * 100));
      }
    });

    xhr.addEventListener("load", () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          resolve(JSON.parse(xhr.responseText) as DocumentRead);
        } catch {
          reject(new Error("Invalid JSON response from upload"));
        }
      } else {
        let detail = xhr.statusText || "Upload failed";
        try {
          const body = JSON.parse(xhr.responseText) as {
            detail?: string | Array<{ msg: string }>;
          };
          if (typeof body?.detail === "string") {
            detail = body.detail;
          } else if (Array.isArray(body?.detail)) {
            detail = body.detail.map((e) => e.msg).join("; ");
          }
        } catch {
          // ignore parse failure; use statusText
        }
        reject(new ApiError(xhr.status, detail));
      }
    });

    xhr.addEventListener("error", () =>
      reject(new Error("Network error during upload"))
    );
    xhr.addEventListener("abort", () =>
      reject(new DOMException("Upload aborted", "AbortError"))
    );

    // Respect AbortSignal
    if (opts.signal) {
      opts.signal.addEventListener("abort", () => xhr.abort());
    }

    xhr.open("POST", `${API_BASE_URL}/chats/${chatId}/documents`);
    xhr.send(form);
  });
}
