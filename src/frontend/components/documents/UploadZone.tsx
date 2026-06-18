"use client";

import { useRef, useState, useCallback } from "react";
import { UploadCloud, X } from "lucide-react";

const MAX_SIZE_BYTES = 50 * 1024 * 1024; // 50 MB

interface UploadZoneProps {
  onUpload: (file: File) => void;
  isUploading: boolean;
  uploadProgress: number; // 0-100
  uploadError: string | null;
  onDismissError: () => void;
}

export function UploadZone({
  onUpload,
  isUploading,
  uploadProgress,
  uploadError,
  onDismissError,
}: UploadZoneProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragOver, setDragOver] = useState(false);
  const [validationError, setValidationError] = useState<string | null>(null);

  function validateAndUpload(file: File) {
    setValidationError(null);
    if (file.type !== "application/pdf") {
      setValidationError("Only PDF files are accepted.");
      return;
    }
    if (file.size > MAX_SIZE_BYTES) {
      setValidationError("File exceeds the 50 MB limit.");
      return;
    }
    onUpload(file);
  }

  const handleDrop = useCallback(
    (e: React.DragEvent<HTMLDivElement>) => {
      e.preventDefault();
      setDragOver(false);
      const file = e.dataTransfer.files[0];
      if (file) validateAndUpload(file);
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [onUpload]
  );

  function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (file) validateAndUpload(file);
    // Reset so the same file can be re-selected after an error
    e.target.value = "";
  }

  const displayError = validationError ?? uploadError;

  return (
    <div className="px-3 pb-3">
      {/* Drop zone */}
      <div
        role="button"
        tabIndex={0}
        aria-label="Upload PDF"
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
        onClick={() => !isUploading && inputRef.current?.click()}
        onKeyDown={(e) => {
          if ((e.key === "Enter" || e.key === " ") && !isUploading) {
            inputRef.current?.click();
          }
        }}
        className={`
          flex flex-col items-center justify-center gap-1.5 rounded-lg border-2 border-dashed
          px-4 py-5 cursor-pointer transition-colors select-none
          ${dragOver
            ? "border-[var(--accent)] bg-[var(--accent)]/10"
            : "border-[var(--border)] hover:border-[var(--accent)]/60 hover:bg-[var(--surface-raised)]"
          }
          ${isUploading ? "pointer-events-none opacity-60" : ""}
        `}
      >
        <UploadCloud
          size={22}
          className={dragOver ? "text-[var(--accent)]" : "text-[var(--muted)]"}
        />
        {isUploading ? (
          <span className="text-xs text-[var(--muted)]">
            Uploading… {uploadProgress}%
          </span>
        ) : (
          <span className="text-xs text-[var(--muted)] text-center">
            Drop PDF here or{" "}
            <span className="text-[var(--accent)]">click to browse</span>
          </span>
        )}

        {/* Progress bar */}
        {isUploading && (
          <div className="w-full h-1 rounded-full bg-[var(--border)] overflow-hidden mt-1">
            <div
              className="h-full bg-[var(--accent)] transition-all duration-200"
              style={{ width: `${uploadProgress}%` }}
            />
          </div>
        )}
      </div>

      {/* Error message */}
      {displayError && (
        <div className="mt-2 flex items-start gap-1.5 rounded-md bg-rose-900/30 px-3 py-2 text-[11px] text-rose-300 ring-1 ring-rose-500/20">
          <span className="flex-1">{displayError}</span>
          <button
            onClick={() => {
              setValidationError(null);
              onDismissError();
            }}
            className="shrink-0 text-rose-400 hover:text-rose-200"
            aria-label="Dismiss error"
          >
            <X size={12} />
          </button>
        </div>
      )}

      <input
        ref={inputRef}
        type="file"
        accept="application/pdf"
        className="hidden"
        onChange={handleFileChange}
        aria-hidden="true"
        tabIndex={-1}
      />
    </div>
  );
}
