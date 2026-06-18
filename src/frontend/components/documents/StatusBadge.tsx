"use client";

import type { DocumentStatus } from "@/lib/api/types";

interface StatusBadgeProps {
  status: DocumentStatus;
  className?: string;
}

const STATUS_CONFIG: Record<
  DocumentStatus,
  { label: string; classes: string }
> = {
  uploaded: {
    label: "Uploaded",
    classes: "bg-gray-700 text-gray-300 ring-gray-600/40",
  },
  parsing: {
    label: "Parsing",
    classes: "bg-blue-900/60 text-blue-300 ring-blue-500/30",
  },
  parsed: {
    label: "Parsed",
    classes: "bg-indigo-900/60 text-indigo-300 ring-indigo-500/30",
  },
  enriching: {
    label: "Enriching",
    classes: "bg-violet-900/60 text-violet-300 ring-violet-500/30",
  },
  indexed: {
    label: "Indexed",
    classes: "bg-emerald-900/60 text-emerald-300 ring-emerald-500/30",
  },
  failed: {
    label: "Failed",
    classes: "bg-rose-900/60 text-rose-300 ring-rose-500/30",
  },
};

export function StatusBadge({ status, className = "" }: StatusBadgeProps) {
  const config = STATUS_CONFIG[status] ?? STATUS_CONFIG.uploaded;
  return (
    <span
      className={`inline-flex items-center rounded-full px-2 py-0.5 text-[10px] font-semibold ring-1 ring-inset ${config.classes} ${className}`}
    >
      {config.label}
    </span>
  );
}
