"use client";

import { useEffect, useState } from "react";
import { AppShell } from "@/components/layout/AppShell";
import { healthCheck } from "@/lib/api/client";

export default function Home() {
  const [backendStatus, setBackendStatus] = useState<"checking" | "ok" | "unreachable">(
    "checking"
  );

  useEffect(() => {
    healthCheck()
      .then(() => setBackendStatus("ok"))
      .catch(() => setBackendStatus("unreachable"));
  }, []);

  return (
    <div className="h-full flex flex-col">
      {/* Backend status banner */}
      <div
        className={`px-3 py-1 text-xs text-center font-medium transition-colors ${
          backendStatus === "ok"
            ? "bg-green-900/40 text-green-300"
            : backendStatus === "unreachable"
            ? "bg-red-900/40 text-red-300"
            : "bg-gray-800 text-gray-400"
        }`}
      >
        {backendStatus === "checking" && "Backend: checking…"}
        {backendStatus === "ok" && "Backend: ok"}
        {backendStatus === "unreachable" && "Backend: unreachable"}
      </div>

      {/* Main workspace */}
      <div className="flex-1 overflow-hidden">
        <AppShell />
      </div>
    </div>
  );
}
