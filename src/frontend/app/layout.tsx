import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";
import { QueryProvider } from "@/lib/queries/provider";

export const metadata: Metadata = {
  title: "Paper Notebook Agent",
  description: "NotebookLM-like multi-document Agentic QA for arXiv papers",
};

/**
 * Thin top nav — provides a link to /settings without touching AppShell internals.
 * Rendered inside QueryProvider so children can use TanStack Query hooks.
 */
function TopNav() {
  return (
    <nav className="h-8 shrink-0 flex items-center justify-end px-4 border-b border-[var(--border)] bg-[var(--surface)] z-10">
      <Link
        href="/settings"
        className="text-xs text-[var(--muted)] hover:text-[var(--foreground)] transition-colors"
      >
        Model Settings
      </Link>
    </nav>
  );
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark h-full">
      <body className="h-full overflow-hidden flex flex-col">
        <QueryProvider>
          <TopNav />
          <div className="flex-1 overflow-hidden">{children}</div>
        </QueryProvider>
      </body>
    </html>
  );
}
