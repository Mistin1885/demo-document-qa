import { type NextRequest, NextResponse } from "next/server";
import { buildBackendTarget } from "./route-utils";

// force-dynamic: disable caching so SSE (text/event-stream) streams correctly.
export const dynamic = "force-dynamic";

const BACKEND = process.env.BACKEND_URL ?? "http://backend:8000";

async function proxy(req: NextRequest): Promise<NextResponse> {
  const target = buildBackendTarget(
    BACKEND,
    req.nextUrl.pathname,
    req.nextUrl.search
  );

  const headers = new Headers();
  req.headers.forEach((value, key) => {
    const k = key.toLowerCase();
    if (!["host", "connection", "keep-alive", "transfer-encoding"].includes(k)) {
      headers.set(key, value);
    }
  });

  const hasBody = req.method !== "GET" && req.method !== "HEAD";
  // Stream the request body directly. Redirect retries are intentionally
  // avoided because streamed request bodies cannot be replayed safely.
  const init: RequestInit & { duplex?: string } = {
    method: req.method,
    headers,
    ...(hasBody ? { body: req.body, duplex: "half" } : {}),
  };

  let upstream: Response;
  try {
    upstream = await fetch(target, init);
  } catch (err) {
    const raw = err instanceof Error ? err.cause : err;
    const cause =
      raw instanceof Error
        ? `${raw.constructor.name}: ${raw.message || "(no message)"} [code=${(raw as NodeJS.ErrnoException).code ?? "none"}]`
        : String(raw);
    console.error("[proxy] fetch failed", { target, error: String(err), cause });
    return NextResponse.json(
      { detail: "backend unreachable", target, error: String(err), cause },
      { status: 503 }
    );
  }

  const resHeaders = new Headers();
  upstream.headers.forEach((value, key) => {
    const k = key.toLowerCase();
    if (!["transfer-encoding", "connection", "keep-alive"].includes(k)) {
      resHeaders.set(key, value);
    }
  });

  return new NextResponse(upstream.body, {
    status: upstream.status,
    statusText: upstream.statusText,
    headers: resHeaders,
  });
}

export {
  proxy as GET,
  proxy as POST,
  proxy as PUT,
  proxy as PATCH,
  proxy as DELETE,
};
