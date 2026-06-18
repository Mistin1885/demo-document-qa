# Paper Notebook Agent — Frontend

Next.js 14 App Router · TypeScript · Tailwind CSS · TanStack Query v5

## Prerequisites

- Node.js 20+
- Backend running at `http://localhost:8000` (or set `NEXT_PUBLIC_API_BASE_URL`)

## Setup

```bash
# Install dependencies
npm --prefix src/frontend install

# Copy env template (edit if needed)
cp src/frontend/.env.example src/frontend/.env.local
```

## Development

```bash
npm --prefix src/frontend run dev     # http://localhost:3000
```

## Build & Type-check

```bash
npm --prefix src/frontend run build      # production build
npm --prefix src/frontend run typecheck  # tsc --noEmit
npm --prefix src/frontend run lint       # ESLint
npm --prefix src/frontend run test       # vitest (SSE unit tests)
```
