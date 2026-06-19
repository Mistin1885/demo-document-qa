# syntax=docker/dockerfile:1.7
#
# Paper Notebook Agent — Frontend image
# -----------------------------------------------------------------------------
# Three-stage Next.js 15 build:
#   1. deps    — install all npm dependencies (cached on package.json hash)
#   2. builder — `next build` with the standalone output
#   3. runner  — copy the standalone bundle + static assets into a slim image
#
# `NEXT_PUBLIC_API_BASE_URL` is a *build-time* variable: Next inlines public
# env vars into the client bundle, so it must be set as a build ARG and not
# only at runtime.

FROM node:20-alpine AS deps
WORKDIR /app
COPY src/frontend/package.json src/frontend/package-lock.json* ./
RUN npm ci --no-audit --no-fund


FROM node:20-alpine AS builder
WORKDIR /app

ARG NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
ENV NEXT_PUBLIC_API_BASE_URL=${NEXT_PUBLIC_API_BASE_URL}
ENV NEXT_TELEMETRY_DISABLED=1

COPY --from=deps /app/node_modules ./node_modules
COPY src/frontend/ ./
RUN npm run build


FROM node:20-alpine AS runner
WORKDIR /app

ENV NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1 \
    PORT=3000 \
    HOSTNAME=0.0.0.0

RUN addgroup --system --gid 1001 nodejs \
    && adduser --system --uid 1001 nextjs

# Next.js standalone output already contains a minimal node_modules.
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static
# `public/` is optional in this project; copy it only if it exists in the builder stage.
# (Using a wildcard avoids a missing-source error from a plain COPY.)

USER nextjs
EXPOSE 3000

HEALTHCHECK --interval=15s --timeout=5s --retries=10 --start-period=15s \
    CMD wget -qO- http://127.0.0.1:3000/ >/dev/null 2>&1 || exit 1

CMD ["node", "server.js"]
