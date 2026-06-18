# syntax=docker/dockerfile:1.7
#
# Paper Notebook Agent — Backend image
# -----------------------------------------------------------------------------
# Stage 1 (builder): installs the project + all runtime deps with `uv sync` into
# a self-contained virtualenv at /app/.venv.
# Stage 2 (runtime): copies the venv + source code into a slim Python image.
#
# CLAUDE.md §0 rule: package management goes through `uv`. We install `uv`
# from its official slim image and reuse it inside the builder stage.

FROM ghcr.io/astral-sh/uv:0.5-python3.12-bookworm-slim AS builder

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv

WORKDIR /app

# Install only dependencies first so that pure code edits don't bust the cache.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

# Now copy the rest of the project and install the package itself.
COPY src ./src
COPY migrations ./migrations
COPY alembic.ini ./alembic.ini
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev


FROM python:3.12-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:${PATH}" \
    PYTHONPATH="/app/src"

# curl: healthcheck. libpq5: psycopg runtime. No build toolchain in runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl libpq5 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src
COPY --from=builder /app/migrations /app/migrations
COPY --from=builder /app/alembic.ini /app/alembic.ini

# Runtime-only directories (mounted as volumes in docker-compose).
RUN mkdir -p /app/data/storage /app/data/parsed

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --retries=10 --start-period=20s \
    CMD curl -fsS http://localhost:8000/docs >/dev/null || exit 1

# Run Alembic migrations on every start, then launch Uvicorn.
CMD ["sh", "-c", "alembic upgrade head && exec uvicorn app.main:app --host 0.0.0.0 --port 8000"]
