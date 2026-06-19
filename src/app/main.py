"""FastAPI application entrypoint.

This module creates the ``app`` instance and wires together:
- Routers (thin HTTP layer; no domain logic)
- Exception handlers mapping domain errors → HTTP status codes
- Lifespan hook (currently a no-op placeholder)

Phase progression
-----------------
Phase 3.1 added the ``/chats`` router.
Phase 3.2 added the ``/chats/{chat_id}/sessions`` router.
Phase 3.3 adds the ``/chats/{chat_id}/documents`` router.
Later phases (3.4+) will add messages, agent QA, etc.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.errors import (
    ChatDocumentAlreadyAssociated,
    ChatNotFound,
    DocumentAlreadyExists,
    DocumentNotFound,
    DocumentStorageError,
    FactNotFound,
    InvalidFactFilter,
    InvalidUpload,
    SessionNotFound,
)


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan hook — placeholder for startup/shutdown tasks."""
    # Phase 6+ will initialise the Vespa connection pool here.
    yield


app = FastAPI(
    title="Paper Notebook Agent",
    version="0.1.0",
    description="NotebookLM-like multi-document Agentic QA over arXiv PDFs.",
    lifespan=lifespan,
    # Disable Starlette's automatic trailing-slash redirects. Collection routes
    # are defined without relying on 307 redirects, so streamed proxied requests
    # from the frontend never need to replay a consumed request body.
    redirect_slashes=False,
)

# CORS — Phase 8 frontend (Next.js dev server) calls the API directly.
# In production behind a reverse proxy this list should be tightened or removed.
_cors_origins: list[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]
_cors_extra = get_settings().cors_extra_origins.strip()
if _cors_extra:
    _cors_origins.extend(o.strip() for o in _cors_extra.split(",") if o.strip())

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------


@app.exception_handler(ChatNotFound)
async def chat_not_found_handler(request: Request, exc: ChatNotFound) -> JSONResponse:
    return JSONResponse(status_code=404, content={"detail": "chat not found"})


@app.exception_handler(SessionNotFound)
async def session_not_found_handler(
    request: Request, exc: SessionNotFound
) -> JSONResponse:
    return JSONResponse(status_code=404, content={"detail": str(exc)})


@app.exception_handler(DocumentNotFound)
async def document_not_found_handler(
    request: Request, exc: DocumentNotFound
) -> JSONResponse:
    return JSONResponse(status_code=404, content={"detail": str(exc)})


@app.exception_handler(InvalidUpload)
async def invalid_upload_handler(request: Request, exc: InvalidUpload) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(DocumentAlreadyExists)
async def document_already_exists_handler(
    request: Request, exc: DocumentAlreadyExists
) -> JSONResponse:
    return JSONResponse(
        status_code=409,
        content={
            "detail": str(exc),
            "document_id": str(exc.document_id),
        },
    )


@app.exception_handler(ChatDocumentAlreadyAssociated)
async def chat_document_already_associated_handler(
    request: Request, exc: ChatDocumentAlreadyAssociated
) -> JSONResponse:
    return JSONResponse(
        status_code=409,
        content={
            "detail": str(exc),
            "document_id": str(exc.document_id),
        },
    )


@app.exception_handler(DocumentStorageError)
async def document_storage_error_handler(
    request: Request, exc: DocumentStorageError
) -> JSONResponse:
    return JSONResponse(status_code=500, content={"detail": str(exc)})


@app.exception_handler(FactNotFound)
async def fact_not_found_handler(request: Request, exc: FactNotFound) -> JSONResponse:
    return JSONResponse(status_code=404, content={"detail": str(exc)})


@app.exception_handler(InvalidFactFilter)
async def invalid_fact_filter_handler(
    request: Request, exc: InvalidFactFilter
) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": str(exc)})


# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

from app.api import chats  # noqa: E402,I001
from app.api import documents  # noqa: E402,I001
from app.api import facts  # noqa: E402,I001
from app.api import manifest  # noqa: E402,I001
from app.api import messages  # noqa: E402,I001
from app.api import provider_profiles  # noqa: E402,I001
from app.api import sessions  # noqa: E402,I001

app.include_router(chats.router, prefix="/chats", tags=["chats"])
app.include_router(
    sessions.router,
    prefix="/chats/{chat_id}/sessions",
    tags=["sessions"],
)
app.include_router(
    documents.router,
    prefix="/chats/{chat_id}/documents",
    tags=["documents"],
)
app.include_router(
    facts.router,
    prefix="/chats/{chat_id}/facts",
    tags=["facts"],
)
app.include_router(
    manifest.router,
    prefix="/chats/{chat_id}/manifest",
    tags=["manifest"],
)
app.include_router(
    messages.router,
    prefix="/chats/{chat_id}/sessions/{session_id}/messages",
    tags=["messages"],
)
app.include_router(
    provider_profiles.router,
    prefix="/provider_profiles",
    tags=["provider_profiles"],
)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


@app.get("/health", tags=["meta"])
async def health() -> dict[str, str]:
    return {"status": "ok"}
