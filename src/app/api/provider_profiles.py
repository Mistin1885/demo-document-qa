"""FastAPI router for /provider_profiles — stateless test_connection only.

Why stateless?
--------------
Provider profile CRUD lives entirely in the frontend's localStorage (see
``src/frontend/lib/api/providers.ts``). The only thing the frontend cannot
do on its own is actually call the upstream LLM / embedding / reranker
endpoint, because browser-side CORS + key handling would expose the key.

This router takes the profile body inline, builds an ephemeral provider
via :mod:`app.providers.registry`, calls ``test_connection()``, and returns
the result. Nothing is persisted; the key never touches the DB or logs.
"""

from __future__ import annotations

import logging
from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.providers.base import (
    ChatProvider,
    EmbeddingProvider,
    ProviderTestResult,
    RerankerProvider,
)
from app.providers.registry import (
    ProviderProfileLike,
    build_chat_provider,
    build_embedding_provider,
    build_reranker_provider,
)

logger = logging.getLogger(__name__)
router = APIRouter()


class TestConnectionRequest(BaseModel):
    """Inline profile body — never persisted."""

    kind: Literal["chat", "embedding", "reranker"]
    provider_type: str = Field(..., description="openai | gemini_native | openai_compat | vllm | mock")
    model: str
    base_url: str | None = None
    api_key_plaintext: str | None = None
    context_window: int | None = None
    embedding_dim: int | None = None


class _EphemeralProfile:
    """In-memory ProviderProfileLike for one test call."""

    def __init__(self, req: TestConnectionRequest) -> None:
        self.provider_type = req.provider_type
        self.model_name = req.model
        self.api_key_encrypted: bytes | None = None
        self.api_key_plain: str | None = req.api_key_plaintext
        self.base_url: str | None = req.base_url
        self.context_window: int | None = req.context_window
        self.embedding_dim: int | None = req.embedding_dim
        self.provider_name: str | None = req.provider_type


@router.post("/test_connection", response_model=ProviderTestResult)
async def test_connection(req: TestConnectionRequest) -> ProviderTestResult:
    """Build an ephemeral provider from *req* and return its test result.

    The provider key is never logged. On configuration errors (missing
    base_url, unknown provider_type) we surface the message via the
    ``error`` field so the UI can render it inline.
    """
    profile: ProviderProfileLike = _EphemeralProfile(req)
    try:
        provider: ChatProvider | EmbeddingProvider | RerankerProvider
        if req.kind == "chat":
            provider = build_chat_provider(profile)
        elif req.kind == "embedding":
            provider = build_embedding_provider(profile)
        else:
            provider = build_reranker_provider(profile)
        return await provider.test_connection()
    except ValueError as exc:
        return ProviderTestResult(ok=False, error=str(exc))
    except Exception as exc:  # noqa: BLE001
        logger.exception("test_connection failed", extra={"kind": req.kind})
        return ProviderTestResult(ok=False, error=f"{type(exc).__name__}: {exc}")
