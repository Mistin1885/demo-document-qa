"""Async VLM (Vision Language Model) client for figure/table narration.

This client targets the same self-hosted OpenAI-compatible Gemma-4 vLLM
endpoint that already powers ``LLM_*`` in the chat pipeline, but it is kept
in its own module because:

- It always sends multimodal payloads (image + text).
- It is invoked synchronously during ingestion, not at QA time.
- The retry / timeout policy differs (image inference is slower).

Provider support
----------------
- ``openai_compatible`` / ``openai`` (default) — sends a chat-completion call
  with ``{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}``.
- ``gemini_native`` — uses Google's ``inline_data`` payload.

Design
------
- Pure async via ``httpx.AsyncClient``.
- Picks up configuration from ``app.config.get_settings()`` with explicit
  ``VLM_*`` overrides (so the VLM can point at a different server than the
  chat LLM if needed).
- Never logs API keys; failures raise ``VLMError`` with a sanitised message.
- Returns plain strings; callers decide whether to treat them as HTML, free
  text, or JSON.

The module re-implements the small subset of the reference
``src/app/modules/document_parser/infra/vision/vlm_client.py`` that this
project needs, while staying inside the project's "async + Pydantic v2"
conventions.
"""

from __future__ import annotations

import base64
import io
import logging
from dataclasses import dataclass
from pathlib import Path

import httpx
from PIL import Image

from app.config import get_settings

logger = logging.getLogger(__name__)


class VLMError(RuntimeError):
    """Raised when the VLM endpoint returns an error or invalid payload."""


@dataclass(frozen=True)
class VLMSettings:
    """Resolved VLM configuration (env-driven; see ``app.config.Settings``)."""

    provider: str
    api_url: str
    api_key: str
    model: str
    max_tokens: int
    temperature: float
    timeout: float
    image_max_side: int
    image_jpeg_quality: int

    @classmethod
    def from_app_settings(cls) -> VLMSettings:
        s = get_settings()
        return cls(
            provider=s.vlm_provider,
            api_url=s.vlm_api_url,
            api_key=s.vlm_api_key,
            model=s.vlm_model,
            max_tokens=s.vlm_max_tokens,
            temperature=s.vlm_temperature,
            timeout=s.vlm_timeout,
            image_max_side=s.vlm_image_max_side,
            image_jpeg_quality=s.vlm_image_jpeg_quality,
        )


# ---------------------------------------------------------------------------
# Image preparation
# ---------------------------------------------------------------------------


def _encode_image_to_b64(image: Image.Image, max_side: int, jpeg_quality: int) -> str:
    """Resize (preserving aspect ratio) and JPEG-encode the image as base64."""
    if image.mode != "RGB":
        image = image.convert("RGB")

    width, height = image.size
    scale = min(1.0, max_side / max(width, height))
    if scale < 1.0:
        image = image.resize((int(width * scale), int(height * scale)))

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=jpeg_quality, optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def encode_path_to_b64(image_path: str | Path, settings: VLMSettings | None = None) -> str:
    """Open an image from disk and return its base64-encoded JPEG payload."""
    settings = settings or VLMSettings.from_app_settings()
    path = Path(image_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"image not found: {path}")
    with Image.open(path) as img:
        return _encode_image_to_b64(img, settings.image_max_side, settings.image_jpeg_quality)


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class VLMClient:
    """Async multimodal client for Gemma-4-class endpoints.

    Use ``call_with_image`` for single-image prompts.  The class is cheap to
    instantiate; reuse a single instance across a document if possible so the
    underlying HTTP client can pool connections.
    """

    def __init__(self, settings: VLMSettings | None = None) -> None:
        self._s = settings or VLMSettings.from_app_settings()
        if not self._s.api_url and self._s.provider != "gemini_native":
            raise VLMError(
                "VLM_API_URL is required for provider "
                f"{self._s.provider!r} but is empty"
            )
        if not self._s.model:
            raise VLMError("VLM_MODEL is required but is empty")

    @property
    def settings(self) -> VLMSettings:
        return self._s

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def call_with_image(
        self,
        *,
        system_prompt: str,
        user_text: str,
        image_b64: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Send a single-image multimodal request and return the text response."""
        if self._s.provider == "gemini_native":
            return await self._call_gemini(
                system_prompt=system_prompt,
                user_text=user_text,
                image_b64=image_b64,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        return await self._call_openai_compatible(
            system_prompt=system_prompt,
            user_text=user_text,
            image_b64=image_b64,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    async def call_with_image_path(
        self,
        image_path: str | Path,
        *,
        system_prompt: str,
        user_text: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        image_b64 = encode_path_to_b64(image_path, self._s)
        return await self.call_with_image(
            system_prompt=system_prompt,
            user_text=user_text,
            image_b64=image_b64,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    # ------------------------------------------------------------------
    # Internal — OpenAI-compatible (vLLM Gemma, OpenAI gpt-4o, …)
    # ------------------------------------------------------------------

    async def _call_openai_compatible(
        self,
        *,
        system_prompt: str,
        user_text: str,
        image_b64: str,
        temperature: float | None,
        max_tokens: int | None,
    ) -> str:
        url = self._normalize_openai_url(self._s.api_url)
        payload = {
            "model": self._s.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                        },
                    ],
                },
            ],
            "max_tokens": max_tokens if max_tokens is not None else self._s.max_tokens,
            "temperature": (
                temperature if temperature is not None else self._s.temperature
            ),
            "stream": False,
        }
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self._s.api_key and self._s.api_key.lower() != "not-needed":
            headers["Authorization"] = f"Bearer {self._s.api_key}"

        async with httpx.AsyncClient(timeout=self._s.timeout) as client:
            try:
                resp = await client.post(url, json=payload, headers=headers)
            except httpx.HTTPError as exc:
                raise VLMError(f"VLM request failed: {exc.__class__.__name__}") from exc

        if resp.status_code != 200:
            snippet = resp.text[:200].replace("\n", " ")
            raise VLMError(f"VLM HTTP {resp.status_code}: {snippet}")

        try:
            data = resp.json()
            return str(data["choices"][0]["message"]["content"]).strip()
        except (KeyError, IndexError, ValueError) as exc:
            raise VLMError(f"VLM response malformed: {exc}") from exc

    # ------------------------------------------------------------------
    # Internal — Gemini native
    # ------------------------------------------------------------------

    async def _call_gemini(
        self,
        *,
        system_prompt: str,
        user_text: str,
        image_b64: str,
        temperature: float | None,
        max_tokens: int | None,
    ) -> str:
        if not self._s.api_key:
            raise VLMError("GEMINI / VLM_API_KEY is required for gemini_native provider")
        url = self._s.api_url or (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self._s.model}:generateContent"
        )
        payload = {
            "system_instruction": {"parts": [{"text": system_prompt}]},
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": user_text},
                        {
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": image_b64,
                            }
                        },
                    ],
                }
            ],
            "generationConfig": {
                "temperature": (
                    temperature if temperature is not None else self._s.temperature
                ),
                "maxOutputTokens": (
                    max_tokens if max_tokens is not None else self._s.max_tokens
                ),
            },
        }
        async with httpx.AsyncClient(timeout=self._s.timeout) as client:
            try:
                resp = await client.post(
                    url,
                    params={"key": self._s.api_key},
                    json=payload,
                )
            except httpx.HTTPError as exc:
                raise VLMError(f"Gemini request failed: {exc.__class__.__name__}") from exc

        if resp.status_code != 200:
            snippet = resp.text[:200].replace("\n", " ")
            raise VLMError(f"Gemini HTTP {resp.status_code}: {snippet}")

        data = resp.json()
        candidates = data.get("candidates") or []
        if not candidates:
            raise VLMError("Gemini returned no candidates")
        parts = candidates[0].get("content", {}).get("parts", [])
        texts = [p.get("text", "") for p in parts if isinstance(p, dict) and p.get("text")]
        if not texts:
            finish = candidates[0].get("finishReason", "unknown")
            raise VLMError(f"Gemini returned no text (finishReason={finish})")
        return "\n".join(texts).strip()

    @staticmethod
    def _normalize_openai_url(api_url: str) -> str:
        """Ensure URL ends with ``/chat/completions`` for OpenAI-compatible servers.

        ``LLM_API_URL`` is often configured as either ``http://host:port/v1`` or
        ``http://host:port/v1/chat/completions``; accept both for convenience.
        """
        url = api_url.rstrip("/")
        if url.endswith("/chat/completions"):
            return url
        if url.endswith("/v1"):
            return f"{url}/chat/completions"
        return f"{url}/chat/completions"


__all__ = [
    "VLMClient",
    "VLMError",
    "VLMSettings",
    "encode_path_to_b64",
]
