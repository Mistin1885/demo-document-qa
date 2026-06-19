"""Async MinerU hybrid-backend client.

This module wraps the MinerU CLI (``mineru -b hybrid-http-client``) as
a reusable async service that:

1. Invokes MinerU via ``asyncio.create_subprocess_exec``.
2. Post-processes the output (image rename, page-marker injection, cleanup)
   using the shared helpers in ``app.parsing._postprocess``.
3. Runs the Phase-1 gate check and raises ``MinerUGateFailure`` on failure.
4. Is **idempotent** by default — if a valid (gate-passing) result already
   exists it is returned without re-running MinerU (use ``force=True`` to
   override).

Usage example (service layer)::

    from app.parsing import MinerUClient

    client = MinerUClient()          # reads server_url from app.config
    result = await client.parse_pdf(Path("data/storage/…/paper.pdf"))
    middle = json.loads(result.middle_json_path.read_text())

Security note: no API keys are used or logged here.  The *server_url* is
a plain HTTP endpoint (vLLM / MinerU OpenAI-compatible server) with no auth.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel

from app.parsing._postprocess import check, postprocess

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class MinerUError(Exception):
    """Base class for all MinerU client errors."""


class MinerUServerUnavailable(MinerUError):
    """Raised when the MinerU/vLLM HTTP server cannot be reached."""


class MinerUInvocationError(MinerUError):
    """Raised when the ``mineru`` subprocess exits with a non-zero code or
    the expected output directory is not produced."""

    def __init__(self, message: str, *, returncode: int | None = None, stderr: str = "") -> None:
        super().__init__(message)
        self.returncode = returncode
        self.stderr = stderr


class MinerUGateFailure(MinerUError):
    """Raised when the post-processed output fails the Phase-1 gate check."""

    def __init__(self, message: str, *, gate_summary: dict[str, Any]) -> None:
        super().__init__(message)
        self.gate_summary = gate_summary


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------


class MinerUParseResult(BaseModel):
    """Structured result returned by :meth:`MinerUClient.parse_pdf`.

    All paths are absolute and refer to files that exist on disk at the time
    this object is constructed.
    """

    output_dir: Path
    """``data/parsed/<basename>/hybrid_auto/`` — the post-processed directory."""

    markdown_path: Path
    """``<output_dir>/<basename>.md``"""

    middle_json_path: Path
    """``<output_dir>/<basename>_middle.json``"""

    image_paths: list[Path]
    """All renamed image files under ``<output_dir>/images/``."""

    pages: int
    """Page count derived from ``pdf_info`` length in middle.json."""

    gate_summary: dict[str, Any]
    """Raw output of :func:`app.parsing._postprocess.check`, includes ``gate_pass``."""

    duration_seconds: float
    """Wall-clock seconds for the MinerU invocation + post-processing."""

    mineru_version: str
    """Value of ``_version_name`` from middle.json (e.g. ``"MinerU-3.3.1"``).
    Empty string if the field is absent."""

    mineru_backend: str
    """Value of ``_backend`` from middle.json (should be ``"hybrid"``)."""

    model_config = {"arbitrary_types_allowed": True}


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

_DEFAULT_PARSED_ROOT = Path("data/parsed")
_HEALTH_CHECK_TIMEOUT = 2.0  # seconds
_EXPECTED_MODEL_ID = "opendatalab/MinerU2.5-2509-1.2B"


class MinerUClient:
    """Async client for MinerU hybrid-backend PDF parsing.

    Parameters
    ----------
    server_url:
        URL of the vLLM / MinerU OpenAI-compatible server.  Defaults to
        ``app.config.get_settings().mineru_server_url``.
    parsed_root:
        Root directory passed to MinerU's ``-o`` argument. MinerU itself
        creates ``<parsed_root>/<pdf_basename>/hybrid_auto/``.
    timeout_seconds:
        Maximum wall-clock time (seconds) allowed for a single
        ``mineru`` subprocess call.
    """

    def __init__(
        self,
        *,
        server_url: str | None = None,
        parsed_root: Path = _DEFAULT_PARSED_ROOT,
        timeout_seconds: int = 600,
    ) -> None:
        if server_url is None:
            from app.config import get_settings

            server_url = get_settings().mineru_server_url
        self._server_url = server_url
        self._parsed_root = parsed_root
        self._timeout_seconds = timeout_seconds

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def parse_pdf(
        self,
        pdf_path: Path,
        *,
        document_id: str | None = None,
        force: bool = False,
    ) -> MinerUParseResult:
        """Parse *pdf_path* with MinerU and return a :class:`MinerUParseResult`.

        Parameters
        ----------
        pdf_path:
            Absolute path to the source PDF file.
        document_id:
            Optional identifier added to ``gate_summary["document_id"]`` for
            downstream tracing (Phase 5).  Does not affect disk paths.
        force:
            When *True*, remove any existing output directory and re-run
            MinerU unconditionally.  When *False* (default), return the
            existing result if it passes the gate check.

        Raises
        ------
        MinerUInvocationError
            The subprocess exited with a non-zero return code, or the
            expected output directory was not produced.
        MinerUGateFailure
            Post-processing completed but the gate check failed (e.g.,
            page marker count mismatch).
        """
        doc_stem = pdf_path.stem
        output_dir = self._parsed_root / doc_stem / "hybrid_auto"

        # --- Idempotency check -------------------------------------------------
        if not force and self._result_is_valid(output_dir, doc_stem):
            return self._build_result(
                output_dir=output_dir,
                doc_stem=doc_stem,
                duration_seconds=0.0,
                document_id=document_id,
            )

        # --- Optionally purge stale output ------------------------------------
        if force and output_dir.exists():
            import shutil as _shutil

            _shutil.rmtree(output_dir)

        # --- Run MinerU -------------------------------------------------------
        output_parent = self._parsed_root
        output_parent.mkdir(parents=True, exist_ok=True)

        t0 = time.monotonic()
        await self._run_mineru_subprocess(pdf_path, output_parent)
        duration = time.monotonic() - t0

        if not output_dir.is_dir():
            raise MinerUInvocationError(
                f"MinerU finished but expected output directory is missing: {output_dir}",
                returncode=0,
                stderr="",
            )

        # --- Post-process -----------------------------------------------------
        postprocess(output_dir, doc_stem)

        # --- Gate check -------------------------------------------------------
        gate_summary = check(output_dir)
        if document_id is not None:
            gate_summary["document_id"] = document_id

        if not gate_summary.get("gate_pass"):
            raise MinerUGateFailure(
                f"MinerU output for '{doc_stem}' failed gate check: "
                f"page_markers_open={gate_summary.get('page_markers_open')} "
                f"pages={gate_summary.get('pages')}",
                gate_summary=gate_summary,
            )

        return self._build_result(
            output_dir=output_dir,
            doc_stem=doc_stem,
            duration_seconds=duration,
            document_id=document_id,
            gate_summary=gate_summary,
        )

    async def health_check(self) -> bool:
        """Return *True* if the MinerU/vLLM server is reachable and the
        expected model is loaded.

        Sends a GET to ``<server_url>/v1/models`` with a 2-second timeout.
        Returns *False* on any error (connection refused, timeout, unexpected
        model id, non-200 response).
        """
        url = self._server_url.rstrip("/") + "/v1/models"
        try:
            async with httpx.AsyncClient(timeout=_HEALTH_CHECK_TIMEOUT) as client:
                resp = await client.get(url)
            if resp.status_code != 200:
                return False
            data = resp.json()
            models = data.get("data", [])
            if not models:
                return False
            return models[0].get("id") == _EXPECTED_MODEL_ID
        except Exception:  # noqa: BLE001
            return False

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _result_is_valid(self, output_dir: Path, doc_stem: str) -> bool:
        """Return *True* if the output directory already contains a gate-passing result."""
        middle = output_dir / f"{doc_stem}_middle.json"
        md = output_dir / f"{doc_stem}.md"
        if not middle.exists() or not md.exists():
            return False
        try:
            summary = check(output_dir)
            return bool(summary.get("gate_pass"))
        except Exception:  # noqa: BLE001
            return False

    async def _run_mineru_subprocess(self, pdf_path: Path, output_parent: Path) -> None:
        """Invoke ``mineru`` asynchronously and raise on failure."""
        cmd = [
            "mineru",
            "-p",
            str(pdf_path),
            "-o",
            str(output_parent),
            "-b",
            "hybrid-http-client",
            "-u",
            self._server_url,
        ]
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=self._timeout_seconds,
                )
            except TimeoutError as exc:
                proc.kill()
                await proc.communicate()
                raise MinerUInvocationError(
                    f"MinerU subprocess timed out after {self._timeout_seconds}s",
                    returncode=None,
                    stderr="",
                ) from exc
        except MinerUInvocationError:
            raise
        except Exception as exc:
            raise MinerUInvocationError(
                f"Failed to start MinerU subprocess: {exc}",
                returncode=None,
                stderr="",
            ) from exc

        stderr_text = stderr_bytes.decode("utf-8", errors="replace")
        if proc.returncode != 0:
            raise MinerUInvocationError(
                f"MinerU exited with code {proc.returncode}",
                returncode=proc.returncode,
                stderr=stderr_text,
            )

    def _build_result(
        self,
        *,
        output_dir: Path,
        doc_stem: str,
        duration_seconds: float,
        document_id: str | None = None,
        gate_summary: dict[str, Any] | None = None,
    ) -> MinerUParseResult:
        """Construct a :class:`MinerUParseResult` from the post-processed directory."""
        if gate_summary is None:
            gate_summary = check(output_dir)
            if document_id is not None:
                gate_summary["document_id"] = document_id

        middle_path = output_dir / f"{doc_stem}_middle.json"
        middle_data: dict[str, Any] = json.loads(middle_path.read_text(encoding="utf-8"))

        images_dir = output_dir / "images"
        image_paths: list[Path] = []
        if images_dir.is_dir():
            image_paths = sorted(p for p in images_dir.iterdir() if p.is_file())

        return MinerUParseResult(
            output_dir=output_dir,
            markdown_path=output_dir / f"{doc_stem}.md",
            middle_json_path=middle_path,
            image_paths=image_paths,
            pages=len(middle_data.get("pdf_info", [])),
            gate_summary=gate_summary,
            duration_seconds=duration_seconds,
            mineru_version=middle_data.get("_version_name", ""),
            mineru_backend=middle_data.get("_backend", ""),
        )
