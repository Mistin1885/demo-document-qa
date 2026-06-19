#!/usr/bin/env python
"""Deploy the Vespa application package to a running Vespa config server.

This script:
1. Reads ``Settings.embedding_dim`` from the application config.
2. Renders the full application package (schema + XML files) to
   ``deploy/vespa/application/`` via ``app_package.write_application_files``.
3. Zips the application directory and POSTs it to the Vespa config server
   ``/application/v2/tenant/default/prepareandactivate`` endpoint.
4. Polls the returned session URL until the deployment reaches state
   ``active``.

Usage
-----
    uv run python scripts/deploy_vespa.py
    uv run python scripts/deploy_vespa.py --dry-run   # render files only
    uv run python scripts/deploy_vespa.py --config-url http://localhost:19071 --application-dir deploy/vespa/application

Flags
-----
--dry-run           Render application files but skip upload to Vespa.
--application-dir   Path to the Vespa application directory.
                    Default: ``deploy/vespa/application``
--config-url        Vespa config server base URL.
                    Default: ``http://localhost:19071``
--embedding-dim     Override the embedding dimension (overrides Settings).
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import time
import zipfile
from pathlib import Path
from urllib.parse import urlparse, urlunparse

import httpx

# ---------------------------------------------------------------------------
# Ensure repo root is on sys.path so we can import ``app``
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from app.vespa.app_package import write_application_files  # noqa: E402, I001


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_POLL_INTERVAL_S = 2.0
_MAX_POLL_ATTEMPTS = 60  # 2 minutes total


def _zip_application_dir(application_dir: Path) -> bytes:
    """Zip an application directory into an in-memory bytes buffer."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in sorted(application_dir.rglob("*")):
            if file_path.is_file():
                arcname = file_path.relative_to(application_dir)
                zf.write(file_path, arcname)
    return buf.getvalue()


def _deploy_zip(config_url: str, zip_bytes: bytes) -> str:
    """POST the zipped application to Vespa and return the session URL.

    The config server returns a ``Location`` header or a JSON body with
    ``session-id`` that we use to poll for activation status.

    Returns the session URL to poll.
    """
    url = f"{config_url.rstrip('/')}/application/v2/tenant/default/prepareandactivate"
    headers = {
        "Content-Type": "application/zip",
    }
    with httpx.Client(timeout=60.0) as client:
        resp = client.post(url, content=zip_bytes, headers=headers)
        resp.raise_for_status()
        data = resp.json()

    # Vespa returns something like:
    # {"session-id": 2, "tenant": "default", "url": "http://…/session/2/prepared"}
    # The final activation URL differs slightly between versions; we poll the
    # session status URL.
    session_url: str | None = data.get("url")
    if not session_url:
        session_id = data.get("session-id")
        if session_id is not None:
            session_url = (
                f"{config_url.rstrip('/')}/application/v2/tenant/default"
                f"/session/{session_id}/status"
            )
        else:
            # Fallback: try the tenant application URL
            session_url = (
                f"{config_url.rstrip('/')}/application/v2/tenant/default"
                "/application/default/environment/default/region/default/instance/default"
            )
    return session_url


def _wait_for_active(config_url: str, session_url: str) -> None:
    """Poll *session_url* until Vespa reports ``active`` deployment."""
    # Vespa may return a session URL using the container's internal host/port
    # (for compose: localhost:19071). Normalize it to the user-supplied
    # config_url so polling works from the host (e.g. localhost:19072).
    session_url = _normalize_session_url(config_url, session_url)

    # Also check the tenant application status directly as fallback
    app_status_url = (
        f"{config_url.rstrip('/')}/application/v2/tenant/default"
        "/application/default/environment/default/region/default/instance/default"
    )
    print(f"Polling deployment status: {session_url}", flush=True)
    with httpx.Client(timeout=10.0) as client:
        for attempt in range(1, _MAX_POLL_ATTEMPTS + 1):
            try:
                resp = client.get(session_url)
                if resp.status_code == 200:
                    data = resp.json()
                    status = data.get("status", data.get("generation", "unknown"))
                    print(f"  [{attempt:02d}] status={status!r}", flush=True)
                    if status == "active" or (
                        isinstance(status, int) and status > 0
                    ):
                        print("Deployment active.", flush=True)
                        return
                else:
                    # Session URL might be a "prepare" URL; try app status
                    resp2 = client.get(app_status_url)
                    if resp2.status_code == 200:
                        print("  Deployment confirmed via app status.", flush=True)
                        return
            except httpx.HTTPError as exc:
                print(f"  [{attempt:02d}] HTTP error: {exc}", flush=True)
            time.sleep(_POLL_INTERVAL_S)
    raise TimeoutError(
        f"Vespa deployment did not reach 'active' after {_MAX_POLL_ATTEMPTS} polls."
    )


def _normalize_session_url(config_url: str, session_url: str) -> str:
    """Return *session_url* with scheme/netloc from *config_url*.

    Config servers commonly return URLs that are only valid from inside the
    Vespa container. Keeping the returned path but replacing scheme/netloc lets
    the deploy script work with host port mappings.
    """
    config_parts = urlparse(config_url.rstrip("/"))
    session_parts = urlparse(session_url)
    if not session_parts.scheme or not session_parts.netloc:
        return f"{config_url.rstrip('/')}/{session_url.lstrip('/')}"
    return urlunparse(
        session_parts._replace(
            scheme=config_parts.scheme,
            netloc=config_parts.netloc,
        )
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render + deploy the Vespa application package."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Render application files without uploading to Vespa.",
    )
    parser.add_argument(
        "--application-dir",
        default="deploy/vespa/application",
        help="Path to the Vespa application directory (default: deploy/vespa/application).",
    )
    parser.add_argument(
        "--config-url",
        default=None,
        help="Vespa config server URL (default: http://localhost:19071 or VESPA_CONFIG_URL env).",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=None,
        help="Override embedding dimension (default: Settings.embedding_dim).",
    )
    return parser


def main() -> int:
    parser = _build_arg_parser()
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Resolve settings
    # ------------------------------------------------------------------
    # Load embedding_dim from settings (with env override)
    embedding_dim: int
    if args.embedding_dim is not None:
        embedding_dim = args.embedding_dim
    else:
        # Import settings lazily so that the script doesn't fail when
        # DATABASE_URL / APP_ENCRYPTION_KEY are not set in --dry-run mode
        # with defaults.  We only need embedding_dim here.
        try:
            from app.config import get_settings
            embedding_dim = get_settings().embedding_dim
        except Exception:
            # Fallback default; Settings might raise if required vars absent
            embedding_dim = int(os.environ.get("EMBEDDING_DIM", "384"))

    config_url: str
    if args.config_url:
        config_url = args.config_url
    else:
        config_url = os.environ.get("VESPA_CONFIG_URL", "http://localhost:19071")

    application_dir = (_REPO_ROOT / args.application_dir).resolve()

    print(f"embedding_dim     : {embedding_dim}")
    print(f"application_dir   : {application_dir}")
    print(f"config_url        : {config_url}")
    print(f"dry_run           : {args.dry_run}")
    print()

    # ------------------------------------------------------------------
    # Step 1 — Render application files
    # ------------------------------------------------------------------
    print("Rendering application files...", flush=True)
    write_application_files(application_dir, embedding_dim)

    # List generated files
    for f in sorted(application_dir.rglob("*")):
        if f.is_file():
            print(f"  Generated: {f}", flush=True)

    if args.dry_run:
        print("\nDry-run mode — skipping Vespa upload.")
        return 0

    # ------------------------------------------------------------------
    # Step 2 — Zip and upload
    # ------------------------------------------------------------------
    print("\nZipping application...", flush=True)
    zip_bytes = _zip_application_dir(application_dir)
    print(f"  ZIP size: {len(zip_bytes):,} bytes", flush=True)

    print("Uploading to Vespa config server...", flush=True)
    try:
        session_url = _deploy_zip(config_url, zip_bytes)
    except httpx.HTTPStatusError as exc:
        print(f"Deploy failed (HTTP {exc.response.status_code}): {exc.response.text}")
        return 1
    except httpx.ConnectError as exc:
        print(f"Cannot connect to Vespa config server at {config_url}: {exc}")
        print("Is Vespa running?  Try: docker compose -f deploy/docker-compose.yml up -d vespa")
        return 1

    # ------------------------------------------------------------------
    # Step 3 — Wait for active
    # ------------------------------------------------------------------
    try:
        _wait_for_active(config_url, session_url)
    except TimeoutError as exc:
        print(f"Warning: {exc}")
        print("The deployment may still succeed — check Vespa logs.")
        return 1

    print("\nDone. Vespa application package deployed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
