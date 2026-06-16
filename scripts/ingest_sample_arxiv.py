"""Download sample arXiv PDFs for parser evaluation.

Usage
-----
    uv run python scripts/ingest_sample_arxiv.py                  # default list
    uv run python scripts/ingest_sample_arxiv.py 2410.05779v3     # specific IDs
    uv run python scripts/ingest_sample_arxiv.py --help

Behaviour
---------
- Downloads to ``data/sample/arxiv/<id>.pdf``.
- Skips papers that already exist (idempotent).
- Polite: 3-second delay between requests.
- Uses HTTPX async client with a descriptive User-Agent.
- Does NOT require any API key.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import httpx

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_DEFAULT_PAPER_IDS: list[str] = ["2410.05779v3"]
_ARXIV_PDF_URL = "https://arxiv.org/pdf/{id}.pdf"
_OUTPUT_DIR = Path("data/sample/arxiv")
_USER_AGENT = "PaperNotebookAgent/0.1 (research)"
_POLITE_DELAY_SECONDS = 3.0
_DOWNLOAD_TIMEOUT = 120.0  # seconds


# ---------------------------------------------------------------------------
# Download helper
# ---------------------------------------------------------------------------


async def _download_paper(
    client: httpx.AsyncClient,
    paper_id: str,
    output_dir: Path,
) -> None:
    """Download a single arXiv PDF to ``output_dir/<paper_id>.pdf``.

    Skips gracefully if the file already exists.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    dest = output_dir / f"{paper_id}.pdf"

    if dest.exists():
        size_kb = dest.stat().st_size // 1024
        print(f"[SKIP] {paper_id} — already exists ({size_kb} KB): {dest}")
        return

    url = _ARXIV_PDF_URL.format(id=paper_id)
    print(f"[DOWNLOAD] {paper_id} <- {url}")

    response = await client.get(url, follow_redirects=True)
    response.raise_for_status()

    dest.write_bytes(response.content)
    size_kb = len(response.content) // 1024
    print(f"[OK] {paper_id} -> {dest} ({size_kb} KB)")


# ---------------------------------------------------------------------------
# Main async entry point
# ---------------------------------------------------------------------------


async def _main(paper_ids: list[str]) -> None:
    """Download papers sequentially with polite delays."""
    headers = {"User-Agent": _USER_AGENT}

    async with httpx.AsyncClient(
        headers=headers,
        timeout=_DOWNLOAD_TIMEOUT,
    ) as client:
        for i, paper_id in enumerate(paper_ids):
            await _download_paper(client, paper_id, _OUTPUT_DIR)
            if i < len(paper_ids) - 1:
                print(f"[WAIT] sleeping {_POLITE_DELAY_SECONDS:.0f}s (polite delay)...")
                await asyncio.sleep(_POLITE_DELAY_SECONDS)

    print(f"\n[DONE] Downloaded / verified {len(paper_ids)} paper(s).")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ingest_sample_arxiv",
        description=(
            "Download arXiv PDF samples for parser evaluation.\n\n"
            "Examples:\n"
            "  uv run python scripts/ingest_sample_arxiv.py\n"
            "  uv run python scripts/ingest_sample_arxiv.py 2410.05779v3\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "paper_ids",
        nargs="*",
        metavar="ARXIV_ID",
        help=(
            "One or more arXiv IDs to download (e.g. '2410.05779v3'). "
            f"Defaults to: {_DEFAULT_PAPER_IDS}"
        ),
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    paper_ids: list[str] = args.paper_ids if args.paper_ids else _DEFAULT_PAPER_IDS
    asyncio.run(_main(paper_ids))


if __name__ == "__main__":
    sys.exit(main())  # type: ignore[func-returns-value]
