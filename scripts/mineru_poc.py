"""Phase 1 MinerU PoC driver + post-processor.

Workflow per PDF:

    1. Invoke ``mineru -b hybrid-http-client -u $MINERU_SERVER_URL``.
    2. Post-process the per-doc output dir:
       a. Rename ``images/<sha256>.jpg`` → ``<doc>_p<page>_<short_hash>.jpg``
          and update every reference (markdown + middle.json).
       b. Rebuild ``<doc>.md`` from the ordered ``content_list.json`` blocks,
          wrapping each PDF page in ``<Page N>...</Page N>`` markers.
       c. Delete every output file except ``<doc>.md``, ``<doc>_middle.json``,
          and the ``images/`` directory.
    3. Run a gate check (markdown + middle.json only) and emit a JSON summary.

Usage:
    uv run python scripts/mineru_poc.py data/2410.05779v3.pdf data/parsed

Note: post-processing pure functions live in ``app.parsing._postprocess`` and
are re-exported here for backwards-compatibility.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

# Re-export pure functions from the canonical module so any existing code that
# does ``from scripts.mineru_poc import postprocess, check`` (if scripts were
# ever on sys.path) keeps working.  The PoC CLI itself uses them directly.
from app.parsing._postprocess import (  # noqa: F401 (re-exports)
    apply_rename_to_disk,
    build_rename_map,
    check,
    cleanup_extras,
    postprocess,
    render_md_with_page_markers,
    rewrite_middle_image_paths,
)

MINERU_URL = os.environ.get("MINERU_SERVER_URL", "http://localhost:8001")


# ---------------------------------------------------------------------------
# 1) Run MinerU (synchronous CLI helper — not used by the async client)
# ---------------------------------------------------------------------------


def run_mineru(pdf: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "uv", "run", "mineru",
        "-p", str(pdf),
        "-o", str(out_dir),
        "-b", "hybrid-http-client",
        "-u", MINERU_URL,
    ]
    print(f"[poc] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    parsed = out_dir / pdf.stem / "hybrid_auto"
    if not parsed.is_dir():
        raise SystemExit(f"expected output dir {parsed} missing")
    return parsed


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    pdf = Path(argv[1]).resolve()
    out_dir = Path(argv[2] if len(argv) > 2 else "data/parsed").resolve()
    parsed = run_mineru(pdf, out_dir)
    post = postprocess(parsed, pdf.stem)
    summary = check(parsed) | {"postprocess": post}
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
