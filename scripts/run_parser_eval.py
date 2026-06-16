"""CLI entry point for running the parser evaluation harness.

Usage
-----
    uv run python scripts/run_parser_eval.py
    uv run python scripts/run_parser_eval.py --paper-id 2410.05779v3
    uv run python scripts/run_parser_eval.py --force-reparse
    uv run python scripts/run_parser_eval.py --help

Behaviour
---------
- Reads golden annotations from ``data/fixtures/golden/*.json``.
- Resolves each paper's PDF from ``data/sample/arxiv/<id>.pdf`` (primary)
  or ``data/<id>.pdf`` (fallback — Phase 1 PoC location).
- Connects to MinerUClient using ``app.config.get_settings().mineru_server_url``.
  If the MinerU server is unavailable and a parsed cache exists, the cached
  result is reused automatically (idempotent behaviour from Phase 4.1).
- Writes ``artifacts/evaluation/parser-report.json`` and
  ``artifacts/evaluation/parser-report.md``.
- Exit code 0 if all papers pass their gate; 1 if any fail.

Stand-alone: no DB / Vespa required.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path bootstrap: ensure 'src' is on sys.path so 'app.*' imports work when
# this script is run directly (without the pytest test runner).
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Inject minimal env vars so Settings validation doesn't fail in stand-alone mode.
os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://noop:noop@localhost:5432/noop")
os.environ.setdefault("APP_ENCRYPTION_KEY", "standalone-eval-key")

from app.config import get_settings  # noqa: E402
from app.evaluation.parser_eval import (  # noqa: E402
    CorpusReport,
    GoldenAnnotation,
    evaluate_corpus,
    load_golden,
)
from app.parsing.mineru_client import MinerUClient  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_GOLDEN_DIR = _REPO_ROOT / "data" / "fixtures" / "golden"
_SAMPLE_DIR = _REPO_ROOT / "data" / "sample" / "arxiv"
_LEGACY_DATA_DIR = _REPO_ROOT / "data"
_OUTPUT_DIR = _REPO_ROOT / "artifacts" / "evaluation"


# ---------------------------------------------------------------------------
# PDF resolver
# ---------------------------------------------------------------------------


def _resolve_pdf(golden: GoldenAnnotation) -> Path:
    """Find the PDF on disk for the given golden annotation.

    Search order:
    1. ``data/sample/arxiv/<pdf_filename>``
    2. ``data/<pdf_filename>``

    Raises
    ------
    FileNotFoundError
        With a hint to run ``ingest_sample_arxiv.py``.
    """
    candidates = [
        _SAMPLE_DIR / golden.pdf_filename,
        _LEGACY_DATA_DIR / golden.pdf_filename,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"PDF not found for '{golden.paper_id}' (tried: "
        + ", ".join(str(c) for c in candidates)
        + "). Run:\n"
        + f"  uv run python scripts/ingest_sample_arxiv.py {golden.paper_id}"
    )


# ---------------------------------------------------------------------------
# Main async logic
# ---------------------------------------------------------------------------


async def _run_eval(
    paper_ids: list[str] | None,
    *,
    force_reparse: bool,
) -> CorpusReport:
    settings = get_settings()
    client = MinerUClient(server_url=settings.mineru_server_url)

    # ------------------------------------------------------------------ #
    # Load golden annotations
    # ------------------------------------------------------------------ #
    if not _GOLDEN_DIR.is_dir():
        print(
            f"[ERROR] Golden annotations directory not found: {_GOLDEN_DIR}",
            file=sys.stderr,
        )
        sys.exit(2)

    golden_paths = sorted(_GOLDEN_DIR.glob("*.json"))
    if not golden_paths:
        print(f"[ERROR] No golden annotation files in {_GOLDEN_DIR}", file=sys.stderr)
        sys.exit(2)

    goldens: list[GoldenAnnotation] = [load_golden(p) for p in golden_paths]

    # Filter by paper_id if specified
    if paper_ids:
        goldens = [g for g in goldens if g.paper_id in paper_ids]
        if not goldens:
            print(
                f"[ERROR] No golden annotations found for paper IDs: {paper_ids}",
                file=sys.stderr,
            )
            sys.exit(2)

    # ------------------------------------------------------------------ #
    # Resolve PDFs
    # ------------------------------------------------------------------ #
    pairs: list[tuple[Path, GoldenAnnotation]] = []
    for golden in goldens:
        try:
            pdf_path = _resolve_pdf(golden)
            pairs.append((pdf_path, golden))
        except FileNotFoundError as exc:
            print(f"[ERROR] {exc}", file=sys.stderr)
            sys.exit(2)

    # ------------------------------------------------------------------ #
    # Run evaluation
    # ------------------------------------------------------------------ #
    print(f"[INFO] Running parser eval on {len(pairs)} paper(s)...")
    print(f"[INFO] MinerU server: {settings.mineru_server_url}")
    print(f"[INFO] Output dir:    {_OUTPUT_DIR}")
    if force_reparse:
        print("[INFO] --force-reparse is set; ignoring existing cache")
    print()

    report = await evaluate_corpus(
        pairs,
        client=client,
        output_dir=_OUTPUT_DIR,
        force_reparse=force_reparse,
    )

    return report


# ---------------------------------------------------------------------------
# Report printer
# ---------------------------------------------------------------------------


def _print_summary(report: CorpusReport) -> None:
    print(f"\n{'='*60}")
    print("PARSER EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total papers : {report.total_papers}")
    print(f"Passed       : {report.passed}")
    print(f"Failed       : {report.failed}")
    print(f"Pass rate    : {report.pass_rate:.1%}")
    print(f"Avg H-F1     : {report.avg_heading_f1:.3f}")
    print(f"Avg Math Rec : {report.avg_inline_math_recall:.3f}")
    print()

    for r in report.papers:
        status = "PASS" if r.gate_passed else "FAIL"
        print(f"  [{status}] {r.paper_id} — H-F1={r.heading_f1:.3f}, refs={r.reference_count}, "
              f"figs={r.figure_count}, tables={r.table_count}")
        if r.failure_reasons:
            for reason in r.failure_reasons:
                print(f"       ! {reason}")

    print()
    json_out = _OUTPUT_DIR / "parser-report.json"
    md_out = _OUTPUT_DIR / "parser-report.md"
    print("Reports written to:")
    print(f"  {json_out}")
    print(f"  {md_out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_parser_eval",
        description=(
            "Run the Phase 4.4 parser evaluation harness.\n\n"
            "Reads golden annotations from data/fixtures/golden/*.json,\n"
            "runs MinerU + mapping + hierarchy, computes metrics, and writes\n"
            "artifacts/evaluation/parser-report.{json,md}.\n\n"
            "Stand-alone: no DB or Vespa required.\n"
            "Dry-run: if data/parsed/<id>/ cache exists, MinerU is NOT re-run.\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--paper-id",
        dest="paper_ids",
        metavar="ID",
        action="append",
        help="Evaluate only this paper ID (may be repeated). Default: all golden files.",
    )
    parser.add_argument(
        "--force-reparse",
        action="store_true",
        default=False,
        help="Ignore existing parse cache and re-run MinerU (requires server).",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    report = asyncio.run(
        _run_eval(
            paper_ids=args.paper_ids,
            force_reparse=args.force_reparse,
        )
    )
    _print_summary(report)

    sys.exit(0 if report.failed == 0 else 1)


if __name__ == "__main__":
    main()
