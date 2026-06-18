"""scripts/run_qa_eval.py — run the Phase 9.1 Golden QA corpus and write the report.

Mirrors ``run_parser_eval.py`` / ``run_retrieval_eval.py``: runs every case in
``data/fixtures/qa_cases.json`` against the LangGraph StateGraph with a
deterministic mock retrieval + CitingMockChatProvider and writes::

    artifacts/evaluation/qa-report.json
    artifacts/evaluation/qa-report.md

Exit code is non-zero when ``pass_rate < 0.85`` so CI can gate the run.

Usage::

    uv run python scripts/run_qa_eval.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from app.evaluation.qa_eval import evaluate_corpus, load_qa_cases

_DEFAULT_FIXTURE = Path("data/fixtures/qa_cases.json")
_DEFAULT_ARTIFACTS = Path("artifacts/evaluation")
_GATE_MIN_PASS_RATE = 0.85


def _render_markdown(report) -> str:  # type: ignore[no-untyped-def]
    lines: list[str] = []
    lines.append("# Golden QA Evaluation Report (GUIDE §19)")
    lines.append("")
    lines.append(
        f"**Pass rate: {report.passed}/{report.total} = {report.pass_rate * 100:.1f}%** "
        f"(gate: ≥ {_GATE_MIN_PASS_RATE * 100:.0f}%)"
    )
    lines.append("")
    lines.append("| case_id | kind | passed | citations | doc_labels | coverage |")
    lines.append("|---|---|---|---|---|---|")
    for r in report.results:
        lines.append(
            f"| {r.case_id} | {r.kind} | {'✓' if r.passed else '✗'} | "
            f"{r.citation_count} | {', '.join(r.citation_doc_labels) or '-'} | "
            f"{r.coverage_state} |"
        )
    lines.append("")
    if any(not r.passed for r in report.results):
        lines.append("## Failures")
        for r in report.results:
            if r.passed:
                continue
            lines.append(f"- **{r.case_id}**: {'; '.join(r.failures)}")
    return "\n".join(lines) + "\n"


async def _amain(fixture: Path, artifacts_dir: Path) -> int:
    cases = load_qa_cases(fixture)
    report = await evaluate_corpus(cases)

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    (artifacts_dir / "qa-report.json").write_text(
        report.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    (artifacts_dir / "qa-report.md").write_text(_render_markdown(report), encoding="utf-8")

    print(json.dumps(
        {
            "total": report.total,
            "passed": report.passed,
            "failed": report.failed,
            "pass_rate": report.pass_rate,
            "gate_min_pass_rate": _GATE_MIN_PASS_RATE,
            "gate_passed": report.pass_rate >= _GATE_MIN_PASS_RATE,
        },
        indent=2,
    ))
    return 0 if report.pass_rate >= _GATE_MIN_PASS_RATE else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, default=_DEFAULT_FIXTURE)
    parser.add_argument("--artifacts-dir", type=Path, default=_DEFAULT_ARTIFACTS)
    args = parser.parse_args(argv)
    return asyncio.run(_amain(args.fixture, args.artifacts_dir))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
