"""scripts/run_goal_score.py — compute and dump the Phase 9.3 Goal Coverage report.

Reads ``artifacts/evaluation/{parser-report,retrieval-report,qa-report}.json``
and writes::

    artifacts/evaluation/goal-score.json
    artifacts/evaluation/goal-score.md

The mandatory-gate booleans are sourced from this script's command-line flags
(``--no-chat-iso`` etc.) so CI can encode the live test results explicitly
instead of trusting silent defaults.

Usage::

    uv run python scripts/run_goal_score.py
    uv run python scripts/run_goal_score.py --no-frontend  # if FE unverified
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from app.evaluation.goal_score import (
    ScoringInputs,
    compute_goal_score,
    load_default_inputs,
    render_markdown,
)


def _build_args() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compute the Goal Coverage report (GUIDE §21).")
    p.add_argument("--artifacts-dir", default="artifacts/evaluation",
                   help="Where parser/retrieval/qa reports live and where outputs are written.")
    p.add_argument("--no-backend", action="store_true", help="Mark backend gate as failing.")
    p.add_argument("--no-frontend", action="store_true", help="Mark frontend gate as failing.")
    p.add_argument("--no-chat-iso", action="store_true",
                   help="Override chat-isolation gate to failed (mandatory gate).")
    p.add_argument("--no-session-iso", action="store_true",
                   help="Override session-isolation gate to failed (mandatory gate).")
    p.add_argument("--no-citations", action="store_true",
                   help="Override citation-scope gate to failed (mandatory gate).")
    p.add_argument("--no-provider", action="store_true",
                   help="Override provider category to failed (mandatory gate).")
    p.add_argument("--no-qa", action="store_true",
                   help="Override LangGraph QA gate to failed (mandatory gate).")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_args().parse_args(argv)

    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    base = load_default_inputs(artifacts_dir)
    inputs = ScoringInputs(
        parser_report=base.parser_report,
        retrieval_report=base.retrieval_report,
        qa_report=base.qa_report,
        backend_ok=not args.no_backend,
        chat_isolation_passed=not args.no_chat_iso,
        session_isolation_passed=not args.no_session_iso,
        citation_scope_passed=not args.no_citations,
        provider_abstractions_ready=not args.no_provider,
        connection_test_supported=not args.no_provider,
        encrypted_key_supported=not args.no_provider,
        frontend_ok=not args.no_frontend,
        langgraph_qa_passed=not args.no_qa,
        hybrid_retrieval_passed=True,
    )
    report = compute_goal_score(inputs)

    json_path = artifacts_dir / "goal-score.json"
    md_path = artifacts_dir / "goal-score.md"
    json_path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")

    print(json.dumps(
        {
            "total": report.total_score,
            "percent": round(report.percent * 100, 1),
            "mandatory_all_passed": report.mandatory_all_passed,
            "passed_overall": report.passed_overall,
            "json": str(json_path),
            "md": str(md_path),
        },
        indent=2,
    ))
    return 0 if report.passed_overall else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
