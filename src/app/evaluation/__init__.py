"""Evaluation sub-package for the Paper Notebook Agent.

Phase 4.4 — Parser evaluation harness.
Phase 6.5 — Retrieval evaluation harness.
Phase 9.1 — Golden QA evaluation harness.

Re-exports each harness's public API.
"""

from app.evaluation.goal_score import GoalScoreReport as GoalScoreReport
from app.evaluation.goal_score import ScoringInputs as ScoringInputs
from app.evaluation.goal_score import compute_goal_score as compute_goal_score
from app.evaluation.goal_score import render_markdown as render_goal_score_markdown
from app.evaluation.parser_eval import CorpusReport as CorpusReport
from app.evaluation.parser_eval import GoldenAnnotation as GoldenAnnotation
from app.evaluation.parser_eval import PaperEvalResult as PaperEvalResult
from app.evaluation.parser_eval import evaluate_corpus as evaluate_corpus
from app.evaluation.parser_eval import evaluate_paper as evaluate_paper
from app.evaluation.parser_eval import load_golden as load_golden
from app.evaluation.qa_eval import QACaseResult as QACaseResult
from app.evaluation.qa_eval import QACaseSpec as QACaseSpec
from app.evaluation.qa_eval import QACorpusReport as QACorpusReport
from app.evaluation.qa_eval import evaluate_corpus as evaluate_qa_corpus
from app.evaluation.qa_eval import load_qa_cases as load_qa_cases
from app.evaluation.qa_eval import run_case as run_qa_case

__all__ = [
    "CorpusReport",
    "GoalScoreReport",
    "GoldenAnnotation",
    "PaperEvalResult",
    "QACaseResult",
    "QACaseSpec",
    "QACorpusReport",
    "ScoringInputs",
    "compute_goal_score",
    "evaluate_corpus",
    "evaluate_paper",
    "evaluate_qa_corpus",
    "load_golden",
    "load_qa_cases",
    "render_goal_score_markdown",
    "run_qa_case",
]
