"""Evaluation sub-package for the Paper Notebook Agent.

Phase 4.4 — Parser evaluation harness.
Phase 6   — Retrieval evaluation harness (planned; separate module).

Re-exports the parser evaluation public API so callers can do::

    from app.evaluation import evaluate_paper, evaluate_corpus, GoldenAnnotation
"""

from app.evaluation.parser_eval import CorpusReport as CorpusReport
from app.evaluation.parser_eval import GoldenAnnotation as GoldenAnnotation
from app.evaluation.parser_eval import PaperEvalResult as PaperEvalResult
from app.evaluation.parser_eval import evaluate_corpus as evaluate_corpus
from app.evaluation.parser_eval import evaluate_paper as evaluate_paper
from app.evaluation.parser_eval import load_golden as load_golden

__all__ = [
    "GoldenAnnotation",
    "PaperEvalResult",
    "CorpusReport",
    "load_golden",
    "evaluate_paper",
    "evaluate_corpus",
]
