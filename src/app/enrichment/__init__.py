"""Section-level enrichment for Paper Notebook Agent.

Phase 5.1 — Section / subsection / appendix nodes are enriched with
structured metadata (summaries, keywords, entities, claims, definitions,
methods, limitations, performance facts, related figure/table IDs).

Public API
----------
- :mod:`app.enrichment.models` — ``SectionEnrichment`` and item types.
- :mod:`app.enrichment.section` — ``enrich_sections`` async entry-point.

Design constraints (CLAUDE.md §12)
-------------------------------------
- No FastAPI or SQLAlchemy imports here.
- All LLM calls go through ``app.providers.base.ChatProvider``.
- Outputs are deterministic for a given (input, provider) pair.
"""

from app.enrichment.facts import extract_structured_facts
from app.enrichment.models import (
    ClaimItem,
    DefinitionItem,
    LimitationItem,
    MethodItem,
    PerformanceFactItem,
    SectionEnrichment,
)
from app.enrichment.section import enrich_sections

__all__ = [
    "SectionEnrichment",
    "ClaimItem",
    "DefinitionItem",
    "MethodItem",
    "LimitationItem",
    "PerformanceFactItem",
    "enrich_sections",
    # Phase 5.3
    "extract_structured_facts",
]
