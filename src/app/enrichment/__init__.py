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

from app.enrichment.document_overview import enrich_document_overview
from app.enrichment.facts import extract_structured_facts
from app.enrichment.manifest import build_chat_manifest
from app.enrichment.models import (
    ChatManifest,
    ChatManifestEntry,
    ChatManifestV2,
    ClaimItem,
    ConclusionItem,
    Contribution,
    DatasetMention,
    DefinitionItem,
    DocumentManifestEntry,
    DocumentOverview,
    FindingItem,
    LimitationItem,
    LimitationMention,
    ManifestTopic,
    MethodItem,
    MethodMention,
    MetricMention,
    PerformanceFactItem,
    SectionEnrichment,
)
from app.enrichment.section import (
    EnrichmentParseError,
    enrich_document_sections,
    enrich_section,
    enrich_sections,
)

__all__ = [
    "SectionEnrichment",
    "ClaimItem",
    "DefinitionItem",
    "MethodItem",
    "LimitationItem",
    "PerformanceFactItem",
    "enrich_sections",
    # Phase 5.1 spec-compliant API
    "enrich_section",
    "enrich_document_sections",
    "EnrichmentParseError",
    # Phase 5.2 LLM-JSON path
    "enrich_document_overview",
    "DocumentOverview",
    "Contribution",
    "FindingItem",
    "DatasetMention",
    "MetricMention",
    "ConclusionItem",
    "MethodMention",
    "LimitationMention",
    # Phase 5.3
    "extract_structured_facts",
    # Phase 5.4 — Chat-level manifest
    "build_chat_manifest",
    "ChatManifest",
    "ChatManifestEntry",
    "ChatManifestV2",
    "DocumentManifestEntry",
    "ManifestTopic",
]
