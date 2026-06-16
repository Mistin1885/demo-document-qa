"""Document parsing sub-package.

Public surface
--------------
Phase 4.1 — MinerU async client:
- :class:`MinerUClient` — async client for MinerU hybrid-backend PDF parsing.
- :class:`MinerUParseResult` — structured result returned by
  :meth:`MinerUClient.parse_pdf`.

Phase 4.2 — middle.json → ParsedBlock mapping:
- :class:`ParsedBlock`, :class:`BBox`, :class:`BlockType`,
  :class:`ImageRef`, :class:`TableRef` — domain models.
- :func:`map_middle_to_parsed_blocks` — pure mapping function.
- :func:`load_middle_json` — read + JSON-parse helper.
- :func:`extract_text_from_block` — span text extractor (reused by Phase 4.3).

Phase 4.3 — hierarchy derivation:
- :class:`NodeType` — document node type enum.
- :class:`DocumentNodeOut` — output node Pydantic model.
- :class:`HierarchyResult` — full hierarchy result for one document.
- :func:`derive_hierarchy` — pure function: ``list[ParsedBlock] → HierarchyResult``.

Error hierarchy
---------------
- :exc:`MinerUError` — base exception (Phase 4.1).
- :exc:`MinerUServerUnavailable` — server not reachable.
- :exc:`MinerUInvocationError` — subprocess failure.
- :exc:`MinerUGateFailure` — post-processing passed but gate check failed.
- :exc:`ParsingError` — middle.json load/parse failure (Phase 4.2).
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Phase 4.3 — hierarchy derivation
# ---------------------------------------------------------------------------
from app.parsing.hierarchy import derive_hierarchy

# ---------------------------------------------------------------------------
# Phase 4.2 — mapping functions
# ---------------------------------------------------------------------------
from app.parsing.mapping import (
    ParsingError,
    extract_text_from_block,
    load_middle_json,
    map_middle_to_parsed_blocks,
)

# ---------------------------------------------------------------------------
# Phase 4.1 — MinerU async client
# ---------------------------------------------------------------------------
from app.parsing.mineru_client import (
    MinerUClient,
    MinerUError,
    MinerUGateFailure,
    MinerUInvocationError,
    MinerUParseResult,
    MinerUServerUnavailable,
)

# ---------------------------------------------------------------------------
# Phase 4.2 — domain models
# ---------------------------------------------------------------------------
from app.parsing.models import (
    BBox,
    BlockType,
    DocumentNodeOut,
    HierarchyResult,
    ImageRef,
    NodeType,
    ParsedBlock,
    TableRef,
)

__all__ = [
    # Phase 4.1
    "MinerUClient",
    "MinerUError",
    "MinerUGateFailure",
    "MinerUInvocationError",
    "MinerUParseResult",
    "MinerUServerUnavailable",
    # Phase 4.2 models
    "BBox",
    "BlockType",
    "ImageRef",
    "ParsedBlock",
    "TableRef",
    # Phase 4.2 mapping
    "ParsingError",
    "extract_text_from_block",
    "load_middle_json",
    "map_middle_to_parsed_blocks",
    # Phase 4.3 models
    "NodeType",
    "DocumentNodeOut",
    "HierarchyResult",
    # Phase 4.3 hierarchy
    "derive_hierarchy",
]
