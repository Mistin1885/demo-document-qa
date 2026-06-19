"""Programmatic Vespa ApplicationPackage builder for the document_chunk schema.

This module is the **single source of truth** for the Vespa schema used by the
Paper Notebook Agent.  All field definitions, rank profiles, and deployment
helpers are expressed here so that:

1. The embedding dimension (``DIM``) is driven by ``Settings.embedding_dim`` at
   build time — it is never hard-coded in production logic.
2. The ``source_type`` enumeration (14 values per CLAUDE.md §5.2) and the
   source-type boost map are defined once in Python and cannot drift between
   the ``.sd`` file and application code.
3. ``build_application_package(embedding_dim)`` returns a pyvespa
   ``ApplicationPackage`` that can be (a) deployed via the deploy script and
   (b) written to ``deploy/vespa/application/`` for human review / git diff.

Usage
-----
    from app.vespa.app_package import build_application_package
    from app.config import get_settings

    pkg = build_application_package(get_settings().embedding_dim)
    pkg.to_files("deploy/vespa/application/")   # write .sd / services.xml / …

Rank profiles (CLAUDE.md §7):
- ``bm25_only``            – pure BM25, weighted across content/title/…
- ``semantic_only``        – pure vector closeness
- ``hybrid_first_phase``   – BM25 + semantic linear combination
- ``hybrid_with_native_rerank`` – same first-phase + Vespa-native second-phase boost
- ``hybrid_for_cross_encoder``  – first-phase only; Python cross-encoder reranks

All profiles expose ``match_features`` / ``summary_features`` so that
``SearchHit`` can carry per-stage scores (bm25(content), bm25(title),
closeness, etc.).

Source-type boost weights (``source_type_boost`` function in hybrid profiles):
Higher weight = ranked higher when other scores are equal.

    document_overview       → 2.0
    section_summary         → 1.8
    compact_section_summary → 1.5
    chapter_summary         → 1.6
    compact_chapter_summary → 1.3
    technology_card         → 1.4
    claim                   → 1.2
    definition              → 1.2
    performance_fact        → 1.3
    table_record            → 1.1
    figure_caption          → 1.0
    raw_block               → 0.9
    chunk                   → 1.0
    (default)               → 1.0
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from vespa.package import (
    HNSW,
    ApplicationPackage,
    Document,
    Field,
    FieldSet,
    Function,
    RankProfile,
    Schema,
    SecondPhaseRanking,
)

# ---------------------------------------------------------------------------
# Source-type enumeration (CLAUDE.md §5.2 — 14 values)
# ---------------------------------------------------------------------------

SOURCE_TYPES: Final[tuple[str, ...]] = (
    "raw_block",
    "chunk",
    "section_summary",
    "compact_section_summary",
    "chapter_summary",
    "compact_chapter_summary",
    "document_overview",
    "technology_card",
    "claim",
    "definition",
    "performance_fact",
    "table_record",
    "figure_caption",
    # The 14th type — a generic document-level compact variant
    "compact_document_overview",
)

# Boost weight per source_type.  The ``source_type_boost`` Vespa function maps
# the stored string to a numeric multiplier used in second-phase ranking.
# Default (any unrecognised value) = 1.0.
SOURCE_TYPE_BOOST: Final[dict[str, float]] = {
    "document_overview": 2.0,
    "compact_document_overview": 1.6,
    "section_summary": 1.8,
    "compact_section_summary": 1.5,
    "chapter_summary": 1.6,
    "compact_chapter_summary": 1.3,
    "technology_card": 1.4,
    "performance_fact": 1.3,
    "claim": 1.2,
    "definition": 1.2,
    "table_record": 1.1,
    "figure_caption": 1.0,
    "chunk": 1.0,
    "raw_block": 0.9,
}


def _build_source_type_boost_expression() -> str:
    """Build the Vespa YQL expression for the source-type boost map.

    Generates a nested ``if(attribute(source_type) == "X", W, ...)`` chain
    that returns the float weight for each known ``source_type``, falling
    through to 1.0 for any unrecognised value.
    """
    # Build from innermost (default) outward.
    expr = "1.0"
    for stype, weight in reversed(SOURCE_TYPE_BOOST.items()):
        expr = f'if(attribute(source_type) == "{stype}", {weight}, {expr})'
    return expr


def _build_bm25_first_phase() -> str:
    """Weighted BM25 first-phase expression.

    Weights per CLAUDE.md §7 task requirement:
    title 2.0 | heading_path 1.5 | content 1.0 |
    keywords 1.2 | technical_keywords 1.2 | entities 1.0
    """
    return (
        "2.0 * bm25(title)"
        " + 1.5 * bm25(heading_path)"
        " + 1.0 * bm25(content)"
        " + 1.2 * bm25(keywords)"
        " + 1.2 * bm25(technical_keywords)"
        " + 1.0 * bm25(entities)"
    )


def _build_schema(embedding_dim: int) -> Schema:
    """Return the ``document_chunk`` Schema for the given embedding dimension."""

    tensor_type = f"tensor<float>(x[{embedding_dim}])"

    # ------------------------------------------------------------------
    # Common features exposed by every rank profile so that SearchHit can
    # carry per-stage scores (CLAUDE.md §7).
    # ------------------------------------------------------------------
    common_match_features = [
        "bm25(content)",
        "bm25(title)",
        "bm25(heading_path)",
        "bm25(keywords)",
        "bm25(technical_keywords)",
        "closeness(field, embedding)",
    ]
    common_summary_features = common_match_features  # expose same set
    common_inputs = [("query(qvec)", tensor_type)]

    # ------------------------------------------------------------------
    # Shared Function definitions reused across profiles
    # ------------------------------------------------------------------
    fn_bm25_weighted = Function(name="bm25_weighted", expression=_build_bm25_first_phase())
    fn_semantic = Function(name="semantic_score", expression="closeness(field, embedding)")
    fn_source_type_boost = Function(
        name="source_type_boost", expression=_build_source_type_boost_expression()
    )
    fn_heading_match_boost = Function(
        name="heading_match_boost",
        expression=(
            "0.3 * bm25(title)"
            " + 0.2 * bm25(heading_path)"
        ),
    )

    # ------------------------------------------------------------------
    # Rank profiles
    # ------------------------------------------------------------------

    # 1. bm25_only
    rp_bm25_only = RankProfile(
        name="bm25_only",
        first_phase=_build_bm25_first_phase(),
        functions=[fn_bm25_weighted, fn_semantic],
        inputs=common_inputs,
        match_features=common_match_features,
        summary_features=common_summary_features,
    )

    # 2. semantic_only
    rp_semantic_only = RankProfile(
        name="semantic_only",
        first_phase="closeness(field, embedding)",
        functions=[fn_bm25_weighted, fn_semantic],
        inputs=common_inputs,
        match_features=common_match_features,
        summary_features=common_summary_features,
    )

    # 3. hybrid_first_phase — linear BM25 + semantic (no rerank inside Vespa)
    rp_hybrid_first_phase = RankProfile(
        name="hybrid_first_phase",
        first_phase=f"{_build_bm25_first_phase()} + 100.0 * closeness(field, embedding)",
        functions=[
            fn_bm25_weighted,
            fn_semantic,
            fn_source_type_boost,
            fn_heading_match_boost,
        ],
        inputs=common_inputs,
        second_phase=SecondPhaseRanking(
            expression=(
                "firstPhase"
                " + 0.5 * heading_match_boost"
                " + firstPhase * (source_type_boost - 1.0)"
            ),
            rerank_count=200,
        ),
        match_features=common_match_features,
        summary_features=common_summary_features,
    )

    # 4. hybrid_with_native_rerank — Vespa-native rerank (no cross-encoder)
    rp_hybrid_native = RankProfile(
        name="hybrid_with_native_rerank",
        inherits="hybrid_first_phase",
        # Override second-phase with a more explicit rerank expression
        second_phase=SecondPhaseRanking(
            expression=(
                "firstPhase * source_type_boost"
                " + 0.5 * heading_match_boost"
            ),
            rerank_count=200,
        ),
        functions=[
            fn_bm25_weighted,
            fn_semantic,
            fn_source_type_boost,
            fn_heading_match_boost,
        ],
        inputs=common_inputs,
        match_features=common_match_features,
        summary_features=common_summary_features,
    )

    # 5. hybrid_for_cross_encoder — first-phase only; Python cross-encoder reranks
    rp_hybrid_cross_encoder = RankProfile(
        name="hybrid_for_cross_encoder",
        first_phase=f"{_build_bm25_first_phase()} + 100.0 * closeness(field, embedding)",
        # No second_phase: all reranking is handled in Python
        functions=[fn_bm25_weighted, fn_semantic],
        inputs=common_inputs,
        match_features=common_match_features,
        summary_features=common_summary_features,
    )

    # ------------------------------------------------------------------
    # Schema assembly
    # ------------------------------------------------------------------
    return Schema(
        name="document_chunk",
        document=Document(
            fields=[
                # ---------- identity / routing ----------
                Field(
                    name="vespa_document_id",
                    type="string",
                    indexing=["attribute", "summary"],
                    attribute=["fast-search"],
                ),
                Field(
                    name="chat_id",
                    type="string",
                    indexing=["attribute", "summary"],
                    attribute=["fast-search"],
                ),
                Field(
                    name="document_id",
                    type="string",
                    indexing=["attribute", "summary"],
                    attribute=["fast-search"],
                ),
                Field(
                    name="source_node_id",
                    type="string",
                    indexing=["attribute", "summary"],
                ),
                Field(
                    name="parent_node_id",
                    type="string",
                    indexing=["attribute", "summary"],
                ),
                Field(
                    name="source_type",
                    type="string",
                    indexing=["attribute", "summary"],
                    attribute=["fast-search"],
                ),
                # ---------- searchable text fields ----------
                Field(
                    name="title",
                    type="string",
                    indexing=["index", "summary"],
                    index="enable-bm25",
                ),
                Field(
                    name="heading_path",
                    type="string",
                    indexing=["index", "summary"],
                    index="enable-bm25",
                ),
                Field(
                    name="content",
                    type="string",
                    indexing=["index", "summary"],
                    index="enable-bm25",
                ),
                # ---------- keyword / entity arrays ----------
                Field(
                    name="keywords",
                    type="array<string>",
                    indexing=["attribute", "index", "summary"],
                    index="enable-bm25",
                ),
                Field(
                    name="technical_keywords",
                    type="array<string>",
                    indexing=["attribute", "index", "summary"],
                    index="enable-bm25",
                ),
                Field(
                    name="entities",
                    type="array<string>",
                    indexing=["attribute", "index", "summary"],
                    index="enable-bm25",
                ),
                # ---------- page / order metadata ----------
                Field(
                    name="page_start",
                    type="int",
                    indexing=["attribute", "summary"],
                ),
                Field(
                    name="page_end",
                    type="int",
                    indexing=["attribute", "summary"],
                ),
                Field(
                    name="order_index",
                    type="int",
                    indexing=["attribute", "summary"],
                ),
                Field(
                    name="token_count",
                    type="int",
                    indexing=["attribute", "summary"],
                ),
                # ---------- embedding ----------
                Field(
                    name="embedding",
                    type=tensor_type,
                    indexing=["input content | embed e5", "attribute", "index"],
                    ann=HNSW(distance_metric="angular"),
                    is_document_field=False,
                ),
                # ---------- timestamp ----------
                Field(
                    name="created_at",
                    type="long",
                    indexing=["attribute", "summary"],
                ),
            ]
        ),
        fieldsets=[
            FieldSet(
                name="default",
                fields=[
                    "content",
                    "title",
                    "heading_path",
                    "keywords",
                    "technical_keywords",
                    "entities",
                ],
            )
        ],
        rank_profiles=[
            rp_bm25_only,
            rp_semantic_only,
            rp_hybrid_first_phase,
            rp_hybrid_native,
            rp_hybrid_cross_encoder,
        ],
    )


def _services_xml() -> str:
    """Return a single-node services.xml for Docker / local development.

    Hardcoded for redundancy=1 / single-node because this is a dev-only
    deployment (CLAUDE.md §5.2).  The container name is ``default``; the
    content cluster name is ``documents`` per the spec.

    """
    return """\
<?xml version="1.0" encoding="UTF-8"?>
<services version="1.0">
  <container id="default" version="1.0">
    <search/>
    <document-api/>
    <component id="e5" type="hugging-face-embedder">
      <transformer-model url="https://huggingface.co/intfloat/e5-small-v2/resolve/main/model.onnx"/>
      <tokenizer-model url="https://huggingface.co/intfloat/e5-small-v2/raw/main/tokenizer.json"/>
      <prepend>
        <query>query: </query>
        <document>passage: </document>
      </prepend>
    </component>
    <document-processing/>
  </container>
  <content id="documents" version="1.0">
    <redundancy>1</redundancy>
    <documents>
      <document type="document_chunk" mode="index"/>
    </documents>
    <group>
      <node distribution-key="0" hostalias="node1"/>
    </group>
  </content>
</services>
"""


def _hosts_xml() -> str:
    """Return a single-host hosts.xml for Docker / local development."""
    return """\
<?xml version="1.0" encoding="UTF-8"?>
<hosts>
  <host name="localhost">
    <alias>node1</alias>
  </host>
</hosts>
"""


def _validation_overrides_xml(until: str) -> str:
    """Return validation-overrides.xml allowing common schema-change validations.

    Parameters
    ----------
    until:
        ISO-8601 date (``YYYY-MM-DD``) until which the overrides are valid.
        Recommended: today + 29 days. Vespa rejects overrides more than 30
        days in the future, and time-zone differences can make exactly 30
        days too far.
    """
    return f"""\
<?xml version="1.0" encoding="UTF-8"?>
<validation-overrides>
  <!-- Allow tensor-type changes (e.g., when embedding_dim is updated) -->
  <allow until="{until}" comment="allow embedding dim changes">tensor-type-change</allow>
  <!-- Allow field-type changes during development -->
  <allow until="{until}" comment="allow field-type changes">field-type-change</allow>
  <!-- Allow indexing-mode changes (e.g., attribute ↔ index) -->
  <allow until="{until}" comment="allow indexing changes">indexing-change</allow>
</validation-overrides>
"""


def write_application_files(
    application_dir: Path,
    embedding_dim: int,
    *,
    validation_until: str | None = None,
) -> None:
    """Render the full Vespa application package to *application_dir*.

    This function:
    1. Calls ``build_application_package(embedding_dim).to_files(…)`` to
       produce ``schemas/document_chunk.sd`` (and pyvespa default XMLs).
    2. Overwrites ``services.xml``, ``hosts.xml``, and
       ``validation-overrides.xml`` with our custom single-node versions that
       match the spec (container name ``default``, content cluster
       ``documents``).

    Parameters
    ----------
    application_dir:
        Root directory of the Vespa application package (must exist).
    embedding_dim:
        Tensor dimension; passed directly to ``build_application_package``.
    validation_until:
        ISO date string for validation-overrides.xml. Defaults to today
        + 29 days if ``None``.
    """
    import datetime

    application_dir = Path(application_dir)
    application_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: let pyvespa write the schema (and default XMLs)
    pkg = build_application_package(embedding_dim)
    pkg.to_files(application_dir)

    # Step 2: overwrite XML files with our customised versions
    if validation_until is None:
        validation_until = (datetime.date.today() + datetime.timedelta(days=29)).isoformat()

    (application_dir / "services.xml").write_text(_services_xml(), encoding="utf-8")
    (application_dir / "hosts.xml").write_text(_hosts_xml(), encoding="utf-8")
    (application_dir / "validation-overrides.xml").write_text(
        _validation_overrides_xml(validation_until), encoding="utf-8"
    )


def build_application_package(embedding_dim: int) -> ApplicationPackage:
    """Build and return the Vespa ``ApplicationPackage`` for this project.

    Parameters
    ----------
    embedding_dim:
        The vector dimension for the ``embedding`` tensor field.  Must match
        the dimension produced by the configured embedding provider.  Never
        hard-code this value — always pass ``get_settings().embedding_dim``.

    Returns
    -------
    ApplicationPackage
        A fully configured pyvespa ``ApplicationPackage`` containing the
        ``document_chunk`` schema with all required fields and rank profiles.

    Example
    -------
    ::

        from app.vespa.app_package import build_application_package
        from app.config import get_settings

        pkg = build_application_package(get_settings().embedding_dim)
        pkg.to_files("deploy/vespa/application/")
    """
    if embedding_dim <= 0:
        raise ValueError(f"embedding_dim must be positive, got {embedding_dim}")

    schema = _build_schema(embedding_dim)
    return ApplicationPackage(name="documentchunk", schema=[schema])
