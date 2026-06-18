"""Unit tests for ``app.vespa.app_package``.

Deterministic — no external dependencies (no Vespa, no network).
"""

from __future__ import annotations

import pytest

from app.vespa.app_package import (
    SOURCE_TYPE_BOOST,
    SOURCE_TYPES,
    build_application_package,
    write_application_files,
)

DEFAULT_DIM = 1024

REQUIRED_FIELDS = [
    "vespa_document_id", "chat_id", "document_id", "source_node_id",
    "parent_node_id", "source_type", "title", "heading_path", "content",
    "keywords", "technical_keywords", "entities", "page_start", "page_end",
    "order_index", "token_count", "embedding", "created_at",
]

REQUIRED_RANK_PROFILES = [
    "bm25_only",
    "semantic_only",
    "hybrid_first_phase",
    "hybrid_with_native_rerank",
    "hybrid_for_cross_encoder",
]


@pytest.fixture(scope="module")
def schema_1024():
    return build_application_package(DEFAULT_DIM).schema


# ---------------------------------------------------------------------------
# Schema basics: fields + embedding tensor type
# ---------------------------------------------------------------------------


def test_schema_required_fields_and_tensor(schema_1024) -> None:
    """All required fields present; embedding tensor type reflects the dim."""
    field_names = {f.name for f in schema_1024.document.fields}
    assert not (set(REQUIRED_FIELDS) - field_names)
    emb = next(f for f in schema_1024.document.fields if f.name == "embedding")
    assert emb.type == f"tensor<float>(x[{DEFAULT_DIM}])"

    # Custom dim updates the type string
    pkg_512 = build_application_package(512)
    emb_512 = next(f for f in pkg_512.schema.document.fields if f.name == "embedding")
    assert emb_512.type == "tensor<float>(x[512])"


def test_invalid_dim_raises() -> None:
    with pytest.raises(ValueError, match="embedding_dim must be positive"):
        build_application_package(0)
    with pytest.raises(ValueError, match="embedding_dim must be positive"):
        build_application_package(-1)


# ---------------------------------------------------------------------------
# Rank profiles
# ---------------------------------------------------------------------------


def test_rank_profiles_structure(schema_1024) -> None:
    """All 5 profiles present; correct first/second phase; match_features have bm25+closeness."""
    profile_names = list(schema_1024.rank_profiles.keys())
    for name in REQUIRED_RANK_PROFILES:
        assert name in profile_names

    rp_bm25 = schema_1024.rank_profiles["bm25_only"]
    assert "bm25" in str(rp_bm25.first_phase) and "closeness" not in str(rp_bm25.first_phase)

    for name in ("hybrid_first_phase", "hybrid_for_cross_encoder"):
        expr = str(schema_1024.rank_profiles[name].first_phase)
        assert "closeness" in expr and "bm25" in expr

    assert schema_1024.rank_profiles["hybrid_with_native_rerank"].second_phase is not None
    assert schema_1024.rank_profiles["hybrid_for_cross_encoder"].second_phase is None

    for name in REQUIRED_RANK_PROFILES:
        rp = schema_1024.rank_profiles[name]
        feature_strs = [str(f) for f in rp.match_features]
        assert any("bm25" in f for f in feature_strs)
        assert any("closeness" in f for f in feature_strs)


# ---------------------------------------------------------------------------
# source_type enumeration + boost map
# ---------------------------------------------------------------------------


def test_source_types_count_and_boost(schema_1024) -> None:
    """Exactly 14 source types; all have positive boost (explicit or default 1.0);
    source_type_boost function in hybrid_first_phase references all boosted entries."""
    assert len(SOURCE_TYPES) == 14, f"Expected 14 source types, got {len(SOURCE_TYPES)}"

    for stype in SOURCE_TYPES:
        w = SOURCE_TYPE_BOOST.get(stype, 1.0)
        assert isinstance(w, (int, float)) and w > 0

    rp = schema_1024.rank_profiles["hybrid_first_phase"]
    fn_map = {fn.name: fn for fn in (rp.functions or [])}
    assert "source_type_boost" in fn_map
    expr = str(fn_map["source_type_boost"].expression)
    for stype in SOURCE_TYPE_BOOST:
        assert stype in expr


# ---------------------------------------------------------------------------
# write_application_files — disk output
# ---------------------------------------------------------------------------


def test_write_application_files(tmp_path) -> None:
    """Creates document_chunk.sd with correct tensor type; services.xml, hosts.xml,
    validation-overrides.xml all created with required content."""
    write_application_files(tmp_path, embedding_dim=1024, validation_until="2026-12-31")

    # Schema file
    sd = tmp_path / "schemas" / "document_chunk.sd"
    assert sd.exists()
    sd_content = sd.read_text()
    assert "tensor<float>(x[1024])" in sd_content
    for name in REQUIRED_RANK_PROFILES:
        assert name in sd_content

    # Dim override
    d256 = tmp_path / "dim256"
    write_application_files(d256, embedding_dim=256)
    sd256 = (d256 / "schemas" / "document_chunk.sd").read_text()
    assert "tensor<float>(x[256])" in sd256 and "tensor<float>(x[1024])" not in sd256

    # services.xml
    svc = (tmp_path / "services.xml").read_text()
    assert 'id="default"' in svc and 'id="documents"' in svc

    # hosts.xml
    hosts = (tmp_path / "hosts.xml").read_text()
    assert "node1" in hosts and "localhost" in hosts

    # validation-overrides.xml
    vo = (tmp_path / "validation-overrides.xml").read_text()
    assert "tensor-type-change" in vo and "2026-12-31" in vo
