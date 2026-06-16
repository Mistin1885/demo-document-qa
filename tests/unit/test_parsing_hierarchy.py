"""Unit tests for app.parsing.hierarchy.derive_hierarchy.

Coverage
--------
- happy-path on synthetic 8-page fixture: node count, root existence, tree validity.
- references_start_index and appendix_start_index are set correctly.
- abstract and authors heuristics fire on expected blocks.
- section vs subsection discrimination ("3" → section, "3.1" → subsection).
- one-owner invariant: no ParsedBlock appears in more than one node's source_block_ids.
- coverage: paragraph/title/ref_text/image/table/equation blocks are all owned;
  discarded blocks are not owned.
- deterministic id: two calls with the same inputs produce identical node ids.
- real-data smoke test against data/parsed/2410.05779v3 (skipped when absent).
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from app.parsing.hierarchy import derive_hierarchy
from app.parsing.mapping import map_middle_to_parsed_blocks
from app.parsing.models import BlockType, DocumentNodeOut, NodeType, ParsedBlock

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PAPER_FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "mineru_sample_paper"
PAPER_MIDDLE = PAPER_FIXTURE_DIR / "middle.json"

REAL_MIDDLE = (
    Path(__file__).parent.parent.parent
    / "data"
    / "parsed"
    / "2410.05779v3"
    / "hybrid_auto"
    / "2410.05779v3_middle.json"
)

# Stable UUIDs for determinism tests
_CHAT_ID = UUID("cccccccc-cccc-cccc-cccc-cccccccccccc")
_DOC_ID = UUID("dddddddd-dddd-dddd-dddd-dddddddddddd")


# ---------------------------------------------------------------------------
# Module-scoped fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def paper_blocks() -> list[ParsedBlock]:
    middle = json.loads(PAPER_MIDDLE.read_text(encoding="utf-8"))
    return map_middle_to_parsed_blocks(middle, chat_id=_CHAT_ID, document_id=_DOC_ID)


@pytest.fixture(scope="module")
def paper_result(paper_blocks: list[ParsedBlock]):
    return derive_hierarchy(paper_blocks, document_id=_DOC_ID, chat_id=_CHAT_ID)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _nodes_by_type(nodes: list[DocumentNodeOut], nt: NodeType) -> list[DocumentNodeOut]:
    return [n for n in nodes if n.node_type == nt]


# ---------------------------------------------------------------------------
# Happy-path: basic structure
# ---------------------------------------------------------------------------


class TestHappyPath:
    def test_result_not_empty(self, paper_result) -> None:
        assert paper_result.nodes, "Expected at least one node"

    def test_root_document_node_exists(self, paper_result) -> None:
        doc_nodes = _nodes_by_type(paper_result.nodes, NodeType.document)
        assert len(doc_nodes) == 1, f"Expected exactly 1 document node, got {len(doc_nodes)}"

    def test_root_is_first_node(self, paper_result) -> None:
        assert paper_result.nodes[0].node_type == NodeType.document

    def test_root_parent_is_none(self, paper_result) -> None:
        root = paper_result.nodes[0]
        assert root.parent_id is None

    def test_root_title_populated(self, paper_result) -> None:
        root = paper_result.nodes[0]
        assert root.title is not None
        assert "Adaptive Retrieval" in root.title

    def test_root_level_zero(self, paper_result) -> None:
        root = paper_result.nodes[0]
        assert root.level == 0

    def test_total_node_count_reasonable(self, paper_result) -> None:
        # Synthetic fixture: 1 doc + 1 authors + 1 abstract + sections/subsections
        # + paragraphs + 1 figure + 1 table + 1 equation + 5 refs + appendix nodes
        # Expect at least 15 and no more than 60 nodes
        n = len(paper_result.nodes)
        assert 15 <= n <= 60, f"Unexpected node count: {n}"

    def test_heuristics_applied_non_empty(self, paper_result) -> None:
        assert paper_result.heuristics_applied, "heuristics_applied should not be empty"

    def test_heuristics_applied_sorted(self, paper_result) -> None:
        h = paper_result.heuristics_applied
        assert h == sorted(h), "heuristics_applied must be sorted"


# ---------------------------------------------------------------------------
# Tree structure validity
# ---------------------------------------------------------------------------


class TestTreeStructure:
    def test_all_parent_ids_resolve(self, paper_result) -> None:
        """Every non-root node's parent_id must exist in the node list."""
        node_ids = {n.id for n in paper_result.nodes}
        for node in paper_result.nodes:
            if node.node_type == NodeType.document:
                assert node.parent_id is None
            else:
                assert node.parent_id in node_ids, (
                    f"Node {node.id} ({node.node_type}) has unknown parent_id {node.parent_id}"
                )

    def test_order_index_unique(self, paper_result) -> None:
        indices = [n.order_index for n in paper_result.nodes]
        assert len(indices) == len(set(indices)), "Duplicate order_index values found"

    def test_order_index_starts_at_zero(self, paper_result) -> None:
        assert min(n.order_index for n in paper_result.nodes) == 0

    def test_no_self_parent(self, paper_result) -> None:
        for node in paper_result.nodes:
            assert node.parent_id != node.id, f"Node {node.id} is its own parent"

    def test_page_start_le_page_end(self, paper_result) -> None:
        for node in paper_result.nodes:
            assert node.page_start <= node.page_end, (
                f"Node {node.id} has page_start {node.page_start} > page_end {node.page_end}"
            )

    def test_root_page_range_covers_document(self, paper_result) -> None:
        """Document root must cover the full page range of all children."""
        root = paper_result.nodes[0]
        all_starts = [n.page_start for n in paper_result.nodes]
        all_ends = [n.page_end for n in paper_result.nodes]
        assert root.page_start <= min(all_starts)
        assert root.page_end >= max(all_ends)


# ---------------------------------------------------------------------------
# References and appendix boundaries
# ---------------------------------------------------------------------------


class TestBoundaries:
    def test_references_start_index_set(self, paper_result) -> None:
        assert paper_result.references_start_index is not None, (
            "references_start_index should be set for a paper with References"
        )

    def test_appendix_start_index_set(self, paper_result) -> None:
        assert paper_result.appendix_start_index is not None, (
            "appendix_start_index should be set for a paper with Appendix"
        )

    def test_reference_nodes_exist(self, paper_result) -> None:
        refs = _nodes_by_type(paper_result.nodes, NodeType.reference)
        assert len(refs) >= 5, f"Expected >= 5 reference nodes, got {len(refs)}"

    def test_references_start_index_points_to_reference_node(self, paper_result) -> None:
        idx = paper_result.references_start_index
        assert idx is not None
        # Find the node with that order_index
        ref_nodes = [n for n in paper_result.nodes if n.order_index == idx]
        assert len(ref_nodes) == 1
        assert ref_nodes[0].node_type == NodeType.reference, (
            f"references_start_index points to {ref_nodes[0].node_type}, expected reference"
        )

    def test_appendix_start_index_points_to_appendix_node(self, paper_result) -> None:
        idx = paper_result.appendix_start_index
        assert idx is not None
        app_nodes = [n for n in paper_result.nodes if n.order_index == idx]
        assert len(app_nodes) == 1
        assert app_nodes[0].node_type == NodeType.appendix, (
            f"appendix_start_index points to {app_nodes[0].node_type}, expected appendix"
        )

    def test_appendix_node_exists(self, paper_result) -> None:
        app_nodes = _nodes_by_type(paper_result.nodes, NodeType.appendix)
        assert len(app_nodes) >= 1

    def test_references_section_has_boundary_metadata(self, paper_result) -> None:
        """The references section heading must carry boundary='references' in metadata."""
        ref_sections = [
            n
            for n in paper_result.nodes
            if n.node_type == NodeType.section and n.metadata_.get("boundary") == "references"
        ]
        assert len(ref_sections) >= 1, "Expected a references section with boundary metadata"

    def test_appendix_node_boundary_metadata(self, paper_result) -> None:
        app_nodes = _nodes_by_type(paper_result.nodes, NodeType.appendix)
        assert any(n.metadata_.get("boundary") == "appendix" for n in app_nodes), (
            "Expected appendix node with boundary='appendix' in metadata"
        )


# ---------------------------------------------------------------------------
# Abstract and authors heuristics
# ---------------------------------------------------------------------------


class TestAbstractAndAuthors:
    def test_abstract_node_exists(self, paper_result) -> None:
        abs_nodes = _nodes_by_type(paper_result.nodes, NodeType.abstract)
        assert len(abs_nodes) == 1, f"Expected 1 abstract node, got {len(abs_nodes)}"

    def test_abstract_content_non_empty(self, paper_result) -> None:
        abs_nodes = _nodes_by_type(paper_result.nodes, NodeType.abstract)
        assert abs_nodes[0].content.strip(), "Abstract content should not be empty"

    def test_abstract_level_one(self, paper_result) -> None:
        abs_nodes = _nodes_by_type(paper_result.nodes, NodeType.abstract)
        assert abs_nodes[0].level == 1

    def test_abstract_parent_is_document(self, paper_result) -> None:
        root = paper_result.nodes[0]
        abs_nodes = _nodes_by_type(paper_result.nodes, NodeType.abstract)
        assert abs_nodes[0].parent_id == root.id

    def test_authors_node_exists(self, paper_result) -> None:
        auth_nodes = _nodes_by_type(paper_result.nodes, NodeType.authors)
        assert len(auth_nodes) >= 1, "Expected at least 1 authors node"

    def test_authors_content_has_sup(self, paper_result) -> None:
        auth_nodes = _nodes_by_type(paper_result.nodes, NodeType.authors)
        assert any("<sup>" in n.content for n in auth_nodes), (
            "Authors node content should contain <sup> affiliation markers"
        )

    def test_authors_heuristic_in_applied(self, paper_result) -> None:
        assert "authors-by-sup-numeric-or-star" in paper_result.heuristics_applied

    def test_abstract_heading_heuristic_in_applied(self, paper_result) -> None:
        assert "abstract-by-heading" in paper_result.heuristics_applied


# ---------------------------------------------------------------------------
# Section vs subsection discrimination
# ---------------------------------------------------------------------------


class TestSectionSubsectionDiscrimination:
    def test_plain_numbered_is_section(self, paper_result) -> None:
        """'3 Methodology' must be a section, not a subsection."""
        sections = _nodes_by_type(paper_result.nodes, NodeType.section)
        section_titles = [n.title or "" for n in sections]
        assert any("3 Methodology" in t for t in section_titles), (
            f"Expected '3 Methodology' as section; found sections: {section_titles}"
        )

    def test_dotted_numbered_is_subsection(self, paper_result) -> None:
        """'3.1 Adaptive Retrieval Module' must be a subsection."""
        subsections = _nodes_by_type(paper_result.nodes, NodeType.subsection)
        sub_titles = [n.title or "" for n in subsections]
        assert any("3.1" in t for t in sub_titles), (
            f"Expected '3.1 ...' as subsection; found subsections: {sub_titles}"
        )

    def test_subsection_parent_is_section(self, paper_result) -> None:
        """All subsections must have a section (or appendix) as parent."""
        node_by_id = {n.id: n for n in paper_result.nodes}
        subsections = _nodes_by_type(paper_result.nodes, NodeType.subsection)
        for sub in subsections:
            assert sub.parent_id is not None
            parent = node_by_id[sub.parent_id]
            assert parent.node_type in (NodeType.section, NodeType.appendix, NodeType.document), (
                f"Subsection '{sub.title}' has parent of type {parent.node_type}"
            )

    def test_subsection_level_is_two(self, paper_result) -> None:
        subsections = _nodes_by_type(paper_result.nodes, NodeType.subsection)
        assert all(n.level == 2 for n in subsections), "All subsections should be at level 2"

    def test_section_level_is_one(self, paper_result) -> None:
        sections = _nodes_by_type(paper_result.nodes, NodeType.section)
        assert all(n.level == 1 for n in sections), "All sections should be at level 1"

    def test_appendix_subsection_dotted_a1(self, paper_result) -> None:
        """'A.1 Ablation Study' must be a subsection."""
        subsections = _nodes_by_type(paper_result.nodes, NodeType.subsection)
        sub_titles = [n.title or "" for n in subsections]
        assert any("A.1" in t for t in sub_titles), (
            f"Expected 'A.1 ...' subsection; got subsections: {sub_titles}"
        )


# ---------------------------------------------------------------------------
# One-owner invariant (no ParsedBlock in multiple nodes)
# ---------------------------------------------------------------------------


class TestOneOwnerInvariant:
    def test_no_duplicate_block_ids_across_nodes(self, paper_result) -> None:
        """The union of all source_block_ids must have no duplicates."""
        all_ids: list[UUID] = []
        for node in paper_result.nodes:
            all_ids.extend(node.source_block_ids)
        counts = Counter(all_ids)
        duplicates = {bid: cnt for bid, cnt in counts.items() if cnt > 1}
        assert not duplicates, f"ParsedBlock ids claimed by multiple nodes: {duplicates}"

    def test_discarded_blocks_not_owned(
        self, paper_blocks: list[ParsedBlock], paper_result
    ) -> None:
        """Discarded blocks must not appear in any node's source_block_ids."""
        discarded_ids = {b.block_id for b in paper_blocks if b.block_type == BlockType.discarded}
        owned_ids = {bid for node in paper_result.nodes for bid in node.source_block_ids}
        overlap = discarded_ids & owned_ids
        assert not overlap, f"Discarded blocks should not be owned: {overlap}"

    def test_non_discarded_blocks_owned(
        self, paper_blocks: list[ParsedBlock], paper_result
    ) -> None:
        """All non-discarded blocks should be owned by exactly one node."""
        non_discarded_ids = {
            b.block_id for b in paper_blocks if b.block_type != BlockType.discarded
        }
        owned_ids = {bid for node in paper_result.nodes for bid in node.source_block_ids}
        unowned = non_discarded_ids - owned_ids
        assert not unowned, f"Non-discarded blocks not owned by any node: {len(unowned)} block(s)"

    def test_coverage_rate(self, paper_blocks: list[ParsedBlock], paper_result) -> None:
        """Coverage: owned / total-non-discarded should be 100%."""
        non_discarded = [b for b in paper_blocks if b.block_type != BlockType.discarded]
        owned_ids = {bid for node in paper_result.nodes for bid in node.source_block_ids}
        owned_count = sum(1 for b in non_discarded if b.block_id in owned_ids)
        assert owned_count == len(non_discarded), (
            f"Coverage: {owned_count}/{len(non_discarded)} non-discarded blocks owned"
        )


# ---------------------------------------------------------------------------
# Deterministic id
# ---------------------------------------------------------------------------


class TestDeterministicId:
    def test_two_calls_produce_identical_node_ids(self, paper_blocks: list[ParsedBlock]) -> None:
        result_a = derive_hierarchy(paper_blocks, document_id=_DOC_ID, chat_id=_CHAT_ID)
        result_b = derive_hierarchy(paper_blocks, document_id=_DOC_ID, chat_id=_CHAT_ID)
        ids_a = [n.id for n in result_a.nodes]
        ids_b = [n.id for n in result_b.nodes]
        assert ids_a == ids_b, "Second call produced different node ids"

    def test_different_document_id_gives_different_node_ids(
        self, paper_blocks: list[ParsedBlock]
    ) -> None:
        other_doc = uuid4()
        result_a = derive_hierarchy(paper_blocks, document_id=_DOC_ID, chat_id=_CHAT_ID)
        result_b = derive_hierarchy(paper_blocks, document_id=other_doc, chat_id=_CHAT_ID)
        ids_a = {n.id for n in result_a.nodes}
        ids_b = {n.id for n in result_b.nodes}
        assert ids_a.isdisjoint(ids_b), (
            "Different document_id must produce non-overlapping node ids"
        )

    def test_node_ids_unique_within_result(self, paper_result) -> None:
        ids = [n.id for n in paper_result.nodes]
        assert len(ids) == len(set(ids)), "Duplicate node ids within one result"


# ---------------------------------------------------------------------------
# Metadata invariants
# ---------------------------------------------------------------------------


class TestMetadata:
    def test_all_nodes_have_source_heuristic(self, paper_result) -> None:
        for node in paper_result.nodes:
            assert node.metadata_.get("source") == "heuristic", (
                f"Node {node.id} ({node.node_type}) missing source='heuristic'"
            )

    def test_figure_node_has_image_path(self, paper_result) -> None:
        figures = _nodes_by_type(paper_result.nodes, NodeType.figure)
        assert figures, "Expected at least one figure node"
        for fig in figures:
            assert "image_path" in fig.metadata_, (
                f"Figure node {fig.id} missing image_path in metadata"
            )

    def test_table_node_has_html_body(self, paper_result) -> None:
        tables = _nodes_by_type(paper_result.nodes, NodeType.table)
        assert tables, "Expected at least one table node"
        for tbl in tables:
            assert "html_body" in tbl.metadata_, (
                f"Table node {tbl.id} missing html_body in metadata"
            )
            assert "<table>" in str(tbl.metadata_["html_body"]), (
                f"Table node {tbl.id} html_body does not contain <table>"
            )

    def test_equation_node_content_has_latex(self, paper_result) -> None:
        equations = _nodes_by_type(paper_result.nodes, NodeType.equation)
        assert equations, "Expected at least one equation node"
        for eq in equations:
            assert eq.content.strip(), f"Equation node {eq.id} has empty content"


# ---------------------------------------------------------------------------
# Real-data smoke test
# ---------------------------------------------------------------------------


# Module-level fixtures for real-data smoke tests (avoids class-scoped instance method warning)
_real_smoke_result: tuple | None = None


def _get_real_smoke_result() -> tuple:
    global _real_smoke_result
    if _real_smoke_result is None:
        middle = json.loads(REAL_MIDDLE.read_text(encoding="utf-8"))
        chat_id = uuid4()
        document_id = uuid4()
        blocks = map_middle_to_parsed_blocks(middle, chat_id=chat_id, document_id=document_id)
        result = derive_hierarchy(blocks, document_id=document_id, chat_id=chat_id)
        _real_smoke_result = (result, blocks)
    return _real_smoke_result


@pytest.mark.skipif(
    not REAL_MIDDLE.exists(),
    reason=f"Real middle.json not available at {REAL_MIDDLE}",
)
class TestRealDataSmoke:
    def test_has_document_root(self) -> None:
        result, _ = _get_real_smoke_result()
        doc_nodes = _nodes_by_type(result.nodes, NodeType.document)
        assert len(doc_nodes) >= 1

    def test_section_count_above_threshold(self) -> None:
        result, _ = _get_real_smoke_result()
        sections = _nodes_by_type(result.nodes, NodeType.section)
        # LightRAG paper has many sections
        assert len(sections) >= 5, f"Expected >= 5 sections, got {len(sections)}"

    def test_reference_count_above_threshold(self) -> None:
        result, _ = _get_real_smoke_result()
        refs = _nodes_by_type(result.nodes, NodeType.reference)
        assert len(refs) >= 10, f"Expected >= 10 reference nodes, got {len(refs)}"

    def test_appendix_start_index_set(self) -> None:
        result, _ = _get_real_smoke_result()
        assert result.appendix_start_index is not None, (
            "Expected appendix_start_index to be set for LightRAG paper"
        )

    def test_no_duplicate_block_ownership(self) -> None:
        result, blocks = _get_real_smoke_result()
        all_ids: list[UUID] = []
        for node in result.nodes:
            all_ids.extend(node.source_block_ids)
        counts = Counter(all_ids)
        duplicates = {bid: cnt for bid, cnt in counts.items() if cnt > 1}
        assert not duplicates, f"Duplicate block ownership in real data: {len(duplicates)} block(s)"

    def test_heuristics_applied_non_empty(self) -> None:
        result, _ = _get_real_smoke_result()
        assert result.heuristics_applied
