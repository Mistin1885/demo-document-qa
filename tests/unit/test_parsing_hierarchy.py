"""Unit tests for app.parsing.hierarchy.derive_hierarchy.

Kept invariants
---------------
- happy-path: node count, root existence, tree validity, heuristics populated.
- references/appendix boundary detection + boundary metadata.
- abstract and authors heuristics.
- section vs subsection discrimination (plain "3" vs dotted "3.1").
- one-owner invariant: no ParsedBlock in >1 node; discarded blocks not owned.
- deterministic uuid5 idempotency: same inputs → same node ids.
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
# Paths / constants
# ---------------------------------------------------------------------------

PAPER_FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "mineru_sample_paper"
PAPER_MIDDLE = PAPER_FIXTURE_DIR / "middle.json"

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
# 1. Happy-path: basic structure + tree validity (merged)
# ---------------------------------------------------------------------------


def test_happy_path_structure(paper_result) -> None:
    """Root doc node exists first, has level 0 and no parent; total node count sane;
    all parent_ids resolve; order_index unique and starts at 0; no self-parents."""
    nodes = paper_result.nodes
    assert nodes, "Expected at least one node"

    # Root
    root = nodes[0]
    assert root.node_type == NodeType.document
    assert root.level == 0
    assert root.parent_id is None
    assert root.title is not None and "Adaptive Retrieval" in root.title

    # Count
    assert 15 <= len(nodes) <= 60, f"Unexpected node count: {len(nodes)}"

    # Heuristics
    assert paper_result.heuristics_applied
    assert paper_result.heuristics_applied == sorted(paper_result.heuristics_applied)

    # Tree validity
    node_ids = {n.id for n in nodes}
    for n in nodes:
        if n.node_type == NodeType.document:
            assert n.parent_id is None
        else:
            assert n.parent_id in node_ids
        assert n.parent_id != n.id  # no self-parent
        assert n.page_start <= n.page_end

    # order_index
    indices = [n.order_index for n in nodes]
    assert len(indices) == len(set(indices))
    assert min(indices) == 0


# ---------------------------------------------------------------------------
# 2. References and appendix boundary detection
# ---------------------------------------------------------------------------


def test_boundaries(paper_result) -> None:
    """references_start_index and appendix_start_index are set and point to the
    correct node types; boundary metadata is present."""
    assert paper_result.references_start_index is not None
    assert paper_result.appendix_start_index is not None

    node_by_order = {n.order_index: n for n in paper_result.nodes}

    ref_node = node_by_order[paper_result.references_start_index]
    assert ref_node.node_type == NodeType.reference

    app_node = node_by_order[paper_result.appendix_start_index]
    assert app_node.node_type == NodeType.appendix
    assert app_node.metadata_.get("boundary") == "appendix"

    # At least one reference section heading carries boundary metadata
    ref_sections = [
        n for n in paper_result.nodes
        if n.node_type == NodeType.section and n.metadata_.get("boundary") == "references"
    ]
    assert len(ref_sections) >= 1


# ---------------------------------------------------------------------------
# 3. Abstract and authors heuristics
# ---------------------------------------------------------------------------


def test_abstract_and_authors(paper_result) -> None:
    """Single abstract node at level 1 under root; authors node has <sup> markers."""
    root = paper_result.nodes[0]

    abs_nodes = _nodes_by_type(paper_result.nodes, NodeType.abstract)
    assert len(abs_nodes) == 1
    assert abs_nodes[0].level == 1
    assert abs_nodes[0].parent_id == root.id
    assert abs_nodes[0].content.strip()

    auth_nodes = _nodes_by_type(paper_result.nodes, NodeType.authors)
    assert len(auth_nodes) >= 1
    assert any("<sup>" in n.content for n in auth_nodes)

    assert "authors-by-sup-numeric-or-star" in paper_result.heuristics_applied
    assert "abstract-by-heading" in paper_result.heuristics_applied


# ---------------------------------------------------------------------------
# 4. Section vs subsection discrimination
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "node_type, title_fragment, expected_level",
    [
        (NodeType.section, "3 Methodology", 1),
        (NodeType.subsection, "3.1", 2),
        (NodeType.subsection, "A.1", 2),
    ],
)
def test_section_subsection_discrimination(
    paper_result, node_type: NodeType, title_fragment: str, expected_level: int
) -> None:
    matching = [
        n for n in paper_result.nodes
        if n.node_type == node_type and title_fragment in (n.title or "")
    ]
    assert matching, f"Expected node type={node_type} with title containing '{title_fragment}'"
    assert all(n.level == expected_level for n in matching)


# ---------------------------------------------------------------------------
# 5. One-owner invariant + coverage
# ---------------------------------------------------------------------------


def test_one_owner_invariant(
    paper_blocks: list[ParsedBlock], paper_result
) -> None:
    """No ParsedBlock appears in >1 node; discarded blocks are not owned;
    every non-discarded block is owned."""
    all_ids: list[UUID] = []
    for node in paper_result.nodes:
        all_ids.extend(node.source_block_ids)

    duplicates = {bid: cnt for bid, cnt in Counter(all_ids).items() if cnt > 1}
    assert not duplicates, f"Blocks claimed by multiple nodes: {duplicates}"

    discarded_ids = {b.block_id for b in paper_blocks if b.block_type == BlockType.discarded}
    non_discarded_ids = {b.block_id for b in paper_blocks if b.block_type != BlockType.discarded}
    owned_ids = set(all_ids)

    assert not (discarded_ids & owned_ids), "Discarded blocks must not be owned"
    unowned = non_discarded_ids - owned_ids
    assert not unowned, f"{len(unowned)} non-discarded block(s) not owned by any node"


# ---------------------------------------------------------------------------
# 6. Deterministic uuid5 idempotency
# ---------------------------------------------------------------------------


def test_deterministic_ids(paper_blocks: list[ParsedBlock]) -> None:
    """Same inputs → identical node ids; different document_id → disjoint ids."""
    result_a = derive_hierarchy(paper_blocks, document_id=_DOC_ID, chat_id=_CHAT_ID)
    result_b = derive_hierarchy(paper_blocks, document_id=_DOC_ID, chat_id=_CHAT_ID)
    assert [n.id for n in result_a.nodes] == [n.id for n in result_b.nodes]

    other_doc = uuid4()
    result_c = derive_hierarchy(paper_blocks, document_id=other_doc, chat_id=_CHAT_ID)
    assert {n.id for n in result_a.nodes}.isdisjoint({n.id for n in result_c.nodes})

    # No duplicates within a single result
    ids = [n.id for n in result_a.nodes]
    assert len(ids) == len(set(ids))
