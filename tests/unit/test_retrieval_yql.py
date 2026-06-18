"""Unit tests for RetrievalService YQL WHERE-clause construction.

Tests focus on:
1. ``_yql_where`` always embeds ``chat_id contains "<uuid>"`` — the core isolation invariant.
2. Correct inclusion of document_id and source_type filters.
3. Rejection of invalid UUIDs, non-whitelisted source types, and injection attempts.
4. The ``query`` text is NEVER concatenated into YQL (uses userQuery() instead).

CLAUDE.md §2 / §7: chat_id filter is mandatory; any code path that omits it is a defect.
"""

from __future__ import annotations

import inspect

import pytest

from app.errors import InvalidRetrievalFilter
from app.retrieval.service import _yql_where

CHAT_A = "11111111-1111-1111-1111-111111111111"
CHAT_B = "22222222-2222-2222-2222-222222222222"
DOC_1 = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
DOC_2 = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"


def test_chat_id_always_present_in_all_filter_combinations() -> None:
    """chat_id must appear in WHERE regardless of optional filters."""
    assert f'chat_id contains "{CHAT_A}"' in _yql_where(CHAT_A)
    assert f'chat_id contains "{CHAT_A}"' in _yql_where(CHAT_A, document_id_strs=[DOC_1])
    where = _yql_where(CHAT_A, document_id_strs=[DOC_1], source_type_strs=["chunk"])
    assert f'chat_id contains "{CHAT_A}"' in where
    assert DOC_1 in where and "chunk" in where and " and " in where


def test_different_chat_ids_do_not_bleed() -> None:
    where = _yql_where(CHAT_B)
    assert f'chat_id contains "{CHAT_B}"' in where
    assert CHAT_A not in where


def test_document_ids_filter() -> None:
    """Multiple doc IDs appear; empty list omits filter."""
    where = _yql_where(CHAT_A, document_id_strs=[DOC_1, DOC_2])
    assert "document_id in" in where and DOC_1 in where and DOC_2 in where
    assert "document_id" not in _yql_where(CHAT_A, document_id_strs=[])


def test_all_valid_source_types_accepted() -> None:
    from app.vespa.feed import VALID_SOURCE_TYPES

    for st in sorted(VALID_SOURCE_TYPES):
        assert st in _yql_where(CHAT_A, source_type_strs=[st])


# Rejection: invalid chat_id (non-UUID and injection representative cases)
@pytest.mark.parametrize(
    "bad_chat_id",
    ["not-a-uuid", '"; drop table chats; --'],
)
def test_invalid_chat_id_raises(bad_chat_id: str) -> None:
    with pytest.raises(InvalidRetrievalFilter):
        _yql_where(bad_chat_id)


# Rejection: invalid source_type (unknown and injection representative cases)
@pytest.mark.parametrize(
    "bad_source_type",
    ["not_a_real_type", "'; drop table documents; --"],
)
def test_invalid_source_type_raises(bad_source_type: str) -> None:
    with pytest.raises(InvalidRetrievalFilter, match="source_type"):
        _yql_where(CHAT_A, source_type_strs=[bad_source_type])


def test_invalid_document_id_injection_raises() -> None:
    with pytest.raises(InvalidRetrievalFilter):
        _yql_where(CHAT_A, document_id_strs=['"; drop table documents; --'])


def test_query_not_in_yql() -> None:
    """_yql_where has no query param; generated clause has no injected text."""
    assert "query" not in inspect.signature(_yql_where).parameters
    where = _yql_where(CHAT_A)
    assert "drop table" not in where.lower()
    assert "SELECT" not in where
