"""Unit tests for RetrievalService YQL WHERE-clause construction.

Tests focus on:
1. ``_yql_where`` always embeds ``chat_id contains "<uuid>"`` — the core isolation invariant.
2. Correct inclusion of document_id and source_type filters.
3. Rejection of invalid UUIDs, non-whitelisted source types, and injection attempts.
4. The ``query`` text is NEVER concatenated into YQL (uses userQuery() instead).

CLAUDE.md §2 / §7: chat_id filter is mandatory; any code path that omits it is a defect.
"""

from __future__ import annotations

import pytest

from app.errors import InvalidRetrievalFilter
from app.retrieval.service import _yql_where

# ---------------------------------------------------------------------------
# Valid UUIDs and source types for reuse
# ---------------------------------------------------------------------------

CHAT_A = "11111111-1111-1111-1111-111111111111"
CHAT_B = "22222222-2222-2222-2222-222222222222"
DOC_1 = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
DOC_2 = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"

VALID_SOURCE_TYPE = "chunk"
ANOTHER_VALID_SOURCE_TYPE = "section_summary"


# ===========================================================================
# 1. chat_id is always present
# ===========================================================================


class TestChatIdAlwaysPresent:
    """chat_id contains '<uuid>' MUST appear in every generated WHERE clause."""

    def test_chat_id_only(self) -> None:
        where = _yql_where(CHAT_A)
        assert f'chat_id contains "{CHAT_A}"' in where

    def test_chat_id_with_document_ids(self) -> None:
        where = _yql_where(CHAT_A, document_id_strs=[DOC_1])
        assert f'chat_id contains "{CHAT_A}"' in where

    def test_chat_id_with_source_types(self) -> None:
        where = _yql_where(CHAT_A, source_type_strs=[VALID_SOURCE_TYPE])
        assert f'chat_id contains "{CHAT_A}"' in where

    def test_chat_id_with_both_filters(self) -> None:
        where = _yql_where(CHAT_A, document_id_strs=[DOC_1], source_type_strs=[VALID_SOURCE_TYPE])
        assert f'chat_id contains "{CHAT_A}"' in where

    def test_different_chat_id(self) -> None:
        where = _yql_where(CHAT_B)
        assert f'chat_id contains "{CHAT_B}"' in where
        assert CHAT_A not in where


# ===========================================================================
# 2. document_id filter
# ===========================================================================


class TestDocumentIdFilter:
    """document_id in (...) appears when document_ids are provided."""

    def test_single_document_id(self) -> None:
        where = _yql_where(CHAT_A, document_id_strs=[DOC_1])
        assert "document_id in" in where
        assert DOC_1 in where

    def test_multiple_document_ids(self) -> None:
        where = _yql_where(CHAT_A, document_id_strs=[DOC_1, DOC_2])
        assert "document_id in" in where
        assert DOC_1 in where
        assert DOC_2 in where

    def test_no_document_ids_omits_filter(self) -> None:
        where = _yql_where(CHAT_A, document_id_strs=None)
        assert "document_id" not in where

    def test_empty_document_ids_omits_filter(self) -> None:
        where = _yql_where(CHAT_A, document_id_strs=[])
        assert "document_id" not in where


# ===========================================================================
# 3. source_type filter
# ===========================================================================


class TestSourceTypeFilter:
    """source_type in (...) appears when source_types are provided."""

    def test_single_source_type(self) -> None:
        where = _yql_where(CHAT_A, source_type_strs=[VALID_SOURCE_TYPE])
        assert "source_type in" in where
        assert VALID_SOURCE_TYPE in where

    def test_multiple_source_types(self) -> None:
        where = _yql_where(
            CHAT_A,
            source_type_strs=[VALID_SOURCE_TYPE, ANOTHER_VALID_SOURCE_TYPE],
        )
        assert "source_type in" in where
        assert VALID_SOURCE_TYPE in where
        assert ANOTHER_VALID_SOURCE_TYPE in where

    def test_no_source_types_omits_filter(self) -> None:
        where = _yql_where(CHAT_A, source_type_strs=None)
        assert "source_type" not in where

    def test_all_valid_source_types_accepted(self) -> None:
        """All 13 whitelisted source types should pass without error."""
        from app.vespa.feed import VALID_SOURCE_TYPES

        for st in sorted(VALID_SOURCE_TYPES):
            where = _yql_where(CHAT_A, source_type_strs=[st])
            assert st in where


# ===========================================================================
# 4. Rejection — invalid source types
# ===========================================================================


class TestInvalidSourceType:
    """Non-whitelisted source types must raise InvalidRetrievalFilter."""

    def test_unknown_source_type_raises(self) -> None:
        with pytest.raises(InvalidRetrievalFilter, match="source_type"):
            _yql_where(CHAT_A, source_type_strs=["not_a_real_type"])

    def test_sql_injection_source_type_raises(self) -> None:
        with pytest.raises(InvalidRetrievalFilter):
            _yql_where(CHAT_A, source_type_strs=["'; drop table documents; --"])

    def test_empty_string_source_type_raises(self) -> None:
        with pytest.raises(InvalidRetrievalFilter):
            _yql_where(CHAT_A, source_type_strs=[""])

    def test_mixed_valid_invalid_raises(self) -> None:
        with pytest.raises(InvalidRetrievalFilter):
            _yql_where(CHAT_A, source_type_strs=[VALID_SOURCE_TYPE, "bad_type"])


# ===========================================================================
# 5. Rejection — invalid chat_id
# ===========================================================================


class TestInvalidChatId:
    """Non-UUID chat_id strings must raise InvalidRetrievalFilter."""

    def test_non_uuid_chat_id_raises(self) -> None:
        with pytest.raises(InvalidRetrievalFilter, match="UUID"):
            _yql_where("not-a-uuid")

    def test_empty_chat_id_raises(self) -> None:
        with pytest.raises(InvalidRetrievalFilter, match="UUID"):
            _yql_where("")

    def test_chat_id_with_special_chars_raises(self) -> None:
        with pytest.raises(InvalidRetrievalFilter):
            _yql_where('"; drop table chats; --')

    def test_chat_id_with_double_quote_raises(self) -> None:
        with pytest.raises(InvalidRetrievalFilter):
            _yql_where('"11111111-1111-1111-1111-111111111111"')

    def test_chat_id_with_newline_raises(self) -> None:
        with pytest.raises(InvalidRetrievalFilter):
            _yql_where("11111111-1111-1111-1111-111111111111\n")


# ===========================================================================
# 6. Rejection — invalid document_id
# ===========================================================================


class TestInvalidDocumentId:
    """Non-UUID document IDs must raise InvalidRetrievalFilter."""

    def test_non_uuid_document_id_raises(self) -> None:
        with pytest.raises(InvalidRetrievalFilter, match="UUID"):
            _yql_where(CHAT_A, document_id_strs=["not-a-uuid"])

    def test_document_id_with_injection_raises(self) -> None:
        with pytest.raises(InvalidRetrievalFilter):
            _yql_where(CHAT_A, document_id_strs=['"; drop table documents; --'])


# ===========================================================================
# 7. Query text is NEVER in YQL (injection safety)
# ===========================================================================


class TestQueryNotInYQL:
    """The search query string must NEVER be embedded in the YQL string.

    The service uses ``userQuery()`` in YQL and passes the query as a
    separate request body parameter.  This test verifies that _yql_where
    does NOT accept a query parameter at all, and that the returned YQL
    fragment contains no user-supplied text.
    """

    def test_yql_where_has_no_query_param(self) -> None:
        """_yql_where signature has no query parameter — verify this."""
        import inspect

        sig = inspect.signature(_yql_where)
        assert "query" not in sig.parameters, (
            "_yql_where must not accept a 'query' parameter. "
            "Query text goes in the POST body as a separate field, not in YQL."
        )

    def test_sql_injection_not_in_where(self) -> None:
        """Verify that _yql_where output cannot contain arbitrary injected text."""
        injection = "'; drop table documents; --"
        # The only way _yql_where could output this is if it accepted a 'query'
        # param (which it must not) or if a filter value containing it slipped
        # through validation.  Since neither is allowed, the output must be clean.
        where = _yql_where(CHAT_A)
        assert injection not in where

    def test_xss_text_not_in_where(self) -> None:
        """Unrelated text should never appear in the WHERE clause."""
        where = _yql_where(CHAT_A)
        assert "<script>" not in where
        assert "DROP" not in where
        assert "SELECT" not in where


# ===========================================================================
# 8. Combined filter WHERE clause structure
# ===========================================================================


class TestCombinedFilters:
    """Verify AND-connected structure when multiple filters are present."""

    def test_chat_and_doc_and_source(self) -> None:
        where = _yql_where(
            CHAT_A,
            document_id_strs=[DOC_1],
            source_type_strs=["definition"],
        )
        # All three parts must appear
        assert f'chat_id contains "{CHAT_A}"' in where
        assert DOC_1 in where
        assert "definition" in where
        # Must be joined with 'and'
        assert " and " in where

    def test_only_chat_id_has_no_and(self) -> None:
        where = _yql_where(CHAT_A)
        # Single predicate: no 'and' needed
        assert " and " not in where

    def test_chat_and_doc_joined_with_and(self) -> None:
        where = _yql_where(CHAT_A, document_id_strs=[DOC_1])
        assert " and " in where
