"""Integration tests for the /chats/{chat_id}/facts API (Phase 5.3).

Strategy
--------
- Uses ``httpx.AsyncClient`` wired to the FastAPI app via ASGI transport.
- ``api_client`` and ``db_session`` fixtures come from ``tests/conftest.py``.
- All DB changes run inside a SAVEPOINT that is rolled back after each test.

Coverage
--------
1. POST /chats/{A}/facts/search body={kinds:["metric"]} -> 200 + list.
2. POST /chats/{B}/facts/search -> does NOT contain chat A's facts.
3. POST /chats/{A}/facts/search body contains unknown field -> 422.
4. POST /chats/{A}/facts/extract with monkeypatched fixture path -> 201
   + at least 1 fact extracted.
5. GET /chats/{A}/facts/{fact_id} -> 200.
6. GET /chats/{A}/facts/{fact_id} with wrong chat -> 404.
"""

from __future__ import annotations

import io
import shutil
import uuid
from pathlib import Path
from unittest.mock import patch

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.domain import ChatCreate
from app.services import chat_service
from app.services.vespa_indexer import NullVespaIndexer
from app.storage.local import LocalBlobStorage

# ---------------------------------------------------------------------------
# Fixture path
# ---------------------------------------------------------------------------

_FIXTURE_MIDDLE_JSON = (
    Path(__file__).parent.parent / "fixtures" / "mineru_sample_paper" / "middle.json"
)

# ---------------------------------------------------------------------------
# Dependency override helpers
# ---------------------------------------------------------------------------


def _override_storage(tmp_path: Path):  # type: ignore[no-untyped-def]
    def _get_storage() -> LocalBlobStorage:
        return LocalBlobStorage(root=tmp_path)

    return _get_storage


def _override_indexer():  # type: ignore[no-untyped-def]
    def _get_indexer() -> NullVespaIndexer:
        return NullVespaIndexer()

    return _get_indexer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _create_chat(db_session: AsyncSession, name: str) -> uuid.UUID:
    """Create a chat via the service layer and return its ID."""
    chat = await chat_service.create_chat(db_session, ChatCreate(name=name))
    return chat.id


async def _upload_doc(
    api_client: AsyncClient,
    chat_id: uuid.UUID,
    filename: str,
    tmp_path: Path,
) -> uuid.UUID:
    """Upload a dummy PDF document (with storage/indexer overrides) and return its ID."""
    from app.api.documents import get_indexer, get_storage
    from app.main import app

    app.dependency_overrides[get_storage] = _override_storage(tmp_path)
    app.dependency_overrides[get_indexer] = _override_indexer()

    try:
        pdf_bytes = b"%PDF-1.4 test content"
        resp = await api_client.post(
            f"/chats/{chat_id}/documents",
            files=[("file", (filename, io.BytesIO(pdf_bytes), "application/pdf"))],
        )
        assert resp.status_code == 201, f"Doc upload failed ({resp.status_code}): {resp.text}"
        return uuid.UUID(resp.json()["id"])
    finally:
        app.dependency_overrides.pop(get_storage, None)
        app.dependency_overrides.pop(get_indexer, None)


async def _seed_fact(
    session: AsyncSession,
    *,
    chat_id: uuid.UUID,
    document_id: uuid.UUID,
    kind: str,
    key: str,
    unit: str | None = None,
    context_excerpt: str | None = None,
    page: int | None = None,
    seq: int = 0,
) -> uuid.UUID:
    """Insert one StructuredFact row via ORM and return its ID."""
    from uuid import NAMESPACE_OID, uuid5

    from app.models.orm import StructuredFact

    page_str = str(page) if page is not None else "None"
    fact_id = uuid5(NAMESPACE_OID, f"{document_id}:fact:{kind}:{key}:{page_str}:{seq}")
    value_dict: dict = {"raw": f"{key} test", "numeric": 42.0}  # type: ignore[type-arg]

    row = StructuredFact(
        id=fact_id,
        chat_id=chat_id,
        document_id=document_id,
        source_node_id=None,
        kind=kind,
        key=key,
        value=value_dict,
        unit=unit,
        context_excerpt=context_excerpt or f"test excerpt for {key}",
        page=page,
    )
    session.add(row)
    await session.flush()
    return fact_id


# ---------------------------------------------------------------------------
# 1. Search returns chat A's metric facts
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_metric_facts_chat_a(
    api_client: AsyncClient,
    db_session: AsyncSession,
    tmp_path: Path,
) -> None:
    """POST /chats/{A}/facts/search with kinds=["metric"] returns metric facts."""
    chat_id = await _create_chat(db_session, "Search Metric Chat A")
    doc_id = await _upload_doc(api_client, chat_id, "search_metric.pdf", tmp_path)

    await _seed_fact(
        db_session,
        chat_id=chat_id,
        document_id=doc_id,
        kind="metric",
        key="f1",
        unit="%",
        context_excerpt="F1 score of 87.3%",
        page=5,
    )

    resp = await api_client.post(
        f"/chats/{chat_id}/facts/search",
        json={"kinds": ["metric"]},
    )
    assert resp.status_code == 200, f"Unexpected status: {resp.status_code} {resp.text}"
    body = resp.json()
    assert "count" in body
    assert "items" in body
    assert body["count"] >= 1
    kinds = [item["kind"] for item in body["items"]]
    assert all(k == "metric" for k in kinds), f"Non-metric facts returned: {kinds}"


# ---------------------------------------------------------------------------
# 2. Chat B search does NOT return Chat A's facts
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_does_not_leak_chat_a_into_chat_b(
    api_client: AsyncClient,
    db_session: AsyncSession,
    tmp_path: Path,
) -> None:
    """Chat B's search must never return Chat A's facts (isolation invariant)."""
    chat_a_id = await _create_chat(db_session, "Isolation Chat A Facts")
    doc_a_id = await _upload_doc(api_client, chat_a_id, "iso_a.pdf", tmp_path)

    chat_b_id = await _create_chat(db_session, "Isolation Chat B Facts")

    # Seed a fact with a unique key only under chat A
    unique_key = f"unique_leak_{uuid.uuid4().hex[:8]}"
    await _seed_fact(
        db_session,
        chat_id=chat_a_id,
        document_id=doc_a_id,
        kind="metric",
        key=unique_key,
        page=1,
    )

    # Chat B search — must not return the unique key
    resp = await api_client.post(
        f"/chats/{chat_b_id}/facts/search",
        json={"limit": 200},
    )
    assert resp.status_code == 200
    body = resp.json()
    keys = [item["key"] for item in body["items"]]
    assert unique_key not in keys, (
        f"Chat A's fact (key={unique_key!r}) leaked into Chat B results: {keys}"
    )


# ---------------------------------------------------------------------------
# 3. Unknown field in filter body -> 422
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_unknown_field_returns_422(
    api_client: AsyncClient,
    db_session: AsyncSession,
) -> None:
    """An unknown field in the filter body must return 422 (extra=forbid)."""
    chat_id = await _create_chat(db_session, "422 Test Chat Facts")

    resp = await api_client.post(
        f"/chats/{chat_id}/facts/search",
        json={"evil": "DROP TABLE structured_facts;--"},
    )
    assert resp.status_code == 422, (
        f"Expected 422 for unknown field; got {resp.status_code}: {resp.text}"
    )


# ---------------------------------------------------------------------------
# 4. Extract endpoint with monkeypatched fixture path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_extract_facts_from_fixture(
    api_client: AsyncClient,
    db_session: AsyncSession,
    tmp_path: Path,
) -> None:
    """POST /extract with a fixture middle.json -> 201 + >= 1 fact extracted.

    Monkeypatches ``app.api.facts.get_settings`` so that ``app_data_root``
    points to a tmp_path with the fixture file at the expected location:
    {app_data_root}/parsed/doc/hybrid_auto/doc_middle.json
    """
    chat_id = await _create_chat(db_session, "Extract Test Chat Facts")
    # "doc.pdf" -> stem = "doc"
    document_id = await _upload_doc(api_client, chat_id, "doc.pdf", tmp_path)

    # Set up the fake parsed dir
    stem = "doc"
    parsed_dir = tmp_path / "parsed" / stem / "hybrid_auto"
    parsed_dir.mkdir(parents=True)
    shutil.copy(_FIXTURE_MIDDLE_JSON, parsed_dir / f"{stem}_middle.json")

    # Monkeypatch get_settings used inside the router
    class _FakeSettings:
        app_data_root = str(tmp_path)

    with patch("app.api.facts.get_settings", return_value=_FakeSettings()):
        resp = await api_client.post(
            f"/chats/{chat_id}/facts/extract",
            json={"document_id": str(document_id), "use_llm": False},
        )

    assert resp.status_code == 201, f"Extract failed ({resp.status_code}): {resp.text}"
    body = resp.json()
    assert body["facts_extracted"] >= 1, (
        f"Expected at least 1 fact from fixture; got {body['facts_extracted']}"
    )
    assert len(body["fact_ids"]) == body["facts_extracted"]


# ---------------------------------------------------------------------------
# 5. GET /chats/{A}/facts/{fact_id} -> 200
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_fact_by_id(
    api_client: AsyncClient,
    db_session: AsyncSession,
    tmp_path: Path,
) -> None:
    """GET /facts/{fact_id} within correct chat -> 200."""
    chat_id = await _create_chat(db_session, "Get Fact Chat Facts")
    doc_id = await _upload_doc(api_client, chat_id, "get_fact.pdf", tmp_path)

    fact_id = await _seed_fact(
        db_session,
        chat_id=chat_id,
        document_id=doc_id,
        kind="metric",
        key="precision",
        unit="%",
        page=3,
    )

    resp = await api_client.get(f"/chats/{chat_id}/facts/{fact_id}")
    assert resp.status_code == 200, f"Unexpected status: {resp.status_code} {resp.text}"
    body = resp.json()
    assert body["id"] == str(fact_id)
    assert body["kind"] == "metric"
    assert body["key"] == "precision"


# ---------------------------------------------------------------------------
# 6. GET /chats/{A}/facts/{fact_id} with wrong chat -> 404
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_fact_wrong_chat_returns_404(
    api_client: AsyncClient,
    db_session: AsyncSession,
    tmp_path: Path,
) -> None:
    """GET /chats/{B}/facts/{fact_id_from_A} must return 404 (isolation)."""
    chat_a_id = await _create_chat(db_session, "GetFact Chat A Facts")
    doc_a_id = await _upload_doc(api_client, chat_a_id, "gf_a.pdf", tmp_path)

    chat_b_id = await _create_chat(db_session, "GetFact Chat B Facts")

    # Seed a fact under chat A
    fact_id = await _seed_fact(
        db_session,
        chat_id=chat_a_id,
        document_id=doc_a_id,
        kind="benchmark",
        key="bert_score",
        page=7,
    )

    # Request via chat B -> should be 404
    resp = await api_client.get(f"/chats/{chat_b_id}/facts/{fact_id}")
    assert resp.status_code == 404, (
        f"Expected 404 for cross-chat access; got {resp.status_code}: {resp.text}"
    )
