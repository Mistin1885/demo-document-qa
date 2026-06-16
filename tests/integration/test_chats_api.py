"""Integration tests for the /chats API endpoints.

Uses ``httpx.AsyncClient`` wired to the FastAPI app via ASGI transport.  The
``api_client`` and ``db_session`` fixtures are provided by
``tests/conftest.py``.  All DB changes run inside a SAVEPOINT that is rolled
back after each test.
"""

from __future__ import annotations

import uuid

import pytest
from httpx import AsyncClient

# ---------------------------------------------------------------------------
# POST /chats
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_chat_returns_201(api_client: AsyncClient) -> None:
    response = await api_client.post("/chats/", json={"name": "My Notebook"})
    assert response.status_code == 201
    body = response.json()
    assert body["name"] == "My Notebook"
    assert body["description"] is None
    assert "id" in body
    assert "created_at" in body
    assert "updated_at" in body


@pytest.mark.asyncio
async def test_create_chat_with_description(api_client: AsyncClient) -> None:
    response = await api_client.post(
        "/chats/",
        json={"name": "AI Papers", "description": "Arxiv papers on LLMs"},
    )
    assert response.status_code == 201
    body = response.json()
    assert body["description"] == "Arxiv papers on LLMs"


# ---------------------------------------------------------------------------
# GET /chats/{chat_id}
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_chat_found(api_client: AsyncClient) -> None:
    create_resp = await api_client.post("/chats/", json={"name": "Findable"})
    assert create_resp.status_code == 201
    chat_id = create_resp.json()["id"]

    get_resp = await api_client.get(f"/chats/{chat_id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["id"] == chat_id
    assert get_resp.json()["name"] == "Findable"


@pytest.mark.asyncio
async def test_get_chat_not_found_returns_404(api_client: AsyncClient) -> None:
    response = await api_client.get(f"/chats/{uuid.uuid4()}")
    assert response.status_code == 404
    assert response.json()["detail"] == "chat not found"


# ---------------------------------------------------------------------------
# GET /chats
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_chats_returns_200(api_client: AsyncClient) -> None:
    response = await api_client.get("/chats/")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


@pytest.mark.asyncio
async def test_list_chats_includes_created(api_client: AsyncClient) -> None:
    create_resp = await api_client.post("/chats/", json={"name": "Listed"})
    chat_id = create_resp.json()["id"]

    list_resp = await api_client.get("/chats/")
    ids = [c["id"] for c in list_resp.json()]
    assert chat_id in ids


@pytest.mark.asyncio
async def test_list_chats_pagination(api_client: AsyncClient) -> None:
    for i in range(3):
        await api_client.post("/chats/", json={"name": f"Paginate{i}"})

    page1 = await api_client.get("/chats/?limit=2&offset=0")
    page2 = await api_client.get("/chats/?limit=2&offset=2")

    assert page1.status_code == 200
    assert page2.status_code == 200
    ids1 = {c["id"] for c in page1.json()}
    ids2 = {c["id"] for c in page2.json()}
    assert ids1.isdisjoint(ids2)


# ---------------------------------------------------------------------------
# PATCH /chats/{chat_id}
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_patch_chat_name(api_client: AsyncClient) -> None:
    create_resp = await api_client.post(
        "/chats/", json={"name": "Old Name", "description": "Keep this"}
    )
    chat_id = create_resp.json()["id"]

    patch_resp = await api_client.patch(f"/chats/{chat_id}", json={"name": "New Name"})
    assert patch_resp.status_code == 200
    body = patch_resp.json()
    assert body["name"] == "New Name"
    assert body["description"] == "Keep this"  # unchanged


@pytest.mark.asyncio
async def test_patch_chat_not_found_returns_404(api_client: AsyncClient) -> None:
    response = await api_client.patch(f"/chats/{uuid.uuid4()}", json={"name": "Ghost"})
    assert response.status_code == 404
    assert response.json()["detail"] == "chat not found"


# ---------------------------------------------------------------------------
# DELETE /chats/{chat_id}
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delete_chat_returns_204(api_client: AsyncClient) -> None:
    create_resp = await api_client.post("/chats/", json={"name": "DeleteMe"})
    chat_id = create_resp.json()["id"]

    del_resp = await api_client.delete(f"/chats/{chat_id}")
    assert del_resp.status_code == 204

    get_resp = await api_client.get(f"/chats/{chat_id}")
    assert get_resp.status_code == 404


@pytest.mark.asyncio
async def test_delete_chat_not_found_returns_404(api_client: AsyncClient) -> None:
    response = await api_client.delete(f"/chats/{uuid.uuid4()}")
    assert response.status_code == 404
    assert response.json()["detail"] == "chat not found"
