"""Integration cases 1-7: versioning, activation, failures, duplicates, concurrency."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from tests.conftest import ADMIN_HEADERS, AGENT_HEADERS
from tests.integration.conftest import (
    BOOKSHOP_TEXT,
    ClientFactory,
    active_version_ids,
    app_of,
    create_company,
    upload,
)

BASE = "/api/v1/companies/bookshop/guidelines"


async def _versions(client: httpx.AsyncClient) -> list[dict[str, Any]]:
    r = await client.get(BASE, headers=AGENT_HEADERS)
    assert r.status_code == 200
    return r.json()  # type: ignore[no-any-return]


async def _assert_single_active(
    client: httpx.AsyncClient, sm: async_sessionmaker[AsyncSession], expected_number: int | None
) -> None:
    """Case 4: exactly one active version per company, matching the pointer in the DB."""
    versions = await _versions(client)
    active = [v for v in versions if v["is_active"]]
    pointer = await active_version_ids(sm, "bookshop")
    assert len(pointer) == 1
    if expected_number is None:
        assert active == [] and pointer == [None]
    else:
        assert len(active) == 1 and active[0]["version_number"] == expected_number
        assert str(pointer[0]) == active[0]["id"]


async def test_upload_creates_v1_and_activates(
    client: httpx.AsyncClient, db_sessionmaker: async_sessionmaker[AsyncSession]
) -> None:
    """Case 1."""
    await create_company(client, "bookshop")
    r = await upload(client, "bookshop", BOOKSHOP_TEXT, notes="initial import")
    assert r.status_code == 201, r.text  # INGESTION_SYNC=true in tests
    v1 = r.json()
    assert v1["version_number"] == 1 and v1["status"] == "ready" and v1["is_active"]
    assert v1["chunk_count"] > 0
    assert v1["notes"] == "initial import" and v1["uploaded_by"] == "tester"
    assert v1["embedding_model"] == "fake-embedding"
    vs = app_of(client).state.resources.vector_store
    assert await vs.collection_exists(v1["vector_collection"])
    assert await vs.collection_count(v1["vector_collection"]) == v1["chunk_count"]
    await _assert_single_active(client, db_sessionmaker, 1)

    r = await client.get(f"{BASE}/active", headers=AGENT_HEADERS)
    assert r.json()["version_number"] == 1 and r.json()["raw_text"] is None
    r = await client.get(f"{BASE}/1", params={"include_text": "true"}, headers=AGENT_HEADERS)
    assert r.json()["raw_text"].startswith("TONE")
    detail = (await client.get("/api/v1/companies/bookshop", headers=AGENT_HEADERS)).json()
    assert detail["active_version_number"] == 1


async def test_second_upload_becomes_active_and_rollback(
    client: httpx.AsyncClient, db_sessionmaker: async_sessionmaker[AsyncSession]
) -> None:
    """Cases 2, 3 and 4."""
    await create_company(client, "bookshop")
    v1 = (await upload(client, "bookshop", BOOKSHOP_TEXT)).json()
    await _assert_single_active(client, db_sessionmaker, 1)

    v2 = (await upload(client, "bookshop", BOOKSHOP_TEXT + "\nNEW RULE:\nFree bookmarks.\n")).json()
    assert v2["version_number"] == 2 and v2["is_active"]
    versions = await _versions(client)
    assert [v["version_number"] for v in versions] == [2, 1]  # newest first
    assert versions[1]["status"] == "ready" and not versions[1]["is_active"]
    await _assert_single_active(client, db_sessionmaker, 2)

    # Case 3: roll back to v1, then a chat must record v1.
    r = await client.post(f"{BASE}/1/activate", headers=ADMIN_HEADERS)
    assert r.status_code == 200 and r.json()["is_active"]
    await _assert_single_active(client, db_sessionmaker, 1)

    r = await client.post(
        "/api/v1/chats",
        json={"company": "bookshop", "customer_message": "Can I return a damaged book?"},
        headers=AGENT_HEADERS,
    )
    assert r.status_code == 201, r.text
    assert r.json()["guideline_version"] == 1
    detail = (
        await client.get(f"/api/v1/chats/{r.json()['chat_id']}", headers=AGENT_HEADERS)
    ).json()
    assert detail["guideline_version_id"] == v1["id"]
    assert all(c["chunk_id"].startswith(v1["id"]) for c in detail["retrieved_chunks"])

    # Roll forward again.
    assert (await client.post(f"{BASE}/2/activate", headers=ADMIN_HEADERS)).status_code == 200
    await _assert_single_active(client, db_sessionmaker, 2)

    # Upload without activation keeps v2 active.
    v3 = (await upload(client, "bookshop", BOOKSHOP_TEXT + "\nX:\ny\n", activate=False)).json()
    assert v3["status"] == "ready" and not v3["is_active"]
    await _assert_single_active(client, db_sessionmaker, 2)


async def test_failed_ingestion_keeps_previous_active(
    client: httpx.AsyncClient, db_sessionmaker: async_sessionmaker[AsyncSession], monkeypatch: Any
) -> None:
    """Case 5."""
    await create_company(client, "bookshop")
    await upload(client, "bookshop", BOOKSHOP_TEXT)
    vs = app_of(client).state.resources.vector_store

    async def boom(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("vector store is down")

    monkeypatch.setattr(vs, "add_chunks", boom)
    r = await upload(client, "bookshop", BOOKSHOP_TEXT + "\nCHANGED:\nyes\n")
    assert r.status_code == 201
    failed = r.json()
    assert failed["version_number"] == 2 and failed["status"] == "failed"
    assert "vector store is down" in failed["error_message"]
    assert not failed["is_active"]
    assert not await vs.collection_exists(failed["vector_collection"])
    await _assert_single_active(client, db_sessionmaker, 1)

    # failed versions can't be activated
    r = await client.post(f"{BASE}/2/activate", headers=ADMIN_HEADERS)
    assert r.status_code == 409 and r.json()["error"]["code"] == "VERSION_NOT_READY"
    # chats keep working off v1
    r = await client.post(
        "/api/v1/chats",
        json={"company": "bookshop", "customer_message": "hi"},
        headers=AGENT_HEADERS,
    )
    assert r.status_code == 201 and r.json()["guideline_version"] == 1


async def test_duplicate_upload_of_active_content_409(client: httpx.AsyncClient) -> None:
    """Case 6."""
    await create_company(client, "bookshop")
    await upload(client, "bookshop", BOOKSHOP_TEXT)
    r = await upload(client, "bookshop", BOOKSHOP_TEXT, filename="again.txt")
    assert r.status_code == 409
    err = r.json()["error"]
    assert err["code"] == "DUPLICATE_GUIDELINES"
    assert err["details"]["existing_version"]["version_number"] == 1
    assert len(await _versions(client)) == 1

    # Same content as an older, NON-active version is allowed (with an info hint).
    await upload(client, "bookshop", BOOKSHOP_TEXT + "\nV2:\nx\n")
    r = await upload(client, "bookshop", BOOKSHOP_TEXT)
    assert r.status_code == 201
    assert r.json()["version_number"] == 3 and "v1" in r.json()["info"]


async def test_concurrent_uploads_get_distinct_versions(
    client: httpx.AsyncClient, db_sessionmaker: async_sessionmaker[AsyncSession]
) -> None:
    """Case 7: SELECT ... FOR UPDATE serialises version allocation."""
    await create_company(client, "bookshop")
    results = await asyncio.gather(
        *(upload(client, "bookshop", BOOKSHOP_TEXT + f"\nVARIANT {i}:\nx{i}\n") for i in range(4))
    )
    assert all(r.status_code == 201 for r in results), [r.text for r in results]
    numbers = sorted(r.json()["version_number"] for r in results)
    assert numbers == [1, 2, 3, 4]
    # "latest wins" even if versions finish out of order
    await _assert_single_active(client, db_sessionmaker, 4)


async def test_delete_version(client: httpx.AsyncClient) -> None:
    await create_company(client, "bookshop")
    v1 = (await upload(client, "bookshop", BOOKSHOP_TEXT)).json()
    await upload(client, "bookshop", BOOKSHOP_TEXT + "\nMORE:\nx\n")
    r = await client.delete(f"{BASE}/2", headers=ADMIN_HEADERS)
    assert r.status_code == 409 and r.json()["error"]["code"] == "VERSION_IS_ACTIVE"
    r = await client.delete(f"{BASE}/1", headers=ADMIN_HEADERS)
    assert r.status_code == 204
    assert not await app_of(client).state.resources.vector_store.collection_exists(
        v1["vector_collection"]
    )
    assert [v["version_number"] for v in await _versions(client)] == [2]
    assert (await client.get(f"{BASE}/1", headers=AGENT_HEADERS)).status_code == 404


async def test_upload_validation_errors(client: httpx.AsyncClient) -> None:
    await create_company(client, "bookshop")
    r = await upload(client, "bookshop", "hello", filename="guide.pdf")
    assert r.status_code == 422 and r.json()["error"]["code"] == "INVALID_FILE"
    r = await upload(client, "bookshop", "   ")
    assert r.json()["error"]["code"] == "INVALID_FILE"
    r = await upload(client, "nope", BOOKSHOP_TEXT)
    assert r.status_code == 404 and r.json()["error"]["code"] == "COMPANY_NOT_FOUND"


async def test_upload_too_large(client_factory: ClientFactory) -> None:
    async with client_factory(MAX_UPLOAD_BYTES=100) as client:
        await create_company(client, "bookshop")
        r = await upload(client, "bookshop", "x" * 101)
        assert r.status_code == 413 and r.json()["error"]["code"] == "FILE_TOO_LARGE"


async def test_async_mode_returns_202_then_ready(client_factory: ClientFactory) -> None:
    async with client_factory(INGESTION_SYNC=False) as client:
        await create_company(client, "bookshop")
        r = await upload(client, "bookshop", BOOKSHOP_TEXT)
        assert r.status_code == 202
        assert r.json()["status"] == "processing"
        # httpx's ASGI transport runs background tasks before returning the response
        v = (await client.get(f"{BASE}/1", headers=AGENT_HEADERS)).json()
        assert v["status"] == "ready" and v["is_active"]
