"""Original guideline files are kept in object storage (in-memory fake standing in for Supabase
Storage) and can be downloaded byte-for-byte."""

from __future__ import annotations

import httpx
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.storage.memory import InMemoryFileStore
from tests.conftest import ADMIN_HEADERS, AGENT_HEADERS
from tests.integration.conftest import BOOKSHOP_TEXT, ClientFactory, create_company, upload

BASE = "/api/v1/companies/bookshop/guidelines"


async def test_upload_stores_original_bytes_and_download_returns_them(
    client_factory: ClientFactory,
) -> None:
    store = InMemoryFileStore()
    # BOM + CRLF: the decoded raw_text differs from the upload, the stored file must not.
    original = b"\xef\xbb\xbf" + BOOKSHOP_TEXT.replace("\n", "\r\n").encode()
    async with client_factory(file_store=store) as client:
        company = await create_company(client, "bookshop")
        r = await upload(client, "bookshop", original, filename="../Book shop.txt")
        assert r.status_code == 201, r.text
        v = r.json()
        assert v["storage_backend"] == "memory"
        assert v["storage_bucket"] == store.bucket
        assert v["storage_key"] == f"{company['id']}/{v['id']}/Book_shop.txt"
        assert v["content_type"] == "text/plain"
        assert store.objects[v["storage_key"]].data == original

        r = await client.get(f"{BASE}/1/file", headers=AGENT_HEADERS)
        assert r.status_code == 200
        assert r.content == original
        assert r.headers["content-type"].startswith("text/plain")
        assert 'filename="Book_shop.txt"' in r.headers["content-disposition"]


async def test_storage_outage_fails_upload_without_creating_a_version(
    client_factory: ClientFactory, db_sessionmaker: async_sessionmaker[AsyncSession]
) -> None:
    store = InMemoryFileStore()
    store.fail_puts = True
    async with client_factory(file_store=store) as client:
        await create_company(client, "bookshop")
        r = await upload(client, "bookshop", BOOKSHOP_TEXT)
        assert r.status_code == 502 and r.json()["error"]["code"] == "STORAGE_ERROR"
        async with db_sessionmaker() as s:
            n = (await s.execute(text("SELECT count(*) FROM guideline_versions"))).scalar_one()
        assert n == 0
        # Recovers once storage is back; version numbering is unaffected.
        store.fail_puts = False
        r = await upload(client, "bookshop", BOOKSHOP_TEXT)
        assert r.status_code == 201 and r.json()["version_number"] == 1


async def test_duplicate_upload_does_not_leave_an_object(client_factory: ClientFactory) -> None:
    store = InMemoryFileStore()
    async with client_factory(file_store=store) as client:
        await create_company(client, "bookshop")
        await upload(client, "bookshop", BOOKSHOP_TEXT)
        r = await upload(client, "bookshop", BOOKSHOP_TEXT)
        assert r.status_code == 409
        assert len(store.objects) == 1


async def test_delete_version_removes_the_stored_file(client_factory: ClientFactory) -> None:
    store = InMemoryFileStore()
    async with client_factory(file_store=store) as client:
        await create_company(client, "bookshop")
        v1 = (await upload(client, "bookshop", BOOKSHOP_TEXT)).json()
        v2 = (await upload(client, "bookshop", BOOKSHOP_TEXT + "\nMORE:\nx\n")).json()
        assert set(store.objects) == {v1["storage_key"], v2["storage_key"]}
        r = await client.delete(f"{BASE}/1", headers=ADMIN_HEADERS)
        assert r.status_code == 204
        assert set(store.objects) == {v2["storage_key"]}


async def test_download_without_storage_is_404(client: httpx.AsyncClient) -> None:
    await create_company(client, "bookshop")
    v = (await upload(client, "bookshop", BOOKSHOP_TEXT)).json()
    assert v["storage_backend"] is None and v["storage_key"] is None
    r = await client.get(f"{BASE}/1/file", headers=AGENT_HEADERS)
    assert r.status_code == 404 and r.json()["error"]["code"] == "FILE_NOT_STORED"


async def test_object_missing_from_bucket_is_404(client_factory: ClientFactory) -> None:
    store = InMemoryFileStore()
    async with client_factory(file_store=store) as client:
        await create_company(client, "bookshop")
        await upload(client, "bookshop", BOOKSHOP_TEXT)
        store.objects.clear()
        r = await client.get(f"{BASE}/1/file", headers=AGENT_HEADERS)
        assert r.status_code == 404 and r.json()["error"]["code"] == "FILE_NOT_STORED"
