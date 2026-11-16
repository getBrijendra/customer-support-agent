"""Original-file storage: key layout, Supabase Storage REST client (mocked transport), factory."""

from __future__ import annotations

import json
import uuid

import httpx
import pytest

from app.services.ingestion_service import stored_content_type
from app.storage.base import (
    StorageUnavailable,
    StoredObjectNotFound,
    guideline_object_key,
    safe_file_name,
)
from app.storage.factory import build_file_store
from app.storage.memory import InMemoryFileStore
from app.storage.supabase import SupabaseFileStore, auth_headers
from tests.conftest import make_settings

URL = "http://storage.test"


def _store(handler: httpx.MockTransport, key: str = "sb_secret_abc") -> SupabaseFileStore:
    return SupabaseFileStore(URL, key, "guideline-files", transport=handler, max_file_bytes=1000)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("bookshop.txt", "bookshop.txt"),
        ("../../etc/passwd", "passwd"),
        ("C:\\docs\\My Guide (v2).txt", "My_Guide_v2_.txt"),
        ("", "guideline.txt"),
        (None, "guideline.txt"),
        ("...", "guideline.txt"),
    ],
)
def test_safe_file_name(name: str | None, expected: str) -> None:
    assert safe_file_name(name) == expected


def test_object_key_is_tenant_and_version_scoped() -> None:
    c, v = uuid.uuid4(), uuid.uuid4()
    assert guideline_object_key(c, v, "a b.txt") == f"{c}/{v}/a_b.txt"


@pytest.mark.parametrize(
    ("filename", "declared", "expected"),
    [
        ("g.txt", "text/plain; charset=utf-8", "text/plain"),
        ("g.txt", None, "text/plain"),
        ("g.txt", "application/octet-stream", "text/plain"),
        ("g.unknownext", None, "application/octet-stream"),
    ],
)
def test_stored_content_type(filename: str, declared: str | None, expected: str) -> None:
    assert stored_content_type(filename, declared) == expected


def test_auth_headers_for_new_and_legacy_keys() -> None:
    assert auth_headers("sb_secret_x") == {"apikey": "sb_secret_x"}
    assert auth_headers("eyJhbGc") == {"apikey": "eyJhbGc", "Authorization": "Bearer eyJhbGc"}


async def test_put_uploads_with_upsert() -> None:
    seen: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        return httpx.Response(200, json={"Key": "guideline-files/x"})

    store = _store(httpx.MockTransport(handler))
    await store.put("c1/v1/my file.txt", b"hello", "text/plain")
    await store.aclose()
    req = seen[0]
    assert req.method == "POST"
    assert req.url.raw_path == b"/storage/v1/object/guideline-files/c1/v1/my%20file.txt"
    assert req.headers["x-upsert"] == "true"
    assert req.headers["content-type"] == "text/plain"
    assert req.headers["apikey"] == "sb_secret_abc"
    assert "authorization" not in req.headers
    assert req.content == b"hello"


async def test_put_failure_raises_unavailable() -> None:
    store = _store(httpx.MockTransport(lambda r: httpx.Response(403, text="denied")))
    with pytest.raises(StorageUnavailable, match="HTTP 403"):
        await store.put("k", b"x", "text/plain")


async def test_network_error_raises_unavailable() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom", request=req)

    store = _store(httpx.MockTransport(handler))
    with pytest.raises(StorageUnavailable):
        await store.put("k", b"x", "text/plain")
    assert await store.healthcheck() is False


async def test_get_returns_bytes_and_content_type() -> None:
    store = _store(
        httpx.MockTransport(
            lambda r: httpx.Response(
                200, content=b"\xef\xbb\xbfhi", headers={"content-type": "text/plain"}
            )
        )
    )
    obj = await store.get("k")
    assert obj.data == b"\xef\xbb\xbfhi" and obj.content_type == "text/plain"


@pytest.mark.parametrize(
    "resp",
    [
        httpx.Response(404),
        httpx.Response(
            400, json={"statusCode": "404", "error": "not_found", "message": "Object not found"}
        ),
    ],
)
async def test_get_missing_object(resp: httpx.Response) -> None:
    store = _store(httpx.MockTransport(lambda r: resp))
    with pytest.raises(StoredObjectNotFound):
        await store.get("k")


async def test_delete_sends_prefixes() -> None:
    seen: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        return httpx.Response(200, json=[])

    await _store(httpx.MockTransport(handler)).delete("c/v/f.txt")
    assert seen[0].method == "DELETE"
    assert seen[0].url.path == "/storage/v1/object/guideline-files"
    assert json.loads(seen[0].content) == {"prefixes": ["c/v/f.txt"]}


async def test_ensure_bucket_creates_private_bucket_when_missing() -> None:
    seen: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        if req.method == "GET":
            return httpx.Response(400, json={"error": "Bucket not found"})
        return httpx.Response(200, json={"name": "guideline-files"})

    await _store(httpx.MockTransport(handler)).ensure_bucket()
    create = seen[1]
    assert create.method == "POST" and create.url.path == "/storage/v1/bucket"
    assert json.loads(create.content) == {
        "id": "guideline-files",
        "name": "guideline-files",
        "public": False,
        "file_size_limit": 1000,
    }


async def test_ensure_bucket_is_idempotent() -> None:
    calls: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(req.method)
        return httpx.Response(200, json={"id": "guideline-files"})

    await _store(httpx.MockTransport(handler)).ensure_bucket()
    assert calls == ["GET"]  # exists: no create


async def test_ensure_bucket_tolerates_create_race() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        if req.method == "GET":
            return httpx.Response(404)
        return httpx.Response(409, json={"message": "The resource already exists"})

    await _store(httpx.MockTransport(handler)).ensure_bucket()


async def test_in_memory_store_round_trip() -> None:
    store = InMemoryFileStore()
    await store.put("k", b"abc", "text/plain")
    assert (await store.get("k")).data == b"abc"
    await store.delete("k")
    with pytest.raises(StoredObjectNotFound):
        await store.get("k")


def test_factory() -> None:
    assert build_file_store(make_settings()) is None
    assert isinstance(
        build_file_store(make_settings(GUIDELINE_STORAGE_BACKEND="memory")), InMemoryFileStore
    )
    store = build_file_store(
        make_settings(
            GUIDELINE_STORAGE_BACKEND="supabase",
            SUPABASE_URL=URL,
            SUPABASE_SERVICE_ROLE_KEY="sb_secret_x",
        )
    )
    assert isinstance(store, SupabaseFileStore) and store.bucket == "guideline-files"
    with pytest.raises(RuntimeError, match="SUPABASE_URL"):
        build_file_store(make_settings(GUIDELINE_STORAGE_BACKEND="supabase"))
