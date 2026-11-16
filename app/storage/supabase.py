"""`FileStore` backed by Supabase Storage (REST API, private bucket, service-role key).

Endpoints used (`{SUPABASE_URL}/storage/v1/...`):
  GET/POST   bucket[/{id}]                 check / create the private bucket
  POST       object/{bucket}/{key}         upload (x-upsert, so a retried upload is idempotent)
  GET        object/{bucket}/{key}         download
  DELETE     object/{bucket}  {prefixes}   delete
"""

from __future__ import annotations

from typing import Any
from urllib.parse import quote

import httpx

from app.core.logging import get_logger
from app.storage.base import StorageUnavailable, StoredObject, StoredObjectNotFound

logger = get_logger(__name__)


def auth_headers(key: str) -> dict[str, str]:
    headers = {"apikey": key}
    # Legacy service_role keys are JWTs and also go in Authorization. New `sb_secret_...` keys
    # are not JWTs; the API gateway accepts them in `apikey` only.
    if key.startswith("eyJ"):
        headers["Authorization"] = f"Bearer {key}"
    return headers


def _is_not_found(resp: httpx.Response) -> bool:
    if resp.status_code == 404:
        return True
    # Storage has historically answered 400 with {"statusCode": "404", "error": "not_found"}
    return resp.status_code == 400 and "not found" in resp.text.lower()


class SupabaseFileStore:
    backend = "supabase"

    def __init__(
        self,
        url: str,
        service_key: str,
        bucket: str,
        *,
        timeout: float = 30.0,
        max_file_bytes: int | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.bucket = bucket
        self._max_file_bytes = max_file_bytes
        self._client = httpx.AsyncClient(
            base_url=url.rstrip("/") + "/storage/v1",
            headers=auth_headers(service_key),
            timeout=timeout,
            transport=transport,
        )

    def _bucket_q(self) -> str:
        return quote(self.bucket, safe="")

    def _object_path(self, key: str) -> str:
        return f"/object/{self._bucket_q()}/{quote(key, safe='/')}"

    async def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        try:
            return await self._client.request(method, path, **kwargs)
        except httpx.HTTPError as exc:
            raise StorageUnavailable(
                f"Supabase Storage request failed: {type(exc).__name__}"
            ) from exc

    @staticmethod
    def _fail(action: str, resp: httpx.Response) -> StorageUnavailable:
        return StorageUnavailable(
            f"Supabase Storage {action} failed with HTTP {resp.status_code}: {resp.text[:200]}"
        )

    async def ensure_bucket(self) -> None:
        """Create the private bucket if it doesn't exist (idempotent)."""
        resp = await self._request("GET", f"/bucket/{self._bucket_q()}")
        if resp.status_code == 200:
            return
        body: dict[str, Any] = {"id": self.bucket, "name": self.bucket, "public": False}
        if self._max_file_bytes:
            body["file_size_limit"] = self._max_file_bytes
        resp = await self._request("POST", "/bucket", json=body)
        if resp.status_code in (200, 201) or (
            resp.status_code in (400, 409) and "already exists" in resp.text.lower()
        ):
            logger.info("storage.bucket_ready", extra={"bucket": self.bucket})
            return
        raise self._fail("bucket creation", resp)

    async def put(self, key: str, data: bytes, content_type: str) -> None:
        resp = await self._request(
            "POST",
            self._object_path(key),
            content=data,
            headers={"Content-Type": content_type, "x-upsert": "true"},
        )
        if resp.status_code not in (200, 201):
            raise self._fail("upload", resp)

    async def get(self, key: str) -> StoredObject:
        resp = await self._request("GET", self._object_path(key))
        if resp.status_code == 200:
            return StoredObject(
                data=resp.content,
                content_type=resp.headers.get("content-type", "application/octet-stream"),
            )
        if _is_not_found(resp):
            raise StoredObjectNotFound(key)
        raise self._fail("download", resp)

    async def delete(self, key: str) -> None:
        resp = await self._request(
            "DELETE", f"/object/{self._bucket_q()}", json={"prefixes": [key]}
        )
        if resp.status_code not in (200, 204) and not _is_not_found(resp):
            raise self._fail("delete", resp)

    async def healthcheck(self) -> bool:
        try:
            resp = await self._request("GET", f"/bucket/{self._bucket_q()}")
        except StorageUnavailable:
            return False
        return resp.status_code == 200

    async def aclose(self) -> None:
        await self._client.aclose()
