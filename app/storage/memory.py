"""In-process `FileStore` for tests and offline demos (data is lost on restart)."""

from __future__ import annotations

from app.storage.base import StorageUnavailable, StoredObject, StoredObjectNotFound


class InMemoryFileStore:
    backend = "memory"

    def __init__(self, bucket: str = "guideline-files") -> None:
        self.bucket = bucket
        self.objects: dict[str, StoredObject] = {}
        self.fail_puts = False  # tests flip this to simulate an outage

    async def ensure_bucket(self) -> None:
        return None

    async def put(self, key: str, data: bytes, content_type: str) -> None:
        if self.fail_puts:
            raise StorageUnavailable("simulated storage outage")
        self.objects[key] = StoredObject(data=bytes(data), content_type=content_type)

    async def get(self, key: str) -> StoredObject:
        try:
            return self.objects[key]
        except KeyError:
            raise StoredObjectNotFound(key) from None

    async def delete(self, key: str) -> None:
        self.objects.pop(key, None)

    async def healthcheck(self) -> bool:
        return True

    async def aclose(self) -> None:
        return None
