"""`FileStore` interface: where the original bytes of uploaded guideline files live.

The database keeps the decoded text (`raw_text`) as the system of record for compilation; the
file store keeps the exact uploaded bytes so the original can be downloaded and re-parsed later
(e.g. once PDF/DOCX parsers exist). Nothing outside `app/storage/` knows which backend is used.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from typing import Protocol

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


class StorageUnavailable(Exception):
    """The backend could not be reached or rejected the request."""


class StoredObjectNotFound(Exception):
    """The requested object does not exist in the bucket."""


@dataclass(frozen=True)
class StoredObject:
    data: bytes
    content_type: str


class FileStore(Protocol):
    backend: str
    bucket: str

    async def ensure_bucket(self) -> None: ...

    async def put(self, key: str, data: bytes, content_type: str) -> None: ...

    async def get(self, key: str) -> StoredObject: ...

    async def delete(self, key: str) -> None: ...

    async def healthcheck(self) -> bool: ...

    async def aclose(self) -> None: ...


def safe_file_name(filename: str | None, default: str = "guideline.txt") -> str:
    """Basename only, restricted to `[A-Za-z0-9._-]`, at most 100 chars."""
    base = (filename or "").replace("\\", "/").rsplit("/", 1)[-1]
    cleaned = _UNSAFE.sub("_", base).strip("._")[:100]
    return cleaned or default


def guideline_object_key(company_id: uuid.UUID, version_id: uuid.UUID, filename: str | None) -> str:
    """`<company_id>/<version_id>/<file name>`: tenant-prefixed and unique per version."""
    return f"{company_id}/{version_id}/{safe_file_name(filename)}"
