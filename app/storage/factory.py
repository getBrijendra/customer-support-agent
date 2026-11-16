from __future__ import annotations

from app.core.config import Settings
from app.storage.base import FileStore
from app.storage.memory import InMemoryFileStore
from app.storage.supabase import SupabaseFileStore


def build_file_store(settings: Settings) -> FileStore | None:
    """None when `GUIDELINE_STORAGE_BACKEND=none`: original files are then not kept."""
    backend = settings.GUIDELINE_STORAGE_BACKEND
    if backend == "none":
        return None
    if backend == "memory":
        return InMemoryFileStore(settings.GUIDELINE_STORAGE_BUCKET)
    if not settings.SUPABASE_URL or not settings.SUPABASE_SERVICE_ROLE_KEY:
        raise RuntimeError(
            "GUIDELINE_STORAGE_BACKEND=supabase needs SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY"
        )
    return SupabaseFileStore(
        settings.SUPABASE_URL,
        settings.SUPABASE_SERVICE_ROLE_KEY.get_secret_value(),
        settings.GUIDELINE_STORAGE_BUCKET,
        timeout=settings.STORAGE_TIMEOUT_SECONDS,
        max_file_bytes=settings.MAX_UPLOAD_BYTES,
    )
