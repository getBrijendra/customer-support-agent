"""Guideline upload -> validate -> chunk -> embed -> activate, plus version management.

`IngestionService.run_ingestion` has no FastAPI dependency so it can move to a worker queue
(Arq/Celery) unchanged: it only needs a version id and the injected collaborators.
"""

from __future__ import annotations

import hashlib
import time
import uuid
from dataclasses import dataclass

from sqlalchemy.exc import IntegrityError

from app.ai.chunker import chunk_guideline
from app.core.config import Settings
from app.core.errors import (
    CompanyNotFound,
    DuplicateGuidelines,
    FileTooLarge,
    InvalidFile,
    NoActiveGuidelines,
    VersionIsActive,
    VersionNotFound,
    VersionNotReady,
)
from app.core.logging import get_logger
from app.db.models import Company, GuidelineVersion, VersionStatus, utcnow
from app.services.common import AppMetrics, CompanyContextCache, UnitOfWork, UowFactory
from app.vectorstore.base import VectorStoreGateway

logger = get_logger(__name__)

_ALLOWED_CONTROL = {"\t", "\n", "\r", "\x0c"}


def _has_bad_control_chars(text: str) -> bool:
    return any(
        (ord(ch) < 32 and ch not in _ALLOWED_CONTROL) or 0x7F <= ord(ch) <= 0x9F for ch in text
    )


def decode_text(content: bytes) -> str:
    """UTF-8 (BOM tolerant) first; latin-1 only if it yields clean text."""
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = content.decode("latin-1")
        if _has_bad_control_chars(text):
            raise InvalidFile("File is not valid UTF-8 or Latin-1 text") from None
        return text
    if "\x00" in text:
        raise InvalidFile("File looks binary, not text")
    return text


def validate_upload(filename: str | None, content: bytes, max_bytes: int) -> str:
    """Validate an uploaded guideline file and return its decoded text."""
    if not filename or not filename.lower().endswith(".txt"):
        raise InvalidFile("Only .txt guideline files are accepted")
    if len(content) > max_bytes:
        raise FileTooLarge(f"File exceeds the {max_bytes} byte limit")
    text = decode_text(content)
    if not text.strip():
        raise InvalidFile("File is empty")
    return text


def sha256_hex(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


@dataclass
class CreatedVersion:
    version: GuidelineVersion
    company_slug: str
    info: str | None = None


def version_summary(v: GuidelineVersion) -> dict[str, object]:
    return {"version_number": v.version_number, "id": str(v.id), "status": v.status}


class IngestionService:
    def __init__(
        self,
        uow_factory: UowFactory,
        vector_store: VectorStoreGateway,
        settings: Settings,
        cache: CompanyContextCache,
        metrics: AppMetrics,
    ) -> None:
        self._uow = uow_factory
        self._vs = vector_store
        self._settings = settings
        self._cache = cache
        self._metrics = metrics

    # ------------------------------------------------------------------ upload
    async def create_version(
        self,
        *,
        slug: str,
        filename: str | None,
        content: bytes,
        notes: str | None,
        uploaded_by: str | None,
    ) -> CreatedVersion:
        text = validate_upload(filename, content, self._settings.MAX_UPLOAD_BYTES)
        digest = sha256_hex(content)
        last_exc: Exception | None = None
        for _attempt in range(2):
            async with self._uow() as uow:
                try:
                    return await self._allocate(
                        uow, slug, filename or "", text, content, digest, notes, uploaded_by
                    )
                except IntegrityError as exc:  # lost a version-number race; retry once
                    await uow.rollback()
                    last_exc = exc
        raise RuntimeError("Could not allocate a version number") from last_exc

    async def _allocate(
        self,
        uow: UnitOfWork,
        slug: str,
        filename: str,
        text: str,
        content: bytes,
        digest: str,
        notes: str | None,
        uploaded_by: str | None,
    ) -> CreatedVersion:
        # Row lock on the company serialises version allocation for that company.
        company: Company | None = await uow.companies.get_by_slug_for_update(slug)
        if company is None:
            raise CompanyNotFound(f"Company '{slug}' not found")
        existing: list[GuidelineVersion] = await uow.versions.list_for_company(company.id)
        active = next((v for v in existing if v.id == company.active_version_id), None)
        if active is not None and active.content_sha256 == digest:
            # (the session is rolled back when the unit of work closes)
            raise DuplicateGuidelines(
                f"Content is identical to the active version v{active.version_number}",
                details={"existing_version": version_summary(active)},
            )
        info = None
        same = next(
            (v for v in existing if v.content_sha256 == digest and v.status == VersionStatus.READY),
            None,
        )
        if same is not None:
            info = (
                f"Content is identical to v{same.version_number}; you could activate that "
                "version instead of re-uploading."
            )
        number: int = await uow.versions.next_version_number(company.id)
        version = GuidelineVersion(
            id=uuid.uuid4(),
            company_id=company.id,
            version_number=number,
            status=VersionStatus.PROCESSING,
            file_name=filename,
            raw_text=text,
            content_sha256=digest,
            byte_size=len(content),
            vector_collection=self._vs.collection_name(company.id, number),
            embedding_model=self._settings.EMBEDDING_MODEL,
            chunk_size=self._settings.CHUNK_SIZE,
            chunk_overlap=self._settings.CHUNK_OVERLAP,
            notes=notes,
            uploaded_by=uploaded_by,
            created_at=utcnow(),
        )
        await uow.versions.add(version)
        await uow.commit()
        logger.info(
            "ingestion.started",
            extra={"company": slug, "version": number, "bytes": len(content)},
        )
        return CreatedVersion(version=version, company_slug=slug, info=info)

    # ------------------------------------------------------------- background
    async def run_ingestion(
        self, version_id: uuid.UUID, *, activate: bool = True
    ) -> GuidelineVersion | None:
        started = time.perf_counter()
        async with self._uow() as uow:
            version: GuidelineVersion | None = await uow.versions.get(version_id)
            if version is None:
                logger.warning("ingestion.version_missing", extra={"version_id": str(version_id)})
                return None
            company: Company | None = await uow.companies.get_by_id(version.company_id)
            assert company is not None
            slug, company_id = company.slug, company.id
            number = version.version_number
            collection = version.vector_collection or self._vs.collection_name(company_id, number)
            raw_text, embedding_model = version.raw_text, version.embedding_model
            chunk_size, chunk_overlap = version.chunk_size, version.chunk_overlap
            file_name = version.file_name

        try:
            chunks = chunk_guideline(
                raw_text,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                version_id=version_id,
                base_metadata={
                    "company_id": str(company_id),
                    "company_slug": slug,
                    "version_id": str(version_id),
                    "version_number": number,
                    "source_file": file_name,
                },
            )
            if not chunks:
                raise ValueError("Guideline produced no chunks")
            await self._vs.add_chunks(
                collection,
                [c.document for c in chunks],
                [c.id for c in chunks],
                embedding_model=embedding_model,
            )
            async with self._uow() as uow:
                locked: Company | None = await uow.companies.lock(company_id)
                version = await uow.versions.get(version_id)
                if locked is None or version is None:
                    raise RuntimeError("Company or version disappeared during ingestion")
                version.status = VersionStatus.READY
                version.chunk_count = len(chunks)
                version.vector_collection = collection
                version.ready_at = utcnow()
                activated = False
                if activate:
                    current = (
                        await uow.versions.get(locked.active_version_id)
                        if locked.active_version_id
                        else None
                    )
                    # "Latest wins": an older, slower concurrent upload never overrides a newer one.
                    if current is None or current.version_number < version.version_number:
                        await uow.companies.set_active_version(locked, version)
                        activated = True
                await uow.commit()
            self._cache.invalidate(slug)
            duration_ms = int((time.perf_counter() - started) * 1000)
            self._metrics.ingestions.labels(status="ready").inc()
            logger.info(
                "ingestion.completed",
                extra={
                    "company": slug,
                    "version": number,
                    "chunk_count": len(chunks),
                    "duration_ms": duration_ms,
                    "activated": activated,
                },
            )
            if activated:
                logger.info("guidelines.activated", extra={"company": slug, "version": number})
            return version
        except Exception as exc:
            await self._vs.delete_collection(collection)  # best effort; never touches active
            async with self._uow() as uow:
                version = await uow.versions.get(version_id)
                if version is not None:
                    version.status = VersionStatus.FAILED
                    version.error_message = f"{type(exc).__name__}: {exc}"[:2000]
                    await uow.commit()
            self._metrics.ingestions.labels(status="failed").inc()
            logger.error(
                "ingestion.failed",
                extra={
                    "company": slug,
                    "version": number,
                    "duration_ms": int((time.perf_counter() - started) * 1000),
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
            return version

    # ------------------------------------------------------------ management
    async def list_versions(self, slug: str) -> tuple[Company, list[GuidelineVersion]]:
        async with self._uow() as uow:
            company = await self._company(uow, slug)
            versions: list[GuidelineVersion] = await uow.versions.list_for_company(company.id)
            return company, versions

    async def get_version(self, slug: str, number: int) -> tuple[Company, GuidelineVersion]:
        async with self._uow() as uow:
            company = await self._company(uow, slug)
            version = await uow.versions.get_by_number(company.id, number)
            if version is None:
                raise VersionNotFound(f"Version {number} not found for '{slug}'")
            return company, version

    async def get_active(self, slug: str) -> tuple[Company, GuidelineVersion]:
        async with self._uow() as uow:
            company = await self._company(uow, slug)
            if company.active_version_id is None:
                raise NoActiveGuidelines(f"Company '{slug}' has no active guideline version")
            version = await uow.versions.get(company.active_version_id)
            assert version is not None
            return company, version

    async def activate(self, slug: str, number: int) -> tuple[Company, GuidelineVersion]:
        async with self._uow() as uow:
            company = await uow.companies.get_by_slug_for_update(slug)
            if company is None:
                raise CompanyNotFound(f"Company '{slug}' not found")
            version = await uow.versions.get_by_number(company.id, number)
            if version is None:
                raise VersionNotFound(f"Version {number} not found for '{slug}'")
            if version.status != VersionStatus.READY:
                raise VersionNotReady(
                    f"Version {number} is '{version.status}'; only ready versions can be activated"
                )
            await uow.companies.set_active_version(company, version)
            await uow.commit()
        self._cache.invalidate(slug)
        logger.info("guidelines.activated", extra={"company": slug, "version": number})
        return company, version

    async def delete_version(self, slug: str, number: int) -> None:
        async with self._uow() as uow:
            company = await uow.companies.get_by_slug_for_update(slug)
            if company is None:
                raise CompanyNotFound(f"Company '{slug}' not found")
            version = await uow.versions.get_by_number(company.id, number)
            if version is None:
                raise VersionNotFound(f"Version {number} not found for '{slug}'")
            if company.active_version_id == version.id:
                raise VersionIsActive("Cannot delete the active version; activate another first")
            collection = version.vector_collection
            await uow.versions.delete(version.id)
            await uow.commit()
        if collection:
            await self._vs.delete_collection(collection)
        self._cache.invalidate(slug)
        logger.info("guidelines.deleted", extra={"company": slug, "version": number})

    @staticmethod
    async def _company(uow: UnitOfWork, slug: str) -> Company:
        company: Company | None = await uow.companies.get_by_slug(slug)
        if company is None:
            raise CompanyNotFound(f"Company '{slug}' not found")
        return company
