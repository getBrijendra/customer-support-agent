"""Review & activation rules for guideline versions.

- approve: ready + (pending | rejected) -> approved; activates if `activate_on_approval`
- reject: not allowed on the active version; rejected versions can't be activated
- activate: only ready + approved (rollback / roll forward)
- delete: refuses the active version; compiled rows cascade
"""

from __future__ import annotations

from app.core.errors import (
    CompanyNotFound,
    Conflict,
    VersionIsActive,
    VersionNotApproved,
    VersionNotFound,
    VersionNotReady,
)
from app.core.logging import get_logger
from app.db.models import Company, GuidelineVersion, ReviewStatus, VersionStatus, utcnow
from app.services.common import CompanyContextCache, UnitOfWork, UowFactory

logger = get_logger(__name__)


async def activate_if_latest(uow: UnitOfWork, company: Company, version: GuidelineVersion) -> bool:
    """Activate `version` unless a NEWER version is already active ("latest wins": an older,
    slower concurrent upload never overrides a newer one). Caller holds the company lock."""
    current = (
        await uow.versions.get(company.active_version_id) if company.active_version_id else None
    )
    if current is not None and current.version_number > version.version_number:
        return False
    await uow.companies.set_active_version(company, version)
    return True


class ReviewService:
    def __init__(self, uow_factory: UowFactory, cache: CompanyContextCache) -> None:
        self._uow = uow_factory
        self._cache = cache

    async def _locked(
        self, uow: UnitOfWork, slug: str, number: int
    ) -> tuple[Company, GuidelineVersion]:
        company: Company | None = await uow.companies.get_by_slug_for_update(slug)
        if company is None:
            raise CompanyNotFound(f"Company '{slug}' not found")
        version: GuidelineVersion | None = await uow.versions.get_by_number(company.id, number)
        if version is None:
            raise VersionNotFound(f"Version {number} not found for '{slug}'")
        return company, version

    async def approve(
        self, slug: str, number: int, *, by: str | None, note: str | None = None
    ) -> tuple[Company, GuidelineVersion]:
        async with self._uow() as uow:
            company, version = await self._locked(uow, slug, number)
            if version.status != VersionStatus.READY:
                raise VersionNotReady(f"Version {number} is '{version.status}'")
            if version.review_status == ReviewStatus.APPROVED:
                raise Conflict(f"Version {number} is already approved")
            version.review_status = ReviewStatus.APPROVED
            version.reviewed_by = by
            version.reviewed_at = utcnow()
            version.review_note = note or "approved by admin"
            activated = False
            if version.activate_on_approval:
                await uow.companies.set_active_version(company, version)
                activated = True
            await uow.commit()
        self._cache.invalidate(slug)
        logger.info("guidelines.approved", extra={"company": slug, "version": number})
        if activated:
            logger.info("guidelines.activated", extra={"company": slug, "version": number})
        return company, version

    async def reject(
        self, slug: str, number: int, *, by: str | None, reason: str | None
    ) -> tuple[Company, GuidelineVersion]:
        async with self._uow() as uow:
            company, version = await self._locked(uow, slug, number)
            if company.active_version_id == version.id:
                raise VersionIsActive("Cannot reject the active version; activate another first")
            version.review_status = ReviewStatus.REJECTED
            version.reviewed_by = by
            version.reviewed_at = utcnow()
            version.review_note = reason or "rejected by admin"
            await uow.commit()
        logger.info("guidelines.rejected", extra={"company": slug, "version": number})
        return company, version

    async def activate(self, slug: str, number: int) -> tuple[Company, GuidelineVersion]:
        async with self._uow() as uow:
            company, version = await self._locked(uow, slug, number)
            if version.status != VersionStatus.READY:
                raise VersionNotReady(
                    f"Version {number} is '{version.status}'; only ready versions can be activated"
                )
            if version.review_status != ReviewStatus.APPROVED:
                raise VersionNotApproved(
                    f"Version {number} is '{version.review_status}'; approve it first"
                )
            await uow.companies.set_active_version(company, version)
            await uow.commit()
        self._cache.invalidate(slug)
        logger.info("guidelines.activated", extra={"company": slug, "version": number})
        return company, version

    async def delete(self, slug: str, number: int) -> None:
        async with self._uow() as uow:
            company, version = await self._locked(uow, slug, number)
            if company.active_version_id == version.id:
                raise VersionIsActive("Cannot delete the active version; activate another first")
            await uow.versions.delete(version.id)  # compiled rows + vectors cascade
            await uow.commit()
        self._cache.invalidate(slug)
        logger.info("guidelines.deleted", extra={"company": slug, "version": number})
