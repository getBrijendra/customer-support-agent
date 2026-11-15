from __future__ import annotations

from fastapi import APIRouter, Depends, Path, status

from app.api.deps import company_service
from app.core.security import Principal, require_admin, require_agent
from app.schemas.api import SLUG_PATTERN, CompanyCreate, CompanyOut, ErrorResponse
from app.services.company_service import CompanyService

router = APIRouter(prefix="/companies", tags=["companies"])

SlugPath = Path(pattern=SLUG_PATTERN, examples=["vpn-saas"])


@router.post(
    "",
    response_model=CompanyOut,
    status_code=status.HTTP_201_CREATED,
    summary="Create a company (admin)",
    responses={409: {"model": ErrorResponse}},
)
async def create_company(
    body: CompanyCreate,
    _: Principal = Depends(require_admin),
    service: CompanyService = Depends(company_service),
) -> CompanyOut:
    created = await service.create(slug=body.slug, name=body.name, description=body.description)
    return CompanyOut.build(created.company, created.active_version)


@router.get("", response_model=list[CompanyOut], summary="List companies")
async def list_companies(
    _: Principal = Depends(require_agent),
    service: CompanyService = Depends(company_service),
) -> list[CompanyOut]:
    return [CompanyOut.build(c.company, c.active_version) for c in await service.list()]


@router.get(
    "/{slug}",
    response_model=CompanyOut,
    summary="Company detail incl. active version",
    responses={404: {"model": ErrorResponse}},
)
async def get_company(
    slug: str = SlugPath,
    _: Principal = Depends(require_agent),
    service: CompanyService = Depends(company_service),
) -> CompanyOut:
    found = await service.get(slug)
    return CompanyOut.build(found.company, found.active_version)
