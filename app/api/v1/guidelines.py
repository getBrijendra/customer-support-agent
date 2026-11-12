from __future__ import annotations

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    Path,
    Query,
    Request,
    Response,
    UploadFile,
    status,
)

from app.api.deps import get_resources, ingestion_service
from app.core.errors import FileTooLarge
from app.core.security import Principal, require_admin, require_agent
from app.schemas.api import SLUG_PATTERN, ErrorResponse, GuidelineVersionOut
from app.services.ingestion_service import IngestionService

router = APIRouter(prefix="/companies/{slug}/guidelines", tags=["guidelines"])

SlugPath = Path(pattern=SLUG_PATTERN, examples=["bookshop"])
VersionPath = Path(ge=1, examples=[1])


@router.post(
    "",
    response_model=GuidelineVersionOut,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload a new guideline version (.txt) (admin)",
    description=(
        "Creates a new immutable version. Returns **202** with `status=processing` and ingests in "
        "the background, or **201** with the final status when `INGESTION_SYNC=true`. "
        "With `activate=true` (default) the version becomes active once ready."
    ),
    responses={
        201: {"model": GuidelineVersionOut, "description": "Ingested synchronously"},
        404: {"model": ErrorResponse},
        409: {"model": ErrorResponse, "description": "DUPLICATE_GUIDELINES"},
        413: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
    },
)
async def upload_guidelines(
    request: Request,
    response: Response,
    background: BackgroundTasks,
    slug: str = SlugPath,
    file: UploadFile = File(..., description="UTF-8 .txt guideline file"),
    activate: bool = Form(True),
    notes: str | None = Form(None, max_length=2000),
    principal: Principal = Depends(require_admin),
    service: IngestionService = Depends(ingestion_service),
) -> GuidelineVersionOut:
    settings = get_resources(request).settings
    content = await file.read(settings.MAX_UPLOAD_BYTES + 1)
    if len(content) > settings.MAX_UPLOAD_BYTES:
        raise FileTooLarge(f"File exceeds the {settings.MAX_UPLOAD_BYTES} byte limit")
    created = await service.create_version(
        slug=slug,
        filename=file.filename,
        content=content,
        notes=notes,
        uploaded_by=principal.label,
    )
    version_id = created.version.id
    if settings.INGESTION_SYNC:
        await service.run_ingestion(version_id, activate=activate)
        response.status_code = status.HTTP_201_CREATED
    else:
        background.add_task(service.run_ingestion, version_id, activate=activate)
    company, version = await service.get_version(slug, created.version.version_number)
    return GuidelineVersionOut.build(version, company=company, info=created.info)


@router.get("", response_model=list[GuidelineVersionOut], summary="List versions, newest first")
async def list_guidelines(
    slug: str = SlugPath,
    _: Principal = Depends(require_agent),
    service: IngestionService = Depends(ingestion_service),
) -> list[GuidelineVersionOut]:
    company, versions = await service.list_versions(slug)
    return [GuidelineVersionOut.build(v, company=company) for v in versions]


@router.get(
    "/active",
    response_model=GuidelineVersionOut,
    summary="Active version metadata",
    responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}},
)
async def get_active_guidelines(
    slug: str = SlugPath,
    include_text: bool = Query(False),
    _: Principal = Depends(require_agent),
    service: IngestionService = Depends(ingestion_service),
) -> GuidelineVersionOut:
    company, version = await service.get_active(slug)
    return GuidelineVersionOut.build(version, company=company, include_text=include_text)


@router.get(
    "/{version_number}",
    response_model=GuidelineVersionOut,
    summary="Version metadata (optionally raw text)",
    responses={404: {"model": ErrorResponse}},
)
async def get_guideline_version(
    slug: str = SlugPath,
    version_number: int = VersionPath,
    include_text: bool = Query(False),
    _: Principal = Depends(require_agent),
    service: IngestionService = Depends(ingestion_service),
) -> GuidelineVersionOut:
    company, version = await service.get_version(slug, version_number)
    return GuidelineVersionOut.build(version, company=company, include_text=include_text)


@router.post(
    "/{version_number}/activate",
    response_model=GuidelineVersionOut,
    summary="Activate a ready version (rollback / roll forward) (admin)",
    responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}},
)
async def activate_guideline_version(
    slug: str = SlugPath,
    version_number: int = VersionPath,
    _: Principal = Depends(require_admin),
    service: IngestionService = Depends(ingestion_service),
) -> GuidelineVersionOut:
    company, version = await service.activate(slug, version_number)
    return GuidelineVersionOut.build(version, company=company)


@router.delete(
    "/{version_number}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete an inactive version and its vectors (admin)",
    responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}},
)
async def delete_guideline_version(
    slug: str = SlugPath,
    version_number: int = VersionPath,
    _: Principal = Depends(require_admin),
    service: IngestionService = Depends(ingestion_service),
) -> Response:
    await service.delete_version(slug, version_number)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
