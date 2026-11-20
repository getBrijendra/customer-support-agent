from __future__ import annotations

from typing import Any

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Body,
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

from app.api.deps import (
    get_resources,
    ingestion_service,
    integration_service,
    review_service,
)
from app.core.errors import FileTooLarge
from app.core.security import Principal, require_admin, require_agent
from app.schemas.api import (
    SLUG_PATTERN,
    ErrorResponse,
    FactOut,
    GuidelineVersionOut,
    IntentOut,
    PlaybookOut,
    RejectIn,
    ReviewIn,
    SectionOut,
)
from app.services.ingestion_service import IngestionService
from app.services.integration_service import IntegrationService
from app.services.review_service import ReviewService

router = APIRouter(prefix="/companies/{slug}/guidelines", tags=["guidelines"])

SlugPath = Path(pattern=SLUG_PATTERN, examples=["bookshop"])
VersionPath = Path(ge=1, examples=[1])
E404: dict[int | str, dict[str, Any]] = {404: {"model": ErrorResponse}}
E409: dict[int | str, dict[str, Any]] = {409: {"model": ErrorResponse}}


@router.post(
    "",
    response_model=GuidelineVersionOut,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload a new guideline version (.txt) (admin)",
    description=(
        "Creates a new immutable version, then parses, compiles (intents, company rules, facts), "
        "indexes and evaluates it. Returns **202** with `status=processing` and works in the "
        "background, or **201** with the final status when `INGESTION_SYNC=true`. By default the "
        "version is auto-approved (if the routing eval passes) and becomes active."
    ),
    responses={
        201: {"model": GuidelineVersionOut, "description": "Ingested synchronously"},
        **E404,
        409: {"model": ErrorResponse, "description": "DUPLICATE_GUIDELINES"},
        413: {"model": ErrorResponse},
        422: {"model": ErrorResponse, "description": "INVALID_FILE / UNSUPPORTED_FORMAT"},
    },
)
async def upload_guidelines(
    request: Request,
    response: Response,
    background: BackgroundTasks,
    slug: str = SlugPath,
    file: UploadFile = File(..., description="UTF-8 .txt guideline file"),
    activate: bool = Form(True, description="Activate once ready and approved"),
    notes: str | None = Form(None, max_length=2000),
    principal: Principal = Depends(require_admin),
    service: IngestionService = Depends(ingestion_service),
) -> GuidelineVersionOut:
    settings = get_resources(request).settings
    content = await file.read(settings.MAX_UPLOAD_BYTES + 1)
    if len(content) > settings.MAX_UPLOAD_BYTES:
        raise FileTooLarge(f"File exceeds the {settings.MAX_UPLOAD_BYTES} byte limit")
    # FastAPI 0.160/python-multipart 0.0.32 coerce an omitted multipart bool to False even when
    # its declared default is True. Presence in the parsed form lets us preserve the API default
    # while still honoring an explicit `activate=false`.
    form = await request.form()
    activate_requested = activate if "activate" in form else True
    created = await service.create_version(
        slug=slug,
        filename=file.filename,
        content=content,
        notes=notes,
        uploaded_by=principal.label,
        activate=activate_requested,
        content_type=file.content_type,
    )
    version_id = created.version.id
    if settings.INGESTION_SYNC:
        await service.run_ingestion(version_id)
        response.status_code = status.HTTP_201_CREATED
    else:
        background.add_task(service.run_ingestion, version_id)
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
    responses={**E404, **E409},
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
    responses=E404,
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


@router.get(
    "/{version_number}/file",
    summary="Download the original uploaded file",
    description=(
        "Returns the exact bytes that were uploaded, from object storage (Supabase Storage). "
        "404 `FILE_NOT_STORED` when file storage was disabled at upload time; 502 "
        "`STORAGE_ERROR` when the storage backend is unreachable."
    ),
    response_class=Response,
    responses={
        200: {"content": {"text/plain": {}, "application/octet-stream": {}}},
        **E404,
        502: {"model": ErrorResponse},
    },
)
async def download_guideline_file(
    slug: str = SlugPath,
    version_number: int = VersionPath,
    _: Principal = Depends(require_agent),
    service: IngestionService = Depends(ingestion_service),
) -> Response:
    original = await service.get_original_file(slug, version_number)
    return Response(
        content=original.data,
        media_type=original.content_type,
        headers={
            "Content-Disposition": f'attachment; filename="{original.file_name}"',
            "X-Content-Type-Options": "nosniff",
        },
    )


@router.get(
    "/{version_number}/playbook",
    response_model=PlaybookOut,
    summary="Compiled playbook: intents + steps, company rules, facts, coverage",
    responses=E404,
)
async def get_playbook(
    slug: str = SlugPath,
    version_number: int = VersionPath,
    _: Principal = Depends(require_agent),
    service: IngestionService = Depends(ingestion_service),
) -> PlaybookOut:
    company, pb = await service.get_playbook(slug, version_number)
    v = pb.version
    return PlaybookOut(
        company=company.slug,
        version_number=v.version_number,
        compiler_model=v.compiler_model,
        intents=[
            IntentOut(
                key=i.key,
                name=i.name,
                description=i.description,
                conditions=list(i.conditions or []),
                steps=i.steps,
                reply_template=i.reply_template,
                placeholders=list(i.placeholders or []),
                source_section_ids=list(i.source_section_ids or []),
            )
            for i in pb.intents
        ],
        company_rules=dict(pb.rules.rules) if pb.rules else {},
        facts=[
            FactOut(
                id=f.id, title=f.title, text=f.text, source_section_ids=f.source_section_ids or []
            )
            for f in pb.facts
        ],
        sections=[SectionOut.model_validate(s, from_attributes=True) for s in pb.sections],
        coverage=v.coverage,
        validation_warnings=v.validation_warnings,
    )


@router.get(
    "/{version_number}/capability-coverage",
    summary="Capabilities the playbook references, and which are live vs manual (admin)",
    responses=E404,
)
async def capability_coverage(
    slug: str = SlugPath,
    version_number: int = VersionPath,
    _: Principal = Depends(require_admin),
    service: IntegrationService = Depends(integration_service),
) -> dict[str, Any]:
    return await service.coverage(slug, version_number)


@router.post(
    "/{version_number}/approve",
    response_model=GuidelineVersionOut,
    summary="Approve a ready version (admin); activates it if it was uploaded with activate=true",
    responses={**E404, **E409},
)
async def approve_guideline_version(
    slug: str = SlugPath,
    version_number: int = VersionPath,
    body: ReviewIn | None = Body(None),
    principal: Principal = Depends(require_admin),
    service: ReviewService = Depends(review_service),
) -> GuidelineVersionOut:
    company, version = await service.approve(
        slug, version_number, by=principal.label, note=body.note if body else None
    )
    return GuidelineVersionOut.build(version, company=company)


@router.post(
    "/{version_number}/reject",
    response_model=GuidelineVersionOut,
    summary="Reject a version (admin); rejected versions can't be activated",
    responses={**E404, **E409},
)
async def reject_guideline_version(
    slug: str = SlugPath,
    version_number: int = VersionPath,
    body: RejectIn | None = Body(None),
    principal: Principal = Depends(require_admin),
    service: ReviewService = Depends(review_service),
) -> GuidelineVersionOut:
    company, version = await service.reject(
        slug, version_number, by=principal.label, reason=body.reason if body else None
    )
    return GuidelineVersionOut.build(version, company=company)


@router.post(
    "/{version_number}/activate",
    response_model=GuidelineVersionOut,
    summary="Activate a ready + approved version (rollback / roll forward) (admin)",
    responses={**E404, **E409},
)
async def activate_guideline_version(
    slug: str = SlugPath,
    version_number: int = VersionPath,
    _: Principal = Depends(require_admin),
    service: ReviewService = Depends(review_service),
) -> GuidelineVersionOut:
    company, version = await service.activate(slug, version_number)
    return GuidelineVersionOut.build(version, company=company)


@router.delete(
    "/{version_number}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete an inactive version and its compiled rows + vectors (admin)",
    responses={**E404, **E409},
)
async def delete_guideline_version(
    slug: str = SlugPath,
    version_number: int = VersionPath,
    _: Principal = Depends(require_admin),
    service: ReviewService = Depends(review_service),
) -> Response:
    await service.delete(slug, version_number)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
