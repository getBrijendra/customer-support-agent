from __future__ import annotations

from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.api.deps import metrics_service
from app.core.security import Principal, require_admin, require_admin_header_or_query
from app.schemas.api import SLUG_PATTERN, MetricsSummary
from app.services.metrics_service import MetricsService

router = APIRouter(prefix="/internal", tags=["internal"])
templates = Jinja2Templates(directory=str(Path(__file__).resolve().parents[2] / "templates"))


@router.get(
    "/metrics/summary",
    response_model=MetricsSummary,
    summary="Usage, latency, error and ingestion metrics (admin)",
)
async def metrics_summary(
    from_: datetime | None = Query(None, alias="from", description="Default: now - 7 days"),
    to: datetime | None = Query(None, description="Default: now"),
    company: str | None = Query(None, pattern=SLUG_PATTERN),
    _: Principal = Depends(require_admin),
    service: MetricsService = Depends(metrics_service),
) -> MetricsSummary:
    data = await service.summary(start=from_, end=to, company=company)
    return MetricsSummary.model_validate(data)


@router.get(
    "/dashboard",
    response_class=HTMLResponse,
    summary="Internal HTML dashboard (admin; `?key=` or X-API-Key)",
)
async def dashboard(
    request: Request,
    from_: datetime | None = Query(None, alias="from"),
    to: datetime | None = Query(None),
    company: str | None = Query(None, pattern=SLUG_PATTERN),
    _: Principal = Depends(require_admin_header_or_query),
    service: MetricsService = Depends(metrics_service),
) -> HTMLResponse:
    data = MetricsSummary.model_validate(
        await service.summary(start=from_, end=to, company=company)
    ).model_dump(mode="json")
    return templates.TemplateResponse(
        request,
        "dashboard.html",
        {"summary": data, "company": company},
    )
